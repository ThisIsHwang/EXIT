#!/usr/bin/env python3
"""Fine-tune the EXIT Gemma classifier with the paper configuration.

The implementation uses the stable Transformers ``Trainer`` API and a small
completion-only collator.  This avoids TRL-version-specific constructor
arguments and ensures loss is computed only on the final ``Yes``/``No``
completion, as described in arXiv:2412.12559.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_id: str = "google/gemma-2b-it"
    model_revision: Optional[str] = None
    train_dataset: Optional[str] = None
    validation_dataset: Optional[str] = None
    output_dir: Optional[str] = None
    cache_dir: str = "./cache"
    resume_from_checkpoint: Optional[str] = None

    # Appendix A: LoRA configuration.
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Appendix A: optimizer and schedule configuration.
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    max_seq_length: int = 8192

    eval_steps: int = 300
    save_steps: int = 300
    save_total_limit: int = 3
    logging_steps: int = 10

    seed: int = 42
    deterministic: bool = True
    wandb_project: Optional[str] = None
    experiment_name: Optional[str] = None


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the EXIT relevance classifier")
    parser.add_argument("--model_id", default="google/gemma-2b-it")
    parser.add_argument("--model_revision")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument(
        "--validation_dataset",
        "--test_dataset",
        dest="validation_dataset",
        required=True,
        help="Validation Dataset saved by datasampling.py (--test_dataset is an alias)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--resume_from_checkpoint")

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=8192)

    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_false",
        dest="deterministic",
        help="Disable deterministic-algorithm requests (seeding remains enabled).",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument("--wandb_project")
    parser.add_argument("--experiment_name")
    return TrainingConfig(**vars(parser.parse_args()))


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG used here and request deterministic kernels when possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def setup_experiment_tracking(config: TrainingConfig) -> Optional[Any]:
    """Initialize W&B only when explicitly requested."""
    if not config.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "--wandb_project requires the optional 'wandb' package"
        ) from exc
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=asdict(config),
    )
    return wandb


def setup_tokenizer(config: TrainingConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def setup_model(config: TrainingConfig) -> Any:
    """Load the 4-bit fp16 base and apply LoRA exactly once."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Paper-faithful 4-bit bitsandbytes training requires a CUDA GPU"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # There is intentionally no peft_config passed to Trainer below: wrapping
    # here and then asking a trainer to wrap again creates nested/double PEFT.
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total,
    )
    return model


class CompletionOnlyDataCollator:
    """Tokenize rows and mask every target except the Yes/No completion."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _prefix(self, feature: Mapping[str, Any], label: str) -> str:
        prefix = feature.get("prompt_prefix")
        if prefix is not None:
            return str(prefix)
        prompt = str(feature.get("prompt", ""))
        if prompt.endswith(label):
            return prompt[: -len(label)]
        raise ValueError(
            "dataset row must contain prompt_prefix or prompt ending in label"
        )

    def _encode(self, feature: Mapping[str, Any]) -> Dict[str, List[int]]:
        label = str(feature["label"])
        if label not in {"Yes", "No"}:
            raise ValueError(f"unexpected label: {label!r}")

        prefix_ids = list(
            self.tokenizer(
                self._prefix(feature, label),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        completion_ids = list(
            self.tokenizer(
                label,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        if len(completion_ids) != 1:
            raise ValueError(
                f"EXIT requires {label!r} to be one classifier token, "
                f"but the tokenizer produced {completion_ids}"
            )
        prefix_budget = self.max_length - len(completion_ids)
        if prefix_budget < 1:
            raise ValueError("max_seq_length is too small for a completion")
        if len(prefix_ids) > prefix_budget:
            raise ValueError(
                f"training prompt is {len(prefix_ids) + len(completion_ids)} tokens, "
                f"exceeding max_seq_length={self.max_length}; EXIT training does "
                "not silently truncate full-document context"
            )

        input_ids = prefix_ids + completion_ids
        labels = [-100] * len(prefix_ids) + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        encoded = [self._encode(feature) for feature in features]
        batch_length = max(len(row["input_ids"]) for row in encoded)
        batch_size = len(encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad token")

        input_ids = torch.full((batch_size, batch_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, batch_length), dtype=torch.long)
        labels = torch.full((batch_size, batch_length), -100, dtype=torch.long)
        for index, row in enumerate(encoded):
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Construct arguments across the evaluation_strategy -> eval_strategy rename."""
    kwargs: Dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "optim": "paged_adamw_8bit",
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": ["wandb"] if config.wandb_project else [],
        "run_name": config.experiment_name,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    kwargs["eval_steps"] = config.eval_steps
    if "full_determinism" in parameters:
        kwargs["full_determinism"] = config.deterministic
    return TrainingArguments(**kwargs)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_training_manifest(
    config: TrainingConfig,
    resolved_model_revision: Optional[str] = None,
) -> None:
    """Record the complete run configuration and input dataset manifest hash."""
    output_dir = Path(str(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = Path(str(config.train_dataset)).resolve().parent / "manifest.json"
    manifest: Dict[str, Any] = {
        "paper": "https://arxiv.org/abs/2412.12559",
        "git_sha": _git_sha(),
        "config": asdict(config),
        "resolved_model_revision": resolved_model_revision,
        "implementation_choices_not_reported_by_paper": {
            "quantization_type": "nf4",
            "double_quantization": True,
            "lora_target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "loss_scope": "single Yes/No completion token",
            "prompt_overflow_policy": "error",
            "lr_scheduler": "transformers TrainingArguments default",
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "peft", "datasets", "bitsandbytes")
        },
        "dataset_manifest": {
            "path": str(data_manifest),
            "sha256": _file_sha256(data_manifest) if data_manifest.is_file() else None,
        },
    }
    with (output_dir / "training_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def train(config: TrainingConfig) -> None:
    if (
        not config.train_dataset
        or not config.validation_dataset
        or not config.output_dir
    ):
        raise ValueError(
            "train_dataset, validation_dataset, and output_dir are required"
        )
    seed_everything(config.seed, config.deterministic)
    wandb_module = setup_experiment_tracking(config)
    try:
        write_training_manifest(config)
        tokenizer = setup_tokenizer(config)
        model = setup_model(config)
        write_training_manifest(
            config,
            resolved_model_revision=getattr(model.config, "_commit_hash", None),
        )
        logger.info("Loading training dataset: %s", config.train_dataset)
        train_data = load_from_disk(config.train_dataset)
        logger.info("Loading validation dataset: %s", config.validation_dataset)
        validation_data = load_from_disk(config.validation_dataset)

        trainer = Trainer(
            model=model,
            args=build_training_arguments(config),
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=CompletionOnlyDataCollator(
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
            ),
        )
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

        final_path = Path(config.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info("Saved final adapter and tokenizer to %s", final_path)
    finally:
        if wandb_module is not None and wandb_module.run is not None:
            wandb_module.finish()


if __name__ == "__main__":
    train(parse_args())
