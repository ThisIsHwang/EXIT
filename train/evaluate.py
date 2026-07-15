#!/usr/bin/env python3
"""Evaluate an EXIT LoRA adapter as a binary Yes/No classifier."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

try:  # ``python -m train.evaluate``
    from .datasampling import generate_prompt
except ImportError:  # ``python train/evaluate.py``
    from datasampling import generate_prompt


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExitEvaluator:
    """Load a base causal LM plus EXIT adapter and score Yes against No."""

    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        device: str = "auto",
        cache_dir: str = "./cache",
        load_in_4bit: bool = True,
        base_revision: str | None = None,
        checkpoint_revision: str | None = None,
    ) -> None:
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        use_cuda = torch.cuda.is_available() and device != "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        if load_in_4bit and not use_cuda:
            raise ValueError("4-bit bitsandbytes evaluation requires CUDA")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=cache_dir,
            revision=base_revision,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "torch_dtype": torch.float16 if use_cuda else torch.float32,
        }
        if use_cuda:
            model_kwargs["device_map"] = "auto"
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model: %s", base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            revision=base_revision,
            **model_kwargs,
        )
        logger.info("Loading adapter: %s", checkpoint_path)
        self.model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            revision=checkpoint_revision,
        )
        if not use_cuda:
            self.model.to("cpu")
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")

    def _single_token_id(self, label: str) -> int:
        token_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"{label!r} is {len(token_ids)} tokens for this tokenizer; "
                "EXIT binary next-token scoring requires one token per label"
            )
        return int(token_ids[0])

    @torch.inference_mode()
    def predict_batch(
        self,
        queries: Sequence[str],
        contexts: Sequence[str],
        sentences: Sequence[str],
        threshold: float = 0.5,
    ) -> Tuple[List[str], List[float]]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not (len(queries) == len(contexts) == len(sentences)):
            raise ValueError("queries, contexts, and sentences must have equal length")
        prompts = [
            generate_prompt(query, context, sentence)
            for query, context, sentence in zip(queries, contexts, sentences)
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        limits = [
            limit
            for limit in (model_limit, tokenizer_limit)
            if isinstance(limit, int) and 0 < limit < 10**9
        ]
        max_input_tokens = min(limits) if limits else None
        lengths = inputs["attention_mask"].sum(dim=1)
        if max_input_tokens is not None and torch.any(lengths > max_input_tokens):
            raise ValueError(
                f"classifier prompt exceeds the model limit of {max_input_tokens}; "
                "evaluation never silently truncates document context"
            )
        inputs = inputs.to(self.input_device)
        logits = self.model(**inputs).logits[:, -1, :]
        binary_logits = logits[:, [self.yes_token_id, self.no_token_id]].float()
        yes_probabilities = torch.softmax(binary_logits, dim=-1)[:, 0]
        probabilities = yes_probabilities.detach().cpu().tolist()
        predictions = [
            "Yes" if probability >= threshold else "No" for probability in probabilities
        ]
        return predictions, [float(probability) for probability in probabilities]


def _metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=["Yes", "No"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=["Yes", "No"])
    return {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(
    base_model_path: str,
    checkpoint_path: str,
    validation_data: Dataset,
    threshold: float = 0.5,
    batch_size: int = 16,
    *,
    device: str = "auto",
    cache_dir: str = "./cache",
    load_in_4bit: bool = True,
    base_revision: str | None = None,
    checkpoint_revision: str | None = None,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(validation_data) == 0:
        raise ValueError("validation dataset is empty")
    evaluator = ExitEvaluator(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        device=device,
        cache_dir=cache_dir,
        load_in_4bit=load_in_4bit,
        base_revision=base_revision,
        checkpoint_revision=checkpoint_revision,
    )

    y_true: List[str] = []
    y_pred: List[str] = []
    yes_probabilities: List[float] = []
    sample_types: List[str] = []
    for start in tqdm(range(0, len(validation_data), batch_size), desc="evaluate"):
        stop = min(start + batch_size, len(validation_data))
        batch = validation_data.select(range(start, stop))
        predictions, probabilities = evaluator.predict_batch(
            batch["query"],
            batch["full_passage"],
            batch["sentence_text"],
            threshold=threshold,
        )
        y_true.extend(str(label) for label in batch["label"])
        y_pred.extend(predictions)
        yes_probabilities.extend(probabilities)
        if "sample_type" in batch.column_names:
            sample_types.extend(str(value) for value in batch["sample_type"])
        else:
            sample_types.extend("unknown" for _ in predictions)

    results: Dict[str, Any] = {
        "threshold": threshold,
        "count": len(y_true),
        **_metrics(y_true, y_pred),
        "by_sample_type": {},
        "predictions": [
            {
                "label": true,
                "prediction": predicted,
                "yes_probability": probability,
                "sample_type": sample_type,
            }
            for true, predicted, probability, sample_type in zip(
                y_true,
                y_pred,
                yes_probabilities,
                sample_types,
            )
        ],
    }
    for sample_type in sorted(set(sample_types)):
        indices = [
            index for index, value in enumerate(sample_types) if value == sample_type
        ]
        results["by_sample_type"][sample_type] = _metrics(
            [y_true[index] for index in indices],
            [y_pred[index] for index in indices],
        )
        results["by_sample_type"][sample_type]["count"] = len(indices)
    return results


def save_results(
    results: Mapping[str, Any],
    output_dir: str,
    dataset_name: str,
) -> Tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_threshold_{float(results['threshold']):.2f}"
    json_path = destination / f"{stem}.json"
    text_path = destination / f"{stem}.txt"

    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    predictions = results["predictions"]
    y_true = [row["label"] for row in predictions]
    y_pred = [row["prediction"] for row in predictions]
    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"Dataset: {dataset_name}\n")
        handle.write(f"Threshold: {results['threshold']}\n\n")
        handle.write(
            classification_report(
                y_true,
                y_pred,
                labels=["Yes", "No"],
                zero_division=0,
            )
        )
        handle.write("\nConfusion matrix (rows/columns: Yes, No):\n")
        handle.write(json.dumps(results["confusion_matrix"]))
        handle.write("\n")
    logger.info("Saved %s and %s", json_path, text_path)
    return json_path, text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EXIT LoRA adapter")
    parser.add_argument("--base_model", default="google/gemma-2b-it")
    parser.add_argument("--base_revision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_revision")
    parser.add_argument(
        "--validation_dataset",
        "--test_dataset",
        dest="validation_dataset",
        required=True,
        help="Validation Dataset path (--test_dataset is a compatibility alias)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument(
        "--no_4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable the paper's 4-bit evaluation profile.",
    )
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger.info("Loading validation data: %s", args.validation_dataset)
    validation_data = load_from_disk(args.validation_dataset)
    results = evaluate_model(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        validation_data=validation_data,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        base_revision=args.base_revision,
        checkpoint_revision=args.checkpoint_revision,
    )
    dataset_name = args.dataset_name or Path(args.validation_dataset).name
    save_results(results, args.output_dir, dataset_name)
    report = results["classification_report"]
    logger.info(
        "Yes F1 %.4f | No F1 %.4f | accuracy %.4f",
        report["Yes"]["f1-score"],
        report["No"]["f1-score"],
        report["accuracy"],
    )


if __name__ == "__main__":
    main()
