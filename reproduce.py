#!/usr/bin/env python3
"""Reproduce EXIT QA metrics from pre-retrieved paper-format datasets.

The runner intentionally starts after retrieval.  It expects the CompAct-style
``question``/``answers``/``ctxs`` artifacts described in the README, then
measures EXIT compression plus reader generation.  This matches the latency
boundary used by the paper tables; rebuilding the December 2018 Wikipedia
Contriever index remains a separate retrieval step.
"""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import statistics
import string
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from exit_rag import Document, ExitRAG


PAPER_SPLITS = {
    "nq": "dev",
    "triviaqa": "test",
    "hotpotqa": "dev",
    "2wiki": "dev",
}


def normalize_answer(text: str) -> str:
    """Use the standard SQuAD-style normalization used for QA EM/F1."""

    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    from collections import Counter

    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)

    overlap = Counter(prediction_tokens) & Counter(answer_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(
    prediction: str, answers: Sequence[str]
) -> Tuple[Optional[float], Optional[float]]:
    if not answers:
        return None, None
    return (
        max(exact_match(prediction, answer) for answer in answers),
        max(token_f1(prediction, answer) for answer in answers),
    )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: Path) -> List[Dict]:
    """Load JSON, JSONL, or a top-level ``{"data": [...]}`` artifact."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as input_file:
            return [json.loads(line) for line in input_file if line.strip()]

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError("input must be a JSON list, JSONL file, or {'data': [...]} object")


def get_query(example: Dict) -> str:
    query = example.get("question", example.get("query"))
    if not isinstance(query, str) or not query.strip():
        raise ValueError("every example needs a non-empty 'question' or 'query'")
    return query.strip()


def get_answers(example: Dict) -> Tuple[str, ...]:
    answers = example.get("answers", example.get("answer", ()))
    if isinstance(answers, str):
        return (answers,)
    if isinstance(answers, list) and all(isinstance(answer, str) for answer in answers):
        return tuple(answers)
    if answers in (None, ()):
        return ()
    raise ValueError("'answers' must be a string or list of strings")


def get_documents(example: Dict, top_k: int) -> List[Document]:
    contexts = example.get("ctxs")
    if not isinstance(contexts, list):
        raise ValueError("every example needs a 'ctxs' list of retrieved documents")
    if len(contexts) < top_k:
        raise ValueError(
            f"example has {len(contexts)} retrieved documents, but Top-{top_k} "
            "evaluation requires at least that many"
        )

    documents = []
    for context in contexts[:top_k]:
        if not isinstance(context, dict) or not isinstance(context.get("text"), str):
            raise ValueError("each retrieved context needs a string 'text' field")
        documents.append(
            Document(
                title=str(context.get("title") or ""),
                text=context["text"],
                score=float(context.get("score", 1.0)),
            )
        )
    return documents


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolved_revision(model) -> Optional[str]:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def resolved_adapter_revision(model) -> Optional[str]:
    """Return an adapter commit only when PEFT exposes one explicitly.

    ``PeftModel.config`` is the base-model config, so reading its commit hash
    would incorrectly report the base revision as the adapter revision.
    """

    peft_configs = getattr(model, "peft_config", {})
    for config in peft_configs.values():
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
    return None


def package_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EXIT on pre-retrieved NQ/TQA/HQA/2Wiki artifacts"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(PAPER_SPLITS), required=True)
    parser.add_argument("--split", help="Defaults to the paper split for the dataset")
    parser.add_argument("--top-k", type=int, choices=(5, 20), default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--compression-base-model", default="google/gemma-2b-it")
    parser.add_argument("--compression-model", default="doubleyyh/exit-gemma-2b")
    parser.add_argument("--reader-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--compression-base-revision")
    parser.add_argument("--compression-revision")
    parser.add_argument("--reader-revision")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("input contains no examples")

    # Validate the full selected slice before spending time loading GPU models.
    for example in examples:
        get_query(example)
        get_answers(example)
        get_documents(example, args.top_k)
    if args.validate_only:
        print(
            f"Validated {len(examples)} {args.dataset}/{args.split or PAPER_SPLITS[args.dataset]} examples"
        )
        return

    rag = ExitRAG(
        compression_base_model=args.compression_base_model,
        compression_model=args.compression_model,
        reader_model=args.reader_model,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        cache_dir=args.cache_dir,
        compression_base_revision=args.compression_base_revision,
        compression_revision=args.compression_revision,
        reader_revision=args.reader_revision,
        load_compressor_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    if args.warmup_runs:
        warmup_example = examples[0]
        warmup_query = get_query(warmup_example)
        warmup_documents = get_documents(warmup_example, args.top_k)
        for _ in range(args.warmup_runs):
            rag.run_rag(
                warmup_query,
                warmup_documents,
                compression_threshold=args.threshold,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    all_compression_times = []
    all_reading_times = []
    all_total_times = []
    exact_matches = []
    f1_scores = []

    for index, example in enumerate(examples):
        query = get_query(example)
        answers = get_answers(example)
        documents = get_documents(example, args.top_k)
        runs = [
            rag.run_rag(query, documents, compression_threshold=args.threshold)
            for _ in range(args.latency_repeats)
        ]
        result = runs[0]
        compression_times = [run["compression_time"] for run in runs]
        reading_times = [run["reading_time"] for run in runs]
        total_times = [run["total_time"] for run in runs]
        em, f1 = score_answer(result["answer"], answers)
        if em is not None:
            exact_matches.append(em)
            f1_scores.append(f1)

        all_compression_times.extend(compression_times)
        all_reading_times.extend(reading_times)
        all_total_times.extend(total_times)
        record = {
            "index": index,
            "id": example.get("id", example.get("_id", index)),
            "query": query,
            "answers": answers,
            "prediction": result["answer"],
            "exact_match": em,
            "f1": f1,
            "compressed_context": result["compressed_context"],
            "sentence_selections": result["sentence_selections"],
            "relevance_scores": result["relevance_scores"],
            "sentences": result["sentences"],
            "sentence_document_indices": result["sentence_document_indices"],
            "sentence_indices": result["sentence_indices"],
            "original_tokens": result["original_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "compression_latency": latency_summary(compression_times),
            "reading_latency": latency_summary(reading_times),
            "total_latency": latency_summary(total_times),
        }
        records.append(record)

    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    original_tokens = [record["original_tokens"] for record in records]
    compressed_tokens = [record["compressed_tokens"] for record in records]
    summary = {
        "paper": "arXiv:2412.12559v3",
        "git_sha": git_sha(),
        "dataset": args.dataset,
        "split": args.split or PAPER_SPLITS[args.dataset],
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "examples": len(records),
        "top_k": args.top_k,
        "metrics_percent": {
            "exact_match": 100.0 * mean(exact_matches),
            "f1": 100.0 * mean(f1_scores),
            "scored_examples": len(exact_matches),
        },
        "tokens": {
            "original_mean": mean(original_tokens),
            "compressed_mean": mean(compressed_tokens),
            "retained_fraction": (
                sum(compressed_tokens) / sum(original_tokens)
                if sum(original_tokens)
                else 0.0
            ),
        },
        "latency_seconds": {
            "compression": latency_summary(all_compression_times),
            "reading": latency_summary(all_reading_times),
            "total": latency_summary(all_total_times),
            "repeats": args.latency_repeats,
            "warmup_runs": args.warmup_runs,
            "backend": "transformers",
            "paper_backend": "vLLM 0.5.5",
        },
        "models": {
            "compression_base": args.compression_base_model,
            "compression_adapter": args.compression_model,
            "reader": args.reader_model,
            "requested_revisions": {
                "compression_base": args.compression_base_revision,
                "compression_adapter": args.compression_revision,
                "reader": args.reader_revision,
            },
            "resolved_revisions": {
                "compression_base": resolved_revision(rag.compressor.base_model),
                "compression_adapter": resolved_adapter_revision(rag.compressor.model),
                "reader": resolved_revision(rag.reader),
            },
        },
        "inference": {
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "compressor_4bit": not args.no_4bit,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpus": [
                torch.cuda.get_device_name(device_index)
                for device_index in range(torch.cuda.device_count())
            ],
            "packages": package_versions(
                ["transformers", "peft", "accelerate", "bitsandbytes", "spacy", "vllm"]
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "notes": [
            "Retrieval latency is excluded, matching the paper's compression+reader tables.",
            "Transformers latency must not be compared directly with the paper's vLLM 0.5.5 latency.",
            "The paper does not publish exact dataset/index/model revisions; pin and record them for strict reruns.",
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
