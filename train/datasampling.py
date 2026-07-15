#!/usr/bin/env python3
"""Build the paper-faithful HotpotQA classifier dataset for EXIT.

The EXIT paper (arXiv:2412.12559) uses a 2:1:1 mixture of positive,
hard-negative, and random-negative sentences.  A random negative is *not* an
ordinary negative sentence for the same question: it is a sentence and its
passage taken from a different query and paired with the current query.

This module deliberately keeps data construction independent of model and
tokenizer packages.  Hugging Face ``datasets`` is imported only when saving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from compressors.baselines.exit.core import (
        COMPRESSION_PROMPT_TEMPLATE,
        build_compression_prompt,
    )
except ModuleNotFoundError:  # Support ``python train/datasampling.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compressors.baselines.exit.core import (
        COMPRESSION_PROMPT_TEMPLATE,
        build_compression_prompt,
    )


PAPER_URL = "https://arxiv.org/abs/2412.12559"
DATASET_SCHEMA_VERSION = 1

COMPRESSION_INSTRUCTION_TEMPLATE = (
    "Query:\n{query}\n"
    "Full context:\n{original_passage}\n"
    "Sentence:\n{sentence}\n"
    'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct the EXIT 2:1:1 classifier dataset from HotpotQA"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to hotpot_train_v1.1.json",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory for train_dataset, validation_dataset, and manifest.json",
    )
    parser.add_argument(
        "--validation_size",
        "--test_size",
        dest="validation_size",
        type=int,
        default=1000,
        help=(
            "Number of whole queries assigned to validation (default: 1000). "
            "--test_size remains as a deprecated alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_titles",
        action="store_true",
        help="Do not prefix each original passage with its HotpotQA title.",
    )
    return parser.parse_args()


def load_hotpotqa(file_path: str) -> List[Dict[str, Any]]:
    """Load and minimally validate a HotpotQA JSON array."""
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("HotpotQA input must be a JSON array of query objects")
    return data


def format_compression_instruction(query: str, context: str, sentence: str) -> str:
    """Render the compression prompt body exactly as printed in the paper."""
    return COMPRESSION_INSTRUCTION_TEMPLATE.format(
        query=query.strip(),
        original_passage=context.strip(),
        sentence=sentence.strip(),
    )


def generate_prompt(query: str, context: str, sentence: str) -> str:
    """Render the paper prompt inside Gemma's user/model turn wrapper."""
    return build_compression_prompt(query.strip(), context.strip(), sentence.strip())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _query_id(example: Mapping[str, Any]) -> str:
    """Return HotpotQA's id, with a content hash fallback for custom exports."""
    raw_id = example.get("_id", example.get("id"))
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    identity = {
        "question": example.get("question"),
        "context": example.get("context"),
        "supporting_facts": example.get("supporting_facts"),
    }
    return "sha256:" + _sha256_bytes(_canonical_json(identity))


def _derive_seed(seed: int, namespace: str) -> int:
    payload = f"EXIT:{seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_queries(
    data: Sequence[Mapping[str, Any]],
    validation_size: int,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Deterministically split whole queries, never sentence rows.

    Ranking ids by a seeded SHA-256 key makes the assignment reproducible
    across Python versions and independent of the input file's row order.
    """
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if validation_size >= len(data):
        raise ValueError("validation_size must be smaller than the dataset")

    identified = [(_query_id(example), example) for example in data]
    ids = [query_id for query_id, _ in identified]
    duplicates = [query_id for query_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:3])
        raise ValueError(f"duplicate query ids in input: {preview}")

    def split_key(item: Tuple[str, Mapping[str, Any]]) -> Tuple[str, str]:
        query_id, _ = item
        digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).hexdigest()
        return digest, query_id

    ranked = sorted(identified, key=split_key)
    validation_ids = {query_id for query_id, _ in ranked[:validation_size]}
    train = [
        example for query_id, example in identified if query_id not in validation_ids
    ]
    validation = [
        example for query_id, example in identified if query_id in validation_ids
    ]
    return train, validation


def process_example(
    example: Mapping[str, Any],
    include_titles: bool = True,
) -> Dict[str, Any]:
    """Extract positive, hard-negative, and random-negative source candidates.

    Labels retain the historical numeric convention (2 positive, 1 hard
    negative, 0 other passage).  Label-0 rows are only a *source pool* here;
    they become paper-style random negatives only after being paired with a
    different query in :func:`create_balanced_dataset`.
    """
    question = str(example["question"]).strip()
    query_id = _query_id(example)
    supporting_facts = {
        (str(title), int(sentence_index))
        for title, sentence_index in example["supporting_facts"]
    }
    supporting_titles = {title for title, _ in supporting_facts}

    queries: List[str] = []
    sentences: List[str] = []
    contexts: List[str] = []
    labels: List[int] = []
    titles: List[str] = []
    sentence_indices: List[int] = []

    for raw_title, raw_sentences in example["context"]:
        title = str(raw_title)
        document_sentences = [str(sentence) for sentence in raw_sentences]
        # HotpotQA exports often retain leading whitespace in each sentence,
        # but whitespace-normalized mirrors do not.  Reinsert an explicit
        # boundary so the full-document context is valid in both forms.
        passage_body = " ".join(
            sentence.strip() for sentence in document_sentences if sentence.strip()
        )
        context = f"{title}\n{passage_body}" if include_titles else passage_body

        for sentence_index, raw_sentence in enumerate(document_sentences):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            if (title, sentence_index) in supporting_facts:
                label = 2
            elif title in supporting_titles:
                label = 1
            else:
                label = 0

            queries.append(question)
            sentences.append(sentence)
            contexts.append(context)
            labels.append(label)
            titles.append(title)
            sentence_indices.append(sentence_index)

    return {
        "query_id": query_id,
        "question": question,
        "queries": queries,
        "sentences": sentences,
        "contexts": contexts,
        "labels": labels,
        "titles": titles,
        "sentence_indices": sentence_indices,
    }


Candidate = Tuple[Dict[str, Any], int]


def _candidate_rows(processed: Dict[str, Any]) -> Iterable[Candidate]:
    for index in range(len(processed["sentences"])):
        yield processed, index


def _make_sample(
    target: Dict[str, Any],
    source: Dict[str, Any],
    source_index: int,
    sample_type: str,
) -> Dict[str, Any]:
    label = "Yes" if sample_type == "positive" else "No"
    query = target["question"]
    context = source["contexts"][source_index]
    sentence = source["sentences"][source_index]
    prompt_prefix = generate_prompt(query, context, sentence)
    identity = {
        "query_id": target["query_id"],
        "source_query_id": source["query_id"],
        "title": source["titles"][source_index],
        "sentence_index": source["sentence_indices"][source_index],
        "sample_type": sample_type,
    }
    return {
        "sample_id": _sha256_bytes(_canonical_json(identity))[:24],
        "query_id": target["query_id"],
        "source_query_id": source["query_id"],
        "source_title": source["titles"][source_index],
        "source_sentence_index": source["sentence_indices"][source_index],
        "query": query,
        "full_passage": context,
        "sentence_text": sentence,
        "sample_type": sample_type,
        "label": label,
        "prompt_prefix": prompt_prefix,
        "prompt": prompt_prefix + label,
    }


def _take_unrelated_sources(
    pool: List[Candidate],
    targets: Sequence[Candidate],
) -> List[Candidate]:
    """Take distinct source rows whose query id differs from each target id."""
    selected: List[Candidate] = []
    for target, _ in targets:
        match_index = len(pool) - 1
        while (
            match_index >= 0 and pool[match_index][0]["query_id"] == target["query_id"]
        ):
            match_index -= 1
        if match_index < 0:
            raise ValueError(
                "cannot construct random negatives: no sentence from a different query remains"
            )
        selected.append(pool.pop(match_index))
    return selected


def create_balanced_dataset(
    examples: Sequence[Dict[str, Any]],
    positive_ratio: float = 0.5,
    *,
    seed: int = 42,
    split_name: str = "train",
) -> List[Dict[str, Any]]:
    """Create an exact 2:1:1 Pos/H-Neg/Neg dataset.

    ``positive_ratio`` remains for source compatibility but the paper fixes it
    at 0.5 (two positives out of every four rows).
    """
    if positive_ratio != 0.5:
        raise ValueError("EXIT's paper configuration fixes positive_ratio at 0.5")
    if len(examples) < 2:
        raise ValueError("at least two queries are required to form random negatives")

    positives: List[Candidate] = []
    hard_negatives: List[Candidate] = []
    random_source_pool: List[Candidate] = []
    for processed in examples:
        for candidate in _candidate_rows(processed):
            label = processed["labels"][candidate[1]]
            random_source_pool.append(candidate)
            if label == 2:
                positives.append(candidate)
            elif label == 1:
                hard_negatives.append(candidate)

    negative_count = min(
        len(positives) // 2,
        len(hard_negatives),
        len(random_source_pool),
    )
    if negative_count == 0:
        raise ValueError("split has insufficient positive or hard-negative sentences")

    rng = random.Random(_derive_seed(seed, split_name))
    rng.shuffle(positives)
    rng.shuffle(hard_negatives)
    rng.shuffle(random_source_pool)

    # Truncating at most one positive is expected for an odd positive count and
    # makes the integer ratio exact.  If hard negatives are scarce, all three
    # categories are reduced together rather than oversampling duplicates.
    selected_positives = positives[: 2 * negative_count]
    selected_hard_negatives = hard_negatives[:negative_count]
    random_targets = rng.sample(selected_positives, negative_count)
    selected_random_sources = _take_unrelated_sources(
        random_source_pool,
        random_targets,
    )

    samples = [
        _make_sample(record, record, index, "positive")
        for record, index in selected_positives
    ]
    samples.extend(
        _make_sample(record, record, index, "hard_negative")
        for record, index in selected_hard_negatives
    )
    samples.extend(
        _make_sample(target, source, source_index, "random_negative")
        for (target, _), (source, source_index) in zip(
            random_targets,
            selected_random_sources,
        )
    )
    rng.shuffle(samples)

    counts = Counter(sample["sample_type"] for sample in samples)
    expected = {
        "positive": 2 * negative_count,
        "hard_negative": negative_count,
        "random_negative": negative_count,
    }
    if dict(counts) != expected:
        raise AssertionError(f"unexpected sample counts: {dict(counts)} != {expected}")
    if any(
        sample["query_id"] == sample["source_query_id"]
        for sample in samples
        if sample["sample_type"] == "random_negative"
    ):
        raise AssertionError("random negative was paired with its own query")
    return samples


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        {
            "sample_id": row["sample_id"],
            "query_id": row["query_id"],
            "source_query_id": row["source_query_id"],
            "sample_type": row["sample_type"],
            "label": row["label"],
        }
        for row in rows
    ]
    return _sha256_bytes(_canonical_json(identities))


def _query_digest(examples: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_json(sorted(_query_id(example) for example in examples))
    )


def _split_manifest(
    raw_examples: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    counts = Counter(row["sample_type"] for row in rows)
    random_pairs_valid = all(
        row["query_id"] != row["source_query_id"]
        for row in rows
        if row["sample_type"] == "random_negative"
    )
    return {
        "query_count": len(raw_examples),
        "query_ids_sha256": _query_digest(raw_examples),
        "row_count": len(rows),
        "rows_sha256": _row_digest(rows),
        "sample_counts": {
            "positive": counts["positive"],
            "hard_negative": counts["hard_negative"],
            "random_negative": counts["random_negative"],
        },
        "random_negative_query_mismatch_verified": random_pairs_valid,
    }


def build_manifest(
    *,
    dataset_path: str,
    seed: int,
    validation_size: int,
    include_titles: bool,
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    datasets_version: str,
) -> Dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "paper": {
            "url": PAPER_URL,
            "sampling_ratio": {
                "positive": 2,
                "hard_negative": 1,
                "random_negative": 1,
            },
        },
        "source": {
            "path": os.path.normpath(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "query_count": len(train_examples) + len(validation_examples),
        },
        "construction": {
            "seed": seed,
            "validation_size_queries": validation_size,
            "split_unit": "query",
            "split_algorithm": "ascending sha256(seed + NUL + query_id)",
            "include_titles": include_titles,
            "random_negative_rule": "query_id != source_query_id",
            "sentence_boundaries": (
                "HotpotQA-provided sentence arrays, retained to align with "
                "supporting-fact indices"
            ),
        },
        "prompt": {
            "compression_template": COMPRESSION_PROMPT_TEMPLATE,
            "sha256": _sha256_bytes(COMPRESSION_PROMPT_TEMPLATE.encode("utf-8")),
        },
        "splits": {
            "train": _split_manifest(train_examples, train_rows),
            "validation": _split_manifest(validation_examples, validation_rows),
        },
        "software": {
            "python": platform.python_version(),
            "datasets": datasets_version,
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    data = load_hotpotqa(args.dataset_path)
    train_raw, validation_raw = split_queries(data, args.validation_size, args.seed)

    include_titles = not args.no_titles
    print(f"Processing {len(train_raw):,} train queries...")
    train_processed = [process_example(row, include_titles) for row in train_raw]
    print(f"Processing {len(validation_raw):,} validation queries...")
    validation_processed = [
        process_example(row, include_titles) for row in validation_raw
    ]

    train_rows = create_balanced_dataset(
        train_processed,
        seed=args.seed,
        split_name="train",
    )
    validation_rows = create_balanced_dataset(
        validation_processed,
        seed=args.seed,
        split_name="validation",
    )

    try:
        import datasets
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Saving requires the 'datasets' package; install the project requirements"
        ) from exc

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).save_to_disk(str(save_dir / "train_dataset"))
    Dataset.from_list(validation_rows).save_to_disk(
        str(save_dir / "validation_dataset")
    )

    manifest = build_manifest(
        dataset_path=args.dataset_path,
        seed=args.seed,
        validation_size=args.validation_size,
        include_titles=include_titles,
        train_examples=train_raw,
        validation_examples=validation_raw,
        train_rows=train_rows,
        validation_rows=validation_rows,
        datasets_version=datasets.__version__,
    )
    _write_json(save_dir / "manifest.json", manifest)

    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))
    print(f"Saved dataset and manifest to {save_dir}")


if __name__ == "__main__":
    main()
