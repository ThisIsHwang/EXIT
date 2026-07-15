"""Pure EXIT algorithm primitives shared by training and inference.

This module deliberately has no model or GPU dependencies.  Keeping prompt
construction and document reassembly here prevents the quickstart, evaluator,
and reusable compressor from drifting away from the paper implementation.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable, Sequence, Tuple, TypeVar


COMPRESSION_PROMPT_TEMPLATE = """<start_of_turn>user
Query:
{query}
Full context:
{context}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
<start_of_turn>model
"""

QA_PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query. Do not provide any explanation.
Query: {query}
Answer:"""


@dataclass(frozen=True)
class EXITDocument:
    """One retrieved source document before sentence decomposition."""

    document_id: object
    title: str
    text: str


@dataclass(frozen=True)
class SentenceCandidate:
    """A sentence together with its containing-document context."""

    document_index: int
    sentence_index: int
    context: str
    sentence: str


@dataclass(frozen=True)
class CompressionResult:
    """Detailed result of thresholding EXIT relevance scores."""

    text: str
    sentences: Tuple[str, ...]
    document_indices: Tuple[int, ...]
    sentence_indices: Tuple[int, ...]
    selections: Tuple[bool, ...]
    scores: Tuple[float, ...]
    selected_sentences: Tuple[str, ...]


def build_compression_prompt(query: str, context: str, sentence: str) -> str:
    """Render the compression prompt from Table 6 of the paper."""

    return COMPRESSION_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
        sentence=sentence,
    )


def build_qa_prompt(query: str, context: str) -> str:
    """Render the reader prompt from Table 7 of the paper."""

    return QA_PROMPT_TEMPLATE.format(query=query, context=context)


def validate_threshold(threshold: float) -> None:
    """Validate a sentence-retention threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")


def prepare_candidates(
    documents: Sequence[EXITDocument],
    sentence_splitter: Callable[[str], Iterable[str]],
) -> Tuple[SentenceCandidate, ...]:
    """Split documents and create document-local classification candidates.

    Every sentence from a document receives the same full source document as
    its ``Full context``.  Context from other Top-k documents is never mixed in.
    """

    candidates = []
    for document_index, document in enumerate(documents):
        document_text = document.text.strip()
        if not document_text:
            continue

        title = document.title.strip()
        context = f"{title}\n{document_text}" if title else document_text
        sentences = [
            sentence.strip()
            for sentence in sentence_splitter(document_text)
            if sentence and sentence.strip()
        ]
        candidates.extend(
            SentenceCandidate(
                document_index=document_index,
                sentence_index=sentence_index,
                context=context,
                sentence=sentence,
            )
            for sentence_index, sentence in enumerate(sentences)
        )

    return tuple(candidates)


def assemble_compression(
    candidates: Sequence[SentenceCandidate],
    scores: Sequence[float],
    threshold: float,
) -> CompressionResult:
    """Select and reassemble sentences in source-document order."""

    validate_threshold(threshold)
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")

    normalized_scores = tuple(float(score) for score in scores)
    if any(
        not isfinite(score) or not 0.0 <= score <= 1.0 for score in normalized_scores
    ):
        raise ValueError("relevance scores must be finite values between 0 and 1")

    selections = tuple(score >= threshold for score in normalized_scores)
    selected_by_document = {}
    for candidate, selected in zip(candidates, selections):
        if selected:
            selected_by_document.setdefault(candidate.document_index, []).append(
                candidate.sentence
            )

    compressed_documents = [
        " ".join(selected_by_document[document_index])
        for document_index in sorted(selected_by_document)
    ]
    selected_sentences = tuple(
        candidate.sentence
        for candidate, selected in zip(candidates, selections)
        if selected
    )

    return CompressionResult(
        text="\n\n".join(compressed_documents),
        sentences=tuple(candidate.sentence for candidate in candidates),
        document_indices=tuple(candidate.document_index for candidate in candidates),
        sentence_indices=tuple(candidate.sentence_index for candidate in candidates),
        selections=selections,
        scores=normalized_scores,
        selected_sentences=selected_sentences,
    )


T = TypeVar("T")


def batched(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    """Yield stable, contiguous batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
