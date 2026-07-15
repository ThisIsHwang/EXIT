import re

import pytest
import torch

from compressors.base import SearchResult
from compressors.baselines.exit.compressor import EXITCompressor


def split_sentences(text):
    return [part for part in re.split(r"(?<=\.)\s+", text) if part]


def test_compress_batches_document_local_context_and_preserves_order():
    compressor = EXITCompressor.__new__(EXITCompressor)
    compressor.batch_size = 3
    compressor.threshold = 0.5
    compressor.sentence_splitter = split_sentences
    captured_batches = []

    def fake_predict_batch(queries, contexts, sentences):
        captured_batches.append(list(zip(queries, contexts, sentences)))
        probabilities = torch.tensor(
            [
                [
                    0.9 if sentence in {"A1.", "B1."} else 0.1,
                    0.1 if sentence in {"A1.", "B1."} else 0.9,
                ]
                for sentence in sentences
            ]
        )
        predictions = [
            "Yes" if probability[0] >= 0.5 else "No" for probability in probabilities
        ]
        return predictions, probabilities

    compressor._predict_batch = fake_predict_batch
    # Fragments of one source document need not be adjacent.
    documents = [
        SearchResult(evi_id=1, docid=11, title="Doc A", text="A1.", score=1.0),
        SearchResult(evi_id=2, docid=22, title="Doc B", text="B1.", score=1.0),
        SearchResult(evi_id=1, docid=11, title="Doc A", text="A2.", score=1.0),
        SearchResult(evi_id=2, docid=22, title="Doc B", text="B2.", score=1.0),
    ]

    result = compressor.compress("Question?", documents)
    captured = [item for batch in captured_batches for item in batch]

    assert len(captured_batches) == 2
    assert [context for _, context, _ in captured] == [
        "Doc A\nA1. A2.",
        "Doc A\nA1. A2.",
        "Doc B\nB1. B2.",
        "Doc B\nB1. B2.",
    ]
    assert result[0].text == "A1.\n\nB1."


def test_predict_batch_rejects_overlong_prompt_without_truncating():
    class FakeTokenizer:
        def __call__(self, prompts, **kwargs):
            batch_size = len(prompts)
            assert kwargs["truncation"] is False
            return {
                "input_ids": torch.ones((batch_size, 6), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 6), dtype=torch.long),
            }

    compressor = EXITCompressor.__new__(EXITCompressor)
    compressor.tokenizer = FakeTokenizer()
    compressor.max_input_tokens = 5
    compressor.device = torch.device("cpu")
    compressor.model = None
    compressor.yes_token_id = 1
    compressor.no_token_id = 2

    with pytest.raises(ValueError, match="exceeding the model limit of 5"):
        compressor._predict_batch(["q"], ["context"], ["sentence"])


def test_empty_retrieval_returns_no_result():
    compressor = EXITCompressor.__new__(EXITCompressor)
    assert compressor.compress("Question?", []) == []
