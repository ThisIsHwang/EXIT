import torch
import pytest

from compressors.base import SearchResult
from compressors.baselines.exit.compressor import EXITCompressor


def test_compress_uses_containing_document_context():
    compressor = EXITCompressor.__new__(EXITCompressor)
    compressor.batch_size = 8
    compressor.threshold = 0.5
    captured = []

    def fake_predict_batch(queries, contexts, sentences):
        captured.extend(zip(queries, contexts, sentences))
        probabilities = torch.tensor(
            [
                [0.9 if sentence in {"A1.", "B1."} else 0.1,
                 0.1 if sentence in {"A1.", "B1."} else 0.9]
                for sentence in sentences
            ]
        )
        predictions = [
            "Yes" if probability[0] >= 0.5 else "No"
            for probability in probabilities
        ]
        return predictions, probabilities

    compressor._predict_batch = fake_predict_batch
    documents = [
        SearchResult(evi_id=1, docid=11, title="Doc A", text="A1.", score=1.0),
        SearchResult(evi_id=1, docid=11, title="Doc A", text="A2.", score=1.0),
        SearchResult(evi_id=2, docid=22, title="Doc B", text="B1.", score=1.0),
        SearchResult(evi_id=2, docid=22, title="Doc B", text="B2.", score=1.0),
    ]

    result = compressor.compress("Question?", documents)

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
