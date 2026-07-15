import json

import pytest

from reproduce import (
    exact_match,
    get_answers,
    get_documents,
    get_query,
    load_examples,
    score_answer,
    token_f1,
)


def test_standard_qa_metrics_use_best_alias():
    assert exact_match("The Eiffel Tower!", "Eiffel Tower") == 1.0
    assert token_f1("red blue", "blue green") == pytest.approx(0.5)
    assert score_answer("NYC", ["New York", "NYC"]) == (1.0, 1.0)


def test_reproduction_input_schema_and_top_k(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Question?",
                    "answers": ["Answer"],
                    "ctxs": [
                        {"title": f"Doc {index}", "text": f"Text {index}"}
                        for index in range(7)
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    example = load_examples(path)[0]
    assert get_query(example) == "Question?"
    assert get_answers(example) == ("Answer",)
    assert len(get_documents(example, top_k=5)) == 5


def test_reproduction_rejects_incomplete_top_k_and_normalizes_null_title():
    example = {
        "question": "Question?",
        "ctxs": [{"title": None, "text": f"Text {index}"} for index in range(5)],
    }

    assert get_documents(example, top_k=5)[0].title == ""
    with pytest.raises(ValueError, match="Top-20"):
        get_documents(example, top_k=20)
