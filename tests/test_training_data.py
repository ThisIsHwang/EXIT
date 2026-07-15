from collections import Counter

from compressors.baselines.exit.core import build_compression_prompt
from train.datasampling import (
    create_balanced_dataset,
    generate_prompt,
    process_example,
    split_queries,
)


def hotpot_example(index):
    title = f"Supporting {index}"
    return {
        "_id": f"query-{index}",
        "question": f"Question {index}?",
        "supporting_facts": [[title, 0]],
        "context": [
            [title, [f"Evidence {index}.", f"Hard negative {index}."]],
            [f"Distractor {index}", [f"Distractor sentence {index}."]],
        ],
    }


def test_query_split_is_stable_and_has_no_leakage():
    examples = [hotpot_example(index) for index in range(6)]
    train_rows, validation_rows = split_queries(examples, validation_size=2, seed=42)
    reversed_train, reversed_validation = split_queries(
        list(reversed(examples)), validation_size=2, seed=42
    )

    validation_ids = {row["_id"] for row in validation_rows}
    assert validation_ids == {row["_id"] for row in reversed_validation}
    assert {row["_id"] for row in train_rows}.isdisjoint(validation_ids)
    assert {row["_id"] for row in train_rows} == {row["_id"] for row in reversed_train}


def test_balancing_is_two_to_one_to_one_with_cross_query_negatives():
    processed = [process_example(hotpot_example(index)) for index in range(4)]
    assert processed[0]["contexts"][0] == ("Supporting 0\nEvidence 0. Hard negative 0.")
    rows = create_balanced_dataset(processed, seed=7, split_name="test")
    repeated = create_balanced_dataset(processed, seed=7, split_name="test")

    assert Counter(row["sample_type"] for row in rows) == {
        "positive": 4,
        "hard_negative": 2,
        "random_negative": 2,
    }
    assert [row["sample_id"] for row in rows] == [row["sample_id"] for row in repeated]
    assert all(
        row["query_id"] != row["source_query_id"]
        for row in rows
        if row["sample_type"] == "random_negative"
    )


def test_training_and_inference_share_the_exact_prompt():
    expected = build_compression_prompt("Question?", "Document.", "Sentence.")
    assert generate_prompt("Question?", "Document.", "Sentence.") == expected
