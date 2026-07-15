import pytest

from compressors.baselines.exit.core import (
    EXITDocument,
    assemble_compression,
    build_compression_prompt,
    build_qa_prompt,
    prepare_candidates,
)


def test_prompt_templates_match_paper_fields():
    compression_prompt = build_compression_prompt("q", "document", "sentence")
    assert "Query:\nq" in compression_prompt
    assert "Full context:\ndocument" in compression_prompt
    assert "Sentence:\nsentence" in compression_prompt
    assert compression_prompt.endswith("<start_of_turn>model\n")

    qa_prompt = build_qa_prompt("q", "context")
    assert qa_prompt.startswith("Context information is below.")
    assert "\ncontext\n" in qa_prompt
    assert qa_prompt.endswith("Query: q\nAnswer:")


def test_candidates_use_only_their_containing_document():
    documents = [
        EXITDocument("a", "A", "A1. A2."),
        EXITDocument("b", "B", "B1. B2."),
    ]
    candidates = prepare_candidates(documents, lambda text: text.split(" "))

    assert [candidate.context for candidate in candidates] == [
        "A\nA1. A2.",
        "A\nA1. A2.",
        "B\nB1. B2.",
        "B\nB1. B2.",
    ]


def test_thresholding_keeps_original_document_and_sentence_order():
    documents = [
        EXITDocument("a", "", "A1. A2."),
        EXITDocument("b", "", "B1. B2."),
    ]
    candidates = prepare_candidates(documents, lambda text: text.split(" "))
    result = assemble_compression(candidates, [0.8, 0.1, 0.5, 0.9], 0.5)

    assert result.text == "A1.\n\nB1. B2."
    assert result.selections == (True, False, True, True)


def test_thresholding_allows_empty_output_and_validates_scores():
    candidates = prepare_candidates(
        [EXITDocument("a", "", "Only sentence.")],
        lambda text: [text],
    )
    assert assemble_compression(candidates, [0.1], 0.5).text == ""
    with pytest.raises(ValueError, match="between 0 and 1"):
        assemble_compression(candidates, [1.1], 0.5)
