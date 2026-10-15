from app.services.generation.parser import parse_llm_response


def test_plain_answer_with_single_citation():
    parsed = parse_llm_response("Product X supports PostgreSQL. [1]")
    assert parsed.insufficient_evidence is False
    assert parsed.cited_markers == {1}


def test_multiple_citations_on_one_sentence():
    parsed = parse_llm_response("It supports both databases. [1][3]")
    assert parsed.cited_markers == {1, 3}


def test_citations_across_multiple_sentences_are_unioned():
    parsed = parse_llm_response("First fact. [1] Second fact. [2]")
    assert parsed.cited_markers == {1, 2}


def test_insufficient_evidence_prefix_is_detected():
    parsed = parse_llm_response("INSUFFICIENT_EVIDENCE: The documents do not mention pricing.")
    assert parsed.insufficient_evidence is True
    assert parsed.insufficient_reason == "The documents do not mention pricing."
    assert parsed.cited_markers == set()


def test_insufficient_evidence_with_no_reason_gets_a_default():
    parsed = parse_llm_response("INSUFFICIENT_EVIDENCE:")
    assert parsed.insufficient_evidence is True
    assert parsed.insufficient_reason == "Not specified by the model."


def test_answer_with_no_citations_has_empty_marker_set():
    parsed = parse_llm_response("This is an answer with no markers.")
    assert parsed.cited_markers == set()
    assert parsed.insufficient_evidence is False


def test_leading_trailing_whitespace_is_stripped():
    parsed = parse_llm_response("   Answer text. [1]   \n")
    assert parsed.answer == "Answer text. [1]"
