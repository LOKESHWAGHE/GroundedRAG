from app.services.verification.claim_extraction import extract_claims


def test_sentence_mode_one_claim_per_sentence():
    claims = extract_claims("First fact. [1] Second fact. [2]", mode="sentence")
    assert [c.text for c in claims] == ["First fact.", "Second fact."]
    assert [c.claim_id for c in claims] == ["claim_1", "claim_2"]
    assert claims[0].source_markers == frozenset({1})
    assert claims[1].source_markers == frozenset({2})


def test_llm_mode_decomposes_sentence_into_multiple_claims_inheriting_markers():
    def fake_llm(system, user):
        return '["The company was founded in 1998.", "The company has offices in Pune."]'

    claims = extract_claims(
        "The company was founded in 1998 and has offices in Pune. [1]", mode="llm", llm_fn=fake_llm
    )
    assert len(claims) == 2
    assert all(c.source_markers == frozenset({1}) for c in claims)
    assert claims[0].claim_id == "claim_1" and claims[1].claim_id == "claim_2"


def test_llm_mode_falls_back_gracefully_per_sentence():
    def failing_llm(system, user):
        raise RuntimeError("boom")

    claims = extract_claims("A single fact. [1]", mode="llm", llm_fn=failing_llm)
    assert len(claims) == 1
    assert claims[0].text == "A single fact."


def test_claim_ids_are_sequential_across_multiple_sentences_in_llm_mode():
    def fake_llm(system, user):
        return '["atom a", "atom b"]'

    claims = extract_claims("First sentence. [1] Second sentence. [2]", mode="llm", llm_fn=fake_llm)
    assert [c.claim_id for c in claims] == ["claim_1", "claim_2", "claim_3", "claim_4"]


def test_empty_answer_yields_no_claims():
    assert extract_claims("", mode="sentence") == []
