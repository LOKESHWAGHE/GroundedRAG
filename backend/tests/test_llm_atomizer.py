from app.services.verification.llm_atomizer import atomize_claims_llm


def test_parses_valid_json_array():
    def fake_llm(system, user):
        return '["Claim A.", "Claim B."]'

    assert atomize_claims_llm("Original sentence.", fake_llm) == ["Claim A.", "Claim B."]


def test_strips_markdown_code_fence():
    def fake_llm(system, user):
        return '```json\n["Claim A.", "Claim B."]\n```'

    assert atomize_claims_llm("s", fake_llm) == ["Claim A.", "Claim B."]


def test_falls_back_to_sentence_on_malformed_json():
    def fake_llm(system, user):
        return "not json at all"

    assert atomize_claims_llm("Original sentence.", fake_llm) == ["Original sentence."]


def test_falls_back_to_sentence_when_llm_raises():
    def failing_llm(system, user):
        raise RuntimeError("API down")

    assert atomize_claims_llm("Original sentence.", failing_llm) == ["Original sentence."]


def test_falls_back_when_json_is_not_a_list_of_strings():
    def fake_llm(system, user):
        return '[{"not": "a string"}]'

    assert atomize_claims_llm("Original sentence.", fake_llm) == ["Original sentence."]


def test_empty_array_falls_back_to_sentence():
    def fake_llm(system, user):
        return "[]"

    assert atomize_claims_llm("Original sentence.", fake_llm) == ["Original sentence."]


def test_strips_whitespace_and_drops_empty_claims():
    def fake_llm(system, user):
        return '["  Claim A.  ", "", "Claim B."]'

    assert atomize_claims_llm("s", fake_llm) == ["Claim A.", "Claim B."]
