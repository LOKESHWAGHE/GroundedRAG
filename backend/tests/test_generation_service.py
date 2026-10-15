from dataclasses import dataclass

from app.services.generation.service import generate_answer


@dataclass
class FakeReranked:
    chunk_id: int
    text: str


def meta(doc_id="d1", filename="handbook.pdf", page=1):
    return {"document_id": doc_id, "filename": filename, "page_number": page, "section_title": None}


def test_generate_answer_resolves_markers_to_real_chunk_metadata():
    reranked = [FakeReranked(101, "Product X supports PostgreSQL and MySQL.")]
    chunk_meta = {101: meta(filename="handbook.pdf", page=7)}

    def fake_llm(system, user):
        assert "handbook.pdf" in user
        return "Product X supports PostgreSQL. [1]"

    result = generate_answer("Does X support PostgreSQL?", reranked, chunk_meta, llm_fn=fake_llm)
    assert result.insufficient_evidence is False
    assert len(result.model_claimed_citations) == 1
    citation = result.model_claimed_citations[0]
    assert citation.chunk_id == 101
    assert citation.filename == "handbook.pdf"
    assert citation.page_number == 7


def test_insufficient_evidence_response_yields_no_citations():
    reranked = [FakeReranked(1, "unrelated text")]
    chunk_meta = {1: meta()}

    def fake_llm(system, user):
        return "INSUFFICIENT_EVIDENCE: The documents don't cover this."

    result = generate_answer("unrelated question", reranked, chunk_meta, llm_fn=fake_llm)
    assert result.insufficient_evidence is True
    assert result.model_claimed_citations == []


def test_hallucinated_marker_is_dropped_not_trusted():
    reranked = [FakeReranked(1, "only one chunk")]
    chunk_meta = {1: meta()}

    def fake_llm(system, user):
        return "An answer citing a passage that doesn't exist. [1][7]"

    result = generate_answer("q", reranked, chunk_meta, llm_fn=fake_llm)
    cited_chunk_ids = {c.chunk_id for c in result.model_claimed_citations}
    assert cited_chunk_ids == {1}


def test_no_context_available_still_calls_llm_and_parses_response():
    def fake_llm(system, user):
        assert "No context passages were retrieved" in user
        return "INSUFFICIENT_EVIDENCE: no documents indexed"

    result = generate_answer("q", [], {}, llm_fn=fake_llm)
    assert result.insufficient_evidence is True
    assert result.context.chunks == []


def test_system_prompt_is_passed_to_llm_fn():
    seen = {}

    def fake_llm(system, user):
        seen["system"] = system
        return "answer [1]"

    generate_answer("q", [FakeReranked(1, "text")], {1: meta()}, llm_fn=fake_llm)
    assert "ONLY" in seen["system"]
    assert "INSUFFICIENT_EVIDENCE" in seen["system"]
