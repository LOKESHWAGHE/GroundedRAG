"""
API-layer tests for /api/v1/chat/*. These patch app.api.v1.chat.run_query_pipeline
directly -- the pipeline's own logic (retrieval, generation, verification,
regeneration) is already covered end-to-end in test_pipeline.py with injected
fakes. This file tests the boundary the pipeline tests don't reach: request
validation, response serialization, the retrieval-debug join logic, and
QueryRecord persistence/retrieval.
"""

from app.services.generation.context_selector import ContextChunk
from app.services.reranking.service import RerankedResult
from app.services.rag.pipeline import QueryResult, RetrievalTrace
from app.services.retrieval.hybrid import HybridResult
from app.services.verification.citation_verification import ClaimVerdict, SupportingEvidence, Verdict
from app.services.verification.claim_extraction import Claim
from app.services.verification.groundedness import compute_groundedness

BASE = "/api/v1/chat"


def make_fake_result(insufficient=False, with_claims=True):
    hybrid = [
        HybridResult(chunk_id=1, rrf_score=0.03, dense_rank=1, dense_score=0.9, bm25_rank=2, bm25_score=4.0, hybrid_rank=1),
        HybridResult(chunk_id=2, rrf_score=0.01, dense_rank=3, dense_score=0.4, bm25_rank=None, bm25_score=None, hybrid_rank=2),
    ]
    reranked = [RerankedResult(chunk_id=1, rerank_score=5.0, rerank_rank=1, text="Product X supports PostgreSQL.", hybrid=hybrid[0])]

    if insufficient:
        return QueryResult(
            answer="INSUFFICIENT_EVIDENCE: not covered.", insufficient_evidence=True,
            insufficient_reason="not covered.", claims=[], groundedness=compute_groundedness([]),
            regeneration_count=0, trace=RetrievalTrace(hybrid=hybrid, reranked=reranked), context_chunk_ids=[],
        )

    claims = []
    if with_claims:
        claim = Claim(claim_id="claim_1", text="Product X supports PostgreSQL.", source_markers=frozenset({1}))
        evidence = SupportingEvidence(
            chunk_id=1, document_id="doc-1", filename="handbook.pdf", page_number=3,
            quote="Product X supports PostgreSQL.", entailment_score=0.95, contradiction_score=0.0,
        )
        claims = [ClaimVerdict(claim=claim, verdict=Verdict.SUPPORTED, support_score=0.95, evidence=evidence)]

    return QueryResult(
        answer="Product X supports PostgreSQL. [1]", insufficient_evidence=False, insufficient_reason=None,
        claims=claims, groundedness=compute_groundedness(claims), regeneration_count=0,
        trace=RetrievalTrace(hybrid=hybrid, reranked=reranked), context_chunk_ids=[1],
    )


def patch_pipeline(monkeypatch, result):
    import app.api.v1.chat as chat_module

    def fake_run_query_pipeline(question, chunk_meta_fn, **kwargs):
        chunk_meta_fn([1, 2])  # exercise the real DB-backed chunk_meta_fn so filenames resolve
        return result

    monkeypatch.setattr(chat_module, "run_query_pipeline", fake_run_query_pipeline)


def seed_chunk(db_session):
    """Minimal document+chunk rows so chunk_meta_fn can resolve chunk_id 1 and 2."""
    import uuid
    from app.models.document import Chunk, Document, DocumentStatus

    doc = Document(
        id=uuid.uuid4(), filename="handbook.pdf", file_type="pdf", size_bytes=100,
        sha256="x" * 64, stored_path="/tmp/x.pdf", status=DocumentStatus.INDEXED.value,
        page_count=1, char_count=100, chunk_count=2,
    )
    db_session.add(doc)
    db_session.flush()
    db_session.add_all([
        Chunk(id=1, document_id=doc.id, chunk_index=0, page_number=3, text="Product X supports PostgreSQL.", token_count=4),
        Chunk(id=2, document_id=doc.id, chunk_index=1, page_number=5, text="The office is in Mumbai.", token_count=5),
    ])
    db_session.commit()


def test_query_returns_answer_and_groundedness(client, db_session, monkeypatch):
    seed_chunk(db_session)
    patch_pipeline(monkeypatch, make_fake_result())

    r = client.post(f"{BASE}/query", json={"question": "Does X support PostgreSQL?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Product X supports PostgreSQL. [1]"
    assert body["insufficient_evidence"] is False
    assert body["groundedness"]["groundedness_score"] == 1.0
    assert len(body["claims"]) == 1
    assert body["claims"][0]["verdict"] == "SUPPORTED"
    assert body["claims"][0]["evidence"]["filename"] == "handbook.pdf"


def test_retrieval_debug_rows_resolve_filenames_and_mark_final_context(client, db_session, monkeypatch):
    seed_chunk(db_session)
    patch_pipeline(monkeypatch, make_fake_result())

    r = client.post(f"{BASE}/query", json={"question": "q"})
    rows = r.json()["retrieval_debug"]
    assert len(rows) == 2
    row1 = next(row for row in rows if row["chunk_id"] == 1)
    assert row1["filename"] == "handbook.pdf"
    assert row1["dense_rank"] == 1
    assert row1["bm25_rank"] == 2
    assert row1["hybrid_rank"] == 1
    assert row1["rerank_rank"] == 1
    assert row1["in_final_context"] is True

    row2 = next(row for row in rows if row["chunk_id"] == 2)
    assert row2["rerank_rank"] is None  # not in the reranked top-M
    assert row2["in_final_context"] is False


def test_insufficient_evidence_query_returns_empty_claims(client, db_session, monkeypatch):
    seed_chunk(db_session)
    patch_pipeline(monkeypatch, make_fake_result(insufficient=True))

    r = client.post(f"{BASE}/query", json={"question": "unrelated"})
    body = r.json()
    assert body["insufficient_evidence"] is True
    assert body["claims"] == []
    assert body["groundedness"]["groundedness_score"] is None


def test_empty_question_is_rejected(client):
    r = client.post(f"{BASE}/query", json={"question": ""})
    assert r.status_code == 422


def test_overlong_question_is_rejected(client):
    r = client.post(f"{BASE}/query", json={"question": "x" * 1001})
    assert r.status_code == 422


def test_query_is_persisted_and_retrievable_by_id(client, db_session, monkeypatch):
    seed_chunk(db_session)
    patch_pipeline(monkeypatch, make_fake_result())

    posted = client.post(f"{BASE}/query", json={"question": "Does X support PostgreSQL?"}).json()
    fetched = client.get(f"{BASE}/{posted['id']}").json()
    assert fetched["answer"] == posted["answer"]
    assert fetched["question"] == "Does X support PostgreSQL?"


def test_query_appears_in_history(client, db_session, monkeypatch):
    seed_chunk(db_session)
    patch_pipeline(monkeypatch, make_fake_result())

    client.post(f"{BASE}/query", json={"question": "First question"})
    client.post(f"{BASE}/query", json={"question": "Second question"})

    history = client.get(f"{BASE}/history").json()
    assert len(history) == 2
    assert {h["question"] for h in history} == {"First question", "Second question"}
    assert history[0]["status"] in ("answered", "insufficient_evidence")


def test_unknown_query_id_returns_404(client):
    r = client.get(f"{BASE}/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_malformed_query_id_returns_422(client):
    r = client.get(f"{BASE}/not-a-uuid")
    assert r.status_code == 422
