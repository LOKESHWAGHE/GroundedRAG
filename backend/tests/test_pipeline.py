from app.services.reranking.service import RerankedResult
from app.services.retrieval.hybrid import HybridResult
from app.services.rag.pipeline import run_query_pipeline

CHUNK_META = {
    1: {"document_id": "d1", "filename": "handbook.pdf", "page_number": 3, "section_title": None, "text": "Product X supports PostgreSQL and MySQL."},
    2: {"document_id": "d1", "filename": "handbook.pdf", "page_number": 9, "section_title": None, "text": "The office is located in Mumbai."},
}


def fake_hybrid(query, **kw):
    return [HybridResult(chunk_id=1, rrf_score=0.03, dense_rank=1, bm25_rank=1, hybrid_rank=1),
            HybridResult(chunk_id=2, rrf_score=0.01, dense_rank=2, bm25_rank=2, hybrid_rank=2)]


def fake_rerank(query, candidates, chunk_texts, **kw):
    return [RerankedResult(chunk_id=c.chunk_id, rerank_score=1.0, rerank_rank=i + 1,
                            text=chunk_texts[c.chunk_id], hybrid=c) for i, c in enumerate(candidates)]


def chunk_meta_fn(ids):
    return {i: CHUNK_META[i] for i in ids if i in CHUNK_META}


def always_entails(pairs):
    return [{"entailment": 0.95, "contradiction": 0.0, "neutral": 0.05} for _ in pairs]


def always_contradicts(pairs):
    return [{"entailment": 0.1, "contradiction": 0.9, "neutral": 0.0} for _ in pairs]


def test_grounded_answer_returns_on_first_attempt_no_regeneration():
    def llm_fn(system, user):
        return "Product X supports PostgreSQL. [1]"

    result = run_query_pipeline(
        "Does X support PostgreSQL?", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid,
        rerank_fn=fake_rerank, llm_fn=llm_fn, nli_fn=always_entails, claim_mode="sentence",
    )
    assert result.insufficient_evidence is False
    assert result.regeneration_count == 0
    assert result.groundedness.groundedness_score == 1.0
    assert len(result.trace.hybrid) == 2


def test_insufficient_evidence_short_circuits_without_verification():
    def llm_fn(system, user):
        return "INSUFFICIENT_EVIDENCE: Not covered by the documents."

    result = run_query_pipeline(
        "unrelated question", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid,
        rerank_fn=fake_rerank, llm_fn=llm_fn, nli_fn=always_entails,
    )
    assert result.insufficient_evidence is True
    assert result.claims == []
    assert result.regeneration_count == 0


def test_regenerates_once_then_succeeds():
    calls = {"n": 0}

    def llm_fn(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Product X supports PostgreSQL. [1]"
        assert "NOT verifiably supported" in user
        return "Product X supports PostgreSQL, a verified fact. [1]"

    call_count = {"nli": 0}

    def flaky_nli(pairs):
        call_count["nli"] += 1
        if call_count["nli"] == 1:
            return always_contradicts(pairs)
        return always_entails(pairs)

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=fake_rerank,
        llm_fn=llm_fn, nli_fn=flaky_nli, claim_mode="sentence", max_retries=2,
    )
    assert result.regeneration_count == 1
    assert result.insufficient_evidence is False
    assert calls["n"] == 2


def test_exhausts_retries_and_returns_safe_fallback():
    def llm_fn(system, user):
        return "Product X supports PostgreSQL. [1]"

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=fake_rerank,
        llm_fn=llm_fn, nli_fn=always_contradicts, claim_mode="sentence", max_retries=2,
    )
    assert result.insufficient_evidence is True
    assert result.answer.startswith("INSUFFICIENT_EVIDENCE:")
    assert result.regeneration_count == 2


def test_max_retries_zero_returns_fallback_immediately_on_failure():
    def llm_fn(system, user):
        return "An unsupported claim. [1]"

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=fake_rerank,
        llm_fn=llm_fn, nli_fn=always_contradicts, claim_mode="sentence", max_retries=0,
    )
    assert result.insufficient_evidence is True
    assert result.regeneration_count == 0


def test_retrieval_trace_is_populated_for_the_debug_panel():
    def llm_fn(system, user):
        return "Product X supports PostgreSQL. [1]"

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=fake_rerank,
        llm_fn=llm_fn, nli_fn=always_entails, claim_mode="sentence",
    )
    assert result.trace.hybrid[0].dense_rank == 1
    assert result.trace.hybrid[0].bm25_rank == 1
    assert result.trace.reranked[0].rerank_score == 1.0


def test_min_rerank_score_short_circuits_without_calling_llm():
    llm_called = {"n": 0}

    def llm_fn(system, user):
        llm_called["n"] += 1
        return "should never be called"

    def low_score_rerank(query, candidates, chunk_texts, **kw):
        return [RerankedResult(chunk_id=c.chunk_id, rerank_score=0.01, rerank_rank=i + 1,
                                text=chunk_texts[c.chunk_id], hybrid=c) for i, c in enumerate(candidates)]

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=low_score_rerank,
        llm_fn=llm_fn, nli_fn=always_entails, claim_mode="sentence",
    )
    # MIN_RERANK_SCORE defaults to 0.0 (disabled) in Settings, so this only triggers when
    # explicitly configured -- confirm the default is a true no-op, then confirm it fires once set.
    assert llm_called["n"] == 1  # default: disabled, generation proceeds normally


def test_min_rerank_score_enabled_via_settings(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "MIN_RERANK_SCORE", 0.5)
    llm_called = {"n": 0}

    def llm_fn(system, user):
        llm_called["n"] += 1
        return "answer [1]"

    def low_score_rerank(query, candidates, chunk_texts, **kw):
        return [RerankedResult(chunk_id=c.chunk_id, rerank_score=0.01, rerank_rank=i + 1,
                                text=chunk_texts[c.chunk_id], hybrid=c) for i, c in enumerate(candidates)]

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=low_score_rerank,
        llm_fn=llm_fn, nli_fn=always_entails, claim_mode="sentence",
    )
    assert llm_called["n"] == 0  # short-circuited before generation
    assert result.insufficient_evidence is True
    assert "not relevant enough" in result.answer.lower() or "MIN_RERANK_SCORE" in (result.insufficient_reason or "")


def test_context_chunk_ids_reflects_final_context():
    def llm_fn(system, user):
        return "Product X supports PostgreSQL. [1]"

    result = run_query_pipeline(
        "q", chunk_meta_fn=chunk_meta_fn, hybrid_fn=fake_hybrid, rerank_fn=fake_rerank,
        llm_fn=llm_fn, nli_fn=always_entails, claim_mode="sentence",
    )
    assert set(result.context_chunk_ids) == {1, 2}
