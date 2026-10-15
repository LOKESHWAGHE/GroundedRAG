import pytest

from app.services.reranking.service import rerank
from app.services.retrieval.hybrid import HybridResult


def make_hybrid(chunk_id, rrf_score=0.01):
    return HybridResult(chunk_id=chunk_id, rrf_score=rrf_score, hybrid_rank=chunk_id)


def test_rerank_reorders_by_cross_encoder_score_not_hybrid_order():
    candidates = [make_hybrid(1), make_hybrid(2), make_hybrid(3)]
    texts = {1: "irrelevant text", 2: "highly relevant text", 3: "somewhat relevant"}
    score_map = {"irrelevant text": 0.1, "highly relevant text": 0.9, "somewhat relevant": 0.5}

    def score_fn(pairs):
        return [score_map[chunk_text] for _, chunk_text in pairs]

    results = rerank("q", candidates, texts, top_m=3, score_fn=score_fn)
    assert [r.chunk_id for r in results] == [2, 3, 1]
    assert [r.rerank_rank for r in results] == [1, 2, 3]


def test_rerank_truncates_to_top_m():
    candidates = [make_hybrid(i) for i in range(1, 6)]
    texts = {i: f"text {i}" for i in range(1, 6)}
    score_fn = lambda pairs: [float(i) for i in range(len(pairs))]  # noqa: E731
    results = rerank("q", candidates, texts, top_m=2, score_fn=score_fn)
    assert len(results) == 2


def test_rerank_carries_hybrid_info_through_for_debug_panel():
    hybrid = HybridResult(chunk_id=1, rrf_score=0.5, dense_rank=2, bm25_rank=3, hybrid_rank=1)
    texts = {1: "some text"}
    results = rerank("q", [hybrid], texts, top_m=1, score_fn=lambda pairs: [0.7])
    assert results[0].hybrid.dense_rank == 2
    assert results[0].hybrid.bm25_rank == 3
    assert results[0].hybrid.rrf_score == 0.5


def test_rerank_empty_candidates_returns_empty():
    assert rerank("q", [], {}, top_m=5, score_fn=lambda pairs: []) == []


def test_rerank_mismatched_score_count_raises():
    candidates = [make_hybrid(1), make_hybrid(2)]
    texts = {1: "a", 2: "b"}
    with pytest.raises(ValueError):
        rerank("q", candidates, texts, top_m=2, score_fn=lambda pairs: [0.1])


def test_rerank_preserves_query_and_text_pairing_order_sent_to_score_fn():
    candidates = [make_hybrid(1), make_hybrid(2)]
    texts = {1: "first chunk", 2: "second chunk"}
    seen = {}

    def score_fn(pairs):
        seen["pairs"] = pairs
        return [1.0, 2.0]

    rerank("my query", candidates, texts, top_m=2, score_fn=score_fn)
    assert seen["pairs"] == [("my query", "first chunk"), ("my query", "second chunk")]
