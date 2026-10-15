from dataclasses import dataclass

from app.services.retrieval.hybrid import fuse_with_rrf, hybrid_search


@dataclass
class SC:
    chunk_id: int
    score: float


def test_doc_ranked_first_by_both_retrievers_wins():
    dense = [SC(1, 0.9), SC(2, 0.8), SC(3, 0.7)]
    bm25 = [SC(1, 12.0), SC(3, 8.0), SC(2, 5.0)]
    fused = fuse_with_rrf(dense, bm25, rrf_k=60)
    assert fused[0].chunk_id == 1
    assert fused[0].hybrid_rank == 1


def test_doc_found_by_only_one_retriever_still_included():
    dense = [SC(1, 0.9)]
    bm25 = [SC(2, 10.0)]
    fused = fuse_with_rrf(dense, bm25, rrf_k=60)
    ids = {r.chunk_id for r in fused}
    assert ids == {1, 2}
    assert fused[0].rrf_score == fused[1].rrf_score


def test_rrf_formula_is_exact():
    dense = [SC(1, 0.9), SC(2, 0.5)]
    bm25 = [SC(2, 5.0), SC(1, 1.0)]
    fused = {r.chunk_id: r for r in fuse_with_rrf(dense, bm25, rrf_k=60)}
    assert abs(fused[1].rrf_score - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(fused[2].rrf_score - (1 / 62 + 1 / 61)) < 1e-9


def test_dense_and_bm25_ranks_are_exposed_for_the_debug_panel():
    dense = [SC(5, 0.9), SC(7, 0.4)]
    bm25 = [SC(7, 9.0), SC(5, 1.0)]
    fused = {r.chunk_id: r for r in fuse_with_rrf(dense, bm25, rrf_k=60)}
    assert fused[5].dense_rank == 1 and fused[5].dense_score == 0.9
    assert fused[5].bm25_rank == 2 and fused[5].bm25_score == 1.0
    assert fused[7].dense_rank == 2 and fused[7].bm25_rank == 1


def test_lower_rrf_k_rewards_top_ranks_more_sharply():
    dense = [SC(1, 0.9), SC(2, 0.8)]
    bm25: list[SC] = []
    fused_small_k = {r.chunk_id: r.rrf_score for r in fuse_with_rrf(dense, bm25, rrf_k=1)}
    fused_large_k = {r.chunk_id: r.rrf_score for r in fuse_with_rrf(dense, bm25, rrf_k=1000)}
    gap_small_k = fused_small_k[1] - fused_small_k[2]
    gap_large_k = fused_large_k[1] - fused_large_k[2]
    assert gap_small_k > gap_large_k


def test_empty_retrievers_yield_empty_fusion():
    assert fuse_with_rrf([], [], rrf_k=60) == []


def test_one_retriever_empty_still_returns_the_other():
    dense = [SC(1, 0.9), SC(2, 0.5)]
    fused = fuse_with_rrf(dense, [], rrf_k=60)
    assert {r.chunk_id for r in fused} == {1, 2}
    assert all(r.bm25_rank is None and r.bm25_score is None for r in fused)


def test_hybrid_search_truncates_to_top_n_and_uses_injected_retrievers():
    def fake_dense(query, top_k):
        return [SC(i, 1.0 / i) for i in range(1, 11)]

    def fake_bm25(query, top_k):
        return [SC(i, float(11 - i)) for i in range(1, 11)]

    results = hybrid_search("q", hybrid_top_n=3, rrf_k=60, dense_fn=fake_dense, bm25_fn=fake_bm25)
    assert len(results) == 3
    assert [r.hybrid_rank for r in results] == [1, 2, 3]


def test_results_sorted_descending_by_rrf_score():
    dense = [SC(1, 0.9), SC(2, 0.8), SC(3, 0.1)]
    bm25 = [SC(3, 9.0), SC(2, 1.0)]
    fused = fuse_with_rrf(dense, bm25, rrf_k=60)
    scores = [r.rrf_score for r in fused]
    assert scores == sorted(scores, reverse=True)
