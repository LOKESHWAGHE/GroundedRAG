"""
Hybrid retrieval: run dense and BM25 independently, then fuse with Reciprocal
Rank Fusion (RRF) rather than averaging their raw scores.

Why RRF and not a weighted score average: cosine similarity (roughly [0,1])
and a raw BM25 score (unbounded, query-dependent) live on incompatible
scales, so `0.6 * dense_score + 0.4 * bm25_score` would silently let whichever
retriever happens to produce larger numbers dominate. RRF sidesteps that by
fusing *rankings*, not raw scores:

    RRF_score(d) = sum over retrievers i of  1 / (k + rank_i(d))

A document ranked #1 by one retriever and absent from the other still gets a
meaningful score; `k` (config: RRF_K, default 60 — the value from the original
RRF paper) controls how much weight a top rank carries versus a middling one:
a larger k flattens the curve, a smaller k rewards top ranks more sharply.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.retrieval.dense import ScoredChunkId as DenseScoredChunkId
from app.services.retrieval.dense import dense_search
from app.services.retrieval.sparse import ScoredChunkId as BM25ScoredChunkId
from app.services.retrieval.sparse import bm25_search

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    chunk_id: int
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    hybrid_rank: int = field(default=0)  # 1-indexed, assigned after sorting


def _rank_map(results) -> dict[int, int]:
    """1-indexed rank per chunk_id, in the order the retriever returned them."""
    return {r.chunk_id: i + 1 for i, r in enumerate(results)}


def _score_map(results) -> dict[int, float]:
    return {r.chunk_id: r.score for r in results}


def fuse_with_rrf(
    dense_results: list[DenseScoredChunkId], bm25_results: list[BM25ScoredChunkId], rrf_k: int
) -> list[HybridResult]:
    """Pure fusion logic, separated from the retrieval calls so it's directly unit-testable."""
    dense_rank, dense_score = _rank_map(dense_results), _score_map(dense_results)
    bm25_rank, bm25_score = _rank_map(bm25_results), _score_map(bm25_results)

    all_ids = set(dense_rank) | set(bm25_rank)
    fused: list[HybridResult] = []
    for cid in all_ids:
        rrf = 0.0
        if cid in dense_rank:
            rrf += 1.0 / (rrf_k + dense_rank[cid])
        if cid in bm25_rank:
            rrf += 1.0 / (rrf_k + bm25_rank[cid])
        fused.append(
            HybridResult(
                chunk_id=cid,
                rrf_score=rrf,
                dense_rank=dense_rank.get(cid),
                dense_score=dense_score.get(cid),
                bm25_rank=bm25_rank.get(cid),
                bm25_score=bm25_score.get(cid),
            )
        )
    # Tie-break deterministically on chunk_id so results are stable across runs/tests.
    fused.sort(key=lambda r: (-r.rrf_score, r.chunk_id))
    for i, r in enumerate(fused):
        r.hybrid_rank = i + 1
    return fused


def hybrid_search(
    query: str,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
    hybrid_top_n: int | None = None,
    rrf_k: int | None = None,
    dense_fn: Callable[..., list[DenseScoredChunkId]] = dense_search,
    bm25_fn: Callable[..., list[BM25ScoredChunkId]] = bm25_search,
) -> list[HybridResult]:
    settings = get_settings()
    dense_results = dense_fn(query, top_k=dense_top_k or settings.DENSE_TOP_K)
    bm25_results = bm25_fn(query, top_k=bm25_top_k or settings.BM25_TOP_K)
    fused = fuse_with_rrf(dense_results, bm25_results, rrf_k or settings.RRF_K)
    top_n = hybrid_top_n or settings.HYBRID_TOP_N
    return fused[:top_n]
