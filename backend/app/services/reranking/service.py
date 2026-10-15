"""
Cross-encoder reranking: takes the top-N hybrid candidates, scores each
(query, chunk_text) pair with the cross-encoder, and returns the top-M —
the final context handed to the LLM generator (Phase 8).

`score_fn` is injectable (default: the real CrossEncoder) so this ordering/
truncation logic is unit-testable without loading the actual model.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.reranking.model import cross_encoder_predict
from app.services.retrieval.hybrid import HybridResult

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    chunk_id: int
    rerank_score: float
    rerank_rank: int  # 1-indexed, assigned after sorting — shown in the debug panel
    text: str
    hybrid: HybridResult  # carries dense_rank/bm25_rank/rrf_score through for the debug panel


def rerank(
    query: str,
    candidates: list[HybridResult],
    chunk_texts: dict[int, str],
    top_m: int | None = None,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] = cross_encoder_predict,
) -> list[RerankedResult]:
    if not candidates:
        return []

    pairs = [(query, chunk_texts[c.chunk_id]) for c in candidates]
    scores = score_fn(pairs)
    if len(scores) != len(candidates):
        raise ValueError(f"score_fn returned {len(scores)} scores for {len(candidates)} candidates.")

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_m = top_m or get_settings().RERANK_TOP_M
    results = [
        RerankedResult(
            chunk_id=hybrid.chunk_id,
            rerank_score=score,
            rerank_rank=i + 1,
            text=chunk_texts[hybrid.chunk_id],
            hybrid=hybrid,
        )
        for i, (hybrid, score) in enumerate(scored[:top_m])
    ]
    logger.info("Reranked %d candidate(s) -> top %d", len(candidates), len(results))
    return results
