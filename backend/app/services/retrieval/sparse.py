"""
BM25 orchestration — mirrors dense.py's shape (embed_and_index_chunks / dense_search)
so the hybrid fusion step in Phase 6 can treat both retrieval paths uniformly.
`store` is injectable for testing, same rationale as `index` in dense.py.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.core.config import get_settings
from app.services.retrieval.bm25_index import BM25Index, get_bm25_index, persist_bm25_index

if TYPE_CHECKING:
    from app.models.document import Chunk

logger = logging.getLogger(__name__)


class HasIdAndText(Protocol):
    id: int
    text: str


@dataclass(frozen=True)
class ScoredChunkId:
    chunk_id: int
    score: float  # raw BM25 score (unbounded, NOT comparable across queries)


def index_chunks_bm25(
    chunks: "list[Chunk] | list[HasIdAndText]", store: BM25Index | None = None, persist: bool = True
) -> None:
    if not chunks:
        return
    store = store or get_bm25_index()
    store.remove([c.id for c in chunks])  # idempotent re-indexing, same rationale as dense.py
    store.add([c.id for c in chunks], [c.text for c in chunks])
    if persist:
        persist_bm25_index()
    logger.info("BM25-indexed %d chunk(s) (index size=%d)", len(chunks), store.size)


def bm25_search(query: str, top_k: int | None = None, store: BM25Index | None = None) -> list[ScoredChunkId]:
    top_k = top_k or get_settings().BM25_TOP_K
    store = store or get_bm25_index()
    return [ScoredChunkId(chunk_id=cid, score=score) for cid, score in store.search(query, top_k)]


def remove_chunks_from_bm25_index(chunk_ids: list[int], store: BM25Index | None = None, persist: bool = True) -> None:
    if not chunk_ids:
        return
    store = store or get_bm25_index()
    store.remove(chunk_ids)
    if persist:
        persist_bm25_index()
