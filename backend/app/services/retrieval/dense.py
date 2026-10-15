"""
Dense retrieval: chunks -> embeddings -> FAISS at indexing time; query -> FAISS
search at query time. `index` and `embed_fn` are injectable (default to the
real singletons) purely so this logic can be unit-tested with a fake vector
store and a deterministic embedder, without needing FAISS/torch installed.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.core.config import get_settings
from app.services.embeddings.model import embed_query, embed_texts
from app.services.retrieval.faiss_index import VectorIndex, get_vector_index, persist_vector_index

if TYPE_CHECKING:
    from app.models.document import Chunk

logger = logging.getLogger(__name__)


class HasIdAndText(Protocol):
    """Structural type so this module doesn't need a hard import of the ORM `Chunk` model."""

    id: int
    text: str


@dataclass(frozen=True)
class ScoredChunkId:
    chunk_id: int
    score: float  # cosine similarity


def embed_and_index_chunks(
    chunks: "list[Chunk] | list[HasIdAndText]",
    index: VectorIndex | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] = embed_texts,
    persist: bool = True,
) -> None:
    if not chunks:
        return
    index = index or get_vector_index()
    ids = [c.id for c in chunks]
    vectors = embed_fn([c.text for c in chunks])
    index.remove(ids)  # safe no-op if these ids weren't indexed yet; makes re-indexing idempotent
    index.add(ids, vectors)
    if persist:
        persist_vector_index()
    logger.info("Dense-indexed %d chunk(s) (index size=%d)", len(ids), index.size)


def dense_search(
    query: str,
    top_k: int | None = None,
    index: VectorIndex | None = None,
    embed_fn: Callable[[str], np.ndarray] = embed_query,
) -> list[ScoredChunkId]:
    top_k = top_k or get_settings().DENSE_TOP_K
    index = index or get_vector_index()
    query_vector = embed_fn(query)
    return [ScoredChunkId(chunk_id=cid, score=score) for cid, score in index.search(query_vector, top_k)]


def remove_chunks_from_dense_index(
    chunk_ids: list[int], index: VectorIndex | None = None, persist: bool = True
) -> None:
    if not chunk_ids:
        return
    index = index or get_vector_index()
    index.remove(chunk_ids)
    if persist:
        persist_vector_index()
