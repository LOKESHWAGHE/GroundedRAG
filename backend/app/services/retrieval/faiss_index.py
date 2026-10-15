"""
FAISS wrapper: IndexIDMap2(IndexFlatIP) — inner product over L2-normalized
vectors, which is exactly cosine similarity. IDs used are the SQL `chunks.id`
primary keys, so a search result maps straight back to chunk metadata with no
separate ID-translation table to keep in sync.

`faiss` is imported lazily inside methods (not at module level) so importing
this module — and everything that imports it — doesn't require faiss to be
installed unless an index is actually built or loaded.
"""

import logging
import threading
from pathlib import Path

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._lock = threading.RLock()
        self._index = self._new_index()

    def _new_index(self):
        import faiss

        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        with self._lock:
            self._index.add_with_ids(vectors.astype("float32"), np.array(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> None:
        """No-op for ids not present — safe to call speculatively before a re-index."""
        if not ids:
            return
        with self._lock:
            self._index.remove_ids(np.array(ids, dtype="int64"))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Returns (chunk_id, cosine_similarity) pairs, highest similarity first."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(query_vector.reshape(1, -1).astype("float32"), k)
        return [(int(cid), float(score)) for cid, score in zip(ids[0], scores[0]) if cid != -1]

    @property
    def size(self) -> int:
        return self._index.ntotal

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(path))

    @classmethod
    def load_or_create(cls, path: Path, dim: int) -> "VectorIndex":
        obj = cls(dim)
        if path.exists():
            import faiss

            with obj._lock:
                obj._index = faiss.read_index(str(path))
            if obj._index.d != dim:
                raise RuntimeError(
                    f"Existing FAISS index at {path} has dim={obj._index.d}, "
                    f"but EMBEDDING_DIM={dim}. Delete the index file to rebuild from scratch."
                )
        return obj


_singleton: VectorIndex | None = None
_singleton_lock = threading.Lock()


def get_vector_index() -> VectorIndex:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                settings = get_settings()
                path = settings.index_path / "faiss.index"
                _singleton = VectorIndex.load_or_create(path, settings.EMBEDDING_DIM)
    return _singleton


def persist_vector_index() -> None:
    settings = get_settings()
    get_vector_index().save(settings.index_path / "faiss.index")


def reset_vector_index() -> None:
    """Test/debug helper: drop the cached singleton so the next call rebuilds it from disk."""
    global _singleton
    with _singleton_lock:
        _singleton = None
