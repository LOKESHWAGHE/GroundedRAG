"""Test doubles for dense retrieval — let Phase 4 logic run without FAISS/torch installed."""

import numpy as np
import hashlib

class FakeVectorIndex:
    """In-memory brute-force cosine index with the same surface as VectorIndex."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._vectors: dict[int, np.ndarray] = {}

    def add(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self._vectors[i] = np.asarray(v, dtype="float32")

    def remove(self, ids):
        for i in ids:
            self._vectors.pop(i, None)

    def search(self, query_vector, top_k):
        if not self._vectors:
            return []
        items = list(self._vectors.items())
        scores = [(cid, float(np.dot(query_vector, vec))) for cid, vec in items]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    @property
    def size(self):
        return len(self._vectors)



def deterministic_embed_texts(texts: list[str]) -> np.ndarray:
    """
    Stable hash-based bag-of-words embeddings for retrieval tests.
    """
    dim = 32
    out = np.zeros((len(texts), dim), dtype="float32")

    for row, text in enumerate(texts):
        for word in text.lower().split():
            bucket = int.from_bytes(
                hashlib.sha256(word.encode("utf-8")).digest()[:4],
                byteorder="big",
            ) % dim
            out[row, bucket] += 1.0

        norm = np.linalg.norm(out[row])
        if norm > 0:
            out[row] /= norm

    return out



def deterministic_embed_query(text: str) -> np.ndarray:
    return deterministic_embed_texts([text])[0]
