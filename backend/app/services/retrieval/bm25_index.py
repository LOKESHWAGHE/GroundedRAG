"""
A from-scratch BM25 (Okapi) implementation — deliberately not delegated to a
black-box library, so the scoring formula stays inspectable and the retrieval
debug panel (Phase 12) can show exactly why a document ranked where it did.

Score for query Q against document D:

    score(D, Q) = sum over t in Q of  IDF(t) * f(t, D) * (k1 + 1)
                                       -----------------------------------
                                       f(t, D) + k1 * (1 - b + b * |D| / avgdl)

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )   [BM25+ form; always >= 0]

  N       total number of documents (chunks) in the index
  n(t)    number of documents containing term t
  f(t,D)  number of times t occurs in D
  |D|     length of D in tokens; avgdl = mean document length
  k1, b   tuning constants (config: BM25_K1, BM25_B) — k1 controls how quickly
          term-frequency saturates, b controls length normalization strength

Term statistics (document frequency, average length) are recomputed whenever
the corpus changes. That's O(corpus size) rather than incremental, which is
the right trade-off at this project's scale — simplicity and correctness over
an optimization that only matters at a scale this project doesn't run at.
"""

import logging
import math
import pickle
import re
import threading
from collections import Counter
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. No stemming/stopword removal — keeps exact
    identifiers, codes and rare terms intact, which is precisely what BM25 is
    here to catch that dense embeddings might blur together."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float | None = None, b: float | None = None):
        settings = get_settings()
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self._lock = threading.RLock()
        self._corpus: dict[int, list[str]] = {}
        self._doc_freq: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def _rebuild_stats(self) -> None:
        self._n_docs = len(self._corpus)
        self._doc_freq = Counter()
        total_len = 0
        for tokens in self._corpus.values():
            total_len += len(tokens)
            self._doc_freq.update(set(tokens))
        self._avgdl = (total_len / self._n_docs) if self._n_docs else 0.0

    def add(self, ids: list[int], texts: list[str]) -> None:
        if not ids:
            return
        with self._lock:
            for cid, text in zip(ids, texts):
                self._corpus[cid] = tokenize(text)
            self._rebuild_stats()

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            for cid in ids:
                self._corpus.pop(cid, None)
            self._rebuild_stats()

    def _idf(self, term: str) -> float:
        n_t = self._doc_freq.get(term, 0)
        return math.log((self._n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        """Exposed separately from `search` so the debug panel can show a per-term breakdown."""
        tokens = self._corpus.get(doc_id)
        if tokens is None:
            return 0.0
        doc_len = len(tokens)
        term_freq = Counter(tokens)
        total = 0.0
        for term in query_tokens:
            f = term_freq.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self._avgdl or 1.0))
            total += self._idf(term) * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        with self._lock:
            if not self._corpus:
                return []
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            # Only score documents sharing at least one query term — everything
            # else scores exactly 0 by the formula above, so this is exact, not approximate.
            candidate_ids = {
                cid for cid, tokens in self._corpus.items() if not set(query_tokens).isdisjoint(tokens)
            }
            scored = [(cid, self.score(query_tokens, cid)) for cid in candidate_ids]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    @property
    def size(self) -> int:
        return self._n_docs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump({"corpus": self._corpus, "k1": self.k1, "b": self.b}, f)

    @classmethod
    def load_or_create(cls, path: Path) -> "BM25Index":
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k1=data.get("k1"), b=data.get("b"))
        obj._corpus = data.get("corpus", {})
        obj._rebuild_stats()
        return obj


_singleton: BM25Index | None = None
_singleton_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                settings = get_settings()
                _singleton = BM25Index.load_or_create(settings.index_path / "bm25.pkl")
    return _singleton


def persist_bm25_index() -> None:
    settings = get_settings()
    get_bm25_index().save(settings.index_path / "bm25.pkl")


def reset_bm25_index() -> None:
    """Test/debug helper: drop the cached singleton so the next call rebuilds it from disk."""
    global _singleton
    with _singleton_lock:
        _singleton = None
