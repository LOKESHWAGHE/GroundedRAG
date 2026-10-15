"""
Embedding model loader.

Lazy singleton: the model (~90MB for the default all-MiniLM-L6-v2) is only
loaded into memory on first use, not at import time, so running the test
suite or the ingestion-only endpoints doesn't require torch/sentence-transformers
to actually download anything.
"""

import logging
import threading

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer  # heavy import, deferred

                settings = get_settings()
                logger.info("Loading embedding model '%s'...", settings.EMBEDDING_MODEL_NAME)
                _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                actual_dim = _model.get_sentence_embedding_dimension()
                if actual_dim != settings.EMBEDDING_DIM:
                    raise RuntimeError(
                        f"EMBEDDING_DIM is set to {settings.EMBEDDING_DIM} but "
                        f"'{settings.EMBEDDING_MODEL_NAME}' produces {actual_dim}-dim vectors. "
                        "Update EMBEDDING_DIM to match if you change the model."
                    )
                logger.info("Embedding model loaded (dim=%d)", actual_dim)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Returns float32, L2-normalized vectors (shape [len(texts), EMBEDDING_DIM]).
    Normalizing here means FAISS's inner product IS cosine similarity — no
    separate normalization step needed at index or query time.
    """
    settings = get_settings()
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIM), dtype="float32")
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
