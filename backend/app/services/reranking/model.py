"""
CrossEncoder model loader — lazy singleton, same rationale as the embedding
model loader: heavy imports deferred until first actual use.

Unlike the bi-encoder used for dense retrieval (which embeds the query and
each chunk independently, then compares vectors), a CrossEncoder feeds the
(query, chunk) pair through the model TOGETHER, letting attention directly
compare their tokens. That's slower — you can't precompute chunk
representations — which is exactly why it only runs on the ~20 hybrid
candidates (HYBRID_TOP_N) rather than the whole corpus, and why hybrid
retrieval exists as a cheap first pass to narrow the field for it.
"""

import logging
import threading

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_cross_encoder():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder  # heavy import, deferred

                settings = get_settings()
                logger.info("Loading CrossEncoder model '%s'...", settings.CROSS_ENCODER_MODEL_NAME)
                _model = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME)
                logger.info("CrossEncoder model loaded")
    return _model


def cross_encoder_predict(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Returns raw relevance scores (unbounded; higher = more relevant) for
    (query, chunk_text) pairs — ms-marco-MiniLM-style cross-encoders are
    trained with a ranking loss, not calibrated to [0,1], so scores should be
    compared to each other within one query's candidate set, not against a
    fixed global threshold. (MIN_RERANK_SCORE in config exists for exactly
    this reason — tune it via the evaluation harness, not by inspection.)
    """
    if not pairs:
        return []
    model = get_cross_encoder()
    return [float(s) for s in model.predict(pairs, show_progress_bar=False)]
