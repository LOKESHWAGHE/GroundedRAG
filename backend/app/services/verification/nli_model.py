"""
NLI (Natural Language Inference) model — lazy singleton, same pattern as the
embedding and reranking model loaders. Given a (premise, hypothesis) pair —
here, (evidence chunk text, claim text) — it predicts probabilities for
three labels: contradiction, entailment, neutral.

This is the model that actually decides SUPPORTED vs UNSUPPORTED: entailment
means the evidence logically implies the claim is true; contradiction means
the evidence implies it's false; neutral means the evidence is unrelated or
insufficient either way.
"""

import logging
import threading

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# cross-encoder/nli-deberta-v3-small (and the nli-deberta-v3 family generally)
# outputs logits in this label order.
_LABELS = ["contradiction", "entailment", "neutral"]


def get_nli_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder  # heavy import, deferred

                settings = get_settings()
                logger.info("Loading NLI model '%s'...", settings.NLI_MODEL_NAME)
                _model = CrossEncoder(settings.NLI_MODEL_NAME)
                logger.info("NLI model loaded")
    return _model


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def nli_predict(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """
    pairs: list of (premise, hypothesis) = (evidence_text, claim_text).
    Returns one {"contradiction": p, "entailment": p, "neutral": p} dict per pair.
    """
    if not pairs:
        return []
    model = get_nli_model()
    logits = np.asarray(model.predict(pairs, show_progress_bar=False))
    probs = _softmax(logits)
    return [dict(zip(_LABELS, row.tolist())) for row in probs]
