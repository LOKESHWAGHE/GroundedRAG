"""
LLM-based atomic claim decomposition. "The company was founded in 1998 and
has offices in Mumbai and Pune" becomes three independently-checkable claims
rather than one compound sentence, which matters because a claim that is
half-right (offices in Mumbai but not Pune) should be flagged, not passed
whole as SUPPORTED or UNSUPPORTED.

Falls back to returning the sentence unchanged (as its own single claim) if
the LLM call fails or its response isn't parseable JSON — this is the
"falls back to sentence" behavior CLAIM_EXTRACTION_MODE's docstring refers to.
"""

import json
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Break the given sentence into a list of independent, atomic factual \
claims. Each claim must be a complete, standalone statement that can be true \
or false on its own. Preserve the original meaning and wording as closely as \
possible; do not add information. If the sentence is already a single atomic \
claim, return it unchanged as the only item.

Respond with ONLY a JSON array of strings, nothing else. Example:
["The company was founded in 1998.", "The company has offices in Mumbai.", \
"The company has offices in Pune."]"""


def atomize_claims_llm(sentence: str, llm_fn: Callable[[str, str], str]) -> list[str]:
    try:
        raw = llm_fn(_SYSTEM_PROMPT, sentence)
        claims = json.loads(_extract_json_array(raw))
        if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
            raise ValueError("Response was not a JSON array of strings.")
        cleaned = [c.strip() for c in claims if c.strip()]
        return cleaned or [sentence]
    except Exception as exc:  # noqa: BLE001 — any failure here should degrade gracefully, not break the pipeline
        logger.warning("Atomic claim extraction failed, falling back to whole sentence: %s", exc)
        return [sentence]


def _extract_json_array(text: str) -> str:
    """Models occasionally wrap JSON in a markdown code fence despite instructions; strip it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in response: {text[:200]}")
    return text[start : end + 1]
