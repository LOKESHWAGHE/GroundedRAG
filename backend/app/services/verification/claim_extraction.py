"""
Claim extraction orchestration: answer text -> list[Claim], each with the
citation marker(s) the model attached to its source sentence.

CLAIM_EXTRACTION_MODE:
  "sentence" -> one claim per sentence (fast, no extra LLM call)
  "llm"      -> each sentence is further decomposed into atomic claims via
                the LLM (see llm_atomizer.py); falls back to "sentence"
                behavior per-sentence if that call fails
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.generation.llm_client import generate_completion
from app.services.verification.llm_atomizer import atomize_claims_llm
from app.services.verification.sentence_splitter import split_sentences_with_markers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    source_markers: frozenset[int]  # which cited passage(s) this claim's sentence pointed to


def extract_claims(
    answer_text: str,
    mode: str | None = None,
    llm_fn: Callable[[str, str], str] = generate_completion,
) -> list[Claim]:
    mode = mode or get_settings().CLAIM_EXTRACTION_MODE
    sentences = split_sentences_with_markers(answer_text)

    claims: list[Claim] = []
    counter = 1
    for unit in sentences:
        atoms = atomize_claims_llm(unit.text, llm_fn) if mode == "llm" else [unit.text]
        for atom_text in atoms:
            claims.append(Claim(claim_id=f"claim_{counter}", text=atom_text, source_markers=unit.markers))
            counter += 1

    logger.info("Extracted %d claim(s) from answer (%d sentence(s), mode=%s)", len(claims), len(sentences), mode)
    return claims
