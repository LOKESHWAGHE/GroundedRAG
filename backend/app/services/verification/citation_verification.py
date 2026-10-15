"""
The core differentiator of this project: independently checks whether each
extracted claim is actually supported by its cited evidence, rather than
trusting the LLM's own citations (see generation/service.py's docstring).

For each claim:
  1. Gather evidence chunks — the ones the model cited for that claim's
     sentence (via source_markers), or, if the claim has no markers at all,
     every chunk in the context (a claim with a hallucinated/missing
     citation isn't given a free pass; it still gets checked).
  2. Split each evidence chunk into sentence windows (evidence_windows.py)
     and run NLI on every (window, claim) pair.
  3. Take the window with the highest entailment probability as the claim's
     best support, and separately track the highest contradiction
     probability seen anywhere in its evidence.
  4. Assign a verdict:
       - entailment >= SUPPORT_THRESHOLD                      -> SUPPORTED
       - contradiction >= CONTRADICTION_THRESHOLD              -> UNSUPPORTED
         (an explicit contradiction overrides a middling entailment score)
       - entailment >= PARTIAL_THRESHOLD                       -> PARTIALLY_SUPPORTED
       - otherwise (including "no evidence at all")             -> UNSUPPORTED
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from app.core.config import get_settings
from app.services.generation.context_selector import ContextChunk
from app.services.verification.claim_extraction import Claim
from app.services.verification.evidence_windows import build_evidence_windows
from app.services.verification.nli_model import nli_predict

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class SupportingEvidence:
    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    quote: str  # the specific sentence window that best supports (or contradicts) the claim
    entailment_score: float
    contradiction_score: float


@dataclass(frozen=True)
class ClaimVerdict:
    claim: Claim
    verdict: Verdict
    support_score: float  # best entailment probability found across all evidence, [0, 1]
    evidence: SupportingEvidence | None  # None only when there was no evidence to check at all


def _evidence_chunks_for_claim(claim: Claim, context_chunks: dict[int, ContextChunk]) -> list[ContextChunk]:
    if claim.source_markers:
        return [context_chunks[m] for m in claim.source_markers if m in context_chunks]
    return list(context_chunks.values())  # no citation given: still checked against everything available


def verify_claim(
    claim: Claim,
    context_chunks: dict[int, ContextChunk],
    nli_fn: Callable[[list[tuple[str, str]]], list[dict[str, float]]] = nli_predict,
) -> ClaimVerdict:
    settings = get_settings()
    evidence_chunks = _evidence_chunks_for_claim(claim, context_chunks)

    windows: list[tuple[ContextChunk, str]] = []
    for chunk in evidence_chunks:
        for w in build_evidence_windows(chunk.text, settings.NLI_MAX_WINDOWS_PER_CHUNK):
            windows.append((chunk, w.text))

    if not windows:
        return ClaimVerdict(claim=claim, verdict=Verdict.UNSUPPORTED, support_score=0.0, evidence=None)

    pairs = [(window_text, claim.text) for _, window_text in windows]  # (premise, hypothesis)
    scores = nli_fn(pairs)

    best_entailment_idx = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
    best_entailment = scores[best_entailment_idx]["entailment"]
    max_contradiction = max(s["contradiction"] for s in scores)

    best_chunk, best_window_text = windows[best_entailment_idx]
    evidence = SupportingEvidence(
        chunk_id=best_chunk.chunk_id,
        document_id=best_chunk.document_id,
        filename=best_chunk.filename,
        page_number=best_chunk.page_number,
        quote=best_window_text,
        entailment_score=best_entailment,
        contradiction_score=scores[best_entailment_idx]["contradiction"],
    )

    if max_contradiction >= settings.CONTRADICTION_THRESHOLD:
        verdict = Verdict.UNSUPPORTED
    elif best_entailment >= settings.SUPPORT_THRESHOLD:
        verdict = Verdict.SUPPORTED
    elif best_entailment >= settings.PARTIAL_THRESHOLD:
        verdict = Verdict.PARTIALLY_SUPPORTED
    else:
        verdict = Verdict.UNSUPPORTED

    return ClaimVerdict(claim=claim, verdict=verdict, support_score=best_entailment, evidence=evidence)


def verify_claims(
    claims: list[Claim],
    context_chunks: dict[int, ContextChunk],
    nli_fn: Callable[[list[tuple[str, str]]], list[dict[str, float]]] = nli_predict,
) -> list[ClaimVerdict]:
    return [verify_claim(c, context_chunks, nli_fn) for c in claims]
