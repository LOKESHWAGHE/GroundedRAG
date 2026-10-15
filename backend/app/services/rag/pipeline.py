"""
The full query pipeline: hybrid retrieval -> rerank -> generate -> extract
claims -> verify -> (if ungrounded) regenerate, up to MAX_REGENERATION_RETRIES
-> return the answer with its verified citations and groundedness score.

Every stage function is injectable (defaults to the real implementation) so
this orchestration — including the regeneration loop's control flow — is
testable without FAISS, BM25, a real LLM, or a real NLI model.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.generation.context_selector import ContextChunk, build_context
from app.services.generation.llm_client import generate_completion
from app.services.generation.parser import parse_llm_response
from app.services.generation.prompt import SYSTEM_PROMPT, build_regeneration_prompt, build_user_prompt
from app.services.reranking.service import RerankedResult, rerank
from app.services.retrieval.hybrid import HybridResult, hybrid_search
from app.services.verification.citation_verification import ClaimVerdict, Verdict, verify_claims
from app.services.verification.claim_extraction import Claim, extract_claims
from app.services.verification.groundedness import GroundednessReport, compute_groundedness, has_unsupported_claims
from app.services.verification.nli_model import nli_predict

logger = logging.getLogger(__name__)

SAFE_FALLBACK_PREFIX = "INSUFFICIENT_EVIDENCE:"


@dataclass(frozen=True)
class RetrievalTrace:
    """Everything the Phase 12 debug panel needs: dense/BM25/hybrid/rerank ranks per chunk."""

    hybrid: list[HybridResult]
    reranked: list[RerankedResult]


@dataclass(frozen=True)
class QueryResult:
    answer: str
    insufficient_evidence: bool
    insufficient_reason: str | None
    claims: list[ClaimVerdict]
    groundedness: GroundednessReport
    regeneration_count: int
    trace: RetrievalTrace
    context_chunk_ids: list[int]  # chunk_ids that actually made it into the final LLM context (post word-budget)


def run_query_pipeline(
    question: str,
    chunk_meta_fn: Callable[[list[int]], dict[int, dict]],
    hybrid_fn: Callable[..., list[HybridResult]] = hybrid_search,
    chunk_texts_fn: Callable[[list[int]], dict[int, str]] | None = None,
    rerank_fn: Callable[..., list[RerankedResult]] = rerank,
    llm_fn: Callable[[str, str], str] = generate_completion,
    nli_fn: Callable[..., list[dict[str, float]]] = nli_predict,
    claim_mode: str | None = None,
    max_retries: int | None = None,
) -> QueryResult:
    """
    chunk_meta_fn(chunk_ids) -> {chunk_id: {document_id, filename, page_number,
    section_title, text}}. The caller (the API layer) supplies a version
    backed by the database; tests supply an in-memory dict. If chunk_texts_fn
    is omitted, chunk text is pulled from chunk_meta_fn's "text" field.
    """
    settings = get_settings()
    max_retries = settings.MAX_REGENERATION_RETRIES if max_retries is None else max_retries

    hybrid_results = hybrid_fn(question)
    hybrid_ids = [r.chunk_id for r in hybrid_results]
    meta = chunk_meta_fn(hybrid_ids)
    chunk_texts = chunk_texts_fn(hybrid_ids) if chunk_texts_fn else {cid: m["text"] for cid, m in meta.items()}

    reranked = rerank_fn(question, hybrid_results, chunk_texts)

    # If even the best-reranked chunk scores below MIN_RERANK_SCORE, the retrieved evidence
    # isn't relevant enough to answer from -- skip generation entirely rather than spending an
    # LLM call on a question the context can't support. Disabled by default (threshold 0.0);
    # tune via the evaluation harness (CrossEncoder scores aren't calibrated to a fixed scale
    # out of the box -- see reranking/model.py).
    if settings.MIN_RERANK_SCORE > 0 and (not reranked or reranked[0].rerank_score < settings.MIN_RERANK_SCORE):
        return QueryResult(
            answer=f"{SAFE_FALLBACK_PREFIX} No retrieved passage was relevant enough to answer this question.",
            insufficient_evidence=True,
            insufficient_reason="Best reranked passage scored below MIN_RERANK_SCORE.",
            claims=[],
            groundedness=compute_groundedness([]),
            regeneration_count=0,
            trace=RetrievalTrace(hybrid=hybrid_results, reranked=reranked),
            context_chunk_ids=[],
        )

    context = build_context(reranked, meta)
    marker_lookup: dict[int, ContextChunk] = {c.marker: c for c in context.chunks}
    trace = RetrievalTrace(hybrid=hybrid_results, reranked=reranked)
    context_chunk_ids = [c.chunk_id for c in context.chunks]

    attempt = 0
    failed_claim_texts: list[str] = []
    while True:
        prompt = (
            build_user_prompt(question, context)
            if attempt == 0
            else build_regeneration_prompt(question, context, failed_claim_texts)
        )
        parsed = parse_llm_response(llm_fn(SYSTEM_PROMPT, prompt))

        if parsed.insufficient_evidence:
            return QueryResult(
                answer=parsed.answer,
                insufficient_evidence=True,
                insufficient_reason=parsed.insufficient_reason,
                claims=[],
                groundedness=compute_groundedness([]),
                regeneration_count=attempt,
                trace=trace,
                context_chunk_ids=context_chunk_ids,
            )

        claims: list[Claim] = extract_claims(parsed.answer, mode=claim_mode, llm_fn=llm_fn)
        verdicts = verify_claims(claims, marker_lookup, nli_fn=nli_fn)

        if not has_unsupported_claims(verdicts):
            return QueryResult(
                answer=parsed.answer,
                insufficient_evidence=False,
                insufficient_reason=None,
                claims=verdicts,
                groundedness=compute_groundedness(verdicts),
                regeneration_count=attempt,
                trace=trace,
                context_chunk_ids=context_chunk_ids,
            )

        if attempt >= max_retries:
            logger.info(
                "Query exhausted %d regeneration attempt(s) with unsupported claims remaining; "
                "returning the safe fallback instead of an ungrounded answer.",
                attempt,
            )
            return QueryResult(
                answer=f"{SAFE_FALLBACK_PREFIX} The available documents do not provide reliable, "
                "verifiable support for a complete answer to this question.",
                insufficient_evidence=True,
                insufficient_reason=f"Could not produce a fully grounded answer after {attempt} "
                "regeneration attempt(s); one or more claims remained unsupported.",
                claims=verdicts,
                groundedness=compute_groundedness(verdicts),
                regeneration_count=attempt,
                trace=trace,
                context_chunk_ids=context_chunk_ids,
            )

        failed_claim_texts = [v.claim.text for v in verdicts if v.verdict == Verdict.UNSUPPORTED]
        attempt += 1
