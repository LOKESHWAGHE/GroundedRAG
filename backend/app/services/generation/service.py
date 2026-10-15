"""
Generation orchestration: reranked chunks -> context -> prompt -> LLM ->
parsed answer with resolved citations. The citations resolved here are what
the MODEL claimed to cite — Phase 9 (claim extraction) and Phase 10 (citation
verification) independently check whether each factual claim in the answer
is actually supported by its cited passage. Nothing here is treated as
ground truth.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.services.generation.context_selector import ContextBundle, ContextChunk, build_context
from app.services.generation.llm_client import generate_completion
from app.services.generation.parser import ParsedAnswer, parse_llm_response
from app.services.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.services.reranking.service import RerankedResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelClaimedCitation:
    marker: int
    chunk_id: int
    document_id: str
    filename: str
    page_number: int


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    insufficient_evidence: bool
    insufficient_reason: str | None
    context: ContextBundle
    model_claimed_citations: list[ModelClaimedCitation]  # unverified — see module docstring


def generate_answer(
    question: str,
    reranked: list[RerankedResult],
    chunk_meta: dict[int, dict],
    llm_fn: Callable[[str, str], str] = generate_completion,
) -> GenerationResult:
    context = build_context(reranked, chunk_meta)
    user_prompt = build_user_prompt(question, context)

    raw = llm_fn(SYSTEM_PROMPT, user_prompt)
    parsed: ParsedAnswer = parse_llm_response(raw)

    marker_to_chunk: dict[int, ContextChunk] = {c.marker: c for c in context.chunks}
    citations = [
        ModelClaimedCitation(
            marker=m,
            chunk_id=marker_to_chunk[m].chunk_id,
            document_id=marker_to_chunk[m].document_id,
            filename=marker_to_chunk[m].filename,
            page_number=marker_to_chunk[m].page_number,
        )
        for m in sorted(parsed.cited_markers)
        if m in marker_to_chunk  # a hallucinated marker (e.g. [7] with only 5 passages) is dropped, not trusted
    ]

    return GenerationResult(
        answer=parsed.answer,
        insufficient_evidence=parsed.insufficient_evidence,
        insufficient_reason=parsed.insufficient_reason,
        context=context,
        model_claimed_citations=citations,
    )
