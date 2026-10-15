import uuid

from pydantic import BaseModel, Field

from app.core.config import get_settings


class QueryRequest(BaseModel):
    # max_length is read from settings rather than hardcoded so MAX_QUESTION_CHARS in
    # config.py / .env is the single source of truth, not duplicated here.
    question: str = Field(..., min_length=1, max_length=get_settings().MAX_QUESTION_CHARS)


class SupportingEvidenceOut(BaseModel):
    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    quote: str
    entailment_score: float
    contradiction_score: float


class ClaimVerdictOut(BaseModel):
    claim_id: str
    text: str
    verdict: str  # SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED
    support_score: float
    evidence: SupportingEvidenceOut | None


class GroundednessOut(BaseModel):
    total_claims: int
    supported: int
    partially_supported: int
    unsupported: int
    groundedness_score: float | None
    citation_coverage: float | None


class RetrievedChunkOut(BaseModel):
    """One row of the Phase 12 retrieval debug panel."""

    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    hybrid_rank: int | None
    rrf_score: float | None
    rerank_rank: int | None
    rerank_score: float | None
    in_final_context: bool


class QueryResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str
    insufficient_evidence: bool
    insufficient_reason: str | None
    regeneration_count: int
    groundedness: GroundednessOut
    claims: list[ClaimVerdictOut]
    retrieval_debug: list[RetrievedChunkOut]
    latency_ms: float


class QueryHistoryItem(BaseModel):
    id: uuid.UUID
    question: str
    status: str
    groundedness: float | None
    created_at: str
