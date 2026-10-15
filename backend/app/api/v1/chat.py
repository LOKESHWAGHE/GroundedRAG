"""
POST /api/v1/chat/query — the real, DB-backed entrypoint to the pipeline built
across Phases 4-10. Everything this endpoint calls is injectable in
rag/pipeline.py for tests; this is the one place that wires it to Postgres
and a real session.
"""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Chunk, Document
from app.models.query import QueryRecord
from app.schemas.chat import (
    ClaimVerdictOut,
    GroundednessOut,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
    SupportingEvidenceOut,
)
from app.services.rag.pipeline import QueryResult, run_query_pipeline

router = APIRouter()


def _make_chunk_meta_fn(db: Session):
    def chunk_meta_fn(chunk_ids: list[int]) -> dict[int, dict]:
        if not chunk_ids:
            return {}
        rows = (
            db.query(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Chunk.id.in_(chunk_ids))
            .all()
        )
        return {
            chunk.id: {
                "document_id": str(chunk.document_id),
                "filename": filename,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "text": chunk.text,
            }
            for chunk, filename in rows
        }

    return chunk_meta_fn


def _build_retrieval_debug(result: QueryResult) -> list[RetrievedChunkOut]:
    """Joins the hybrid trace with the rerank trace into one row per chunk, for the debug panel."""
    rerank_by_id = {r.chunk_id: r for r in result.trace.reranked}
    context_ids = set(result.context_chunk_ids)
    rows = []
    for h in result.trace.hybrid:
        rr = rerank_by_id.get(h.chunk_id)
        meta = rr.hybrid if rr else h  # same object either way; kept explicit for clarity
        rows.append(
            RetrievedChunkOut(
                chunk_id=h.chunk_id,
                document_id="",  # filled in by the caller once chunk_meta is available (see below)
                filename="",
                page_number=0,
                dense_rank=h.dense_rank,
                dense_score=h.dense_score,
                bm25_rank=h.bm25_rank,
                bm25_score=h.bm25_score,
                hybrid_rank=h.hybrid_rank,
                rrf_score=h.rrf_score,
                rerank_rank=rr.rerank_rank if rr else None,
                rerank_score=rr.rerank_score if rr else None,
                in_final_context=h.chunk_id in context_ids,
            )
        )
    return rows


@router.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    chunk_meta_fn = _make_chunk_meta_fn(db)

    start = time.perf_counter()
    result: QueryResult = run_query_pipeline(payload.question, chunk_meta_fn=chunk_meta_fn)
    latency_ms = (time.perf_counter() - start) * 1000

    # Fill in filename/document_id on the debug rows now that chunk_meta_fn has been called by the pipeline.
    debug_rows = _build_retrieval_debug(result)
    meta = chunk_meta_fn([r.chunk_id for r in debug_rows])
    for row in debug_rows:
        m = meta.get(row.chunk_id)
        if m:
            row.document_id = m["document_id"]
            row.filename = m["filename"]
            row.page_number = m["page_number"]

    claims_out = [
        ClaimVerdictOut(
            claim_id=v.claim.claim_id,
            text=v.claim.text,
            verdict=v.verdict.value,
            support_score=v.support_score,
            evidence=(
                SupportingEvidenceOut(
                    chunk_id=v.evidence.chunk_id,
                    document_id=v.evidence.document_id,
                    filename=v.evidence.filename,
                    page_number=v.evidence.page_number,
                    quote=v.evidence.quote,
                    entailment_score=v.evidence.entailment_score,
                    contradiction_score=v.evidence.contradiction_score,
                )
                if v.evidence
                else None
            ),
        )
        for v in result.claims
    ]

    record = QueryRecord(
        question=payload.question,
        status="insufficient_evidence" if result.insufficient_evidence else "answered",
        groundedness=result.groundedness.groundedness_score,
        latency_ms=latency_ms,
        result={
            "answer": result.answer,
            "insufficient_evidence": result.insufficient_evidence,
            "insufficient_reason": result.insufficient_reason,
            "regeneration_count": result.regeneration_count,
            "claims": [c.model_dump() for c in claims_out],
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return QueryResponse(
        id=record.id,
        question=payload.question,
        answer=result.answer,
        insufficient_evidence=result.insufficient_evidence,
        insufficient_reason=result.insufficient_reason,
        regeneration_count=result.regeneration_count,
        groundedness=GroundednessOut(**vars(result.groundedness)),
        claims=claims_out,
        retrieval_debug=debug_rows,
        latency_ms=latency_ms,
    )


@router.get("/history", response_model=list[QueryHistoryItem])
def query_history(limit: int = 50, db: Session = Depends(get_db)) -> list[QueryHistoryItem]:
    stmt = select(QueryRecord).order_by(QueryRecord.created_at.desc()).limit(limit)
    records = db.scalars(stmt).all()
    return [
        QueryHistoryItem(
            id=r.id, question=r.question, status=r.status, groundedness=r.groundedness,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]


@router.get("/{query_id}", response_model=QueryResponse)
def get_query(query_id: uuid.UUID, db: Session = Depends(get_db)) -> QueryResponse:
    record = db.get(QueryRecord, query_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Query not found.")
    # Historical queries return the saved answer/claims; retrieval_debug and groundedness
    # breakdown aren't persisted field-by-field (the full result JSON is), so this endpoint
    # focuses on re-surfacing the answer and verdicts rather than re-running retrieval.
    saved = record.result
    claims_out = [ClaimVerdictOut(**c) for c in saved.get("claims", [])]
    return QueryResponse(
        id=record.id,
        question=record.question,
        answer=saved.get("answer", ""),
        insufficient_evidence=saved.get("insufficient_evidence", False),
        insufficient_reason=saved.get("insufficient_reason"),
        regeneration_count=saved.get("regeneration_count", 0),
        groundedness=GroundednessOut(
            total_claims=len(claims_out),
            supported=sum(1 for c in claims_out if c.verdict == "SUPPORTED"),
            partially_supported=sum(1 for c in claims_out if c.verdict == "PARTIALLY_SUPPORTED"),
            unsupported=sum(1 for c in claims_out if c.verdict == "UNSUPPORTED"),
            groundedness_score=record.groundedness,
            citation_coverage=None,
        ),
        claims=claims_out,
        retrieval_debug=[],
        latency_ms=record.latency_ms,
    )
