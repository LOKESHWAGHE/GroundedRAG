"""Chunking orchestration: pages (DB) -> Chunk rows, using the configured strategy."""

import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Chunk, Document, DocumentStatus
from app.services.chunking.splitter import chunk_page_fixed, chunk_page_structure_aware

logger = logging.getLogger(__name__)


def chunk_document(db: Session, doc: Document) -> list[Chunk]:
    """
    Deletes any existing chunks for this document first, so re-running chunking
    (e.g. after a config change) is safe and idempotent.
    """
    settings = get_settings()
    db.execute(delete(Chunk).where(Chunk.document_id == doc.id))

    splitter = chunk_page_structure_aware if settings.CHUNK_STRATEGY == "structure" else chunk_page_fixed
    chunks: list[Chunk] = []
    index = 0
    for page in doc.pages:
        for raw in splitter(
            page.text, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS, settings.CHUNK_MIN_TOKENS
        ):
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    chunk_index=index,
                    page_number=page.page_number,
                    section_title=raw.section_title,
                    text=raw.text,
                    token_count=raw.token_count,
                )
            )
            index += 1

    doc.chunks = chunks
    doc.chunk_count = len(chunks)
    doc.status = DocumentStatus.CHUNKING.value if chunks else DocumentStatus.FAILED.value
    if not chunks:
        doc.error_message = "Chunking produced no chunks (document text may be too short)."
    db.commit()
    logger.info("Chunked '%s' into %d chunk(s)", doc.filename, len(chunks))
    return chunks
