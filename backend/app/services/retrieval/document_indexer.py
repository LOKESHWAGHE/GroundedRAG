"""
Turns a chunked document into index entries in both the dense (FAISS) and
sparse (BM25) indexes. A document reaches status INDEXED only once both have
succeeded — that's the contract the hybrid retriever (Phase 6) relies on.
"""

import logging

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.services.retrieval.dense import embed_and_index_chunks
from app.services.retrieval.sparse import index_chunks_bm25

logger = logging.getLogger(__name__)


def index_document(db: Session, doc: Document) -> Document:
    if not doc.chunks:
        raise ValueError("Document has no chunks to index — run chunking first.")

    doc.status = DocumentStatus.EMBEDDING.value
    doc.error_message = None
    db.commit()

    try:
        embed_and_index_chunks(doc.chunks)
        index_chunks_bm25(doc.chunks)
        doc.status = DocumentStatus.INDEXED.value
        db.commit()
        logger.info("Document '%s' indexed (%d chunks, dense + BM25)", doc.filename, len(doc.chunks))
    except Exception as exc:
        db.rollback()
        doc.status = DocumentStatus.FAILED.value
        doc.error_message = f"Indexing failed: {exc}"
        db.commit()
        logger.exception("Indexing failed for document %s", doc.id)
        raise

    db.refresh(doc)
    return doc
