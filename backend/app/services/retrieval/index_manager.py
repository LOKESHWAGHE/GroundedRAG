"""Removes a document's entries from every index (dense + sparse) when it's deleted."""

import uuid

from sqlalchemy.orm import Session

from app.models.document import Chunk
from app.services.retrieval.dense import remove_chunks_from_dense_index
from app.services.retrieval.sparse import remove_chunks_from_bm25_index


def remove_document_from_indexes(db: Session, document_id: uuid.UUID) -> None:
    chunk_ids = [row[0] for row in db.query(Chunk.id).filter(Chunk.document_id == document_id).all()]
    remove_chunks_from_dense_index(chunk_ids)
    remove_chunks_from_bm25_index(chunk_ids)
