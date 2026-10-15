import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.document import Chunk, Document
from app.schemas.chunk import ChunkOut
from app.schemas.document import DocumentOut, PageOut
from app.services.ingestion.file_validation import FileValidationError, read_upload_limited
from app.services.ingestion.service import DuplicateDocumentError, delete_document, ingest_document
from app.services.retrieval.document_indexer import index_document
from app.services.retrieval.index_manager import remove_document_from_indexes

router = APIRouter()


def _get_or_404(db: Session, document_id: uuid.UUID) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Document:
    """
    Sync `def` on purpose: parsing/chunking is blocking, CPU-bound work, so
    FastAPI runs it in a worker thread instead of stalling the event loop.
    Embedding + indexing are triggered separately via POST /{id}/index.
    """
    max_bytes = get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024
    try:
        data = read_upload_limited(file.file, max_bytes)
        return ingest_document(db, file.filename, data)
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": "This document was already uploaded.", "existing_document_id": str(exc.existing_id)},
        ) from exc
    except FileValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Document:
    return _get_or_404(db, document_id)


@router.get("/{document_id}/pages", response_model=list[PageOut])
def get_document_pages(document_id: uuid.UUID, db: Session = Depends(get_db)) -> list[PageOut]:
    """Debug view of exactly what the parser extracted."""
    doc = _get_or_404(db, document_id)
    return [PageOut(page_number=p.page_number, char_count=len(p.text), text=p.text) for p in doc.pages]


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
def get_document_chunks(document_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Chunk]:
    """Debug view of how the document was split — this is what gets embedded and indexed."""
    _get_or_404(db, document_id)
    stmt = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    return list(db.scalars(stmt))


@router.post("/{document_id}/index", response_model=DocumentOut)
def trigger_indexing(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Document:
    """
    Embeds this document's chunks and adds them to both the dense (FAISS) and
    BM25 indexes. Separate from upload because embedding is comparatively slow
    and worth re-running independently (e.g. after changing EMBEDDING_MODEL_NAME)
    without re-uploading the file.
    """
    doc = _get_or_404(db, document_id)
    if not doc.chunks:
        raise HTTPException(status_code=409, detail="Document has no chunks yet (chunking may have failed).")
    try:
        return index_document(db, doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    doc = _get_or_404(db, document_id)
    remove_document_from_indexes(db, document_id)
    delete_document(db, doc)
