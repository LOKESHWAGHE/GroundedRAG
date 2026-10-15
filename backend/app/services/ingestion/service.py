"""
Ingestion orchestration: validate -> dedupe -> store file -> parse -> clean ->
persist pages -> chunk. Embedding + indexing (Phase 4/5) run as a separate,
explicit step (`index_document`) rather than inline here, so re-indexing after
a config change doesn't require re-uploading the file.
"""

import hashlib
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document, DocumentPage, DocumentStatus
from app.services.chunking.service import chunk_document
from app.services.ingestion.cleaning import clean_text
from app.services.ingestion.file_validation import sanitize_filename, validate_extension, validate_magic_bytes
from app.services.ingestion.parsers import ParseError, parse_document

logger = logging.getLogger(__name__)


class DuplicateDocumentError(Exception):
    def __init__(self, existing_id: uuid.UUID) -> None:
        super().__init__(f"Document already uploaded (id={existing_id}).")
        self.existing_id = existing_id


def ingest_document(db: Session, raw_filename: str | None, data: bytes) -> Document:
    """
    Raises FileValidationError / DuplicateDocumentError for rejected uploads.
    A parse or chunking failure does NOT raise: the document is kept with
    status='failed' and an error_message, so the UI can show what went wrong.
    """
    settings = get_settings()

    filename = sanitize_filename(raw_filename)
    ext = validate_extension(filename, settings.ALLOWED_UPLOAD_EXTENSIONS)
    validate_magic_bytes(ext, data)

    sha256 = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.sha256 == sha256))
    if existing is not None:
        raise DuplicateDocumentError(existing.id)

    # The on-disk name is a server-generated UUID: the client's filename never touches the filesystem.
    doc_id = uuid.uuid4()
    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{doc_id}{ext}"
    stored_path.write_bytes(data)

    try:
        doc = Document(
            id=doc_id,
            filename=filename,
            file_type=ext.lstrip("."),
            size_bytes=len(data),
            sha256=sha256,
            stored_path=str(stored_path),
            status=DocumentStatus.PARSING.value,
        )
        db.add(doc)
        db.commit()
    except IntegrityError:  # two identical uploads racing past the check above
        db.rollback()
        stored_path.unlink(missing_ok=True)
        existing = db.scalar(select(Document).where(Document.sha256 == sha256))
        raise DuplicateDocumentError(existing.id if existing else doc_id) from None
    except Exception:
        db.rollback()
        stored_path.unlink(missing_ok=True)
        raise

    try:
        parsed = parse_document(stored_path, ext)
        pages = [(p.page_number, clean_text(p.text)) for p in parsed]
        if not any(text for _, text in pages):
            raise ParseError(
                "No extractable text found. Scanned/image-only documents need OCR, which is not supported."
            )
        doc.pages = [DocumentPage(page_number=n, text=t) for n, t in pages]
        doc.page_count = len(pages)
        doc.char_count = sum(len(t) for _, t in pages)
        doc.status = DocumentStatus.PARSED.value
        logger.info("Parsed '%s': %d page(s), %d chars", filename, doc.page_count, doc.char_count)
        db.commit()

        chunk_document(db, doc)  # -> status CHUNKING, or FAILED if no chunks resulted
    except ParseError as exc:
        doc.status = DocumentStatus.FAILED.value
        doc.error_message = str(exc)
        logger.warning("Parsing failed for '%s': %s", filename, exc)
        db.commit()

    db.refresh(doc)
    return doc


def delete_document(db: Session, doc: Document) -> None:
    """Vector/BM25 index entries for this document are removed by the caller (see api/v1/documents.py)."""
    Path(doc.stored_path).unlink(missing_ok=True)
    db.delete(doc)  # ORM cascade removes DocumentPage and Chunk rows
    db.commit()
