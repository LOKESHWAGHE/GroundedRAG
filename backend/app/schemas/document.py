import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentOut(BaseModel):
    """Public view of a document. Deliberately omits the server-side stored path."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    error_message: str | None
    page_count: int
    char_count: int
    chunk_count: int
    created_at: datetime


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    char_count: int
    text: str
