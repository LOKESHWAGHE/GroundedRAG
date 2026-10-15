"""ORM models: documents, their cleaned pages, and chunk metadata."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    """uploaded -> parsing -> parsed -> chunking -> embedding -> indexed (any step can end in failed)."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


PROCESSING_STATUSES = {
    DocumentStatus.UPLOADED.value,
    DocumentStatus.PARSING.value,
    DocumentStatus.PARSED.value,
    DocumentStatus.CHUNKING.value,
    DocumentStatus.EMBEDDING.value,
}


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))  # sanitised, display only
    file_type: Mapped[str] = mapped_column(String(10))  # pdf | txt | md | docx
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    stored_path: Mapped[str] = mapped_column(String(1024))  # server-side only, never exposed via the API
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.UPLOADED.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentPage.page_number"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )

    @property
    def embedding_ready(self) -> bool:
        return self.status == DocumentStatus.INDEXED.value

    @property
    def bm25_ready(self) -> bool:
        return self.status == DocumentStatus.INDEXED.value


class DocumentPage(Base):
    """Cleaned text of one page (TXT/MD/DOCX are stored as a single page 1)."""

    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_document_page"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)  # 1-indexed
    text: Mapped[str] = mapped_column(Text)

    document: Mapped["Document"] = relationship(back_populates="pages")


class Chunk(Base):
    """
    Chunk metadata + text. `id` doubles as the FAISS vector id and the BM25 document id,
    which is what keeps the three stores joinable.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)

    document: Mapped["Document"] = relationship(back_populates="chunks")
