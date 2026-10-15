import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class QueryRecord(Base):
    """
    One answered question. The full result (answer, claims, verdicts, retrieval trace)
    is stored as a JSON document: it is always read whole, never queried field-by-field.
    """

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    groundedness: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
