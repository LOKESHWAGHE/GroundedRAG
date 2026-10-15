"""
SQLAlchemy engine + session factory.

The relational DB (PostgreSQL by default) is the source of truth for documents,
pages, chunk metadata and saved query results. Dense vectors live in FAISS and
BM25 term statistics in a pickle file; both are keyed by `chunks.id`.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_factory() -> sessionmaker:
    """FastAPI dependency for background tasks that need their own session."""
    return SessionLocal
