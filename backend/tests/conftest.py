import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.core.database import Base, get_db
from app.main import app
from app.models import Document  # noqa: F401  (registers tables)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    """TestClient wired to an in-memory SQLite DB and a temp upload dir (no Postgres needed)."""
    # DATA_DIR, not UPLOAD_DIR: settings.upload_path/.index_path are computed properties
    # derived from DATA_DIR (see app/core/config.py), not a standalone field.
    monkeypatch.setattr(get_settings(), "DATA_DIR", str(tmp_path / "data"))

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()
