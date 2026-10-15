"""
Regression test for a real bug caught during the Docker review: Settings.data_path
derives from the backend package's own file location (REPO_ROOT = backend/..),
which only resolves correctly when the full repo is on disk. Inside the Docker
container, only backend/ is copied in, so that default silently misses the
docker-compose `./data:/app/data` volume mount unless DATA_DIR is set explicitly
(which docker-compose.yml now does). This test locks in the property logic so the
assumption is explicit, not just asserted in a docker-compose.yml comment.
"""

from pathlib import Path

from app.core.config import Settings


def test_data_path_defaults_to_repo_root_data_dir_when_unset():
    settings = Settings(DATA_DIR="")
    assert settings.data_path.name == "data"
    assert settings.upload_path == settings.data_path / "uploads"
    assert settings.index_path == settings.data_path / "index"


def test_data_path_respects_explicit_override():
    settings = Settings(DATA_DIR="/app/data")
    assert settings.data_path == Path("/app/data").resolve()
    assert settings.upload_path == Path("/app/data/uploads").resolve()


def test_cors_origins_list_parses_comma_separated_string():
    settings = Settings(CORS_ORIGINS="http://localhost:5173, http://example.com")
    assert settings.cors_origins_list == ["http://localhost:5173", "http://example.com"]


def test_cors_origins_list_handles_single_origin():
    settings = Settings(CORS_ORIGINS="http://localhost:5173")
    assert settings.cors_origins_list == ["http://localhost:5173"]
