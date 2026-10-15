"""
Central configuration. Every tunable in GroundedRAG lives here and is read from
environment variables / `backend/.env`, so the evaluation harness can sweep
values (chunk size, RRF k, top-K, thresholds) without touching pipeline code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "GroundedRAG"
    ENVIRONMENT: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:5173"  # comma-separated
    LOG_LEVEL: str = "INFO"

    # --- Database (PostgreSQL by default; sqlite:// URLs also work for zero-setup local runs) ---
    DATABASE_URL: str = "postgresql+psycopg2://grounded_rag:grounded_rag@localhost:5432/grounded_rag"

    # --- Storage ---
    DATA_DIR: str = ""  # empty -> <repo>/data ; uploads/ and index/ live inside
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".docx"]

    # --- Chunking ---
    # Sizes are counted in whitespace-separated words (a cheap proxy for tokens).
    # The default embedding model truncates input at 256 word-pieces (~190 words),
    # so chunks larger than ~180 words would be silently cut off at embedding time.
    CHUNK_SIZE_TOKENS: int = 180
    CHUNK_OVERLAP_TOKENS: int = 30
    CHUNK_MIN_TOKENS: int = 5
    CHUNK_STRATEGY: str = "structure"  # structure (paragraph/heading-aware) | fixed (baseline)
    HEADING_AWARE_CHUNKING: bool = True

    # --- Embeddings / dense retrieval ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # must match the model above
    EMBEDDING_BATCH_SIZE: int = 32
    DENSE_TOP_K: int = 20

    # --- BM25 ---
    BM25_TOP_K: int = 20
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # --- Hybrid fusion ---
    RRF_K: int = 60
    HYBRID_TOP_N: int = 20  # candidates passed to the reranker

    # --- Reranking / context selection ---
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_M: int = 5
    MIN_RERANK_SCORE: float = 0.0  # >0 refuses to answer when even the best chunk scores below it (tune with evaluation)
    CONTEXT_MAX_WORDS: int = 1200

    # --- LLM (provider-agnostic) ---
    LLM_PROVIDER: str = "anthropic"  # anthropic | openai (any OpenAI-compatible server) | extractive (no LLM)
    LLM_MODEL: str = "claude-sonnet-5"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""  # optional override, e.g. http://localhost:11434/v1 for a local OpenAI-compatible server
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_SEND_TEMPERATURE: bool = True  # set false if your model/provider rejects the parameter

    # --- Claim extraction & verification ---
    CLAIM_EXTRACTION_MODE: str = "llm"  # llm (atomic claims, falls back to sentence) | sentence
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"
    SUPPORT_THRESHOLD: float = 0.70  # entailment probability needed for SUPPORTED
    PARTIAL_THRESHOLD: float = 0.30  # entailment probability needed for PARTIALLY_SUPPORTED
    CONTRADICTION_THRESHOLD: float = 0.50  # contradiction probability that overrides entailment -> UNSUPPORTED
    NLI_MAX_WINDOWS_PER_CHUNK: int = 3  # a long chunk is split into at most this many sentence-windows for NLI
    MAX_REGENERATION_RETRIES: int = 2

    # --- Query input limits ---
    MAX_QUESTION_CHARS: int = 1000

    # --- Evaluation ---
    EVALUATION_DIR: str = ""  # empty -> <repo>/evaluation

    # --- Startup ---
    PRELOAD_MODELS: bool = False  # load embedding/reranker/NLI models at startup instead of on first use

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR).resolve() if self.DATA_DIR else REPO_ROOT / "data"

    @property
    def upload_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def index_path(self) -> Path:
        return self.data_path / "index"

    @property
    def evaluation_path(self) -> Path:
        return Path(self.EVALUATION_DIR).resolve() if self.EVALUATION_DIR else REPO_ROOT / "evaluation"


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()
