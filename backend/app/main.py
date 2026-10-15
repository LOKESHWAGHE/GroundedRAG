from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import models as _models  # noqa: F401  (registers ORM tables on Base.metadata)
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.logging import configure_logging, logger

settings = get_settings()
configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info(f"{settings.APP_NAME} starting up in '{settings.ENVIRONMENT}' mode")
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.index_path.mkdir(parents=True, exist_ok=True)
    # Stop-gap: create tables directly. Replaced by Alembic migrations before deployment (Phase 16).
    Base.metadata.create_all(bind=engine)
    if settings.PRELOAD_MODELS:
        from app.services.embeddings.model import get_embedding_model
        from app.services.reranking.model import get_cross_encoder
        from app.services.verification.nli_model import get_nli_model

        get_embedding_model()
        get_cross_encoder()
        get_nli_model()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="Hybrid search RAG with claim-level citation verification and groundedness scoring.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches anything that isn't already an HTTPException (those still get FastAPI's normal
    handling). The full exception is logged server-side with a stack trace; the client only
    ever sees a generic message -- never a raw traceback, file path, or internal exception
    string, which could otherwise leak implementation details (library versions, file system
    layout, DB connection info embedded in a driver error, etc.).
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    body: dict[str, str] = {"detail": "An internal error occurred. Please try again."}
    if settings.ENVIRONMENT == "development":
        body["debug_detail"] = f"{type(exc).__name__}: {exc}"
    return JSONResponse(status_code=500, content=body)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"{settings.APP_NAME} API — see /docs for the interactive API explorer."}
