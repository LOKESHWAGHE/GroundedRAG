
"""Logging setup (stdlib logging; services use `logging.getLogger(__name__)`)."""

import logging
import sys

from app.core.config import get_settings

logger = logging.getLogger("grounded_rag")


def configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().LOG_LEVEL.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "faiss",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
