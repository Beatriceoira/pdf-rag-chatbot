"""Structured logging utilities."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def _request_id() -> str:
    rid = request_id_var.get()
    return f"[{rid}]" if rid else ""


class RequestIdFilter(logging.Filter):
    """Inject the current request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s %(request_id)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(numeric_level)
    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(handler)


@contextmanager
def request_context() -> Iterator[None]:
    """Yield a new request_id for the duration of a context."""
    token = request_id_var.set(uuid.uuid4().hex[:8])
    try:
        yield
    finally:
        request_id_var.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger."""
    return logging.getLogger(f"rag_chatbot.{name}")
