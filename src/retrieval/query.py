"""Query processing: normalization, validation, and conversational rewriting."""

from __future__ import annotations

import re

from src.config.settings import Settings
from src.utils.exceptions import QueryProcessingError
from src.utils.logging import get_logger

logger = get_logger("retrieval")

MAX_QUERY_LENGTH = 2000


def normalize_query(query: str) -> str:
    """Normalize a user query: trim, collapse whitespace, strip control chars."""
    if not query:
        raise QueryProcessingError("Query is empty.")

    # Strip leading/trailing whitespace
    cleaned = query.strip()

    # Remove control characters (keep newline, tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Collapse multiple whitespace into one
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned:
        raise QueryProcessingError("Query is empty (whitespace-only).")

    return cleaned


def validate_query(query: str, *, max_length: int = MAX_QUERY_LENGTH) -> list[str]:
    """Validate a query. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    if not query or not query.strip():
        errors.append("Question cannot be empty.")
        return errors

    if len(query) > max_length:
        errors.append(f"Question is too long ({len(query)} chars). Maximum allowed: {max_length} characters.")

    return errors


def process_query(
    query: str,
    settings: Settings,
    conversation_history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """Process a raw query into a clean, validated search string.

    Args:
        query: Raw user input.
        settings: Application settings.
        conversation_history: Optional list of (role, content) tuples.

    Returns:
        Tuple of (processed_query, list_of_warnings).
    """
    warnings: list[str] = []
    raw = normalize_query(query)

    # Validate
    errors = validate_query(raw)
    if errors:
        raise QueryProcessingError("; ".join(errors))

    # Basic conversational context awareness
    if conversation_history and len(conversation_history) > 0:
        last_user = None
        for role, content in reversed(conversation_history):
            if role == "user":
                last_user = content
                break
        if last_user and len(raw) < 50 and not raw.endswith("?"):
            # Short follow-up without question mark — likely contextual
            warnings.append("Short follow-up detected. Consider rephrasing for clarity.")

    return raw, warnings
