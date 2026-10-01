"""Vector store collection management."""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger("vectorstore")


# Collection name constants
DOCUMENTS_COLLECTION = "pdf_documents"
SESSION_COLLECTION_PREFIX = "session_"


def build_session_collection_name(session_id: str) -> str:
    """Build a collection name for a specific session."""
    return f"{SESSION_COLLECTION_PREFIX}{session_id}"


def sanitize_collection_name(name: str) -> str:
    """Ensure collection name is Qdrant-compatible."""
    import re

    # Qdrant collection names: alphanumeric, underscores, hyphens, dots
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return sanitized[:64]  # Qdrant max collection name length
