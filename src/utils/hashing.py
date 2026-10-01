"""File hashing utilities for duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_hash(filepath: Path | str, algorithm: str = "sha256") -> str:
    """Return hex digest of a file using the specified algorithm."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def bytes_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Return hex digest of in-memory bytes."""
    return hashlib.new(algorithm, data).hexdigest()
