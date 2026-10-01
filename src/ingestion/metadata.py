"""Metadata models for document ingestion."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata attached to each ingested document."""

    document_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    document_name: str = ""
    file_hash: str = ""
    page_count: int = 0
    chunk_count: int = 0
    uploaded_at: float = field(default_factory=time.time)
    indexed: bool = False
    error: str = ""

    def compute_hash(self, filepath: Path | str) -> str:
        """Compute SHA-256 hash from file path."""
        import src.utils.hashing as h

        self.file_hash = h.file_hash(filepath)
        return self.file_hash


@dataclass(slots=True)
class ChunkMetadata(DocumentMetadata):
    """Per-chunk metadata enriched with page and chunk position."""

    page_number: int = 1
    chunk_index: int = 0
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        """Serialize to flat dict for vector store metadata."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "source": self.document_name,
            "file_hash": self.file_hash,
            "uploaded_at": self.uploaded_at,
        }
