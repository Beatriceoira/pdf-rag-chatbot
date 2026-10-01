"""Ingestion pipeline: PDF validation → parsing → chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from src.config.settings import Settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.metadata import DocumentMetadata
from src.ingestion.parser import parse_pdf, parse_pdf_from_bytes
from src.ingestion.pdf_loader import validate_file, validate_in_memory
from src.utils.logging import get_logger

logger = get_logger("ingestion")


@dataclass
class IngestionResult:
    """Result of ingesting a single PDF document."""

    document_meta: DocumentMetadata
    chunks: list[LCDocument] = field(default_factory=list)
    success: bool = False
    error: str = ""


class IngestionPipeline:
    """Orchestrates PDF ingestion: validate → parse → chunk."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def ingest_from_path(self, filepath: Path | str) -> IngestionResult:
        """Ingest a PDF from disk."""
        path = Path(filepath)
        result = IngestionResult(document_meta=DocumentMetadata(document_name=path.name))

        # Validate
        errors = validate_file(path, max_size_mb=self.settings.max_file_size_mb)
        if errors:
            result.error = "; ".join(errors)
            logger.warning("Validation failed for '%s': %s", path.name, errors)
            return result

        # Parse
        try:
            docs, meta = parse_pdf(path)
        except Exception as exc:
            result.error = str(exc)
            return result

        result.document_meta = meta

        # Chunk
        try:
            chunks, chunk_meta = chunk_documents(
                docs,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=self.settings.chunk_separators,
                document_name=path.name,
            )
        except Exception as exc:
            result.error = str(exc)
            return result

        result.chunks = chunks
        result.document_meta.chunk_count = chunk_meta.chunk_count
        result.success = True
        logger.info("Successfully ingested '%s': %d chunks", path.name, len(chunks))
        return result

    def ingest_from_bytes(self, file_bytes: bytes, filename: str) -> IngestionResult:
        """Ingest a PDF from in-memory bytes."""
        result = IngestionResult(document_meta=DocumentMetadata(document_name=filename))

        # Validate
        errors = validate_in_memory(
            file_bytes,
            filename,
            max_size_mb=self.settings.max_file_size_mb,
        )
        if errors:
            result.error = "; ".join(errors)
            logger.warning("Validation failed for '%s': %s", filename, errors)
            return result

        # Parse
        try:
            docs, meta = parse_pdf_from_bytes(file_bytes, filename)
        except Exception as exc:
            result.error = str(exc)
            return result

        result.document_meta = meta

        # Chunk
        try:
            chunks, chunk_meta = chunk_documents(
                docs,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=self.settings.chunk_separators,
                document_name=filename,
            )
        except Exception as exc:
            result.error = str(exc)
            return result

        result.chunks = chunks
        result.document_meta.chunk_count = chunk_meta.chunk_count
        result.success = True
        logger.info("Successfully ingested '%s': %d chunks", filename, len(chunks))
        return result
