"""PDF parsing with error handling and page metadata."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from src.ingestion.metadata import DocumentMetadata
from src.utils.exceptions import PDFProcessingError
from src.utils.logging import get_logger

logger = get_logger("ingestion")

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # pragma: no cover
    from langchain_community.document_loaders import PDFLoader as PyPDFLoader


def parse_pdf(filepath: Path | str) -> tuple[list[LCDocument], DocumentMetadata]:
    """Parse a PDF file into LangChain documents with metadata.

    Returns:
        Tuple of (list of Document, DocumentMetadata).
    """
    path = Path(filepath)
    meta = DocumentMetadata(document_name=path.name)
    meta.compute_hash(path)

    try:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    except Exception as exc:
        logger.error("Failed to parse PDF %s: %s", path.name, exc, exc_info=True)
        meta.error = str(exc)
        raise PDFProcessingError(f"Failed to parse '{path.name}': {exc}") from exc

    meta.page_count = max(d.metadata.get("page", 0) for d in docs) if docs else 0

    # Enrich each document with document-level metadata
    for doc in docs:
        doc.metadata.setdefault("document_name", path.name)
        doc.metadata.setdefault("file_hash", meta.file_hash)

    logger.info(
        "Parsed '%s': %d pages, %d chunks after splitting later",
        path.name,
        meta.page_count,
        0,  # chunk count computed after chunking
    )
    return docs, meta


def parse_pdf_from_bytes(file_bytes: bytes, filename: str) -> tuple[list[LCDocument], DocumentMetadata]:
    """Parse PDF from in-memory bytes (e.g. uploaded file stream)."""
    meta = DocumentMetadata(document_name=filename)
    meta.compute_hash(BytesIO(file_bytes))

    try:
        bio = BytesIO(file_bytes)
        loader = PyPDFLoader(str(bio))
        docs = loader.load()
    except Exception as exc:
        logger.error("Failed to parse in-memory PDF '%s': %s", filename, exc)
        meta.error = str(exc)
        raise PDFProcessingError(f"Failed to parse '{filename}': {exc}") from exc

    meta.page_count = max(d.metadata.get("page", 0) for d in docs) if docs else 0

    for doc in docs:
        doc.metadata.setdefault("document_name", filename)
        doc.metadata.setdefault("file_hash", meta.file_hash)

    return docs, meta
