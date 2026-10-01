"""Document chunking with configurable strategy and page-aware metadata."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.metadata import ChunkMetadata
from src.utils.logging import get_logger

logger = get_logger("ingestion")


@dataclass(slots=True)
class ChunkResult:
    """Result of chunking a single PDF page-document."""

    chunks: list[LCDocument]
    metadata: ChunkMetadata


def chunk_documents(
    documents: list[LCDocument],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
    document_name: str = "",
) -> tuple[list[LCDocument], ChunkMetadata]:
    """Split a list of documents into chunks.

    Returns:
        Tuple of (chunked documents, chunk metadata).
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    chunked = splitter.split_documents(documents)

    # Enrich each chunk with metadata
    page_counts: dict[int, int] = {}
    for i, chunk in enumerate(chunked):
        page = chunk.metadata.get("page", 1)
        page_counts[page] = page_counts.get(page, 0) + 1
        chunk.metadata.setdefault("document_name", document_name)
        chunk.metadata.setdefault("file_hash", "")
        chunk.metadata.setdefault("chunk_index", i)
        chunk.metadata.setdefault("page_number", page)

    total_chunks = len(chunked)
    unique_pages = sum(page_counts.values())

    meta = ChunkMetadata(
        document_name=document_name,
        page_count=unique_pages,
        chunk_count=total_chunks,
    )

    logger.info(
        "Chunked '%s': %d chunks (size=%d, overlap=%d)",
        document_name,
        total_chunks,
        chunk_size,
        chunk_overlap,
    )
    return chunked, meta
