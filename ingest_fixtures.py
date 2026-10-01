#!/usr/bin/env python3
"""Ingest the fixture PDFs into the vector store for evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.embeddings.factory import EmbeddingFactory
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.qdrant_store import QdrantStore


def main():
    # Initialize settings
    settings = Settings()
    # Use local Qdrant mode for simplicity
    settings.qdrant_mode = "local"
    # Use local embeddings to avoid needing an API key
    settings.embedding_provider = "local"
    # Use a default LLM provider (we won't actually call the LLM for ingestion)
    settings.llm_provider = "openai"  # This will fail if no API key, but we don't need the LLM for ingestion

    # Initialize components
    embedding_model = EmbeddingFactory.create(settings)
    store = QdrantStore(settings)
    pipeline = IngestionPipeline(settings)

    # Ingest each fixture PDF
    fixture_dir = Path("benchmark_fixtures")
    pdf_files = [
        fixture_dir / "company_policy.pdf",
        fixture_dir / "product_manual.pdf",
        fixture_dir / "employee_handbook.pdf",
    ]

    for pdf_path in pdf_files:
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            continue

        print(f"Ingesting {pdf_path.name}...")
        result = pipeline.ingest_from_path(pdf_path)
        if not result.success:
            print(f"  Failed to ingest {pdf_path.name}: {result.error}")
            continue

        # Embed the chunks
        from langchain_core.documents import Document as LCDocument
        texts = [doc.page_content for doc in result.chunks]
        embeddings = embedding_model.embed_documents(texts)

        # Insert into vector store
        embedded_docs = []
        for j, doc in enumerate(result.chunks):
            embedded_doc = LCDocument(
                page_content=doc.page_content,
                metadata={**doc.metadata, "embedding": embeddings[j]},
            )
            embedded_docs.append(embedded_doc)
        store.add_documents(embedded_docs)

        print(f"  Successfully ingested {len(result.chunks)} chunks from {pdf_path.name}")

    print("All fixture PDFs ingested.")


if __name__ == "__main__":
    main()