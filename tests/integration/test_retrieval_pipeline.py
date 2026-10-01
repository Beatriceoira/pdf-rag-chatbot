"""Integration tests for the retrieval pipeline."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document as LCDocument

from src.config.settings import Settings
from src.retrieval.retriever import Retriever
from src.vectorstore.qdrant_store import QdrantStore


class TestRetrievalPipeline:
    """Integration tests for retrieval."""

    @pytest.fixture
    def store(self):
        """Create an in-memory Qdrant store."""
        settings = Settings(qdrant_mode="memory", collection_name="test_retrieve")
        return QdrantStore(settings)

    @pytest.fixture
    def retriever(self, store):
        """Create a retriever with the store."""
        settings = Settings(qdrant_mode="memory", collection_name="test_retrieve")
        return Retriever(settings, store)

    def test_basic_retrieval(self, retriever, store):
        """Test basic document retrieval."""
        store.initialize()
        docs = [
            LCDocument(
                page_content="The company provides 20 vacation days per year.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            ),
            LCDocument(
                page_content="Employees receive 10 sick days annually.",
                metadata={"document_name": "policy.pdf", "page_number": 2},
            ),
            LCDocument(
                page_content="The product warranty is 1 year.",
                metadata={"document_name": "manual.pdf", "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        result = retriever.retrieve("How many vacation days?", k=2)
        assert len(result.documents) > 0
        assert result.k == 2
        assert len(result.scores) == len(result.documents)

    def test_retrieval_with_filtering(self, retriever, store):
        """Test retrieval with metadata filtering."""
        store.initialize()
        docs = [
            LCDocument(
                page_content="Policy A: 20 days vacation.",
                metadata={"document_name": "policy_a.pdf", "page_number": 1},
            ),
            LCDocument(
                page_content="Policy B: 30 days vacation.",
                metadata={"document_name": "policy_b.pdf", "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        result = retriever.retrieve(
            "vacation days",
            k=2,
            filter_conditions={"document_name": "policy_a.pdf"},
        )
        # Should only return docs from policy_a
        for doc in result.documents:
            assert doc.metadata.get("document_name") == "policy_a.pdf"

        # Verify filtering metadata is set on the result
        assert result.filtering is not None
        assert result.filtering["document_name"] == "policy_a.pdf"

    def test_empty_retrieval(self, retriever, store):
        """Test retrieval with no documents in store."""
        store.initialize()

        result = retriever.retrieve("anything", k=5)
        assert len(result.documents) == 0
        assert result.scores == []

    def test_retrieve_and_format(self, retriever, store):
        """Test the retrieve_and_format convenience method."""
        store.initialize()
        docs = [
            LCDocument(
                page_content="The refund period is 30 days.",
                metadata={"document_name": "policy.pdf", "page_number": 3},
            ),
        ]
        store.add_documents(docs)

        contexts, sources = retriever.retrieve_and_format("refund policy", k=1)
        assert len(contexts) > 0
        assert len(sources) > 0
        assert "document_name" in sources[0]
        assert "page_number" in sources[0]
