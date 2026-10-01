"""Integration tests for Qdrant vector store."""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.vectorstore.qdrant_store import QdrantStore


class TestQdrantStore:
    """Integration tests for Qdrant vector store operations."""

    @pytest.fixture
    def store(self):
        """Create a Qdrant store in memory mode for testing."""
        settings = Settings(
            qdrant_mode="memory",
            collection_name="test_collection",
            embedding_provider="local",
        )
        return QdrantStore(settings)

    def test_collection_creation(self, store):
        """Test that a collection can be created."""
        store.initialize()
        info = store.get_collection_info()
        assert info["exists"] is True

    def test_collection_exists(self, store):
        """Test collection existence check."""
        store.initialize()
        info = store.get_collection_info()
        assert info["exists"] is True

    def test_vector_insertion(self, store):
        """Test inserting vectors into the store."""
        from langchain_core.documents import Document as LCDocument

        store.initialize()

        docs = [
            LCDocument(
                page_content="The company provides 20 vacation days.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            ),
            LCDocument(
                page_content="Refund period is 30 days.",
                metadata={"document_name": "policy.pdf", "page_number": 2},
            ),
        ]

        store.add_documents(docs)
        info = store.get_collection_info()
        assert info["points_count"] == 2

    def test_vector_search(self, store):
        """Test searching for relevant vectors."""
        from langchain_core.documents import Document as LCDocument

        store.initialize()
        docs = [
            LCDocument(
                page_content="The vacation allowance is 20 days per year.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            ),
            LCDocument(
                page_content="Employees get 10 sick days annually.",
                metadata={"document_name": "policy.pdf", "page_number": 2},
            ),
        ]
        store.add_documents(docs)

        results = store.search("vacation days", k=2)
        assert len(results) > 0
        assert "score" in results[0]
        assert "document" in results[0]

    def test_metadata_filtering(self, store):
        """Test filtering by metadata."""
        from langchain_core.documents import Document as LCDocument

        store.initialize()
        docs = [
            LCDocument(
                page_content="Policy A says 20 days.",
                metadata={"document_name": "policy_a.pdf", "page_number": 1},
            ),
            LCDocument(
                page_content="Policy B says 30 days.",
                metadata={"document_name": "policy_b.pdf", "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        # Search filtered to specific document
        results = store.search("days", k=2, filter_conditions={"document_name": "policy_a.pdf"})
        assert len(results) <= 1
        if results:
            assert results[0]["document"].get("document_name") == "policy_a.pdf"

    def test_search_returns_empty_when_no_match(self, store):
        """Test search returns empty list when nothing matches."""
        store.initialize()
        results = store.search("zzzzzzz", k=5)
        assert results == []

    def test_delete_collection(self, store):
        """Test deleting a collection."""
        store.initialize()
        assert store.get_collection_info()["exists"] is True

        store.delete_collection()
        info = store.get_collection_info()
        assert info["exists"] is False

    def test_clear_collection(self, store):
        """Test clearing a collection (recreate empty)."""
        from langchain_core.documents import Document as LCDocument

        store.initialize()
        docs = [
            LCDocument(
                page_content="Some content.",
                metadata={"document_name": "test.pdf", "page_number": 1},
            ),
        ]
        store.add_documents(docs)
        assert store.get_collection_info()["points_count"] > 0

        store.clear()
        info = store.get_collection_info()
        assert info["exists"] is True
        assert info["points_count"] == 0
