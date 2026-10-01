"""Unit tests for the retriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.retrieval.retriever import RetrievalResult, Retriever
from src.utils.exceptions import RetrievalError


class TestRetriever:
    """Tests for the Retriever class."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock Qdrant store."""
        store = MagicMock()
        store.as_retriever.return_value = MagicMock()
        return store

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(_env_file=None)

    @pytest.fixture
    def retriever(self, settings, mock_store):
        """Create a retriever instance."""
        return Retriever(settings, mock_store)

    def test_initializes_retriever_on_first_use(self, retriever, mock_store):
        """Test that retriever is initialized lazily."""
        retriever._ensure_retriever()
        mock_store.as_retriever.assert_called_once_with(k=retriever.settings.retrieval_k)

    def test_retrieve_returns_result(self, retriever, mock_store):
        """Test basic retrieve returns a RetrievalResult."""
        mock_store.search.return_value = [
            {
                "score": 0.9,
                "document": {
                    "page_content": "The vacation policy is 20 days.",
                    "document_name": "policy.pdf",
                    "page_number": 1,
                },
            }
        ]

        result = retriever.retrieve("vacation days", k=5)

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 1
        assert len(result.scores) == 1
        assert result.documents[0].page_content == "The vacation policy is 20 days."
        assert result.filtering is None

    def test_retrieve_with_filtering(self, retriever, mock_store):
        """Test retrieve with filter conditions."""
        mock_store.search.return_value = [
            {
                "score": 0.85,
                "document": {
                    "page_content": "Policy A content",
                    "document_name": "policy_a.pdf",
                    "page_number": 1,
                },
            }
        ]

        result = retriever.retrieve("days", k=3, filter_conditions={"document_name": "policy_a.pdf"})

        assert result.filtering == {"document_name": "policy_a.pdf"}
        assert len(result.documents) == 1

    def test_retrieve_returns_empty(self, retriever, mock_store):
        """Test retrieve returns empty result when no documents match."""
        mock_store.search.return_value = []

        result = retriever.retrieve("unknown query", k=5)

        assert result.documents == []
        assert result.scores == []

    def test_retrieve_raises_on_store_error(self, retriever, mock_store):
        """Test that retrieval errors are wrapped in RetrievalError."""
        mock_store.search.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RetrievalError, match="Retrieval failed"):
            retriever.retrieve("test query")

    def test_retrieve_uses_custom_k(self, retriever, mock_store):
        """Test that custom k value is passed through."""
        mock_store.search.return_value = []

        retriever.retrieve("query", k=10)

        mock_store.search.assert_called_once_with(query="query", k=10, filter_conditions=None)

    def test_retrieve_and_format(self, retriever, mock_store):
        """Test retrieve_and_format returns contexts and sources."""
        mock_store.search.return_value = [
            {
                "score": 0.9,
                "document": {
                    "page_content": "The refund period is 30 days.",
                    "document_name": "policy.pdf",
                    "page_number": 3,
                },
            }
        ]

        contexts, sources = retriever.retrieve_and_format("refund", k=2)

        assert len(contexts) == 1
        assert contexts[0] == "The refund period is 30 days."
        assert len(sources) == 1
        assert sources[0]["document_name"] == "policy.pdf"
        assert sources[0]["page_number"] == 3

    def test_retrieve_and_format_long_content_truncates(self, retriever, mock_store):
        """Test that long content is truncated in sources."""
        long_content = "x" * 300
        mock_store.search.return_value = [
            {
                "score": 0.8,
                "document": {
                    "page_content": long_content,
                    "document_name": "long.pdf",
                    "page_number": 1,
                },
            }
        ]

        _, sources = retriever.retrieve_and_format("query")

        assert len(sources[0]["text_preview"]) < len(long_content)
        assert "…" in sources[0]["text_preview"]

    def test_retrieve_preserves_scores(self, retriever, mock_store):
        """Test that scores are correctly mapped."""
        mock_store.search.return_value = [
            {"score": 0.95, "document": {"page_content": "A", "document_name": "a.pdf", "page_number": 1}},
            {"score": 0.85, "document": {"page_content": "B", "document_name": "b.pdf", "page_number": 1}},
        ]

        result = retriever.retrieve("query")

        assert result.scores[0] == 0.95
        assert result.scores[1] == 0.85
