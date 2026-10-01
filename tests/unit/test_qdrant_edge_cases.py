"""Unit tests for Qdrant vector store edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.utils.exceptions import VectorStoreError
from src.vectorstore.qdrant_store import QdrantStore


class TestQdrantStoreEdgeCases:
    """Tests for Qdrant store edge cases and error handling."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(qdrant_mode="memory", collection_name="test_edge", _env_file=None)

    @pytest.fixture
    def store(self, settings):
        """Create a Qdrant store."""
        return QdrantStore(settings)

    def test_build_client_local(self):
        """Test building local client."""
        settings = Settings(qdrant_mode="local", _env_file=None)
        store = QdrantStore(settings)

        with patch("src.vectorstore.qdrant_store.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            store._build_client()
            mock_client.assert_called_once_with(path="./qdrant_data")

    def test_build_client_memory(self):
        """Test building memory client."""
        settings = Settings(qdrant_mode="memory", _env_file=None)
        store = QdrantStore(settings)

        with patch("src.vectorstore.qdrant_store.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            store._build_client()
            mock_client.assert_called_once_with(":memory:")

    def test_build_client_server(self):
        """Test building server client."""
        settings = Settings(qdrant_mode="server", qdrant_url="http://example.com", _env_file=None)
        store = QdrantStore(settings)

        with patch("src.vectorstore.qdrant_store.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            store._build_client()
            mock_client.assert_called_once_with(url="http://example.com")

    def test_build_client_server_with_api_key(self):
        """Test building server client with API key."""
        settings = Settings(
            qdrant_mode="server",
            qdrant_url="http://example.com",
            qdrant_api_key="secret-key",
            _env_file=None,
        )
        store = QdrantStore(settings)

        with patch("src.vectorstore.qdrant_store.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            store._build_client()
            mock_client.assert_called_once_with(url="http://example.com", api_key="secret-key")

    def test_build_client_unknown_mode_raises(self):
        """Test that unknown mode raises VectorStoreError."""
        store = QdrantStore(Settings(qdrant_mode="memory", _env_file=None))
        # Manually set invalid mode to bypass pydantic validation
        store.settings.qdrant_mode = "unknown"  # type: ignore[assignment]

        with pytest.raises(VectorStoreError, match="Unknown Qdrant mode"):
            store._build_client()

    def test_initialize_caches_client(self, store):
        """Test that initialize caches the client."""
        store._embedding = MagicMock()
        store._embedding.dimension = 384

        with patch.object(store, "_build_client") as mock_build:
            mock_instance = MagicMock()
            mock_instance.collection_exists.return_value = False
            mock_instance.create_collection.return_value = None
            mock_build.return_value = mock_instance

            mock_qdrant = MagicMock()
            with patch("src.vectorstore.qdrant_store.QdrantVectorStore", mock_qdrant):
                store.initialize()

            assert store._client == mock_instance
            assert store._initialized is True

    def test_initialize_deletes_existing_collection(self, store):
        """Test that initialize deletes an existing collection before recreating."""
        store._embedding = MagicMock()
        store._embedding.dimension = 384

        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True
        mock_instance.create_collection.return_value = None

        with patch.object(store, "_build_client", return_value=mock_instance):
            with patch("src.vectorstore.qdrant_store.QdrantVectorStore"):
                store.initialize()

            mock_instance.delete_collection.assert_called_once_with(store.settings.collection_name)

    def test_add_documents_auto_initializes(self, store):
        """Test that add_documents initializes if needed."""
        with patch.object(store, "initialize") as mock_init:
            store._vectorstore = MagicMock()
            store.add_documents([])
            mock_init.assert_called_once()
            store._vectorstore.add_documents.assert_called_once_with([])

    def test_as_retriever_auto_initializes(self, store):
        """Test that as_retriever initializes if needed."""
        with patch.object(store, "initialize") as mock_init:
            store._vectorstore = MagicMock()
            store.as_retriever(k=5)
            mock_init.assert_called_once()
            store._vectorstore.as_retriever.assert_called_once()

    def test_search_auto_initializes(self, store):
        """Test that search initializes if needed."""
        with patch.object(store, "initialize") as mock_init:
            store._initialized = False
            store._embedding = MagicMock()
            store._embedding.embed_query.return_value = [0.1] * 384
            store._client = MagicMock()
            mock_point = MagicMock()
            mock_point.score = 0.9
            mock_point.payload = {"page_content": "test"}
            mock_result = MagicMock()
            mock_result.points = [mock_point]
            store._client.query_points.return_value = mock_result

            results = store.search("test query", k=5)

            mock_init.assert_called_once()
            assert len(results) == 1
            assert results[0]["score"] == 0.9

    def test_get_collection_info_not_exists(self, store):
        """Test get_collection_info when collection doesn't exist."""
        store._client = MagicMock()
        store._client.collection_exists.return_value = False

        info = store.get_collection_info()
        assert info["exists"] is False
        assert info["points_count"] == 0

    def test_delete_collection_not_exists(self, store):
        """Test delete_collection when collection doesn't exist."""
        store._client = MagicMock()
        store._client.collection_exists.return_value = False

        store.delete_collection()
        store._client.delete_collection.assert_not_called()

    def test_clear_collection_not_exists(self, store):
        """Test clear when collection doesn't exist."""
        store._client = MagicMock()
        store._client.collection_exists.return_value = False

        store.clear()
        store._client.delete_collection.assert_not_called()

    def test_search_with_filter(self, store):
        """Test search with filter conditions."""
        store._initialized = True
        store._embedding = MagicMock()
        store._embedding.embed_query.return_value = [0.1] * 384
        store._client = MagicMock()

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {"document_name": "test.pdf", "page_number": 1}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        store._client.query_points.return_value = mock_result

        results = store.search("query", k=5, filter_conditions={"document_name": "test.pdf"})

        assert len(results) == 1
        store._client.query_points.assert_called_once()
        call_kwargs = store._client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    def test_search_no_filter(self, store):
        """Test search without filter conditions."""
        store._initialized = True
        store._embedding = MagicMock()
        store._embedding.embed_query.return_value = [0.1] * 384
        store._client = MagicMock()

        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {"page_content": "test"}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        store._client.query_points.return_value = mock_result

        results = store.search("query", k=5)

        assert len(results) == 1
        call_kwargs = store._client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None

    def test_search_raises_on_client_error(self, store):
        """Test that search propagates client errors."""
        store._initialized = True
        store._embedding = MagicMock()
        store._embedding.embed_query.return_value = [0.1] * 384
        store._client = MagicMock()
        store._client.query_points.side_effect = RuntimeError("Qdrant unavailable")

        with pytest.raises(RuntimeError, match="Qdrant unavailable"):
            store.search("query", k=5)

    def test_collection_info_with_config(self, store):
        """Test collection info with vector config."""
        store._client = MagicMock()
        store._client.collection_exists.return_value = True

        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_info.config.params.vectors.size = 384
        store._client.get_collection.return_value = mock_info

        info = store.get_collection_info()
        assert info["exists"] is True
        assert info["points_count"] == 100
        assert info["config"]["vector_size"] == 384
