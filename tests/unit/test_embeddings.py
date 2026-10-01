"""Unit tests for embedding factory and wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.embeddings.factory import EmbeddingFactory
from src.utils.exceptions import EmbeddingError


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory."""

    def test_create_openai_with_key(self):
        """Test creating OpenAI embeddings when API key is present."""
        settings = Settings(openai_api_key="sk-test", embedding_provider="openai", _env_file=None)

        with patch("src.embeddings.factory.OpenAIEmbeddingsWrapper") as mock_wrapper:
            EmbeddingFactory.create(settings)
            mock_wrapper.assert_called_once_with(model="text-embedding-3-small")

    def test_create_openai_without_key_raises(self):
        """Test that OpenAI embeddings raise when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(openai_api_key="", embedding_provider="openai", _env_file=None)

            with patch("src.embeddings.factory.OpenAIEmbeddingsWrapper") as mock_wrapper:
                with pytest.raises(EmbeddingError, match="OpenAI API key is required"):
                    EmbeddingFactory.create(settings)
                mock_wrapper.assert_not_called()

    def test_create_local_by_default(self):
        """Test creating local embeddings by default."""
        settings = Settings(embedding_provider="local", _env_file=None)

        with patch("src.embeddings.factory.LocalEmbeddings") as mock_local:
            EmbeddingFactory.create(settings)
            mock_local.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_create_local_with_custom_model(self):
        """Test creating local embeddings with custom model."""
        settings = Settings(
            embedding_provider="local",
            embedding_model="custom-model",
            _env_file=None,
        )

        with patch("src.embeddings.factory.LocalEmbeddings") as mock_local:
            EmbeddingFactory.create(settings)
            mock_local.assert_called_once_with(model_name="custom-model")


class TestOpenAIEmbeddingsWrapper:
    """Tests for OpenAIEmbeddingsWrapper."""

    def test_dimension_probe(self):
        """Test dimension property triggers probe."""
        with patch("src.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_embeddings:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.1] * 384
            mock_embeddings.return_value = mock_instance

            from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper

            wrapper = OpenAIEmbeddingsWrapper()
            dim = wrapper.dimension

            assert dim == 384
            mock_instance.embed_query.assert_called_once_with("dimension probe text")

    def test_embed_documents(self):
        """Test embedding a batch of documents."""
        with patch("src.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_embeddings:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            mock_embeddings.return_value = mock_instance

            from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper

            wrapper = OpenAIEmbeddingsWrapper()
            result = wrapper.embed_documents(["doc1", "doc2"])

            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_instance.embed_documents.assert_called_once_with(["doc1", "doc2"])

    def test_embed_query(self):
        """Test embedding a single query."""
        with patch("src.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_embeddings:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_embeddings.return_value = mock_instance

            from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper

            wrapper = OpenAIEmbeddingsWrapper()
            result = wrapper.embed_query("hello world")

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_query.assert_called_once_with("hello world")

    def test_repr(self):
        """Test string representation."""
        with patch("src.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_embeddings:
            mock_instance = MagicMock()
            mock_instance.model = "text-embedding-3-small"
            mock_embeddings.return_value = mock_instance

            from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper

            wrapper = OpenAIEmbeddingsWrapper()
            assert "text-embedding-3-small" in repr(wrapper)

    def test_init_without_langchain_raises(self):
        """Test that init raises when langchain-openai is not installed."""

        def _create():
            from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper
            OpenAIEmbeddingsWrapper()

        with patch("src.embeddings.openai_embeddings.OpenAIEmbeddings", None), pytest.raises(
            ImportError, match="langchain-openai"
        ):
            _create()


class TestLocalEmbeddings:
    """Tests for LocalEmbeddings."""

    def test_dimension_probe(self):
        """Test dimension property triggers probe."""
        with patch("src.embeddings.local_embeddings._HuggingFaceEmbeddings") as mock_hf:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.1] * 384
            mock_hf.return_value = mock_instance

            from src.embeddings.local_embeddings import LocalEmbeddings

            wrapper = LocalEmbeddings()
            dim = wrapper.dimension
            assert dim == 384

    def test_embed_documents(self):
        """Test embedding a batch of documents."""
        with patch("src.embeddings.local_embeddings._HuggingFaceEmbeddings") as mock_hf:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = [[0.1], [0.2]]
            mock_hf.return_value = mock_instance

            from src.embeddings.local_embeddings import LocalEmbeddings

            wrapper = LocalEmbeddings()
            result = wrapper.embed_documents(["doc1", "doc2"])
            assert result == [[0.1], [0.2]]

    def test_embed_query(self):
        """Test embedding a single query."""
        with patch("src.embeddings.local_embeddings._HuggingFaceEmbeddings") as mock_hf:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.1, 0.2]
            mock_hf.return_value = mock_instance

            from src.embeddings.local_embeddings import LocalEmbeddings

            wrapper = LocalEmbeddings()
            result = wrapper.embed_query("hello")
            assert result == [0.1, 0.2]

    def test_repr(self):
        """Test string representation."""
        with patch("src.embeddings.local_embeddings._HuggingFaceEmbeddings") as mock_hf:
            mock_instance = MagicMock()
            mock_instance.model_name = "all-MiniLM-L6-v2"
            mock_hf.return_value = mock_instance

            from src.embeddings.local_embeddings import LocalEmbeddings

            wrapper = LocalEmbeddings()
            assert "all-MiniLM-L6-v2" in repr(wrapper)

    def test_init_without_langchain_raises(self):
        """Test that init raises when langchain-huggingface is not installed."""

        def _create():
            from src.embeddings.local_embeddings import LocalEmbeddings
            LocalEmbeddings()

        with patch("src.embeddings.local_embeddings._HuggingFaceEmbeddings", None), pytest.raises(
            ImportError, match="langchain-huggingface"
        ):
            _create()
