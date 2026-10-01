"""Unit tests for reranker implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document as LCDocument

from src.retrieval.reranker import CrossEncoderReranker, IdentityReranker, RerankerFactory


class TestIdentityReranker:
    """Tests for the identity reranker."""

    def test_rerank_returns_top_k(self):
        """Test rerank returns top_k documents."""
        docs = [
            LCDocument(page_content="A", metadata={"score": 0.9}),
            LCDocument(page_content="B", metadata={"score": 0.8}),
            LCDocument(page_content="C", metadata={"score": 0.7}),
        ]

        reranker = IdentityReranker()
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "A"
        assert result[1].page_content == "B"

    def test_rerank_top_k_larger_than_docs(self):
        """Test rerank when top_k > len(docs)."""
        docs = [LCDocument(page_content="Only one doc")]
        reranker = IdentityReranker()
        result = reranker.rerank("query", docs, top_k=10)
        assert len(result) == 1

    def test_rerank_empty(self):
        """Test rerank with empty input."""
        reranker = IdentityReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_preserves_order(self):
        """Test that identity reranker preserves original order."""
        docs = [LCDocument(page_content=f"Doc {i}", metadata={"score": 0.9 - i * 0.1}) for i in range(5)]

        reranker = IdentityReranker()
        result = reranker.rerank("query", docs, top_k=3)

        for i, doc in enumerate(result):
            assert doc.page_content == f"Doc {i}"


class TestCrossEncoderReranker:
    """Tests for the cross-encoder reranker."""

    def test_init_sets_model_name(self):
        """Test initialization sets model name."""
        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker._model_name == "custom/model"
        assert reranker._model is None

    def test_load_model_success(self):
        """Test successful model loading."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]

        with patch("sentence_transformers.CrossEncoder") as mock_cross:
            mock_cross.return_value = mock_model
            reranker._load_model()

            assert reranker._model is not None
            mock_cross.assert_called_once_with(reranker._model_name)

    def test_load_model_import_error(self):
        """Test load_model raises on import error."""
        reranker = CrossEncoderReranker()

        with pytest.raises(ImportError, match="sentence-transformers"), patch(
            "sentence_transformers.CrossEncoder", side_effect=ImportError("no module")
        ):
            reranker._load_model()

    def test_rerank_calls_model_predict(self):
        """Test rerank calls model.predict with correct pairs."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.9, 0.7, 0.8]

        docs = [
            LCDocument(page_content="Doc A"),
            LCDocument(page_content="Doc B"),
            LCDocument(page_content="Doc C"),
        ]

        reranker.rerank("query", docs, top_k=2)

        reranker._model.predict.assert_called_once()
        pairs = reranker._model.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0] == ("query", "Doc A")

    def test_rerank_returns_sorted_top_k(self):
        """Test rerank returns documents sorted by score."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.7, 0.9, 0.8]

        docs = [
            LCDocument(page_content="Doc A"),
            LCDocument(page_content="Doc B"),
            LCDocument(page_content="Doc C"),
        ]

        result = reranker.rerank("query", docs, top_k=2)

        assert result[0].page_content == "Doc B"
        assert result[1].page_content == "Doc C"

    def test_rerank_empty_documents(self):
        """Test rerank with empty document list."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()

        result = reranker.rerank("query", [], top_k=5)
        assert result == []
        reranker._model.predict.assert_not_called()

    def test_rerank_top_k_larger_than_docs(self):
        """Test rerank when top_k > len(docs)."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.9]

        docs = [LCDocument(page_content="Only doc")]
        result = reranker.rerank("query", docs, top_k=10)
        assert len(result) == 1

    def test_rerank_uses_custom_model_name(self):
        """Test rerank uses the configured model name."""
        reranker = CrossEncoderReranker(model_name="my-custom-model")
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.9]

        docs = [LCDocument(page_content="Doc")]
        reranker.rerank("query", docs, top_k=1)

        assert reranker._model_name == "my-custom-model"


class TestRerankerFactory:
    """Tests for RerankerFactory."""

    def test_create_disabled_returns_identity(self):
        """Test factory creates IdentityReranker when disabled."""
        reranker = RerankerFactory.create(enabled=False)
        assert isinstance(reranker, IdentityReranker)

    def test_create_enabled_returns_cross_encoder(self):
        """Test factory creates CrossEncoderReranker when enabled."""
        reranker = RerankerFactory.create(enabled=True)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_with_custom_model(self):
        """Test factory passes custom model name."""
        reranker = RerankerFactory.create(enabled=True, model_name="custom/encoder")
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker._model_name == "custom/encoder"

    def test_create_default_disabled(self):
        """Test factory defaults to disabled."""
        reranker = RerankerFactory.create()
        assert isinstance(reranker, IdentityReranker)
