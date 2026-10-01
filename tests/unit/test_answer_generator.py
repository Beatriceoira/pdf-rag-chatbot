"""Unit tests for answer generator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.llm.answer_generator import AnswerGenerator, AnswerResult


class TestAnswerGenerator:
    """Tests for AnswerGenerator."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(_env_file=None)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        response = MagicMock()
        response.content = "The vacation policy is 20 days."
        response.response_metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        llm.invoke.return_value = response
        return llm

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock reranker."""
        reranker = MagicMock()
        reranker.rerank.return_value = []
        return reranker

    @pytest.fixture
    def generator(self, settings, mock_llm, mock_reranker):
        """Create an AnswerGenerator instance."""
        return AnswerGenerator(settings, mock_llm, mock_reranker)

    def test_generate_basic(self, generator):
        """Test basic answer generation."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="The vacation policy is 20 days.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            )
        ]

        result = generator.generate("How many vacation days?", docs)

        assert isinstance(result, AnswerResult)
        assert result.success is True
        assert result.answer == "The vacation policy is 20 days."
        assert len(result.sources) == 1
        assert result.sources[0]["document_name"] == "policy.pdf"
        assert result.latency_ms > 0
        assert result.sources[0]["page_number"] == 1

    def test_generate_with_token_usage(self, generator):
        """Test that token usage is captured."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="Test content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]

        result = generator.generate("Question", docs)

        assert result.token_usage == {"prompt_tokens": 10, "completion_tokens": 20}

    def test_generate_with_reranking(self, settings, mock_llm, mock_reranker):
        """Test generation with reranking enabled."""
        from langchain_core.documents import Document as LCDocument

        settings.enable_reranking = True
        settings.rerank_top_k = 2

        reranker = MagicMock()
        reranked_docs = [
            LCDocument(page_content="Best doc", metadata={"document_name": "a.pdf", "page_number": 1}),
            LCDocument(page_content="Second doc", metadata={"document_name": "b.pdf", "page_number": 1}),
        ]
        reranker.rerank.return_value = reranked_docs

        generator = AnswerGenerator(settings, mock_llm, reranker)

        docs = [
            LCDocument(page_content="Doc 1", metadata={"document_name": "a.pdf", "page_number": 1}),
            LCDocument(page_content="Doc 2", metadata={"document_name": "b.pdf", "page_number": 1}),
            LCDocument(page_content="Doc 3", metadata={"document_name": "c.pdf", "page_number": 1}),
        ]

        result = generator.generate("Question", docs)

        reranker.rerank.assert_called_once()
        assert result.success is True

    def test_generate_without_reranking(self, generator):
        """Test generation with reranking disabled."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="Content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]

        result = generator.generate("Question", docs)

        assert result.success is True
        generator._reranker.rerank.assert_not_called()

    def test_generate_with_conversation_history(self, generator):
        """Test generation with conversation history."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="Policy info",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            )
        ]

        history = [
            {"role": "user", "content": "What is the policy?"},
            {"role": "assistant", "content": "It's 20 days."},
        ]

        result = generator.generate("And sick leave?", docs, conversation_history=history)

        assert result.success is True
        # Check that messages were passed to LLM
        call_args = generator._llm.invoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 4  # history + system + user

    def test_generate_handles_llm_error(self, generator):
        """Test that generation handles LLM errors gracefully."""
        from langchain_core.documents import Document as LCDocument

        generator._llm.invoke.side_effect = Exception("LLM failed")

        docs = [
            LCDocument(
                page_content="Content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]

        result = generator.generate("Question", docs)

        assert result.success is False
        assert "failed" in result.error.lower()
        assert "error" in result.answer.lower()

    def test_generate_streaming(self, generator):
        """Test streaming generation."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="Content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]

        generator._llm.stream.return_value = [
            MagicMock(content="Hello"),
            MagicMock(content=" world"),
        ]

        chunks = list(generator.generate_streaming("Question", docs))
        full_answer = "".join(chunks)

        assert "Hello world" in full_answer
        assert full_answer.endswith("\n")

    def test_generate_streaming_with_error(self, generator):
        """Test streaming handles errors."""
        from langchain_core.documents import Document as LCDocument

        generator._llm.stream.side_effect = Exception("Stream failed")

        docs = [
            LCDocument(
                page_content="Content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]

        chunks = list(generator.generate_streaming("Question", docs))

        assert any("Error" in c for c in chunks)

    def test_generate_streaming_with_reranking(self, settings, mock_llm, mock_reranker):
        """Test streaming applies reranking when enabled."""
        from langchain_core.documents import Document as LCDocument

        settings.enable_reranking = True
        settings.rerank_top_k = 1

        reranker = MagicMock()
        reranked = [LCDocument(page_content="Best", metadata={"document_name": "a.pdf", "page_number": 1})]
        reranker.rerank.return_value = reranked

        generator = AnswerGenerator(settings, mock_llm, reranker)
        generator._llm.stream.return_value = [MagicMock(content="Hi")]

        docs = [
            LCDocument(page_content="Doc 1", metadata={"document_name": "a.pdf", "page_number": 1}),
            LCDocument(page_content="Doc 2", metadata={"document_name": "b.pdf", "page_number": 1}),
        ]

        chunks = list(generator.generate_streaming("Question", docs))
        reranker.rerank.assert_called_once()
        assert "Hi" in "".join(chunks)

    def test_generate_streaming_with_history(self, generator):
        """Test streaming includes conversation history in messages."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="Content",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]
        history = [{"role": "user", "content": "Previous question"}]

        generator._llm.stream.return_value = [MagicMock(content="Answer")]

        list(generator.generate_streaming("Question", docs, conversation_history=history))

        call_args = generator._llm.stream.call_args
        messages = call_args[0][0]
        assert any(m["content"] == "Previous question" for m in messages)

    def test_answer_result_dataclass(self):
        """Test AnswerResult dataclass."""
        result = AnswerResult(
            answer="Test answer",
            sources=[{"document_name": "test.pdf"}],
            token_usage={"prompt_tokens": 10},
            latency_ms=100.5,
            success=True,
        )
        assert result.answer == "Test answer"
        assert result.latency_ms == 100.5
        assert result.success is True
