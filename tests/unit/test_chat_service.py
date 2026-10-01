"""Unit tests for chat service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chat.memory import ConversationMemory
from src.chat.service import ChatResponse, ChatService
from src.config.settings import Settings


class TestChatService:
    """Tests for the ChatService."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.get_collection_info.return_value = {"exists": True, "points_count": 10}
        store.embedding.dimension = 384
        return store

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.model_name = "gpt-4o-mini"
        response = MagicMock()
        response.content = "The vacation policy is 20 days."
        response.response_metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        llm.invoke.return_value = response
        return llm

    @pytest.fixture
    def memory(self, tmp_path):
        """Create in-memory conversation memory."""
        return ConversationMemory(str(tmp_path / "test.db"))

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(_env_file=None)

    @pytest.fixture
    def chat_service(self, settings, mock_store, mock_llm, memory):
        """Create a ChatService instance."""
        with patch("src.chat.service.RerankerFactory.create") as mock_reranker:
            mock_reranker.return_value = MagicMock()
            mock_reranker.return_value.rerank.return_value = []
            service = ChatService(settings, mock_store, mock_llm, memory)
        return service

    def test_ask_returns_answer(self, chat_service, memory):
        """Test that ask returns a ChatResponse with an answer."""
        conv_id = memory.create_conversation()

        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(
                page_content="The vacation policy is 20 days.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            )
        ]

        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=docs, scores=[0.9])

            response = chat_service.ask("How many vacation days?", conv_id)

            assert isinstance(response, ChatResponse)
            assert response.success is True
            assert response.answer == "The vacation policy is 20 days."
            assert len(response.sources) > 0

    def test_ask_with_sources(self, chat_service, memory):
        """Test ask returns sources when documents are found."""
        from langchain_core.documents import Document as LCDocument

        conv_id = memory.create_conversation()

        docs = [
            LCDocument(
                page_content="The vacation policy is 20 days.",
                metadata={"document_name": "policy.pdf", "page_number": 1},
            )
        ]

        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=docs, scores=[0.9])

            response = chat_service.ask("Vacation days?", conv_id)

            assert response.success is True
            assert len(response.sources) > 0
            assert response.sources[0]["document_name"] == "policy.pdf"

    def test_ask_no_retrieval(self, chat_service, memory):
        """Test ask when no documents are retrieved."""
        conv_id = memory.create_conversation()

        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=[], scores=[])

            response = chat_service.ask("Unknown question?", conv_id)

            assert response.success is True
            assert "couldn't find" in response.answer.lower()

    def test_ask_persists_to_memory(self, chat_service, memory):
        """Test that ask persists messages to memory."""
        conv_id = memory.create_conversation()

        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=[], scores=[])

            chat_service.ask("Test question", conv_id)

            messages = memory.get_messages(conv_id)
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Test question"
            assert messages[1]["role"] == "assistant"

    def test_health_check_all_ok(self, chat_service):
        """Test health check when all components are healthy."""
        status = chat_service.health_check()

        assert status["application"] is True
        assert status["qdrant"] is True
        assert status["embedding_model"] is True
        assert status["llm"] is True
        assert status["vectors"] == 10

    def test_health_check_qdrant_fails(self, chat_service):
        """Test health check when Qdrant is down."""
        chat_service._store.get_collection_info.side_effect = Exception("Connection refused")

        status = chat_service.health_check()

        assert status["qdrant"] is False
        assert "qdrant_error" in status

    def test_health_check_embedding_fails(self, chat_service):
        """Test health check when embedding model fails."""
        type(chat_service._store.embedding).dimension = property(
            lambda self: (_ for _ in []).throw(Exception("Embedding failed"))
        )

        status = chat_service.health_check()

        assert status["embedding_model"] is False
        assert "embedding_error" in status

    def test_health_check_llm_fails(self, chat_service):
        """Test health check when LLM fails."""
        chat_service._llm.invoke.side_effect = Exception("LLM error")

        status = chat_service.health_check()

        assert status["llm"] is False
        assert "llm_error" in status

    def test_memory_property(self, chat_service, memory):
        """Test that memory property returns the conversation memory."""
        assert chat_service.memory is memory

    def test_ask_with_reranking(self, settings, mock_store, mock_llm, memory):
        """Test ask with reranking enabled."""
        from langchain_core.documents import Document as LCDocument

        settings.enable_reranking = True
        settings.rerank_top_k = 1

        with patch("src.chat.service.RerankerFactory.create") as mock_reranker:
            reranker_mock = MagicMock()
            reranker_mock.rerank.return_value = []
            mock_reranker.return_value = reranker_mock

            service = ChatService(settings, mock_store, mock_llm, memory)

            # Verify the generator has reranking enabled
            assert service._generator.settings.enable_reranking is True

            docs = [
                LCDocument(page_content="Doc 1", metadata={"document_name": "a.pdf", "page_number": 1}),
                LCDocument(page_content="Doc 2", metadata={"document_name": "b.pdf", "page_number": 1}),
            ]
            with patch.object(service._retriever, "retrieve") as mock_retrieve:
                mock_retrieve.return_value = MagicMock(documents=docs, scores=[0.5, 0.4])

                service.ask("Test", memory.create_conversation())

                # Reranker should have been called since we have more docs than rerank_top_k
                reranker_mock.rerank.assert_called_once()

    def test_ask_streaming_no_docs(self, chat_service, memory):
        """Test streaming when no documents are found."""
        conv_id = memory.create_conversation()

        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=[], scores=[])

            chunks = list(chat_service.ask_streaming("Test question", conv_id))
            assert any("couldn't find" in c.lower() for c in chunks)

    def test_ask_streaming_with_content(self, chat_service, memory):
        """Test streaming with actual content."""
        from langchain_core.documents import Document as LCDocument

        conv_id = memory.create_conversation()

        docs = [
            LCDocument(
                page_content="The answer is 42.",
                metadata={"document_name": "test.pdf", "page_number": 1},
            )
        ]
        with patch.object(chat_service._retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = MagicMock(documents=docs, scores=[0.9])

            chat_service._llm.stream.return_value = [MagicMock(content="The "), MagicMock(content="answer")]

            chunks = list(chat_service.ask_streaming("What is the answer?", conv_id))
            full = "".join(chunks)
            assert "The answer" in full

    def test_response_dataclass(self):
        """Test ChatResponse dataclass."""
        response = ChatResponse(
            answer="Test answer",
            sources=[{"document_name": "test.pdf", "page_number": 1}],
            retrieval_latency_ms=100.5,
            generation_latency_ms=200.3,
            model_name="gpt-4o-mini",
            success=True,
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.retrieval_latency_ms == 100.5
