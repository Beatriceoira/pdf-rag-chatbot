"""Chat service: orchestrates retrieval, reranking, and answer generation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.chat.memory import ConversationMemory
from src.config.settings import Settings
from src.llm.answer_generator import AnswerGenerator
from src.retrieval.reranker import RerankerFactory
from src.retrieval.retriever import Retriever
from src.utils.logging import get_logger, request_context

logger = get_logger("chat")


@dataclass
class ChatResponse:
    """Response from the chat service."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    model_name: str = ""
    success: bool = False
    error: str = ""


class ChatService:
    """High-level chat orchestration service."""

    def __init__(
        self,
        settings: Settings,
        store,
        llm,
        memory: ConversationMemory | None = None,
    ):
        self.settings = settings
        self._store = store
        self._llm = llm
        self._memory = memory or ConversationMemory(settings.conversation_db_path)

        reranker = RerankerFactory.create(
            enabled=settings.enable_reranking,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self._retriever = Retriever(settings, store)
        self._generator = AnswerGenerator(settings, llm, reranker)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def ask(
        self,
        question: str,
        conversation_id: str,
        *,
        k: int | None = None,
    ) -> ChatResponse:
        """Process a question and return a grounded answer.

        Args:
            question: User's question.
            conversation_id: Active conversation ID.
            k: Override retrieval count.

        Returns:
            ChatResponse with answer and metadata.
        """
        with request_context():
            start_retrieve = __import__("time").time()

            # Retrieve relevant context
            result = self._retriever.retrieve(question, k=k)
            retrieval_latency = (__import__("time").time() - start_retrieve) * 1000

            generation_latency = 0.0
            if not result.documents:
                logger.warning("No documents retrieved for query: %s", question[:80])
                answer = "I couldn't find any relevant information in the uploaded documents."
                sources = []
            else:
                # Generate answer
                start_generate = __import__("time").time()
                gen_result = self._generator.generate(
                    query=question,
                    context_documents=result.documents,
                )
                generation_latency = (__import__("time").time() - start_generate) * 1000

                answer = gen_result.answer
                sources = gen_result.sources

            # Persist to memory
            self._memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            self._memory.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources,
            )

            return ChatResponse(
                answer=answer,
                sources=sources,
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                model_name=getattr(self._llm, "model_name", "unknown"),
                success=True,
            )

    def ask_streaming(
        self,
        question: str,
        conversation_id: str,
        *,
        k: int | None = None,
    ) -> Iterator[str]:
        """Stream a question answer. Yields text chunks."""
        start_retrieve = __import__("time").time()

        # Retrieve relevant context
        result = self._retriever.retrieve(question, k=k)
        (__import__("time").time() - start_retrieve) * 1000

        if not result.documents:
            yield "I couldn't find any relevant information in the uploaded documents."
            # Still persist
            self._memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            self._memory.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content="I couldn't find any relevant information in the uploaded documents.",
            )
            return

        # Stream the answer
        full_answer = ""
        for chunk in self._generator.generate_streaming(
            query=question,
            context_documents=result.documents,
        ):
            full_answer += chunk
            yield chunk

        # Persist full response
        sources = [
            {
                "document_name": doc.metadata.get("document_name", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
            }
            for doc in result.documents
        ]
        self._memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        self._memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            sources=sources,
        )

    def health_check(self) -> dict[str, Any]:
        """Return application health status."""
        status = {
            "application": True,
            "qdrant": False,
            "embedding_model": False,
            "llm": False,
            "documents": 0,
            "vectors": 0,
        }
        try:
            info = self._store.get_collection_info()
            status["qdrant"] = info.get("exists", False)
            status["vectors"] = info.get("points_count", 0)
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            status["qdrant_error"] = str(exc)

        # Test embedding
        try:
            dim = self._store.embedding.dimension
            status["embedding_model"] = True
            status["embedding_dimension"] = dim
        except Exception as exc:
            logger.warning("Embedding health check failed: %s", exc)
            status["embedding_error"] = str(exc)

        # Test LLM
        try:
            self._llm.invoke([{"role": "user", "content": "Say OK"}])
            status["llm"] = True
            status["llm_model"] = getattr(self._llm, "model_name", "unknown")
        except Exception as exc:
            logger.warning("LLM health check failed: %s", exc)
            status["llm_error"] = str(exc)

        return status
