"""Answer generation using retrieved context and LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config.settings import Settings
from src.llm.prompts import build_prompt, build_system_prompt
from src.utils.logging import get_logger

logger = get_logger("llm")


@dataclass
class AnswerResult:
    """Result of answer generation."""

    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = False
    error: str = ""


class AnswerGenerator:
    """Generates grounded answers from retrieved context."""

    def __init__(self, settings: Settings, llm, reranker):
        self.settings = settings
        self._llm = llm
        self._reranker = reranker

    def generate(
        self,
        query: str,
        context_documents,
        *,
        conversation_history=None,
    ) -> AnswerResult:
        """Generate an answer from context documents.

        Args:
            query: The user's question.
            context_documents: List of retrieved LangChain documents.
            conversation_history: Optional message history for context.

        Returns:
            AnswerResult with the generated answer and source info.
        """
        start = time.time()
        result = AnswerResult()

        try:
            # Apply reranking if enabled
            if self.settings.enable_reranking and len(context_documents) > self.settings.rerank_top_k:
                context_documents = self._reranker.rerank(
                    query=query,
                    documents=context_documents,
                    top_k=self.settings.rerank_top_k,
                )

            # Extract context texts and source metadata
            contexts = [doc.page_content for doc in context_documents]
            sources = [
                {
                    "document_name": doc.metadata.get("document_name", "unknown"),
                    "page_number": doc.metadata.get("page_number", 1),
                    "score": round(doc.metadata.get("score", 0), 4),
                    "text_preview": (doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content),
                }
                for doc in context_documents
            ]

            # Build prompt
            prompt = build_prompt(contexts, query)
            system_prompt = build_system_prompt()

            # Build messages
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call LLM
            logger.debug("Generating answer for query: %s", query[:100])
            response = self._llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            # Track token usage if available
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("usage", {})
                result.token_usage = usage

            result.answer = answer
            result.sources = sources
            result.success = True

        except Exception as exc:
            logger.error("Answer generation failed: %s", exc, exc_info=True)
            result.error = f"Answer generation failed: {exc}"
            result.answer = "I encountered an error while generating a response. Please try again."

        result.latency_ms = (time.time() - start) * 1000
        logger.info(
            "Generated answer in %.0fms, sources=%d",
            result.latency_ms,
            len(result.sources),
        )
        return result

    def generate_streaming(
        self,
        query: str,
        context_documents,
        *,
        conversation_history=None,
    ):
        """Generate a streaming answer. Yields chunks of the response."""
        try:
            # Apply reranking if enabled
            if self.settings.enable_reranking and len(context_documents) > self.settings.rerank_top_k:
                context_documents = self._reranker.rerank(
                    query=query,
                    documents=context_documents,
                    top_k=self.settings.rerank_top_k,
                )

            contexts = [doc.page_content for doc in context_documents]
            prompt = build_prompt(contexts, query)
            system_prompt = build_system_prompt()

            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Stream the response
            full_answer = ""
            for chunk in self._llm.stream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_answer += content
                yield content

            # Yield final completion marker
            yield "\n"

        except Exception as exc:
            logger.error("Streaming answer generation failed: %s", exc)
            yield f"\nError: {exc}"
