"""Retriever with metadata filtering and score tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document as LCDocument

from src.config.settings import Settings
from src.utils.exceptions import RetrievalError
from src.utils.logging import get_logger

logger = get_logger("retrieval")


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: list[LCDocument]
    scores: list[float]
    query: str
    k: int
    filtering: dict[str, Any] | None = None


class Retriever:
    """Configurable retriever wrapping Qdrant vector store."""

    def __init__(self, settings: Settings, store):
        self.settings = settings
        self._store = store
        self._retriever = None

    def _ensure_retriever(self) -> None:
        if self._retriever is None:
            k = self.settings.retrieval_k
            self._retriever = self._store.as_retriever(k=k)
            logger.info("Initialized retriever with k=%d", k)

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: User question.
            k: Number of results (overrides settings if provided).
            filter_conditions: Optional metadata filters.

        Returns:
            RetrievalResult with documents and scores.
        """
        self._ensure_retriever()
        effective_k = k or self.settings.retrieval_k

        try:
            # Use direct store search for score tracking
            results = self._store.search(
                query=query,
                k=effective_k,
                filter_conditions=filter_conditions,
            )
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query, exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        # Convert to LangChain Documents
        docs: list[LCDocument] = []
        scores: list[float] = []
        for item in results:
            payload = item.get("document", {})
            text = payload.get("page_content", "")
            meta = {k: v for k, v in payload.items() if k != "page_content"}
            doc = LCDocument(page_content=text, metadata=meta)
            docs.append(doc)
            scores.append(item.get("score", 0.0))

        logger.info(
            "Retrieved %d documents for query (k=%d, scores=%s)",
            len(docs),
            effective_k,
            [round(s, 3) for s in scores],
        )

        return RetrievalResult(
            documents=docs,
            scores=scores,
            query=query,
            k=effective_k,
            filtering=filter_conditions,
        )

    def retrieve_and_format(
        self,
        query: str,
        *,
        k: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Retrieve and format as context strings + source metadata.

        Returns:
            Tuple of (context_texts, source_info_list).
        """
        result = self.retrieve(query, k=k, filter_conditions=filter_conditions)

        contexts = [doc.page_content for doc in result.documents]
        sources = [
            {
                "document_name": doc.metadata.get("document_name", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "score": round(result.scores[i], 4) if i < len(result.scores) else None,
                "text_preview": doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content,
            }
            for i, doc in enumerate(result.documents)
        ]
        return contexts, sources
