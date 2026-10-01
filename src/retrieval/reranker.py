"""Reranker abstraction and implementations."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document as LCDocument

from src.utils.logging import get_logger

logger = get_logger("reranking")


class Reranker(Protocol):
    """Protocol for pluggable rerankers."""

    def rerank(
        self,
        query: str,
        documents: list[LCDocument],
        top_k: int,
    ) -> list[LCDocument]:
        """Rerank documents and return top_k results."""
        ...


class IdentityReranker:
    """Default reranker that returns documents in original order."""

    def rerank(
        self,
        query: str,
        documents: list[LCDocument],
        top_k: int,
    ) -> list[LCDocument]:
        return documents[:top_k]


class CrossEncoderReranker:
    """Reranker using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Run: pip install sentence-transformers"
                ) from exc

    def rerank(
        self,
        query: str,
        documents: list[LCDocument],
        top_k: int,
    ) -> list[LCDocument]:
        self._load_model()
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        # Sort by score descending
        indexed = list(zip(documents, scores, strict=False))
        indexed.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, _ in indexed[:top_k]]
        logger.info("Reranked %d documents → top %d", len(documents), len(reranked))
        return reranked


class RerankerFactory:
    """Factory for creating rerankers."""

    @staticmethod
    def create(enabled: bool = False, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
        if not enabled:
            return IdentityReranker()
        return CrossEncoderReranker(model_name=model_name)
