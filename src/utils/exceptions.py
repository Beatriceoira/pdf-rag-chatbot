"""Application-wide exception hierarchy."""

from __future__ import annotations


class RAGChatbotError(Exception):
    """Base exception for all application errors."""


class PDFProcessingError(RAGChatbotError):
    """Raised when PDF parsing or validation fails."""


class EmbeddingError(RAGChatbotError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGChatbotError):
    """Raised when vector store operations fail."""


class RetrievalError(RAGChatbotError):
    """Raised when retrieval fails."""


class LLMError(RAGChatbotError):
    """Raised when LLM calls fail."""


class QueryProcessingError(RAGChatbotError):
    """Raised when query preprocessing fails."""
