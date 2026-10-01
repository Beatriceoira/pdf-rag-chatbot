"""Embedding model factory."""

from __future__ import annotations

from src.config.settings import Settings
from src.embeddings.local_embeddings import LocalEmbeddings
from src.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper
from src.utils.exceptions import EmbeddingError
from src.utils.logging import get_logger

logger = get_logger("embeddings")


class EmbeddingFactory:
    """Factory for creating embedding models based on settings."""

    @staticmethod
    def create(settings: Settings):
        """Create and return the appropriate embedding model."""
        if settings.embedding_provider == "openai":
            if not settings.openai_api_key:
                raise EmbeddingError(
                    "OpenAI API key is required for openai embedding provider. Set OPENAI_API_KEY in your environment."
                )
            return OpenAIEmbeddingsWrapper(model="text-embedding-3-small")

        # Default to local embeddings
        return LocalEmbeddings(model_name=settings.embedding_model)
