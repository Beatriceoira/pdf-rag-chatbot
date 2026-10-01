"""OpenAI embeddings wrapper with batching support."""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger("embeddings")

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore[assignment,misc]


class OpenAIEmbeddingsWrapper:
    """Thin wrapper around langchain OpenAI embeddings."""

    def __init__(self, model: str = "text-embedding-3-small"):
        if OpenAIEmbeddings is None:
            raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
        self._model = OpenAIEmbeddings(model=model)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            probe = self.embed_query("dimension probe text")
            self._dimension = len(probe)
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        logger.debug("Embedding %d documents via OpenAI", len(texts))
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        logger.debug("Embedding query via OpenAI")
        return self._model.embed_query(text)

    def __repr__(self) -> str:
        return f"OpenAIEmbeddingsWrapper(model={self._model.model!r})"
