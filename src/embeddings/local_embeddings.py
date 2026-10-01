"""Local HuggingFace embeddings using sentence-transformers."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.utils.logging import get_logger

logger = get_logger("embeddings")

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    _HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]


class LocalEmbeddings(Embeddings):
    """Thin wrapper around HuggingFace sentence-transformers embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if _HuggingFaceEmbeddings is None:
            raise ImportError("langchain-huggingface is not installed. Run: pip install langchain-huggingface")
        self._model = _HuggingFaceEmbeddings(model_name=model_name)
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
        logger.debug("Embedding %d documents locally", len(texts))
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        logger.debug("Embedding query locally")
        return self._model.embed_query(text)

    def __repr__(self) -> str:
        return f"LocalEmbeddings(model={self._model.model_name!r})"
