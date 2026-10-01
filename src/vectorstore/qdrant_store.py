"""Qdrant vector store integration."""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings
from src.embeddings.factory import EmbeddingFactory
from src.utils.exceptions import VectorStoreError
from src.utils.logging import get_logger

logger = get_logger("vectorstore")

try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:  # pragma: no cover
    QdrantVectorStore = None  # type: ignore
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    Filter = None  # type: ignore
    FieldCondition = None  # type: ignore
    MatchValue = None  # type: ignore


class QdrantStore:
    """Abstracts Qdrant vector store operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: QdrantClient | None = None
        self._embedding = None
        self._vectorstore: QdrantVectorStore | None = None
        self._initialized = False

    def _build_client(self) -> QdrantClient:
        """Create the Qdrant client based on mode."""
        mode = self.settings.qdrant_mode
        if mode == "memory":
            return QdrantClient(":memory:")
        elif mode == "local":
            return QdrantClient(path="./qdrant_data")
        elif mode == "server":
            kwargs: dict[str, Any] = {"url": self.settings.qdrant_url}
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            return QdrantClient(**kwargs)
        else:
            raise VectorStoreError(f"Unknown Qdrant mode: {mode}")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = EmbeddingFactory.create(self.settings)
        return self._embedding

    def initialize(self) -> None:
        """Initialize or recreate the Qdrant collection."""
        client = self.client
        collection_name = self.settings.collection_name

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        dim = self.embedding.dimension
        logger.info("Creating Qdrant collection '%s' with dim=%d", collection_name, dim)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embedding,
            validate_embeddings=False,  # Avoid compatibility issues
        )
        self._initialized = True
        logger.info("Qdrant collection '%s' initialized", collection_name)

    def add_documents(self, documents) -> None:
        """Add documents to the vector store."""
        if not self._initialized:
            self.initialize()
        self._vectorstore.add_documents(documents)
        logger.info("Added documents to Qdrant collection '%s'", self.settings.collection_name)

    def as_retriever(self, k: int = 5) -> Any:
        """Return a LangChain retriever for the collection."""
        if not self._initialized:
            self.initialize()
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    def search(
        self,
        query: str,
        k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list:
        """Direct search returning (score, document) tuples."""
        if not self._initialized:
            self.initialize()

        query_embedding = self.embedding.embed_query(query)
        client = self.client
        collection = self.settings.collection_name

        # Build filter if needed
        # LangChain Qdrant stores document metadata nested under a "metadata" key,
        # so filter fields must be prefixed accordingly.
        search_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                )
            if conditions:
                search_filter = Filter(must=conditions)

        # Use query_points for better compatibility
        result = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=search_filter,
        )

        # Flatten nested metadata so callers can access fields directly
        return [
            {
                "score": point.score,
                "document": {
                    "page_content": payload.get("page_content", ""),
                    **payload.get("metadata", {}),
                },
            }
            for point in result.points
            for payload in [point.payload]
        ]

    def get_collection_info(self) -> dict:
        """Return info about the current collection."""
        client = self.client
        name = self.settings.collection_name
        if not client.collection_exists(name):
            return {"exists": False, "points_count": 0}
        info = client.get_collection(name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "config": {"vector_size": info.config.params.vectors.size if info.config.params.vectors else 0},
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        client = self.client
        name = self.settings.collection_name
        if client.collection_exists(name):
            client.delete_collection(name)
            self._initialized = False
            logger.info("Deleted Qdrant collection '%s'", name)

    def clear(self) -> None:
        """Clear all points from the collection without deleting it."""
        client = self.client
        name = self.settings.collection_name
        if client.collection_exists(name):
            # Re-create the collection to clear it
            client.delete_collection(name)
            dim = self.embedding.dimension
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self._vectorstore = QdrantVectorStore(
                client=client,
                collection_name=name,
                embedding=self.embedding,
                validate_embeddings=False,
            )
            self._initialized = True
            logger.info("Cleared Qdrant collection '%s'", name)
