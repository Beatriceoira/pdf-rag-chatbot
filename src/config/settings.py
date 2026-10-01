"""Application configuration using pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized typed configuration for the RAG chatbot."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = ""

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_mode: Literal["local", "server", "memory"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "pdf_documents"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "local"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "ollama", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", " ", ""])

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_k: int = Field(default=5, ge=1, le=20)
    enable_reranking: bool = False
    rerank_top_k: int = Field(default=3, ge=1, le=10)

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── File Limits ───────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50, ge=1, le=500)

    # ── Conversation ──────────────────────────────────────────────────────
    conversation_db_path: str = "./data/conversations.db"

    @field_validator("openai_api_key")
    @classmethod
    def guard_openai_key(cls, v: str) -> str:
        if not v and os.environ.get("OPENAI_API_KEY"):
            return os.environ["OPENAI_API_KEY"]
        return v

    @model_validator(mode="after")
    def chunk_size_gte_overlap(self) -> Settings:
        if self.chunk_size < self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        return self

    def model_dump_masked(self) -> dict:
        """Return dict with sensitive fields masked."""
        data = self.model_dump()
        if data.get("openai_api_key"):
            data["openai_api_key"] = "***"
        if data.get("qdrant_api_key"):
            data["qdrant_api_key"] = "***"
        return data


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    load_dotenv()
    return Settings()


def reset_settings_cache() -> None:
    """Clear settings cache — useful in tests."""
    get_settings.cache_clear()
