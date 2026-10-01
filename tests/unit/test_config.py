"""Unit tests for configuration."""

from __future__ import annotations

import pytest

from src.config.settings import Settings


class TestSettings:
    """Tests for the settings module."""

    def test_defaults(self):
        """Test default setting values."""
        settings = Settings(_env_file=None)
        assert settings.qdrant_mode == "local"
        assert settings.collection_name == "pdf_documents"
        assert settings.embedding_provider == "local"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_k == 5
        assert settings.enable_reranking is False
        assert settings.rerank_top_k == 3
        assert settings.max_file_size_mb == 50

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("QDRANT_MODE", "server")
        monkeypatch.setenv("COLLECTION_NAME", "my_collection")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("RETRIEVAL_K", "10")

        settings = Settings(_env_file=None)
        assert settings.qdrant_mode == "server"
        assert settings.collection_name == "my_collection"
        assert settings.chunk_size == 500
        assert settings.retrieval_k == 10

    def test_chunk_size_gte_overlap(self):
        """Chunk size must be >= chunk overlap."""
        settings = Settings(chunk_size=500, chunk_overlap=100, _env_file=None)
        assert settings.chunk_size >= settings.chunk_overlap

    def test_chunk_size_less_than_overlap_raises(self):
        """Test that chunk_size < chunk_overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be greater than chunk_overlap"):
            Settings(chunk_size=100, chunk_overlap=200, _env_file=None)

    def test_chunk_overlap_less_than_size_passes(self):
        """Test that overlap <= chunk_size passes."""
        settings = Settings(chunk_size=1000, chunk_overlap=200, _env_file=None)
        assert settings.chunk_overlap <= settings.chunk_size

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Test that OPENAI_API_KEY is picked up from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        settings = Settings(_env_file=None)
        assert settings.openai_api_key == "sk-test-key-123"

    def test_llm_provider_values(self):
        """Test valid LLM provider values."""
        for provider in ["openai", "ollama", "anthropic"]:
            settings = Settings(llm_provider=provider, _env_file=None)
            assert settings.llm_provider == provider

    def test_qdrant_mode_values(self):
        """Test valid Qdrant mode values."""
        for mode in ["local", "server", "memory"]:
            settings = Settings(qdrant_mode=mode, _env_file=None)
            assert settings.qdrant_mode == mode

    def test_max_file_size_configurable(self):
        """Max file size should be configurable."""
        settings = Settings(max_file_size_mb=10, _env_file=None)
        assert settings.max_file_size_mb == 10

    def test_conversation_db_path(self):
        """Test conversation DB path."""
        settings = Settings(conversation_db_path="/tmp/test.db", _env_file=None)
        assert settings.conversation_db_path == "/tmp/test.db"

    def test_masked_dump(self):
        """Test that API keys are masked in dumps."""
        settings = Settings(openai_api_key="sk-secret", _env_file=None)
        data = settings.model_dump_masked()
        assert data["openai_api_key"] == "***"

    def test_openai_key_from_env_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        """Test that empty openai_api_key falls back to OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        settings = Settings(openai_api_key="", _env_file=None)
        assert settings.openai_api_key == "sk-from-env"

    def test_qdrant_api_key_masked(self):
        """Test that qdrant_api_key is masked in dumps."""
        settings = Settings(qdrant_api_key="secret", _env_file=None)
        data = settings.model_dump_masked()
        assert data["qdrant_api_key"] == "***"

    def test_get_settings_and_reset(self):
        """Test get_settings singleton and reset_settings_cache."""
        from src.config.settings import get_settings, reset_settings_cache

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        reset_settings_cache()
        s3 = get_settings()
        assert s3 is not None
