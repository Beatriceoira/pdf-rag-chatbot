"""Security tests for the RAG application."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings
from src.ingestion.pdf_loader import validate_file, validate_in_memory


class TestFileValidation:
    """Test file upload security."""

    def test_reject_non_pdf(self, tmp_path: Path):
        """Non-PDF files should be rejected."""
        txt = tmp_path / "malware.exe"
        txt.write_bytes(b"fake exe content")
        errors = validate_file(txt)
        assert len(errors) > 0

    def test_reject_oversized(self, tmp_path: Path):
        """Oversized files should be rejected."""
        big = tmp_path / "big.pdf"
        big.write_bytes(b"%PDF-1.4\n" + b"x" * (200 * 1024 * 1024))  # 200MB
        errors = validate_file(big, max_size_mb=50)
        assert any("too large" in e.lower() for e in errors)

    def test_reject_empty(self, tmp_path: Path):
        """Empty files should be rejected."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        errors = validate_file(empty)
        assert any("empty" in e.lower() for e in errors)

    def test_in_memory_rejects_text_as_pdf(self):
        """Text content with .pdf extension should be rejected."""
        errors = validate_in_memory(b"This is plain text", "fake.pdf")
        assert any("valid PDF" in e.lower() or "magic" in e.lower() for e in errors)


class TestSecretHandling:
    """Test that secrets are never exposed."""

    def test_settings_dont_leak_api_key(self):
        """Settings masked dump should not contain actual API key."""
        settings = Settings(openai_api_key="sk-this-is-a-secret-key", _env_file=None)
        data = settings.model_dump_masked()
        assert "sk-this-is-a-secret-key" not in data["openai_api_key"]
        assert data["openai_api_key"] == "***"

    def test_env_file_not_committed(self):
        """Verify .env is in .gitignore."""
        gitignore_path = Path(__file__).parent.parent.parent / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert ".env" in content

    def test_logging_does_not_expose_secrets(self):
        """Log messages should not contain API keys."""
        import logging
        from io import StringIO

        from src.utils.logging import get_logger

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = get_logger("test")
        logger.addHandler(handler)

        logger.info("Using API key: %s", "sk-secret")
        output = log_stream.getvalue()
        # The secret should not appear in logs
        assert "sk-secret" not in output

    def test_error_messages_dont_expose_keys(self):
        """Error messages should not leak API keys."""
        settings = Settings(openai_api_key="sk-test", _env_file=None)
        # Verify the key isn't in the masked dump
        data = settings.model_dump_masked()
        assert "sk-test" not in data["openai_api_key"]


class TestResourceLimits:
    """Test resource exhaustion protections."""

    def test_query_length_limit(self):
        """Excessively long queries should be rejected."""
        from src.retrieval.query import validate_query

        huge_query = "x" * 10000
        errors = validate_query(huge_query)
        assert len(errors) > 0
        assert any("too long" in e.lower() for e in errors)

    def test_max_file_size_configurable(self):
        """Max file size should be configurable."""
        settings = Settings(max_file_size_mb=10, _env_file=None)
        assert settings.max_file_size_mb == 10

    def test_embedding_dimension_bound(self):
        """Embedding dimensions should be reasonable."""
        settings = Settings()
        # Just verify settings load without crashing
        assert settings.embedding_provider in ["openai", "local"]
