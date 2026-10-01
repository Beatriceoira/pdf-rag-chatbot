"""Unit tests for vector store collection management."""

from __future__ import annotations

from src.vectorstore.collections import (
    DOCUMENTS_COLLECTION,
    SESSION_COLLECTION_PREFIX,
    build_session_collection_name,
    sanitize_collection_name,
)


class TestCollectionConstants:
    def test_documents_collection_constant(self):
        assert DOCUMENTS_COLLECTION == "pdf_documents"

    def test_session_collection_prefix_constant(self):
        assert SESSION_COLLECTION_PREFIX == "session_"


class TestBuildSessionCollectionName:
    def test_builds_session_collection_name(self):
        assert build_session_collection_name("abc") == "session_abc"

    def test_builds_with_uuid_like_id(self):
        name = build_session_collection_name("550e8400-e29b-41d4-a716-446655440000")
        assert name == "session_550e8400-e29b-41d4-a716-446655440000"

    def test_builds_with_empty_string(self):
        assert build_session_collection_name("") == "session_"


class TestSanitizeCollectionName:
    """sanitize_collection_name must produce a Qdrant-safe name.

    Qdrant allows: alphanumeric, underscores, hyphens, dots. Max length 64.
    """

    def test_passes_through_clean_name(self):
        assert sanitize_collection_name("pdf_documents") == "pdf_documents"

    def test_replaces_invalid_characters_with_underscore(self):
        assert sanitize_collection_name("hello world!") == "hello_world_"

    def test_replaces_path_traversal_sequences(self):
        # Path traversal attempts must be neutralised, never reach Qdrant
        assert sanitize_collection_name("../../etc/passwd") == ".._.._etc_passwd"

    def test_replaces_special_shell_characters(self):
        result = sanitize_collection_name("doc; rm -rf /")
        assert ";" not in result
        assert " " not in result
        assert "/" not in result

    def test_truncates_to_64_chars(self):
        long_name = "a" * 100
        assert len(sanitize_collection_name(long_name)) == 64

    def test_truncation_preserves_safe_prefix(self):
        long_name = "session_" + "x" * 100
        result = sanitize_collection_name(long_name)
        assert result.startswith("session_")
        assert len(result) == 64

    def test_handles_unicode(self):
        # Non-ASCII characters are replaced with underscores
        result = sanitize_collection_name("café résumé")
        assert "é" not in result
        assert all(c.isalnum() or c in "_-." for c in result)

    def test_handles_empty_string(self):
        assert sanitize_collection_name("") == ""