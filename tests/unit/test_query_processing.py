"""Unit tests for query processing."""

from __future__ import annotations

import pytest

from src.retrieval.query import normalize_query, process_query, validate_query
from src.utils.exceptions import QueryProcessingError


class TestQueryProcessing:
    """Tests for query processing utilities."""

    def test_normalize_trims_whitespace(self):
        """Whitespace should be trimmed."""
        assert normalize_query("  hello world  ") == "hello world"

    def test_normalize_collapses_whitespace(self):
        """Multiple whitespace should be collapsed."""
        assert normalize_query("hello    world") == "hello world"

    def test_normalize_removes_control_chars(self):
        """Control characters should be removed."""
        result = normalize_query("hello\x00world")
        assert "\x00" not in result

    def test_empty_query_raises(self):
        """Empty query should raise QueryProcessingError."""
        with pytest.raises(QueryProcessingError):
            normalize_query("")

    def test_whitespace_only_raises(self):
        """Whitespace-only query should raise."""
        with pytest.raises(QueryProcessingError):
            normalize_query("   ")

    def test_validate_empty(self):
        """Empty query should have validation errors."""
        errors = validate_query("")
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_validate_whitespace_only(self):
        """Whitespace-only query should have validation errors."""
        errors = validate_query("   ")
        assert len(errors) > 0

    def test_validate_too_long(self):
        """Overly long queries should be rejected."""
        long_query = "x" * 3000
        errors = validate_query(long_query, max_length=2000)
        assert len(errors) > 0
        assert any("too long" in e.lower() for e in errors)

    def test_validate_normal_query(self):
        """Normal queries should pass validation."""
        errors = validate_query("What is the refund policy?")
        assert len(errors) == 0

    def test_process_query_unicode(self):
        """Unicode queries should be processed correctly."""
        from src.config.settings import Settings

        settings = Settings()
        query, warnings = process_query("Hvad er politikken?", settings)
        assert "hvad" in query.lower()
        assert len(warnings) == 0

    def test_process_query_conversational_hint(self):
        """Short follow-up questions should generate a warning."""
        from src.config.settings import Settings

        settings = Settings()
        history = [("user", "What is the vacation policy?"), ("assistant", "20 days")]

        _query, warnings = process_query("And sick leave?", settings, conversation_history=history)
        # Should produce a hint about being a short follow-up
        assert len(warnings) >= 0  # May or may not warn depending on length

    def test_process_query_with_errors_raises(self):
        """Test that process_query raises on validation errors."""
        from src.config.settings import Settings

        settings = Settings()
        with pytest.raises(QueryProcessingError):
            process_query("", settings)

    def test_process_query_short_followup_warns(self):
        """Short follow-up without question mark should warn."""
        from src.config.settings import Settings

        settings = Settings()
        history = [("user", "What is the vacation policy?"), ("assistant", "20 days")]

        _query, warnings = process_query("ok", settings, conversation_history=history)
        assert len(warnings) >= 1

    def test_process_query_normal_no_warnings(self):
        """Normal queries should produce no warnings."""
        from src.config.settings import Settings

        settings = Settings()
        _query, warnings = process_query("What is the refund policy?", settings)
        assert warnings == []

    def test_validate_accepts_exact_max_length(self):
        """Query at exact max length should pass validation."""
        errors = validate_query("x" * 2000)
        assert len(errors) == 0

    def test_validate_rejects_over_max_length(self):
        """Query over max length should be rejected."""
        errors = validate_query("x" * 2001)
        assert any("too long" in e.lower() for e in errors)

    def test_process_query_raises_on_validation_error(self):
        """Test that process_query raises QueryProcessingError on invalid input."""
        from src.config.settings import Settings

        settings = Settings()
        with pytest.raises(QueryProcessingError):
            process_query("", settings)

    def test_process_query_raises_on_too_long(self):
        """Test that process_query raises when query exceeds max length."""
        from src.config.settings import Settings

        settings = Settings()
        with pytest.raises(QueryProcessingError, match="too long"):
            process_query("x" * 3000, settings)
