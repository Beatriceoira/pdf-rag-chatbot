"""Unit tests for logging utilities."""

from __future__ import annotations

import logging

from src.utils.logging import RequestIdFilter, get_logger, request_context, setup_logging


class TestRequestIdFilter:
    """Tests for RequestIdFilter."""

    def test_filter_adds_request_id(self):
        """Test that filter injects request_id into log records."""
        filter_obj = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = filter_obj.filter(record)

        assert result is True
        assert hasattr(record, "request_id")

    def test_filter_with_empty_request_id(self):
        """Test filter works when no request_id is set."""
        filter_obj = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Ensure no request_id is set
        from src.utils.logging import request_id_var

        token = request_id_var.set("")
        try:
            result = filter_obj.filter(record)
            assert result is True
            assert record.request_id == ""
        finally:
            request_id_var.reset(token)

    def test_filter_with_active_request_id(self):
        """Test filter uses active request_id."""
        filter_obj = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        from src.utils.logging import request_id_var

        token = request_id_var.set("abc123")
        try:
            result = filter_obj.filter(record)
            assert result is True
            assert record.request_id == "[abc123]"
        finally:
            request_id_var.reset(token)


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_setup_logging_adds_handler(self):
        """Test that setup_logging adds a handler."""
        import logging as _logging

        logger_name = "rag_chatbot.test_setup_new"
        logger = _logging.getLogger(logger_name)

        # Use a fresh logger to avoid interference from other tests
        original_level = logger.level
        original_handlers = logger.handlers.copy()
        logger.handlers.clear()

        try:
            setup_logging(level="DEBUG")
            # Check that handler was added by the root logger
            root_logger = _logging.getLogger("rag_chatbot")
            assert len(root_logger.handlers) >= 1
        finally:
            logger.handlers.clear()
            logger.handlers.extend(original_handlers)
            logger.setLevel(original_level)

    def test_setup_logging_no_duplicate_handlers(self):
        """Test that repeated calls don't add duplicate handlers."""
        logger = get_logger("test_duplicate")

        original_handlers = logger.handlers.copy()
        logger.handlers.clear()

        try:
            setup_logging(level="INFO")
            first_count = len(logger.handlers)
            setup_logging(level="DEBUG")
            second_count = len(logger.handlers)

            # Should not add duplicate handlers
            assert first_count == second_count
        finally:
            logger.handlers.clear()
            logger.handlers.extend(original_handlers)

    def test_setup_logging_sets_level(self):
        """Test that logging level is set correctly."""
        import logging as _logging

        root_logger = _logging.getLogger("rag_chatbot")
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()

        try:
            setup_logging(level="WARNING")
            assert root_logger.level == _logging.WARNING
        finally:
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)


class TestRequestContext:
    """Tests for request_context context manager."""

    def test_context_sets_and_resets_id(self):
        """Test that context sets request_id and resets on exit."""
        from src.utils.logging import request_id_var

        with request_context():
            rid = request_id_var.get()
            assert rid != ""
            assert len(rid) == 8  # hex[:8]

        # After context, should be reset to default
        assert request_id_var.get() == ""

    def test_context_id_is_unique(self):
        """Test that each context gets a unique ID."""
        from src.utils.logging import request_id_var

        ids = []
        for _ in range(5):
            with request_context():
                ids.append(request_id_var.get())

        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_context_id_format(self):
        """Test that request_id has correct format."""
        from src.utils.logging import request_id_var

        with request_context():
            rid = request_id_var.get()
            # Should be hex string of length 8
            assert len(rid) == 8
            int(rid, 16)  # Should not raise

    def test_context_resets_on_exception(self):
        """Test that context resets even if exception occurs."""
        from src.utils.logging import request_id_var

        try:
            with request_context():
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be reset even after exception
        assert request_id_var.get() == ""


class TestGetLogger:
    """Tests for get_logger."""

    def test_logger_name(self):
        """Test that logger has correct name."""
        logger = get_logger("my_module")
        assert logger.name == "rag_chatbot.my_module"

    def test_logger_returns_same_instance(self):
        """Test that same name returns same logger."""
        logger1 = get_logger("test_singleton")
        logger2 = get_logger("test_singleton")
        assert logger1 is logger2
