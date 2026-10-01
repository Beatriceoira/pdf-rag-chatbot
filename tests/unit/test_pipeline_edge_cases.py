"""Unit tests for ingestion pipeline edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.ingestion.metadata import DocumentMetadata
from src.ingestion.pipeline import IngestionPipeline


class TestIngestionPipelineEdgeCases:
    """Tests for ingestion pipeline edge cases."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with test settings."""
        settings = Settings(chunk_size=100, chunk_overlap=20, max_file_size_mb=50, _env_file=None)
        return IngestionPipeline(settings)

    def test_ingest_from_path_validation_failure(self, pipeline, tmp_path: Path):
        """Test ingestion fails on validation error."""
        txt_path = tmp_path / "not_a_pdf.txt"
        txt_path.write_text("This is not a PDF")

        result = pipeline.ingest_from_path(txt_path)

        assert result.success is False
        assert "Unsupported file type" in result.error

    def test_ingest_from_path_empty_file(self, pipeline, tmp_path: Path):
        """Test ingestion fails on empty file."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")

        result = pipeline.ingest_from_path(empty)

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_ingest_from_path_oversized(self, pipeline, tmp_path: Path):
        """Test ingestion fails on oversized file."""
        big = tmp_path / "big.pdf"
        big.write_bytes(b"%PDF-1.4\n" + b"x" * (100 * 1024 * 1024))

        result = pipeline.ingest_from_path(big)

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_ingest_from_path_parse_failure(self, pipeline, tmp_path: Path):
        """Test ingestion fails when parser fails."""
        pdf_path = tmp_path / "bad.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n" + b"x" * 1000)

        with patch("src.ingestion.pipeline.parse_pdf") as mock_parse:
            mock_parse.side_effect = Exception("Parse error")

            result = pipeline.ingest_from_path(pdf_path)

            assert result.success is False
            assert "Parse error" in result.error

    def test_ingest_from_path_chunk_failure(self, pipeline, tmp_path: Path):
        """Test ingestion fails when chunking fails."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.pipeline.parse_pdf") as mock_parse:
            mock_doc = MagicMock()
            mock_doc.metadata = {"page": 1}
            mock_parse.return_value = ([mock_doc], MagicMock(file_hash="abc", page_count=1))

            with patch("src.ingestion.pipeline.chunk_documents") as mock_chunk:
                mock_chunk.side_effect = Exception("Chunk error")

                result = pipeline.ingest_from_path(pdf_path)

                assert result.success is False
                assert "Chunk error" in result.error

    def test_ingest_from_bytes_validation_failure(self, pipeline):
        """Test bytes ingestion fails on validation error."""
        result = pipeline.ingest_from_bytes(b"not a pdf", "test.txt")

        assert result.success is False
        assert "Unsupported file type" in result.error

    def test_ingest_from_bytes_empty(self, pipeline):
        """Test bytes ingestion fails on empty bytes."""
        result = pipeline.ingest_from_bytes(b"", "empty.pdf")

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_ingest_from_bytes_oversized(self, pipeline):
        """Test bytes ingestion fails on oversized content."""
        result = pipeline.ingest_from_bytes(b"x" * (100 * 1024 * 1024), "big.pdf")

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_ingest_from_bytes_parse_failure(self, pipeline):
        """Test bytes ingestion fails when parser fails."""
        with patch("src.ingestion.pipeline.parse_pdf_from_bytes") as mock_parse:
            mock_parse.side_effect = Exception("Parse error")

            result = pipeline.ingest_from_bytes(b"%PDF-1.4\n", "test.pdf")

            assert result.success is False
            assert "Parse error" in result.error

    def test_ingest_from_bytes_chunk_failure(self, pipeline):
        """Test bytes ingestion fails when chunking fails."""
        pdf_bytes = b"%PDF-1.4\n"

        with patch("src.ingestion.pipeline.parse_pdf_from_bytes") as mock_parse:
            mock_doc = MagicMock()
            mock_doc.metadata = {"page": 1}
            meta = MagicMock()
            meta.file_hash = "abc"
            meta.page_count = 1
            mock_parse.return_value = ([mock_doc], meta)

            with patch("src.ingestion.pipeline.chunk_documents") as mock_chunk:
                mock_chunk.side_effect = Exception("Chunk error")

                result = pipeline.ingest_from_bytes(pdf_bytes, "test.pdf")

                assert result.success is False
                assert "Chunk error" in result.error

    def test_ingest_successful_path(self, pipeline, tmp_path: Path):
        """Test successful path ingestion."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.pipeline.parse_pdf") as mock_parse:
            mock_doc = MagicMock()
            mock_doc.metadata = {"page": 1}
            mock_meta = MagicMock()
            mock_meta.file_hash = "abc123"
            mock_meta.page_count = 1
            mock_parse.return_value = ([mock_doc], mock_meta)

            with patch("src.ingestion.pipeline.chunk_documents") as mock_chunk:
                mock_chunk.return_value = ([mock_doc], MagicMock(chunk_count=1))

                result = pipeline.ingest_from_path(pdf_path)

                assert result.success is True
                assert result.chunks == [mock_doc]
                assert result.document_meta.chunk_count == 1

    def test_ingest_successful_bytes(self, pipeline):
        """Test successful bytes ingestion."""
        pdf_bytes = b"%PDF-1.4\n"

        with patch("src.ingestion.pipeline.parse_pdf_from_bytes") as mock_parse:
            mock_doc = MagicMock()
            mock_doc.metadata = {"page": 1}
            mock_meta = MagicMock()
            mock_meta.file_hash = "def456"
            mock_meta.page_count = 1
            mock_meta.document_name = "upload.pdf"
            mock_parse.return_value = ([mock_doc], mock_meta)

            with patch("src.ingestion.pipeline.chunk_documents") as mock_chunk:
                mock_chunk.return_value = ([mock_doc], MagicMock(chunk_count=1))

                result = pipeline.ingest_from_bytes(pdf_bytes, "upload.pdf")

                assert result.success is True
                assert result.document_meta.document_name == "upload.pdf"

    def test_result_dataclass(self, pipeline):
        """Test IngestionResult dataclass."""
        from src.ingestion.pipeline import IngestionResult

        meta = DocumentMetadata(document_name="test.pdf")
        result = IngestionResult(
            document_meta=meta,
            chunks=[],
            success=True,
        )
        assert result.success is True
        assert result.error == ""
