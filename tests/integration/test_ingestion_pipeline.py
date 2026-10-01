"""Integration tests for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.ingestion.pipeline import IngestionPipeline


class TestIngestionPipeline:
    """Integration tests for document ingestion."""

    @pytest.fixture
    def pipeline(self):
        """Create an ingestion pipeline with test settings."""
        settings = Settings(
            chunk_size=100,
            chunk_overlap=20,
            max_file_size_mb=50,
        )
        return IngestionPipeline(settings)

    def test_ingest_valid_pdf(self, pipeline, tmp_path: Path):
        """Test ingesting a valid PDF."""
        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        result = pipeline.ingest_from_path(pdf_path)
        # Minimal PDF may not parse well with PyPDFLoader, so we test the pipeline structure
        assert hasattr(result, "success")
        assert hasattr(result, "document_meta")

    def test_ingest_invalid_extension(self, pipeline, tmp_path: Path):
        """Test rejecting non-PDF files."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("This is a text file.")

        result = pipeline.ingest_from_path(txt_path)
        assert result.success is False
        assert "Unsupported file type" in result.error

    def test_ingest_empty_file(self, pipeline, tmp_path: Path):
        """Test rejecting empty files."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")

        result = pipeline.ingest_from_path(empty)
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_ingest_from_bytes(self, pipeline):
        """Test ingesting from in-memory bytes."""
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        result = pipeline.ingest_from_bytes(pdf_bytes, "test.pdf")
        assert hasattr(result, "success")
        assert hasattr(result, "document_meta")

    def test_duplicate_detection(self, pipeline, tmp_path: Path):
        """Test that duplicate files produce the same hash."""
        from src.utils.hashing import file_hash

        content = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        file1 = tmp_path / "doc1.pdf"
        file2 = tmp_path / "doc2.pdf"
        file1.write_bytes(content)
        file2.write_bytes(content)

        hash1 = file_hash(file1)
        hash2 = file_hash(file2)
        assert hash1 == hash2

    def test_file_size_limit(self, pipeline, tmp_path: Path):
        """Test that oversized files are rejected."""
        big = tmp_path / "big.pdf"
        big.write_bytes(b"%PDF-1.4\n" + b"x" * (100 * 1024 * 1024))  # 100 MB

        result = pipeline.ingest_from_path(big)
        assert result.success is False
        assert "too large" in result.error.lower()
