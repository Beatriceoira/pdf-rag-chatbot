"""Unit tests for PDF parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.parser import parse_pdf, parse_pdf_from_bytes
from src.utils.exceptions import PDFProcessingError


class TestParsePdf:
    """Tests for parse_pdf function."""

    def test_parse_valid_pdf(self, tmp_path: Path):
        """Test parsing a valid PDF file."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        with patch("src.ingestion.parser.PyPDFLoader") as mock_loader:
            mock_doc = MagicMock()
            mock_doc.metadata = {"page": 1}
            mock_loader.return_value.load.return_value = [mock_doc]

            docs, meta = parse_pdf(pdf_path)

            assert len(docs) == 1
            assert meta.document_name == "test.pdf"
            assert meta.page_count == 1
            assert meta.file_hash is not None

    def test_parse_pdf_enriches_metadata(self, tmp_path: Path):
        """Test that parsed documents get enriched metadata."""
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.parser.PyPDFLoader") as mock_loader:
            mock_doc = MagicMock()
            mock_doc.metadata = {}
            mock_loader.return_value.load.return_value = [mock_doc]

            docs, _meta = parse_pdf(pdf_path)

            assert docs[0].metadata.get("document_name") == "doc.pdf"
            assert "file_hash" in docs[0].metadata

    def test_parse_pdf_not_found(self, tmp_path: Path):
        """Test parsing a non-existent file raises an error."""
        pdf_path = tmp_path / "missing.pdf"

        with patch("src.utils.hashing.file_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"

            with pytest.raises((PDFProcessingError, FileNotFoundError)):
                parse_pdf(pdf_path)

    def test_parse_pdf_raises_processing_error(self, tmp_path: Path):
        """Test that parsing errors are wrapped in PDFProcessingError."""
        pdf_path = tmp_path / "bad.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.parser.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.side_effect = Exception("Parse failed")

            with pytest.raises(PDFProcessingError, match="Parse failed"):
                parse_pdf(pdf_path)

    def test_parse_empty_pdf(self, tmp_path: Path):
        """Test parsing a PDF with no pages."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.parser.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = []

            docs, meta = parse_pdf(pdf_path)

            assert len(docs) == 0
            assert meta.page_count == 0

    def test_parse_pdf_with_multiple_pages(self, tmp_path: Path):
        """Test parsing a multi-page PDF."""
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.ingestion.parser.PyPDFLoader") as mock_loader:
            mock_docs = [
                MagicMock(metadata={"page": 1}),
                MagicMock(metadata={"page": 2}),
                MagicMock(metadata={"page": 3}),
            ]
            mock_loader.return_value.load.return_value = mock_docs

            docs, meta = parse_pdf(pdf_path)

            assert len(docs) == 3
            assert meta.page_count == 3


class TestParsePdfFromBytes:
    """Tests for parse_pdf_from_bytes function."""

    def test_parse_from_bytes(self):
        """Test parsing PDF from bytes."""
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        with patch("src.utils.hashing.file_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"

            with patch("src.ingestion.parser.PyPDFLoader") as mock_loader_cls:
                mock_doc = MagicMock()
                mock_doc.metadata = {"page": 1}
                mock_loader_cls.return_value.load.return_value = [mock_doc]

                docs, meta = parse_pdf_from_bytes(pdf_bytes, "uploaded.pdf")

                assert len(docs) == 1
                assert meta.document_name == "uploaded.pdf"
                assert meta.page_count == 1

    def test_parse_from_bytes_enriches_metadata(self):
        """Test that bytes-parsed docs get metadata."""
        pdf_bytes = b"%PDF-1.4\n"

        with patch("src.utils.hashing.file_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"

            with patch("src.ingestion.parser.PyPDFLoader") as mock_loader_cls:
                mock_doc = MagicMock()
                mock_doc.metadata = {}
                mock_loader_cls.return_value.load.return_value = [mock_doc]

                docs, _meta = parse_pdf_from_bytes(pdf_bytes, "upload.pdf")

                assert docs[0].metadata.get("document_name") == "upload.pdf"
                assert "file_hash" in docs[0].metadata

    def test_parse_from_bytes_raises_error(self):
        """Test that parse errors are wrapped."""
        pdf_bytes = b"not a pdf"

        with patch("src.utils.hashing.file_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"

            with patch("src.ingestion.parser.PyPDFLoader") as mock_loader_cls:
                mock_loader_cls.return_value.load.side_effect = Exception("Load failed")

                with pytest.raises(PDFProcessingError, match="Load failed"):
                    parse_pdf_from_bytes(pdf_bytes, "bad.pdf")

    def test_parse_from_bytes_empty(self):
        """Test parsing empty bytes."""
        with patch("src.utils.hashing.file_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"

            with patch("src.ingestion.parser.PyPDFLoader") as mock_loader_cls:
                mock_loader_cls.return_value.load.return_value = []

                docs, meta = parse_pdf_from_bytes(b"", "empty.pdf")

                assert len(docs) == 0
                assert meta.page_count == 0
