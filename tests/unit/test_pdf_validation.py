"""Unit tests for PDF validation."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.pdf_loader import validate_file, validate_in_memory


class TestPDFValidation:
    """Tests for PDF file validation."""

    def test_valid_pdf(self, tmp_path: Path):
        """Valid PDF should pass validation."""
        pdf_path = tmp_path / "test.pdf"
        # Create a minimal valid PDF header
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        errors = validate_file(pdf_path)
        assert len(errors) == 0

    def test_invalid_extension(self, tmp_path: Path):
        """Non-PDF extension should be rejected."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("This is a text file, not a PDF.")

        errors = validate_file(txt_path)
        assert len(errors) == 1
        assert "Unsupported file type" in errors[0]

    def test_empty_file(self, tmp_path: Path):
        """Empty file should be rejected."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")

        errors = validate_file(empty)
        assert any("empty" in e.lower() for e in errors)

    def test_corrupted_pdf(self, tmp_path: Path):
        """File with PDF extension but bad magic bytes should be rejected."""
        fake = tmp_path / "fake.pdf"
        fake.write_bytes(b"This is not a PDF at all!")

        errors = validate_file(fake)
        assert any("valid PDF" in e.lower() or "magic" in e.lower() for e in errors)

    def test_oversized_file(self, tmp_path: Path):
        """Overly large files should be rejected."""
        big = tmp_path / "big.pdf"
        big.write_bytes(b"%PDF-1.4\n" + b"x" * (60 * 1024 * 1024))  # 60 MB

        errors = validate_file(big, max_size_mb=50)
        assert any("too large" in e.lower() for e in errors)

    def test_file_not_found(self, tmp_path: Path):
        """Missing files should be reported."""
        errors = validate_file(tmp_path / "missing.pdf")
        assert any("not found" in e.lower() for e in errors)

    def test_in_memory_validation_valid(self):
        """Valid PDF bytes should pass in-memory validation."""
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        )

        errors = validate_in_memory(pdf_bytes, "test.pdf")
        assert len(errors) == 0

    def test_in_memory_validation_invalid_type(self):
        """Non-PDF bytes should fail in-memory validation."""
        txt_bytes = b"This is not a PDF"

        errors = validate_in_memory(txt_bytes, "document.txt")
        assert len(errors) == 1
        assert "Unsupported file type" in errors[0]

    def test_in_memory_validation_empty(self):
        """Empty bytes should fail validation."""
        errors = validate_in_memory(b"", "empty.pdf")
        assert any("empty" in e.lower() for e in errors)
