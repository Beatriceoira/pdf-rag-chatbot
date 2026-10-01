"""PDF file validation utilities."""

from __future__ import annotations

from pathlib import Path

VALID_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB default


def validate_file(
    filepath: Path | str,
    *,
    max_size_mb: int = MAX_FILE_SIZE // (1024 * 1024),
) -> list[str]:
    """Validate a PDF file. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    path = Path(filepath)

    # Extension check
    if path.suffix.lower() not in VALID_EXTENSIONS:
        errors.append(f"Unsupported file type '{path.suffix}'. Only PDF files are accepted.")
        return errors

    # Size check
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors

    max_bytes = max_size_mb * 1024 * 1024
    if size > max_bytes:
        errors.append(f"File too large: {size / (1024 * 1024):.1f} MB (max {max_size_mb} MB).")

    # Empty file check
    if size == 0:
        errors.append("File is empty.")

    # Quick magic-byte check for PDF
    if not errors and size >= 4:
        with open(path, "rb") as f:
            header = f.read(4)
        if not header.startswith(b"%PDF"):
            errors.append("File does not appear to be a valid PDF (bad magic bytes).")

    return errors


def validate_in_memory(
    file_bytes: bytes,
    filename: str,
    *,
    max_size_mb: int = MAX_FILE_SIZE // (1024 * 1024),
) -> list[str]:
    """Validate a PDF provided as in-memory bytes (for upload streams)."""
    errors: list[str] = []

    # Extension check
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        errors.append(f"Unsupported file type '{ext}'. Only PDF files are accepted.")
        return errors

    # Size check
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        errors.append(f"File too large: {len(file_bytes) / (1024 * 1024):.1f} MB (max {max_size_mb} MB).")

    # Empty check
    if len(file_bytes) == 0:
        errors.append("File is empty.")

    # Magic byte check
    if not errors and len(file_bytes) >= 4 and not file_bytes.startswith(b"%PDF"):
        errors.append("File does not appear to be a valid PDF (bad magic bytes).")

    return errors
