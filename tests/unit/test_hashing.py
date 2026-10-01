"""Unit tests for file hashing."""

from __future__ import annotations

from pathlib import Path

from src.utils.hashing import bytes_hash, file_hash


class TestHashing:
    """Tests for the hashing utilities."""

    def test_same_file_same_hash(self, tmp_path: Path):
        """Same file content should produce same hash."""
        file1 = tmp_path / "a.txt"
        file2 = tmp_path / "b.txt"
        content = b"hello world this is a test"

        file1.write_bytes(content)
        file2.write_bytes(content)

        hash1 = file_hash(file1)
        hash2 = file_hash(file2)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_different_file_different_hash(self, tmp_path: Path):
        """Different file content should produce different hash."""
        file1 = tmp_path / "a.txt"
        file2 = tmp_path / "b.txt"

        file1.write_bytes(b"hello world")
        file2.write_bytes(b"goodbye world")

        hash1 = file_hash(file1)
        hash2 = file_hash(file2)

        assert hash1 != hash2

    def test_bytes_hash_consistency(self):
        """Hashing same bytes should always produce same result."""
        data = b"test data for hashing"
        h1 = bytes_hash(data)
        h2 = bytes_hash(data)
        assert h1 == h2

    def test_bytes_hash_different_data(self):
        """Different byte sequences should produce different hashes."""
        h1 = bytes_hash(b"alpha")
        h2 = bytes_hash(b"beta")
        assert h1 != h2

    def test_sha256_default(self):
        """Default algorithm should be SHA-256 (64 hex chars)."""
        h = bytes_hash(b"test")
        assert len(h) == 64  # SHA-256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in h)
