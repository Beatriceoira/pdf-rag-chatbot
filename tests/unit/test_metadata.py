"""Unit tests for document metadata."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.metadata import ChunkMetadata, DocumentMetadata


class TestMetadata:
    """Tests for metadata dataclasses."""

    def test_document_metadata_defaults(self):
        """Test default values for DocumentMetadata."""
        meta = DocumentMetadata(document_name="test.pdf")
        assert meta.document_name == "test.pdf"
        assert meta.file_hash == ""
        assert meta.page_count == 0
        assert meta.chunk_count == 0
        assert meta.indexed is False
        assert meta.error == ""
        assert len(meta.document_id) == 12

    def test_chunk_metadata_to_dict(self):
        """Test ChunkMetadata serialization."""
        chunk = ChunkMetadata(
            document_name="test.pdf",
            page_number=5,
            chunk_index=2,
            file_hash="abc123",
        )
        d = chunk.to_dict()
        assert d["document_name"] == "test.pdf"
        assert d["page_number"] == 5
        assert d["chunk_index"] == 2
        assert d["file_hash"] == "abc123"
        assert "chunk_id" in d
        assert "source" in d

    def test_chunk_metadata_unique_ids(self):
        """Each ChunkMetadata should have a unique chunk_id."""
        c1 = ChunkMetadata(document_name="a.pdf")
        c2 = ChunkMetadata(document_name="a.pdf")
        assert c1.chunk_id != c2.chunk_id
        assert c1.document_id != c2.document_id

    def test_document_hash_computation(self, tmp_path: Path):
        """Test that file hash is computed correctly."""
        meta = DocumentMetadata(document_name="test.pdf")
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content for hashing")

        hash_val = meta.compute_hash(test_file)
        assert len(hash_val) == 64  # SHA-256
        assert hash_val == meta.file_hash
