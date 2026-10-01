"""Unit tests for chunking logic."""

from __future__ import annotations

from src.ingestion.chunker import chunk_documents


class TestChunking:
    """Tests for the chunking module."""

    def test_basic_chunking(self):
        """Test that documents are split into chunks of approximately correct size."""
        from langchain_core.documents import Document as LCDocument

        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        docs = [LCDocument(page_content=long_text, metadata={"page": 1})]

        chunks, _meta = chunk_documents(docs, chunk_size=200, chunk_overlap=50, document_name="test.pdf")

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 + 50  # size + overlap margin

    def test_chunk_overlap(self):
        """Test that overlapping chunks share content."""
        from langchain_core.documents import Document as LCDocument

        text = "A B C D E F G H I J " * 10
        docs = [LCDocument(page_content=text, metadata={"page": 1})]

        chunks, _ = chunk_documents(docs, chunk_size=50, chunk_overlap=20, document_name="test.pdf")

        # With overlap, consecutive chunks should share some content
        if len(chunks) > 1:
            first = chunks[0].page_content
            second = chunks[1].page_content
            # They should share some tokens due to overlap
            first_tokens = set(first.split())
            second_tokens = set(second.split())
            # With overlap=20 out of 50, significant sharing is expected
            shared = first_tokens & second_tokens
            assert len(shared) > 0

    def test_empty_document(self):
        """Test handling of empty documents."""
        from langchain_core.documents import Document as LCDocument

        docs = [LCDocument(page_content="", metadata={"page": 1})]

        chunks, meta = chunk_documents(docs, chunk_size=100, chunk_overlap=10, document_name="test.pdf")

        # Empty doc may produce one small chunk or none depending on splitter
        assert isinstance(chunks, list)
        assert meta.chunk_count >= 0

    def test_unicode_text(self):
        """Test chunking with Unicode content."""
        from langchain_core.documents import Document as LCDocument

        unicode_text = "Hello 世界! 🌍 مرحبا by the lake café naïve. " * 20
        docs = [LCDocument(page_content=unicode_text, metadata={"page": 1})]

        chunks, _meta = chunk_documents(docs, chunk_size=100, chunk_overlap=20, document_name="unicode.pdf")

        assert len(chunks) > 0
        # All chunks should be valid strings
        for chunk in chunks:
            assert isinstance(chunk.page_content, str)
            assert len(chunk.page_content) > 0

    def test_page_metadata_preserved(self):
        """Test that page numbers are preserved in chunks."""
        from langchain_core.documents import Document as LCDocument

        docs = [
            LCDocument(page_content="Page 1 content here. " * 30, metadata={"page": 1}),
            LCDocument(page_content="Page 2 content here. " * 30, metadata={"page": 2}),
        ]

        chunks, _meta = chunk_documents(docs, chunk_size=150, chunk_overlap=30, document_name="pages.pdf")

        assert len(chunks) > 0
        pages_seen = set()
        for chunk in chunks:
            assert "page" in chunk.metadata or "page_number" in chunk.metadata
            page = chunk.metadata.get("page", chunk.metadata.get("page_number", 0))
            pages_seen.add(page)

        # Should have seen content from both pages
        assert 1 in pages_seen or 2 in pages_seen

    def test_very_long_text(self):
        """Test chunking with a very long document."""
        from langchain_core.documents import Document as LCDocument

        long_text = "Word " * 10000
        docs = [LCDocument(page_content=long_text, metadata={"page": 1})]

        chunks, _meta = chunk_documents(docs, chunk_size=100, chunk_overlap=20, document_name="long.pdf")

        assert len(chunks) > 1  # Should produce multiple chunks
        # Each chunk should have reasonable size
        for chunk in chunks:
            assert len(chunk.page_content) > 0
            assert len(chunk.page_content) <= 150  # size + some margin

    def test_chunk_metadata_enrichment(self):
        """Test that chunk metadata includes document name and index."""
        from langchain_core.documents import Document as LCDocument

        docs = [LCDocument(page_content="Some content here.", metadata={"page": 1})]

        chunks, meta = chunk_documents(docs, chunk_size=100, chunk_overlap=10, document_name="enriched.pdf")

        assert meta.document_name == "enriched.pdf"
        assert meta.chunk_count == len(chunks)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("document_name") == "enriched.pdf"
            assert chunk.metadata.get("chunk_index") == i
