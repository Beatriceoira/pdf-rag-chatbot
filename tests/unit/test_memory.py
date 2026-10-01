"""Unit tests for conversation memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chat.memory import ConversationMemory


class TestConversationMemory:
    """Tests for ConversationMemory."""

    @pytest.fixture
    def memory(self, tmp_path: Path):
        """Create a ConversationMemory instance."""
        db_path = tmp_path / "test.db"
        return ConversationMemory(str(db_path))

    def test_create_conversation(self, memory):
        """Test creating a conversation."""
        conv_id = memory.create_conversation()
        assert conv_id is not None
        assert len(conv_id) > 0

        conv = memory.get_conversation(conv_id)
        assert conv is not None
        assert conv["id"] == conv_id
        assert "title" in conv
        assert "created_at" in conv

    def test_create_conversation_with_title(self, memory):
        """Test creating a conversation with a title."""
        conv_id = memory.create_conversation(title="My Chat")
        conv = memory.get_conversation(conv_id)
        assert conv["title"] == "My Chat"

    def test_add_message(self, memory):
        """Test adding a message to a conversation."""
        conv_id = memory.create_conversation()
        msg_id = memory.add_message(conv_id, "user", "Hello")

        assert msg_id is not None
        assert len(msg_id) > 0

    def test_add_message_with_sources(self, memory):
        """Test adding a message with sources."""
        conv_id = memory.create_conversation()
        sources = [{"document_name": "test.pdf", "page_number": 1, "score": 0.9}]
        msg_id = memory.add_message(conv_id, "assistant", "Answer", sources=sources)

        assert msg_id is not None
        messages = memory.get_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["sources"] == sources

    def test_get_messages(self, memory):
        """Test retrieving messages."""
        conv_id = memory.create_conversation()

        memory.add_message(conv_id, "user", "Question")
        memory.add_message(conv_id, "assistant", "Answer")

        messages = memory.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_order(self, memory):
        """Test that messages are ordered by creation time."""
        conv_id = memory.create_conversation()

        memory.add_message(conv_id, "user", "First")
        memory.add_message(conv_id, "assistant", "Second")
        memory.add_message(conv_id, "user", "Third")

        messages = memory.get_messages(conv_id)
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    def test_list_conversations(self, memory):
        """Test listing all conversations."""
        id1 = memory.create_conversation("Chat 1")
        id2 = memory.create_conversation("Chat 2")
        id3 = memory.create_conversation("Chat 3")

        conversations = memory.list_conversations()
        assert len(conversations) == 3

        # Should be ordered by updated_at descending
        assert conversations[0]["id"] == id3
        assert conversations[1]["id"] == id2
        assert conversations[2]["id"] == id1

    def test_delete_conversation(self, memory):
        """Test deleting a conversation."""
        conv_id = memory.create_conversation()
        memory.add_message(conv_id, "user", "Message")

        assert memory.delete_conversation(conv_id) is True
        assert memory.get_conversation(conv_id) is None
        assert memory.get_messages(conv_id) == []

    def test_delete_nonexistent_conversation(self, memory):
        """Test deleting a non-existent conversation."""
        result = memory.delete_conversation("nonexistent")
        assert result is False

    def test_register_document(self, memory):
        """Test registering a document."""
        conv_id = memory.create_conversation()
        doc_id = memory.register_document(
            conversation_id=conv_id,
            document_name="test.pdf",
            file_hash="abc123",
            page_count=10,
            chunk_count=25,
        )

        assert doc_id is not None
        docs = memory.get_documents(conv_id)
        assert len(docs) == 1
        assert docs[0]["document_name"] == "test.pdf"
        assert docs[0]["file_hash"] == "abc123"
        assert docs[0]["page_count"] == 10
        assert docs[0]["chunk_count"] == 25

    def test_get_documents(self, memory):
        """Test getting documents for a conversation."""
        conv_id = memory.create_conversation()

        memory.register_document(conv_id, "doc1.pdf", "hash1", page_count=5)
        memory.register_document(conv_id, "doc2.pdf", "hash2", page_count=10)

        docs = memory.get_documents(conv_id)
        assert len(docs) == 2
        assert docs[0]["document_name"] == "doc1.pdf"
        assert docs[1]["document_name"] == "doc2.pdf"

    def test_export_json(self, memory):
        """Test exporting conversation as JSON."""
        conv_id = memory.create_conversation("Test Chat")
        memory.add_message(conv_id, "user", "Hello")
        memory.add_message(conv_id, "assistant", "Hi there", sources=[{"document_name": "test.pdf"}])

        json_str = memory.export_conversation(conv_id, fmt="json")

        import json

        data = json.loads(json_str)
        assert data["conversation"]["title"] == "Test Chat"
        assert len(data["messages"]) == 2

    def test_export_markdown(self, memory):
        """Test exporting conversation as Markdown."""
        conv_id = memory.create_conversation("Test Chat")
        memory.add_message(conv_id, "user", "Hello")
        memory.add_message(conv_id, "assistant", "Hi there")

        md_str = memory.export_conversation(conv_id, fmt="markdown")

        assert "# Test Chat" in md_str
        assert "## User" in md_str
        assert "Hello" in md_str
        assert "## Assistant" in md_str
        assert "Hi there" in md_str

    def test_export_markdown_with_sources(self, memory):
        """Test markdown export includes sources."""
        conv_id = memory.create_conversation()
        memory.add_message(
            conv_id,
            "assistant",
            "Answer",
            sources=[{"document_name": "test.pdf", "page_number": 1}],
        )

        md_str = memory.export_conversation(conv_id, fmt="markdown")

        assert "Sources" in md_str
        assert "test.pdf" in md_str

    def test_export_invalid_format_raises(self, memory):
        """Test that invalid format raises ValueError."""
        conv_id = memory.create_conversation()

        with pytest.raises(ValueError, match="Unsupported export format"):
            memory.export_conversation(conv_id, fmt="xml")

    def test_get_nonexistent_conversation(self, memory):
        """Test getting a non-existent conversation."""
        result = memory.get_conversation("nonexistent")
        assert result is None

    def test_get_messages_nonexistent_conversation(self, memory):
        """Test getting messages for non-existent conversation."""
        messages = memory.get_messages("nonexistent")
        assert messages == []

    def test_conversation_updated_on_message(self, memory):
        """Test that conversation updated_at changes when message is added."""
        conv_id = memory.create_conversation()

        import time

        time.sleep(0.01)

        memory.add_message(conv_id, "user", "Hello")

        conv = memory.get_conversation(conv_id)
        assert conv["updated_at"] > conv["created_at"]

    def test_duplicate_conversations(self, memory):
        """Test that multiple conversations can exist."""
        id1 = memory.create_conversation("First")
        id2 = memory.create_conversation("Second")

        assert id1 != id2

        conv1 = memory.get_conversation(id1)
        conv2 = memory.get_conversation(id2)

        assert conv1["title"] == "First"
        assert conv2["title"] == "Second"

    def test_default_db_path(self, tmp_path: Path):
        """Test default database path."""
        db_path = tmp_path / "default.db"
        memory = ConversationMemory(str(db_path))
        assert memory._db_path == db_path

    def test_db_directory_created(self, tmp_path: Path):
        """Test that database directory is created."""
        db_path = tmp_path / "nested" / "dir" / "test.db"
        ConversationMemory(str(db_path))
        assert db_path.parent.exists()
        assert db_path.parent.parent.exists()

    def test_thread_safety(self, memory):
        """Test thread-safe operations."""
        import threading

        def add_messages(conv_id, count):
            for i in range(count):
                memory.add_message(conv_id, "user", f"Message {i}")

        conv_id = memory.create_conversation()
        threads = [threading.Thread(target=add_messages, args=(conv_id, 10)) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        messages = memory.get_messages(conv_id)
        assert len(messages) == 50
