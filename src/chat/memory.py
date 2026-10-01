"""Conversation memory management."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger("chat")

# SQL schema
_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
)
"""

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS document_registry (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER,
    indexed_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
)
"""


class ConversationMemory:
    """SQLite-backed conversation memory with repository pattern."""

    def __init__(self, db_path: str = "./data/conversations.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.executescript(_CREATE_CONVERSATIONS)
            conn.executescript(_CREATE_MESSAGES)
            conn.executescript(_CREATE_DOCUMENTS)
            conn.commit()

    def create_conversation(self, title: str = "") -> str:
        """Create a new conversation and return its ID."""
        conv_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title or f"Conversation {conv_id[:6]}", now, now),
            )
            conn.commit()
        logger.info("Created conversation: %s", conv_id)
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> str:
        """Add a message to a conversation. Returns message ID."""
        msg_id = uuid.uuid4().hex[:12]
        now = time.time()
        sources_json = json.dumps(sources) if sources else None
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, sources_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()
        return msg_id

    def get_messages(self, conversation_id: str) -> list[dict]:
        """Get all messages for a conversation, ordered by creation time."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            msg = {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources"]) if row["sources"] else None,
            }
            messages.append(msg)
        return messages

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Get conversation metadata."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row:
            return dict(row)
        return None

    def list_conversations(self) -> list[dict]:
        """List all conversations."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
        return cur.rowcount > 0

    def register_document(
        self,
        conversation_id: str,
        document_name: str,
        file_hash: str,
        page_count: int = 0,
        chunk_count: int = 0,
    ) -> str:
        """Register an ingested document."""
        doc_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO document_registry (id, conversation_id, document_name,"
                " file_hash, page_count, chunk_count, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, conversation_id, document_name, file_hash, page_count, chunk_count, now),
            )
            conn.commit()
        return doc_id

    def get_documents(self, conversation_id: str) -> list[dict]:
        """Get all registered documents for a conversation."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, document_name, file_hash, page_count, chunk_count,"
                " indexed_at FROM document_registry"
                " WHERE conversation_id = ? ORDER BY indexed_at ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def export_conversation(self, conversation_id: str, fmt: str = "json") -> str:
        """Export a conversation as JSON or Markdown."""
        conv = self.get_conversation(conversation_id)
        messages = self.get_messages(conversation_id)

        if fmt == "json":
            data = {
                "conversation": conv,
                "messages": messages,
            }
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif fmt == "markdown":
            lines = [f"# {conv['title'] if conv else 'Conversation'}\n"]
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"## {role}\n\n{msg['content']}\n")
                if msg.get("sources"):
                    lines.append("### Sources\n")
                    for src in msg["sources"]:
                        lines.append(f"- {src.get('document_name', 'unknown')} — Page {src.get('page_number', '?')}\n")
            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {fmt}")
