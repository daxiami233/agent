"""Conversation memory stores.

This module owns short-term conversation persistence: conversation metadata,
chat messages, and memory snapshots.

The SDK exposes only a storage choice through ``AgentRuntimeConfig``:

    memory_backend = "sqlite"  # persisted under config.data_dir
    memory_backend = "memory"  # process-local, useful for tests

Direct paths are intentionally kept out of the public config surface. Advanced
callers may still inject a custom store object that follows ``MemoryStoreProtocol``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol


STATE_DIR = Path.home() / ".agent-runtime"
MEMORY_DB = STATE_DIR / "memory.sqlite3"


@dataclass(slots=True)
class ConversationRecord:
    """Persisted conversation metadata."""

    id: str
    title: str
    created_at: float
    updated_at: float
    memory_snapshot: str = ""


@dataclass(slots=True)
class StoredMessage:
    """Persisted conversation message."""

    id: int
    conversation_id: str
    role: str
    content: str
    created_at: float


class MemoryStoreProtocol(Protocol):
    """Storage contract consumed by ``ContextEngine`` and ``Agent``.

    Implementations may persist to SQLite, keep data in memory, or provide a
    custom backend. Methods intentionally mirror the runtime needs instead of a
    generic database abstraction.
    """

    def create_conversation(
        self,
        conversation_id: str,
        title: str = "新对话",
    ) -> ConversationRecord:
        """Create or return a conversation record."""

    def list_conversations(self) -> list[ConversationRecord]:
        """Return all conversations ordered by most recently updated."""

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Return one conversation record, or ``None`` if it does not exist."""

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update the user-facing conversation title."""

    def ensure_memory_snapshot(self, conversation_id: str, snapshot: str) -> str:
        """Persist the first retrieved-memory snapshot for a conversation."""

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all of its messages."""

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete all messages in a conversation."""

    def append_message(self, conversation_id: str, role: str, content: str) -> StoredMessage:
        """Append one role/content message."""

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return messages ordered from oldest to newest."""

    def replace_messages(
        self,
        conversation_id: str,
        messages: list[tuple[str, str]],
    ) -> None:
        """Replace all stored messages for a conversation."""

    def message_count(self, conversation_id: str) -> int:
        """Return the number of messages in a conversation."""

    def conversation_version(self, conversation_id: str) -> tuple[int, int]:
        """Return ``(max_message_id, message_count)`` for cache invalidation."""


class SQLiteMemoryStore:
    """SQLite-backed conversation memory store.

    Args:
        path: Internal SQLite database path. SDK users should normally configure
            ``AgentRuntimeConfig(data_dir=...)`` and ``memory_backend="sqlite"``
            instead of passing this directly.

    Example:
        store = SQLiteMemoryStore(Path.home() / ".agent-runtime" / "memory.sqlite3")
        store.append_message("conv-1", "user", "hello")
    """

    def __init__(self, path: Path | str = MEMORY_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    def create_conversation(self, conversation_id: str, title: str = "新对话") -> ConversationRecord:
        now = time.time()
        with self._connect() as conn:
            self._ensure_conversation(conn, conversation_id, title, now)
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at, memory_snapshot
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return self._conversation_from_row(row)

    def list_conversations(self) -> list[ConversationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at, memory_snapshot
                FROM conversations
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at, memory_snapshot
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return self._conversation_from_row(row) if row is not None else None

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, now, conversation_id),
            )

    def ensure_memory_snapshot(self, conversation_id: str, snapshot: str) -> str:
        now = time.time()
        with self._lock, self._connect() as conn:
            self._ensure_conversation(conn, conversation_id, "新对话", now)
            row = conn.execute(
                "SELECT memory_snapshot FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            current = str(row[0] or "") if row is not None else ""
            if current:
                return current
            conn.execute(
                """
                UPDATE conversations
                SET memory_snapshot = ?, updated_at = ?
                WHERE id = ?
                """,
                (snapshot, now, conversation_id),
            )
            return snapshot

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def clear_conversation(self, conversation_id: str) -> None:
        now = time.time()
        self.create_conversation(conversation_id)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )

    def append_message(self, conversation_id: str, role: str, content: str) -> StoredMessage:
        now = time.time()
        with self._lock, self._connect() as conn:
            self._ensure_conversation(conn, conversation_id, "新对话", now)
            cursor = conn.execute(
                """
                INSERT INTO conversation_messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            message_id = int(cursor.lastrowid)
        return StoredMessage(message_id, conversation_id, role, content, now)

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        params: list[object] = [conversation_id]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, conversation_id, role, content, created_at
                FROM (
                    SELECT id, conversation_id, role, content, created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    {limit_clause}
                )
                ORDER BY id ASC
                """,
                params,
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def replace_messages(
        self,
        conversation_id: str,
        messages: list[tuple[str, str]],
    ) -> None:
        now = time.time()
        self.create_conversation(conversation_id)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            for role, content in messages:
                conn.execute(
                    """
                    INSERT INTO conversation_messages (conversation_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, role, content, now),
                )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def message_count(self, conversation_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return int(row[0])

    def conversation_version(self, conversation_id: str) -> tuple[int, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(id), 0), COUNT(*)
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return int(row[0]), int(row[1])

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    memory_snapshot TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._ensure_conversation_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id
                ON conversation_messages (conversation_id, id)
                """
            )

    def _ensure_conversation_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(conversations)").fetchall()
        columns = {str(row[1]) for row in rows}
        migrations = {
            "memory_snapshot": "ALTER TABLE conversations ADD COLUMN memory_snapshot TEXT NOT NULL DEFAULT ''",
        }
        for column, sql in migrations.items():
            if column not in columns:
                conn.execute(sql)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_conversation(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        title: str,
        now: float,
    ) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now, now),
        )

    def _conversation_from_row(self, row: sqlite3.Row | tuple[object, ...]) -> ConversationRecord:
        return ConversationRecord(
            id=str(row[0]),
            title=str(row[1]),
            created_at=float(row[2]),
            updated_at=float(row[3]),
            memory_snapshot=str(row[4] or "") if len(row) > 4 else "",
        )

    def _message_from_row(self, row: sqlite3.Row | tuple[object, ...]) -> StoredMessage:
        return StoredMessage(
            id=int(row[0]),
            conversation_id=str(row[1]),
            role=str(row[2]),
            content=str(row[3]),
            created_at=float(row[4]),
        )


class InMemoryMemoryStore:
    """Process-local conversation memory store.

    This backend is useful for tests, demos, and short-lived agents. Data is
    lost when the Python process exits.

    Example:
        store = InMemoryMemoryStore()
        store.append_message("conv-1", "user", "hello")
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationRecord] = {}
        self._messages: dict[str, list[StoredMessage]] = {}
        self._next_message_id = 1

    def create_conversation(self, conversation_id: str, title: str = "新对话") -> ConversationRecord:
        now = time.time()
        with self._lock:
            record = self._conversations.get(conversation_id)
            if record is None:
                record = ConversationRecord(
                    id=conversation_id,
                    title=title,
                    created_at=now,
                    updated_at=now,
                )
                self._conversations[conversation_id] = record
                self._messages.setdefault(conversation_id, [])
            return record

    def list_conversations(self) -> list[ConversationRecord]:
        with self._lock:
            return sorted(
                list(self._conversations.values()),
                key=lambda record: (record.updated_at, record.created_at),
                reverse=True,
            )

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        now = time.time()
        with self._lock:
            record = self.create_conversation(conversation_id)
            self._conversations[conversation_id] = replace(
                record,
                title=title,
                updated_at=now,
            )

    def ensure_memory_snapshot(self, conversation_id: str, snapshot: str) -> str:
        now = time.time()
        with self._lock:
            record = self.create_conversation(conversation_id)
            if record.memory_snapshot:
                return record.memory_snapshot
            self._conversations[conversation_id] = replace(
                record,
                memory_snapshot=snapshot,
                updated_at=now,
            )
            return snapshot

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)
            self._messages.pop(conversation_id, None)

    def clear_conversation(self, conversation_id: str) -> None:
        now = time.time()
        with self._lock:
            record = self.create_conversation(conversation_id)
            self._messages[conversation_id] = []
            self._conversations[conversation_id] = replace(
                record,
                updated_at=now,
            )

    def append_message(self, conversation_id: str, role: str, content: str) -> StoredMessage:
        now = time.time()
        with self._lock:
            record = self.create_conversation(conversation_id)
            message = StoredMessage(
                id=self._next_message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                created_at=now,
            )
            self._next_message_id += 1
            self._messages.setdefault(conversation_id, []).append(message)
            self._conversations[conversation_id] = replace(record, updated_at=now)
            return message

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        with self._lock:
            messages = list(self._messages.get(conversation_id, []))
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def replace_messages(
        self,
        conversation_id: str,
        messages: list[tuple[str, str]],
    ) -> None:
        now = time.time()
        with self._lock:
            record = self.create_conversation(conversation_id)
            stored: list[StoredMessage] = []
            for role, content in messages:
                stored.append(
                    StoredMessage(
                        id=self._next_message_id,
                        conversation_id=conversation_id,
                        role=role,
                        content=content,
                        created_at=now,
                    )
                )
                self._next_message_id += 1
            self._messages[conversation_id] = stored
            self._conversations[conversation_id] = replace(record, updated_at=now)

    def message_count(self, conversation_id: str) -> int:
        with self._lock:
            return len(self._messages.get(conversation_id, []))

    def conversation_version(self, conversation_id: str) -> tuple[int, int]:
        with self._lock:
            messages = self._messages.get(conversation_id, [])
            max_id = max((message.id for message in messages), default=0)
            return max_id, len(messages)


# Backwards-compatible name for existing imports. New code should prefer the
# explicit SQLiteMemoryStore or MemoryStoreProtocol names.
MemoryStore = SQLiteMemoryStore
