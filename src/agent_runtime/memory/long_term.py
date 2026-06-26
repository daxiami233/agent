"""Long-term memory backends.

Long-term memory stores cross-conversation facts that the agent may inject into
the system prompt and update through memory tools.

Public SDK configuration intentionally exposes only a backend choice:

    AgentRuntimeConfig(memory_backend="sqlite")
    AgentRuntimeConfig(memory_backend="memory")

Advanced callers can inject any object matching ``LongTermMemoryProtocol`` into
``create_agent(long_term_memory=...)``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


MAX_LINES = 200


@dataclass(slots=True)
class MemoryRecord:
    """A single long-term memory search match."""

    content: str
    line_number: int


class LongTermMemoryProtocol(Protocol):
    """Storage contract used by memory tools and ``ContextEngine``."""

    def read(self) -> str:
        """Return a bounded text snapshot for prompt injection."""

    def search(self, query: str, *, limit: int = 20) -> list[MemoryRecord]:
        """Return ranked memory matches for a search query."""

    def write(self, content: str) -> None:
        """Replace all long-term memory with the provided content."""

    def append(self, content: str) -> None:
        """Append one long-term memory entry."""

    def replace(self, old: str, new: str) -> bool:
        """Replace the first occurrence of text in long-term memory."""

    def clear(self) -> None:
        """Remove all long-term memory entries."""


class LongTermMemory:
    """Process-local long-term memory implementation.

    Args:
        max_lines: Maximum lines returned by ``read()`` for prompt injection.

    Example:
        memory = LongTermMemory()
        memory.append("The user prefers concise Chinese answers.")
        memory.read()
    """

    def __init__(self, *, max_lines: int = MAX_LINES) -> None:
        self.max_lines = max(1, max_lines)
        self._lock = threading.RLock()
        self._entries: list[str] = []
        self.version = 0

    def read(self) -> str:
        with self._lock:
            return "\n".join(self._entries[: self.max_lines])

    def search(self, query: str, *, limit: int = 20) -> list[MemoryRecord]:
        terms = {term.lower() for term in query.split() if term.strip()}
        if not terms:
            return []
        with self._lock:
            lines = list(self._entries)
        scored: list[tuple[int, MemoryRecord]] = []
        for index, line in enumerate(lines, start=1):
            lowered = line.lower()
            score = sum(1 for term in terms if term in lowered)
            if score:
                scored.append((score, MemoryRecord(content=line, line_number=index)))
        scored.sort(key=lambda item: (-item[0], item[1].line_number))
        return [record for _score, record in scored[:limit]]

    def write(self, content: str) -> None:
        with self._lock:
            self._entries = content.splitlines()
            self.version += 1

    def append(self, content: str) -> None:
        with self._lock:
            self._entries.append(content.rstrip("\n"))
            self.version += 1

    def replace(self, old: str, new: str) -> bool:
        with self._lock:
            for index, entry in enumerate(self._entries):
                if old in entry:
                    self._entries[index] = entry.replace(old, new, 1)
                    self.version += 1
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self.version += 1


class SQLiteLongTermMemory:
    """SQLite-backed long-term memory implementation.

    This class is created internally by ``create_agent`` when
    ``memory_backend="sqlite"``. The database path comes from
    ``AgentRuntimeConfig.data_dir`` and is not exposed as a separate config key.

    Args:
        database_path: Internal SQLite database path.
        max_lines: Maximum lines returned by ``read()``.
    """

    def __init__(
        self,
        database_path: Path | str,
        *,
        max_lines: int = MAX_LINES,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_lines = max(1, max_lines)
        self._lock = threading.RLock()
        self._ensure_schema()

    @property
    def version(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(updated_at), 0), COUNT(*) FROM long_term_memory"
            ).fetchone()
        updated_ms = int(float(row[0]) * 1000)
        return updated_ms + int(row[1])

    def read(self) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content
                FROM long_term_memory
                ORDER BY id ASC
                LIMIT ?
                """,
                (self.max_lines,),
            ).fetchall()
        return "\n".join(str(row[0]) for row in rows)

    def search(self, query: str, *, limit: int = 20) -> list[MemoryRecord]:
        terms = {term.lower() for term in query.split() if term.strip()}
        if not terms:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content
                FROM long_term_memory
                ORDER BY id ASC
                """
            ).fetchall()
        scored: list[tuple[int, MemoryRecord]] = []
        for line_number, row in enumerate(rows, start=1):
            content = str(row[1])
            lowered = content.lower()
            score = sum(1 for term in terms if term in lowered)
            if score:
                scored.append((score, MemoryRecord(content=content, line_number=line_number)))
        scored.sort(key=lambda item: (-item[0], item[1].line_number))
        return [record for _score, record in scored[:limit]]

    def write(self, content: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM long_term_memory")
            for line in content.splitlines():
                conn.execute(
                    """
                    INSERT INTO long_term_memory (content, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (line, now, now),
                )

    def append(self, content: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_memory (content, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (content.rstrip("\n"), now, now),
            )

    def replace(self, old: str, new: str) -> bool:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, content
                FROM long_term_memory
                WHERE content LIKE ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (f"%{old}%",),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE long_term_memory
                SET content = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(row[1]).replace(old, new, 1), now, int(row[0])),
            )
            return True

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM long_term_memory")

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)
