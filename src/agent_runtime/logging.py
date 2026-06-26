"""Runtime JSONL trace logging."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


STATE_DIR = Path.home() / ".agent-runtime"
DEFAULT_LOG_FILE = STATE_DIR / "logs" / "runtime.jsonl"
MAX_STRING_LENGTH = 2_000
REDACTED = "[redacted]"

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)

SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|authorization|credential)"
        r"\s*=\s*([^\s\"']+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{12,}\b"),
)

_LOCK = threading.Lock()


def runtime_log_path() -> Path:
    """Return the active runtime log path."""
    configured = os.getenv("AGENT_RUNTIME_LOG_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_LOG_FILE


def runtime_log(event: str, payload: dict[str, Any] | None = None) -> None:
    """Append one JSONL runtime trace event.

    Logging must never break agent execution, so all filesystem and serialization
    errors are intentionally swallowed.
    """
    if os.getenv("AGENT_RUNTIME_LOG_DISABLED") == "1":
        return

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "payload": _safe_value(payload or {}),
    }
    try:
        path = runtime_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except Exception:
        return


def redact_sensitive_data(value: Any) -> Any:
    """Return value with obvious secrets removed for logs and tool output."""

    return _safe_value(value)


def redact_sensitive_text(value: str) -> str:
    """Redact obvious secret values in free-form text."""

    text = value
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(api"):
            text = pattern.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _safe_value(value: Any, *, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {
            str(item_key): _safe_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        value = redact_sensitive_text(value)
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return value[:MAX_STRING_LENGTH] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)
