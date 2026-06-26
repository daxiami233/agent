"""Shared lightweight context typing helpers."""

from __future__ import annotations

from typing import Protocol


class ContextMessageLike(Protocol):
    """Minimal message interface for token counting and compression.

    Any object with role and content attributes satisfies this protocol.
    """

    role: str
    content: str
