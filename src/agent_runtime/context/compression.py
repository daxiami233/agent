"""Model-based context compression helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_runtime.providers import ModelConfig, ModelResponse

from .engine_types import ContextMessageLike


class SummaryProvider(Protocol):
    """Minimal provider surface needed to summarize old context."""

    def generate(
        self,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Generate one non-streaming summary response."""


@dataclass(slots=True)
class CompressionResult:
    """Result returned by a context compressor."""

    summary: str = ""
    compressed: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class ContextCompressor:
    """Base class for model-based context summarizers."""

    def compress(
        self,
        *,
        conversation_id: str,
        messages: list[ContextMessageLike],
        target_tokens: int,
        previous_summary: str = "",
    ) -> CompressionResult:
        raise NotImplementedError


class ModelContextCompressor(ContextCompressor):
    """Use the configured model provider to summarize older conversation turns."""

    def __init__(
        self,
        provider: SummaryProvider,
        *,
        max_summary_tokens: int = 1_200,
        timeout_seconds: float = 60,
    ) -> None:
        self.provider = provider
        self.max_summary_tokens = max_summary_tokens
        self.timeout_seconds = timeout_seconds

    def compress(
        self,
        *,
        conversation_id: str,
        messages: list[ContextMessageLike],
        target_tokens: int,
        previous_summary: str = "",
    ) -> CompressionResult:
        if not messages and not previous_summary.strip():
            return CompressionResult(
                metadata={
                    "conversation_id": conversation_id,
                    "target_tokens": target_tokens,
                    "reason": "nothing_to_summarize",
                }
            )

        response = self.provider.generate(
            [
                {
                    "role": "system",
                    "content": _summary_system_prompt(target_tokens),
                },
                {
                    "role": "user",
                    "content": _summary_user_prompt(
                        previous_summary=previous_summary,
                        messages=messages,
                    ),
                },
            ],
            tools=[],
            model_config=ModelConfig(
                max_tokens=min(self.max_summary_tokens, max(256, target_tokens)),
                timeout_seconds=self.timeout_seconds,
            ),
        )
        summary = (response.content or "").strip()
        return CompressionResult(
            summary=summary,
            compressed=bool(summary),
            metadata={
                "conversation_id": conversation_id,
                "target_tokens": target_tokens,
                "summarized_messages": len(messages),
                "usage": response.usage,
            },
        )


def _summary_system_prompt(target_tokens: int) -> str:
    return (
        "Summarize older conversation context for a local coding agent. "
        "Preserve user goals, decisions, constraints, important files/paths, "
        "tool results, failed attempts, current state, and pending TODOs. "
        "Do not include long logs or repeated raw outputs. "
        f"Keep the summary concise enough to fit about {target_tokens} tokens."
    )


def _summary_user_prompt(
    *,
    previous_summary: str,
    messages: list[ContextMessageLike],
) -> str:
    payload = {
        "previous_summary": previous_summary.strip(),
        "messages_to_summarize": [
            {
                "role": str(getattr(message, "role", "")),
                "content": str(getattr(message, "content", "") or ""),
                "extra": _extra(message),
            }
            for message in messages
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _extra(message: ContextMessageLike) -> dict[str, Any]:
    extra = getattr(message, "extra", {})
    return extra if isinstance(extra, dict) else {}
