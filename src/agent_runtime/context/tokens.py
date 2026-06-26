"""Token counting helpers for context budgeting.

Provides token estimation for context window management.
Users can implement TokenCounterProtocol for custom counting logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .engine_types import ContextMessageLike


# Default context window size in tokens
DEFAULT_CONTEXT_WINDOW_TOKENS = 32_000

# Tokens reserved for model output (not used for input context)
DEFAULT_RESERVED_OUTPUT_TOKENS = 4_000

# Safety margin for tokenizer/provider differences and tool schema growth
DEFAULT_SAFETY_MARGIN_TOKENS = 1_000

# Start compaction before the hard input budget is exhausted
DEFAULT_COMPACT_THRESHOLD_RATIO = 0.8

# tiktoken encoding name used by OpenAI
DEFAULT_TIKTOKEN_ENCODING = "o200k_base"


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Computed context budgeting numbers for one model request."""

    context_window_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    input_budget_tokens: int
    compact_threshold_tokens: int
    used_input_tokens: int = 0

    @property
    def remaining_input_tokens(self) -> int:
        return max(0, self.input_budget_tokens - self.used_input_tokens)

    @property
    def used_percent(self) -> float:
        if self.input_budget_tokens <= 0:
            return 0.0
        return min(100.0, max(0.0, self.used_input_tokens / self.input_budget_tokens * 100))

    @property
    def compact_percent(self) -> float:
        if self.input_budget_tokens <= 0:
            return 0.0
        return min(100.0, max(0.0, self.compact_threshold_tokens / self.input_budget_tokens * 100))

    def with_used(self, used_input_tokens: int) -> "ContextBudget":
        return ContextBudget(
            context_window_tokens=self.context_window_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
            input_budget_tokens=self.input_budget_tokens,
            compact_threshold_tokens=self.compact_threshold_tokens,
            used_input_tokens=used_input_tokens,
        )

    def to_payload(self) -> dict[str, int | float]:
        return {
            "contextWindow": self.context_window_tokens,
            "reservedOutputTokens": self.reserved_output_tokens,
            "safetyMarginTokens": self.safety_margin_tokens,
            "inputBudgetTokens": self.input_budget_tokens,
            "compactThresholdTokens": self.compact_threshold_tokens,
            "contextUsed": self.used_input_tokens,
            "remainingInputTokens": self.remaining_input_tokens,
            "contextPercent": self.used_percent,
            "compactPercent": self.compact_percent,
        }


class TokenCounterProtocol(Protocol):
    """Protocol for custom token counters.

    Implement this to provide custom token counting logic.
    """

    def count_text(self, text: str) -> int:
        """Return estimated token count for text."""

    def count_message(self, message: ContextMessageLike) -> int:
        """Return estimated token count for a chat message."""


class TokenCounter:
    """Default token counter with optional tiktoken support.

    Falls back to byte-based estimation if tiktoken is not installed.

    Args:
        model: Model name for automatic tiktoken encoding selection
        encoding_name: tiktoken encoding name, defaults to "o200k_base"
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    ) -> None:
        self.model = model
        self.encoding_name = encoding_name
        self._encoding = self._load_encoding(model, encoding_name)

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Fallback: estimate from UTF-8 byte count (~4 bytes per token)
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

    def count_message(self, message: ContextMessageLike) -> int:
        """Count tokens in a message (role + content + extra fields)."""
        extra = getattr(message, "extra", {})
        extra_text = ""
        if extra:
            try:
                extra_text = json.dumps(extra, ensure_ascii=False, sort_keys=True)
            except TypeError:
                extra_text = str(extra)
        # 4 is the per-message overhead in OpenAI Chat Completions API
        return (
            4
            + self.count_text(message.role)
            + self.count_text(message.content)
            + self.count_text(extra_text)
        )

    def _load_encoding(self, model: str | None, encoding_name: str):
        """Load tiktoken encoding if available."""
        try:
            import tiktoken
        except ImportError:
            return None

        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass
        try:
            return tiktoken.get_encoding(encoding_name)
        except (KeyError, ValueError):
            return None
