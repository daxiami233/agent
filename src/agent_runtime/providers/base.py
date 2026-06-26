"""Provider interfaces and normalized model response types."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a model provider request or response fails."""


@dataclass(slots=True)
class RetryConfig:
    """Provider retry settings."""

    max_retries: int = 2


@dataclass(slots=True)
class ModelConfig:
    """Per-request model generation settings."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    retry: RetryConfig | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """Provider-neutral representation of a model-requested tool call."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    """Provider-neutral model response consumed by the agent loop."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelStreamEvent:
    """Provider-neutral streaming event."""

    type: str
    delta: str | None = None
    tool_call: ToolCall | None = None
    response: ModelResponse | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    """Base interface for model providers."""

    @abstractmethod
    def generate(
        self,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Generate a model response."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_config: ModelConfig | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """Stream a model response."""
        raise NotImplementedError
