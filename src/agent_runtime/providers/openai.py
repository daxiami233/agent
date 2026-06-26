"""OpenAI SDK provider with Responses and Chat Completions adapters."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import APIStatusError, DefaultHttpxClient, OpenAI, OpenAIError

from .base import (
    ModelConfig,
    ModelResponse,
    ModelStreamEvent,
    Provider,
    ProviderError,
    RetryConfig,
)
from .openai_chat import ChatCompletionsBackend
from .openai_responses import ResponsesAPIBackend


ProviderAPIMode = Literal["auto", "responses", "chat_completions"]


class OpenAIProvider(Provider):
    """Provider for OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        api_mode: ProviderAPIMode = "auto",
        context_window_tokens: int | None = None,
        timeout_seconds: float = 60,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_mode = api_mode
        self.context_window_tokens = context_window_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self._client: OpenAI | None = None
        self._resolved_api_mode: ProviderAPIMode | None = None
        self._responses_backend = ResponsesAPIBackend()
        self._chat_backend = ChatCompletionsBackend()

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        """Create a provider from ``API_KEY``, ``BASE_URL``, and ``MODEL``."""

        load_dotenv()

        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        model = os.getenv("MODEL")
        missing = [
            name
            for name, value in (
                ("API_KEY", api_key),
                ("BASE_URL", base_url),
                ("MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ProviderError(
                "Missing provider environment variables: " + ", ".join(missing)
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_mode=cls._parse_api_mode(os.getenv("PROVIDER_API")),
            context_window_tokens=cls._parse_context_window(os.getenv("CONTEXT_WINDOW")),
            timeout_seconds=cls._parse_timeout_seconds(
                os.getenv("PROVIDER_TIMEOUT_SECONDS")
            ),
            max_retries=cls._parse_max_retries(os.getenv("MAX_RETRIES")),
        )

    @classmethod
    def _parse_api_mode(cls, value: str | None) -> ProviderAPIMode:
        if value is None or not value.strip():
            return "auto"

        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"auto", "responses", "chat_completions"}:
            return normalized  # type: ignore[return-value]
        if normalized in {"chat", "chatcompletions"}:
            return "chat_completions"

        raise ProviderError(
            "PROVIDER_API must be one of: auto, responses, chat_completions."
        )

    @classmethod
    def _parse_max_retries(cls, value: str | None) -> int:
        if value is None:
            return 2
        try:
            return int(value)
        except ValueError as exc:
            raise ProviderError("MAX_RETRIES must be an integer.") from exc

    @classmethod
    def _parse_timeout_seconds(cls, value: str | None) -> float:
        if value is None or not value.strip():
            return 60
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ProviderError("PROVIDER_TIMEOUT_SECONDS must be a number.") from exc
        if parsed <= 0:
            raise ProviderError("PROVIDER_TIMEOUT_SECONDS must be greater than zero.")
        return parsed

    @classmethod
    def _parse_context_window(cls, value: str | None) -> int | None:
        if value is None or not value.strip():
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ProviderError("CONTEXT_WINDOW must be an integer.") from exc
        if parsed <= 0:
            raise ProviderError("CONTEXT_WINDOW must be greater than zero.")
        return parsed

    def generate(
        self,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Generate a model response."""

        request_config = model_config or ModelConfig()

        try:
            return self._generate_with_mode(
                self._effective_api_mode(),
                input,
                instructions=instructions,
                tools=tools or [],
                model_config=request_config,
            )
        except OpenAIError as exc:
            if self._should_fallback_to_chat(exc):
                self._resolved_api_mode = "chat_completions"
                try:
                    return self._generate_with_mode(
                        "chat_completions",
                        input,
                        instructions=instructions,
                        tools=tools or [],
                        model_config=request_config,
                    )
                except OpenAIError as chat_exc:
                    raise ProviderError(
                        f"Provider request failed after chat fallback: {chat_exc}"
                    ) from chat_exc
            raise ProviderError(f"Provider request failed: {exc}") from exc

    def stream(
        self,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_config: ModelConfig | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """Stream a model response."""

        request_config = model_config or ModelConfig()

        try:
            yield from self._stream_with_mode(
                self._effective_api_mode(),
                input,
                instructions=instructions,
                tools=tools or [],
                model_config=request_config,
            )
        except OpenAIError as exc:
            if self._should_fallback_to_chat(exc):
                self._resolved_api_mode = "chat_completions"
                try:
                    yield from self._stream_with_mode(
                        "chat_completions",
                        input,
                        instructions=instructions,
                        tools=tools or [],
                        model_config=request_config,
                    )
                    return
                except OpenAIError as chat_exc:
                    raise ProviderError(
                        f"Provider request failed after chat fallback: {chat_exc}"
                    ) from chat_exc
            raise ProviderError(f"Provider request failed: {exc}") from exc

    def _effective_api_mode(self) -> ProviderAPIMode:
        if self.api_mode == "auto":
            return self._resolved_api_mode or "responses"
        return self.api_mode

    def current_api_mode(self) -> ProviderAPIMode:
        """Return the configured or auto-resolved API mode."""

        return self._resolved_api_mode or self.api_mode

    def _generate_with_mode(
        self,
        mode: ProviderAPIMode,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None,
        tools: list[dict[str, Any]],
        model_config: ModelConfig,
    ) -> ModelResponse:
        backend = self._backend_for_mode(mode)
        response = backend.generate(
            self._get_client(model_config.retry),
            backend.build_payload(
                input,
                model=model_config.model or self.model,
                instructions=instructions,
                tools=tools,
                model_config=model_config,
                timeout=self._request_timeout(model_config.timeout_seconds),
            ),
        )
        self._remember_successful_mode(mode)
        return response

    def _stream_with_mode(
        self,
        mode: ProviderAPIMode,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None,
        tools: list[dict[str, Any]],
        model_config: ModelConfig,
    ) -> Iterator[ModelStreamEvent]:
        backend = self._backend_for_mode(mode)
        stream = backend.stream(
            self._get_client(model_config.retry),
            backend.build_payload(
                input,
                model=model_config.model or self.model,
                instructions=instructions,
                tools=tools,
                model_config=model_config,
                timeout=self._request_timeout(model_config.timeout_seconds),
            ),
        )
        for event in stream:
            yield event
        self._remember_successful_mode(mode)

    def _backend_for_mode(
        self,
        mode: ProviderAPIMode,
    ) -> ResponsesAPIBackend | ChatCompletionsBackend:
        if mode == "responses":
            return self._responses_backend
        if mode == "chat_completions":
            return self._chat_backend
        raise ProviderError("auto mode must be resolved before backend selection.")

    def _remember_successful_mode(self, mode: ProviderAPIMode) -> None:
        if self.api_mode == "auto" and mode in {"responses", "chat_completions"}:
            self._resolved_api_mode = mode

    def _should_fallback_to_chat(self, exc: OpenAIError) -> bool:
        if self.api_mode != "auto":
            return False
        if self._resolved_api_mode is not None:
            return False

        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, APIStatusError):
            status_code = exc.status_code
        if status_code in {404, 405}:
            return True
        if status_code == 502 and self._is_local_base_url():
            return True

        message = str(exc).lower()
        unsupported_markers = (
            "not found",
            "unknown endpoint",
            "unsupported endpoint",
            "responses is not supported",
            "invalid endpoint",
        )
        return any(marker in message for marker in unsupported_markers)

    def _is_local_base_url(self) -> bool:
        hostname = urlparse(self.base_url).hostname
        return hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _get_client(self, retry: RetryConfig | None = None) -> OpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "max_retries": self.max_retries,
            }
            if self.default_headers:
                kwargs["default_headers"] = self.default_headers
            if self._is_local_base_url():
                kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
            self._client = OpenAI(**kwargs)

        if retry is not None and retry.max_retries != self.max_retries:
            return self._client.with_options(max_retries=retry.max_retries)
        return self._client

    def _request_timeout(self, timeout_seconds: float | None) -> float | None:
        return timeout_seconds if timeout_seconds is not None else self.timeout_seconds

    def _parse_stream_event(self, event: Any) -> Iterator[ModelStreamEvent]:
        """Compatibility helper for tests and focused parser checks."""

        yield from self._responses_backend.parse_stream_event(event)
