"""Responses API adapter for OpenAI-compatible providers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from .base import ModelConfig, ModelResponse, ModelStreamEvent, ProviderError
from .openai_common import get_value, parse_responses_tool_calls, to_dict


class ResponsesAPIBackend:
    """Adapter implemented with ``client.responses.create``."""

    def build_payload(
        self,
        input: str | list[dict[str, Any]],
        *,
        model: str,
        instructions: str | None,
        tools: list[dict[str, Any]],
        model_config: ModelConfig,
        timeout: float | None,
    ) -> dict[str, Any]:
        if not model:
            raise ProviderError("Model name is required for generation.")

        payload: dict[str, Any] = {
            "model": model,
            "input": self.input_to_responses(input),
        }
        if instructions is not None:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = [self.responses_tool_schema(tool) for tool in tools]
        if model_config.temperature is not None:
            payload["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            payload["max_output_tokens"] = model_config.max_tokens
        if timeout is not None:
            payload["timeout"] = timeout
        if model_config.extra_body:
            payload["extra_body"] = model_config.extra_body
        return payload

    def generate(self, client: OpenAI, payload: dict[str, Any]) -> ModelResponse:
        response = client.responses.create(**payload)
        return self.parse_response(response)

    def input_to_responses(self, input: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(input, str):
            return input

        converted: list[dict[str, Any]] = []
        for item in input:
            role = item.get("role")
            if role == "assistant" and item.get("tool_calls"):
                converted.extend(self.tool_calls_to_responses(item.get("tool_calls")))
                continue
            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("tool_call_id"),
                        "output": item.get("content", ""),
                    }
                )
                continue
            converted.append(item)
        return converted

    def tool_calls_to_responses(self, tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []

        converted: list[dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            converted.append(
                {
                    "type": "function_call",
                    "call_id": item.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments", "{}"),
                }
            )
        return converted

    def responses_tool_schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            return tool
        return {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }

    def stream(
        self,
        client: OpenAI,
        payload: dict[str, Any],
    ) -> Iterator[ModelStreamEvent]:
        payload = dict(payload)
        payload["stream"] = True
        for event in client.responses.create(**payload):
            yield from self.parse_stream_event(event)

    def parse_response(self, response: Any) -> ModelResponse:
        return ModelResponse(
            content=self.response_output_text(response),
            tool_calls=parse_responses_tool_calls(get_value(response, "output")),
            finish_reason=get_value(response, "status"),
            usage=to_dict(get_value(response, "usage") or {}),
            raw=to_dict(response),
        )

    def response_output_text(self, response: Any) -> str | None:
        output_text = get_value(response, "output_text")
        if isinstance(output_text, str):
            return output_text

        output = get_value(response, "output")
        if not isinstance(output, list):
            return None

        text_parts: list[str] = []
        for item in output:
            content = get_value(item, "content")
            if not isinstance(content, list):
                continue
            for part in content:
                text = get_value(part, "text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts) or None

    def parse_stream_event(self, event: Any) -> Iterator[ModelStreamEvent]:
        event_type = get_value(event, "type") or ""
        raw = to_dict(event)

        if event_type == "response.output_text.delta":
            yield ModelStreamEvent(type="content_delta", delta=self.event_delta(event), raw=raw)
            return

        if "reasoning" in event_type or "thinking" in event_type:
            yield ModelStreamEvent(type=event_type, delta=self.event_delta(event), raw=raw)
            return

        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            response = get_value(event, "response")
            yield ModelStreamEvent(
                type="finish",
                response=self.parse_response(response),
                raw=raw,
            )
            return

        if event_type.endswith(".delta"):
            delta = self.event_delta(event)
            yield ModelStreamEvent(type=event_type, delta=delta, raw=raw)
            return

        yield ModelStreamEvent(type=event_type or "raw", raw=raw)

    def event_delta(self, event: Any) -> str | None:
        for key in ("delta", "text", "reasoning_content", "thinking", "content"):
            value = get_value(event, key)
            if isinstance(value, str):
                return value
        return None
