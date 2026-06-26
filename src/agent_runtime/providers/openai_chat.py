"""Chat Completions adapter for OpenAI-compatible providers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from .base import ModelConfig, ModelResponse, ModelStreamEvent, ProviderError, ToolCall
from .openai_common import get_value, parse_arguments, parse_chat_tool_calls, to_dict


class ChatCompletionsBackend:
    """Adapter implemented with ``client.chat.completions.create``."""

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
            "messages": self.input_to_messages(input, instructions),
        }
        if tools:
            payload["tools"] = tools
        if model_config.temperature is not None:
            payload["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            payload["max_tokens"] = model_config.max_tokens
        if timeout is not None:
            payload["timeout"] = timeout
        if model_config.extra_body:
            payload["extra_body"] = model_config.extra_body
        return payload

    def input_to_messages(
        self,
        input: str | list[dict[str, Any]],
        instructions: str | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if instructions is not None:
            messages.append({"role": "system", "content": instructions})

        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
            return messages

        for item in input:
            role = item.get("role", "user")
            content = self.normalize_content(item.get("content", ""))
            if role == "tool":
                message = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": item.get("tool_call_id"),
                }
                messages.append(message)
                continue
            if role == "assistant" and item.get("tool_calls"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": self.normalize_tool_calls(item.get("tool_calls")),
                    }
                )
                continue
            messages.append({"role": role, "content": content})
        return messages

    def normalize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = "{}"
            normalized.append(
                {
                    "id": item.get("id"),
                    "type": item.get("type", "function"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": arguments,
                    },
                }
            )
        return normalized

    def normalize_content(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content

        normalized: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                normalized.append({"type": "text", "text": str(part)})
                continue

            part_type = part.get("type")
            if part_type == "input_text":
                normalized.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "input_image":
                normalized.append(
                    {
                        "type": "image_url",
                        "image_url": part.get("image_url") or part.get("image"),
                    }
                )
            else:
                normalized.append(part)
        return normalized

    def generate(self, client: OpenAI, payload: dict[str, Any]) -> ModelResponse:
        response = client.chat.completions.create(**payload)
        return self.parse_response(response)

    def stream(
        self,
        client: OpenAI,
        payload: dict[str, Any],
    ) -> Iterator[ModelStreamEvent]:
        payload = dict(payload)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        content_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        last_finish_reason: str | None = None
        last_usage: dict[str, Any] = {}
        last_raw: dict[str, Any] = {}

        for chunk in client.chat.completions.create(**payload):
            last_raw = to_dict(chunk)
            usage = get_value(chunk, "usage")
            if usage:
                last_usage = to_dict(usage)

            choices = get_value(chunk, "choices")
            if not choices:
                continue

            choice = choices[0]
            finish_reason = get_value(choice, "finish_reason")
            if finish_reason:
                last_finish_reason = finish_reason

            delta = get_value(choice, "delta")
            reasoning = self.reasoning_delta(delta)
            if reasoning:
                yield ModelStreamEvent(
                    type="chat.reasoning.delta",
                    delta=reasoning,
                    raw=last_raw,
                )

            content = get_value(delta, "content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                yield ModelStreamEvent(type="content_delta", delta=content, raw=last_raw)

            self.collect_tool_call_deltas(
                tool_call_parts,
                get_value(delta, "tool_calls"),
            )

        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content="".join(content_parts) or None,
                tool_calls=self.tool_calls_from_deltas(tool_call_parts),
                finish_reason=last_finish_reason,
                usage=last_usage,
                raw=last_raw,
            ),
            raw=last_raw,
        )

    def parse_response(self, response: Any) -> ModelResponse:
        choices = get_value(response, "choices") or []
        choice = choices[0] if choices else None
        message = get_value(choice, "message") if choice is not None else None

        return ModelResponse(
            content=get_value(message, "content"),
            tool_calls=parse_chat_tool_calls(get_value(message, "tool_calls")),
            finish_reason=get_value(choice, "finish_reason") if choice is not None else None,
            usage=to_dict(get_value(response, "usage") or {}),
            raw=to_dict(response),
        )

    def reasoning_delta(self, delta: Any) -> str | None:
        for key in ("reasoning_content", "thinking", "reasoning"):
            value = get_value(delta, key)
            if isinstance(value, str):
                return value
        return None

    def collect_tool_call_deltas(
        self,
        collected: dict[int, dict[str, Any]],
        deltas: Any,
    ) -> None:
        if not isinstance(deltas, list):
            return

        for fallback_index, item in enumerate(deltas):
            index = get_value(item, "index")
            if not isinstance(index, int):
                index = fallback_index
            current = collected.setdefault(
                index,
                {"id": "", "name": "", "arguments": "", "raw": []},
            )
            tool_id = get_value(item, "id")
            if isinstance(tool_id, str) and tool_id:
                current["id"] = tool_id

            function = get_value(item, "function")
            name = get_value(function, "name")
            if isinstance(name, str) and name:
                current["name"] = name
            arguments = get_value(function, "arguments")
            if isinstance(arguments, str):
                current["arguments"] += arguments
            current["raw"].append(to_dict(item))

    def tool_calls_from_deltas(
        self,
        collected: dict[int, dict[str, Any]],
    ) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for index in sorted(collected):
            item = collected[index]
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            parsed.append(
                ToolCall(
                    id=str(item.get("id") or f"chat_tool_call_{index}"),
                    name=name,
                    arguments=parse_arguments(item.get("arguments")),
                    raw={"chunks": item.get("raw", [])},
                )
            )
        return parsed
