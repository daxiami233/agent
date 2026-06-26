"""Shared helpers for OpenAI-compatible providers."""

from __future__ import annotations

import json
from typing import Any

from .base import ProviderError, ToolCall


def get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def parse_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise ProviderError("Provider tool call arguments are invalid.")

    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider tool call arguments are not valid JSON.") from exc

    if not isinstance(decoded, dict):
        raise ProviderError("Provider tool call arguments must decode to an object.")
    return decoded


def parse_responses_tool_calls(output: Any) -> list[ToolCall]:
    if not isinstance(output, list):
        return []

    parsed: list[ToolCall] = []
    for index, item in enumerate(output):
        item_type = get_value(item, "type")
        if item_type != "function_call":
            continue

        name = get_value(item, "name")
        if not isinstance(name, str) or not name:
            continue

        parsed.append(
            ToolCall(
                id=str(
                    get_value(item, "call_id")
                    or get_value(item, "id")
                    or f"response_tool_call_{index}"
                ),
                name=name,
                arguments=parse_arguments(get_value(item, "arguments")),
                raw=to_dict(item),
            )
        )
    return parsed


def parse_chat_tool_calls(tool_calls: Any) -> list[ToolCall]:
    if not isinstance(tool_calls, list):
        return []

    parsed: list[ToolCall] = []
    for index, item in enumerate(tool_calls):
        function = get_value(item, "function")
        name = get_value(function, "name")
        if not isinstance(name, str) or not name:
            continue

        parsed.append(
            ToolCall(
                id=str(get_value(item, "id") or f"chat_tool_call_{index}"),
                name=name,
                arguments=parse_arguments(get_value(function, "arguments")),
                raw=to_dict(item),
            )
        )
    return parsed
