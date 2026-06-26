"""Long-term memory tools."""

from __future__ import annotations

from typing import Any

from agent_runtime.memory import LongTermMemoryProtocol

from .registry import ToolSpec


def memory_tools(memory: LongTermMemoryProtocol) -> list[ToolSpec]:
    """Return all long-term memory tools."""
    return [
        memory_read_tool(memory),
        memory_search_tool(memory),
        memory_append_tool(memory),
        memory_replace_tool(memory),
    ]


def memory_read_tool(memory: LongTermMemoryProtocol) -> ToolSpec:
    """Build a tool that reads long-term memory."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"content": memory.read()}

    return ToolSpec(
        name="memory_read",
        description="Read the current long-term memory content.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["memory"],
    )


def memory_search_tool(memory: LongTermMemoryProtocol) -> ToolSpec:
    """Build a tool that searches long-term memory."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query is required.")
        limit = int(arguments.get("limit", 20))
        return {
            "matches": [
                {"line_number": record.line_number, "content": record.content}
                for record in memory.search(query, limit=max(1, min(limit, 50)))
            ]
        }

    return ToolSpec(
        name="memory_search",
        description="Search long-term memory entries by keyword.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches to return, capped at 50.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["memory"],
    )


def memory_append_tool(memory: LongTermMemoryProtocol) -> ToolSpec:
    """Build a tool that appends one long-term memory entry."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        content = str(arguments.get("content", "")).strip()
        if not content:
            raise ValueError("content is required.")
        memory.append(content)
        return {"ok": True}

    return ToolSpec(
        name="memory_append",
        description=(
            "Append a long-term memory entry. Use only when the user explicitly "
            "asks the agent to remember something."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory text to append.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["memory"],
    )


def memory_replace_tool(memory: LongTermMemoryProtocol) -> ToolSpec:
    """Build a tool that replaces one long-term memory entry."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        if not old:
            raise ValueError("old is required.")
        return {"ok": memory.replace(old, new)}

    return ToolSpec(
        name="memory_replace",
        description="Replace the first matching long-term memory text with new text.",
        input_schema={
            "type": "object",
            "properties": {
                "old": {
                    "type": "string",
                    "description": "Existing memory text to replace.",
                },
                "new": {
                    "type": "string",
                    "description": "Replacement memory text.",
                },
            },
            "required": ["old", "new"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["memory"],
    )
