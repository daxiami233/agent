"""MCP client host and tool adapter placeholder.

TODO: Implement MCP server process lifecycle, schema discovery, async execution,
and error isolation. The current host only adapts prebuilt ToolSpec instances.
"""

from __future__ import annotations

from agent_runtime.tools import ToolRegistry, ToolSpec


class MCPClientHost:
    """Loads MCP tools and registers them with the runtime tool registry."""

    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools = tools or []

    def tools(self) -> list[ToolSpec]:
        """Return tools exposed by configured MCP servers."""

        return list(self._tools)

    def register_tools(self, registry: ToolRegistry) -> None:
        """Register MCP tools with the agent's tool registry."""

        for tool in self._tools:
            if not registry.has(tool.name):
                registry.register(tool)
