"""Tool specifications and registry."""

from .memory import (
    memory_append_tool,
    memory_read_tool,
    memory_replace_tool,
    memory_search_tool,
    memory_tools,
)
from .registry import (
    ToolArgumentError,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    tool,
    tool_from_function,
)
from .shell import shell_command_tool
from .skills import skill_read_resource_tool, skill_read_tool, skill_tools


def built_in_tool_registry() -> ToolRegistry:
    """Create a registry with always-on built-in tools.

    The core runtime no longer ships a demo weather tool by default. Local
    capabilities are enabled explicitly through memory, skill, and shell tools.
    """

    return ToolRegistry()


__all__ = [
    "ToolAlreadyRegisteredError",
    "ToolArgumentError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "built_in_tool_registry",
    "memory_append_tool",
    "memory_read_tool",
    "memory_replace_tool",
    "memory_search_tool",
    "memory_tools",
    "shell_command_tool",
    "skill_read_resource_tool",
    "skill_read_tool",
    "skill_tools",
    "tool",
    "tool_from_function",
]
