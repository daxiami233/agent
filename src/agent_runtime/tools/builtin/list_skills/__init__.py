"""List available skills tool."""

from __future__ import annotations

from typing import Any

from agent_runtime.skills import SkillRegistry
from agent_runtime.tools.registry import ToolSpec


def list_skills_tool(registry: SkillRegistry) -> ToolSpec:
    """Build a tool that lists all available skills."""

    def handle(arguments: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"name": skill.name, "description": skill.description}
            for skill in registry.list()
        ]

    return ToolSpec(
        name="list_skills",
        description=(
            "List all currently available skills, including system built-in "
            "skills and user-defined skills. Returns an array of objects with "
            "name and description fields."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["skills"],
        effects=["read"],
    )
