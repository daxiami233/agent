"""Skill resource reading tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runtime.skills import SkillManifest, SkillRegistry

from agent_runtime.tools.registry import ToolSpec


DEFAULT_MAX_RESOURCE_BYTES = 64_000


def skill_tools(
    registry: SkillRegistry,
    *,
    max_bytes: int = DEFAULT_MAX_RESOURCE_BYTES,
) -> list[ToolSpec]:
    """Return all skill reading tools."""
    return [
        skill_read_tool(registry, max_bytes=max_bytes),
        skill_read_resource_tool(registry, max_bytes=max_bytes),
    ]


def skill_read_tool(
    registry: SkillRegistry,
    *,
    max_bytes: int = DEFAULT_MAX_RESOURCE_BYTES,
) -> ToolSpec:
    """Build a tool that reads a skill's SKILL.md file."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        skill = _skill_from_args(registry, arguments)
        result = _skill_payload(skill)
        if skill.skill_dir is None:
            result.update({"content": "", "source": "", "truncated": False})
            return result

        content, truncated = _read_skill_file(
            Path(skill.skill_dir),
            "SKILL.md",
            max_bytes=max_bytes,
        )
        result.update(
            {
                "content": content,
                "source": "SKILL.md",
                "truncated": truncated,
            }
        )
        return result

    return ToolSpec(
        name="skill_read",
        description="Read the full SKILL.md instructions for a registered skill.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered skill name.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["skills"],
        effects=["read"],
    )


def skill_read_resource_tool(
    registry: SkillRegistry,
    *,
    max_bytes: int = DEFAULT_MAX_RESOURCE_BYTES,
) -> ToolSpec:
    """Build a tool that reads a file inside a skill directory."""

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        skill = _skill_from_args(registry, arguments)
        if skill.skill_dir is None:
            raise ValueError(f"Skill has no skill_dir: {skill.name}")
        path = str(arguments.get("path", "")).strip()
        if not path:
            raise ValueError("path is required.")

        content, truncated = _read_skill_file(
            Path(skill.skill_dir),
            path,
            max_bytes=max_bytes,
        )
        return {
            **_skill_payload(skill),
            "path": path,
            "content": content,
            "truncated": truncated,
        }

    return ToolSpec(
        name="skill_read_resource",
        description="Read a relative resource file from a registered skill directory.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered skill name.",
                },
                "path": {
                    "type": "string",
                    "description": "Relative path inside the skill directory.",
                },
            },
            "required": ["name", "path"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["skills"],
        effects=["read"],
    )


def _skill_from_args(
    registry: SkillRegistry,
    arguments: dict[str, Any],
) -> SkillManifest:
    name = str(arguments.get("name", "")).strip()
    if not name:
        raise ValueError("name is required.")
    try:
        return registry.get(name)
    except KeyError as exc:
        available = ", ".join(skill.name for skill in registry.list()) or "none"
        raise KeyError(f"Skill not found: {name}. Available skills: {available}") from exc


def _skill_payload(skill: SkillManifest) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "triggers": skill.triggers,
        "context_files": skill.context_files,
        "required_tools": skill.required_tools,
        "skill_dir": str(skill.skill_dir) if skill.skill_dir is not None else "",
    }


def _read_skill_file(
    root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> tuple[str, bool]:
    target = _resolve_resource_path(root, relative_path)
    data = target.read_bytes()
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def _resolve_resource_path(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("path must be relative.")
    root = root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must stay inside the skill directory.") from exc
    if not target.is_file():
        raise FileNotFoundError(f"Skill resource not found: {relative_path}")
    return target
