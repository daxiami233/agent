"""Skill metadata used by the runtime.

Skill content follows a Claude-style progressive disclosure model:
the system prompt receives only name/description level metadata, and tools can
read the full SKILL.md or referenced context files when a task needs them.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SkillManifest:
    """Metadata for one local skill directory."""

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    skill_dir: Path | str | None = None
    source: str = "user"
