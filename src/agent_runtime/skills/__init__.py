"""Skill discovery and loading."""

from pathlib import Path

from .loader import load_skill, load_skills
from .manifest import SkillManifest
from .registry import SkillRegistry

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin"


def load_builtin_skills() -> list[SkillManifest]:
    """Load all system built-in skills from the builtin/ directory."""
    return load_skills(_BUILTIN_SKILLS_DIR, source="system")


__all__ = [
    "SkillManifest",
    "SkillRegistry",
    "load_builtin_skills",
    "load_skill",
    "load_skills",
]
