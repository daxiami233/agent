"""Skill discovery and loading."""

from .examples import example_skill
from .loader import load_skill, load_skills
from .manifest import SkillManifest
from .registry import SkillRegistry

__all__ = [
    "SkillManifest",
    "SkillRegistry",
    "example_skill",
    "load_skill",
    "load_skills",
]
