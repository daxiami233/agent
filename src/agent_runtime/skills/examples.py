"""Small built-in skill examples for tests and demos."""

from __future__ import annotations

from .manifest import SkillManifest


def example_skill() -> SkillManifest:
    """Return a minimal skill manifest used by tests and examples."""
    return SkillManifest(
        name="example",
        description="A tiny example skill used to verify skill registration.",
        triggers=["example", "test skill"],
    )
