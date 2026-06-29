"""Skill registration and system-prompt rendering helpers.

The registry stores skill metadata only. Full skill instructions remain in the
skill directory and can be loaded later through skill tools when a task needs
progressive disclosure.

Example:
    from agent_runtime.skills import SkillManifest, SkillRegistry

    registry = SkillRegistry()
    registry.register(SkillManifest(
        name="weather",
        description="Look up weather information.",
    ))

    prompt = registry.apply_to_system_prompt()
"""

from __future__ import annotations

import re

from .manifest import SkillManifest


# Default system prompt template.
# Supported placeholders: {tools}, {skills}, and {retrieved_memory}.
DEFAULT_SYSTEM_PROMPT = """You are Agent Runtime, a local agent. Answer in Chinese. Be concise, accurate, and actionable.

# Tools
{tools}

# Skills
Use available tools and skills when they help. After receiving tool results, answer the user directly.

{skills}

# Retrieved Memory
{retrieved_memory}"""


class SkillRegistry:
    """Registry for skill metadata and skill prompt rendering.

    A skill is represented as metadata here: name, description, triggers,
    required tools, and resource paths. The rendered system prompt gives the
    model a compact list of available skills without loading every full
    ``SKILL.md`` file into the context.

    Example:
        registry = SkillRegistry()
        registry.register(SkillManifest(
            name="weather",
            description="Look up weather information.",
            triggers=["weather", "temperature"],
        ))

        prompt = registry.apply_to_system_prompt()
    """

    def __init__(self, skills: list[SkillManifest] | None = None) -> None:
        self._skills: dict[str, SkillManifest] = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: SkillManifest) -> None:
        """Register one skill manifest.

        Args:
            skill: Skill metadata loaded from code or a ``SKILL.md`` file.

        Raises:
            ValueError: If the skill name is empty or already registered.
        """
        if not skill.name:
            raise ValueError("Skill name is required.")
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def list(self) -> list[SkillManifest]:
        """Return all registered skills sorted by name.

        Returns:
            Skill manifests sorted by name.
        """
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> SkillManifest:
        """Return a registered skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Skill not found: {name}") from exc

    def has(self, name: str) -> bool:
        """Return whether a skill is registered."""
        return name in self._skills

    def apply_to_system_prompt(self, system_prompt: str | None = None) -> str:
        """Inject compact skill metadata into a system prompt template.

        Registered skills are rendered at the ``{skills}`` placeholder. If the
        template has already been rendered, the previous skill block is removed
        before a fresh block is appended.

        Args:
            system_prompt: Optional prompt template. The default template is
                used when this is ``None``.

        Returns:
            A prompt containing the current compact skill list.
        """
        prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        skills = self.list()
        user_skills = [
            skill
            for skill in skills
            if getattr(skill, "source", "user") != "system"
        ]

        if not skills:
            prompt = prompt.replace(
                "{skills}",
                "No skills are currently available.",
            )
        else:
            lines = []
            if not user_skills:
                lines.append(
                    "No user-defined skills are loaded. Built-in system skills are still available."
                )
                lines.append("")
            lines.append("# Available Skills")
            for skill in skills:
                lines.append(f"- {skill.name}: {skill.description}")
            skills_text = "\n".join(lines)
            if "{skills}" in prompt:
                prompt = prompt.replace("{skills}", skills_text)
            else:
                prompt = f"{_strip_rendered_skills(prompt)}\n{skills_text}"

        return prompt


def _strip_rendered_skills(prompt: str) -> str:
    """Remove a previously rendered skill block before re-rendering."""

    return re.sub(
        r"\n*# Available Skills\n(?:- .*(?:\n|$))*",
        "\n",
        prompt,
    ).strip()
