"""Skill registration and prompt injection helpers.

本模块提供技能注册和系统提示词注入功能。

主要组件：
- SkillRegistry: 技能注册表，管理可用技能
- SkillManifest: 技能元数据定义

使用示例：
    from agent_runtime.skills import SkillRegistry, SkillManifest
    
    registry = SkillRegistry()
    registry.register(SkillManifest(
        name="weather",
        description="查询天气信息",
    ))
    
    # 获取带有技能描述的系统提示词
    prompt = registry.apply_to_system_prompt()
"""

from __future__ import annotations

import re

from .manifest import SkillManifest


# 默认系统提示词模板
# 使用 {tools}、{skills}、{retrieved_memory} 作为占位符
DEFAULT_SYSTEM_PROMPT = """You are Agent Runtime, a local agent. Answer in Chinese. Be concise, accurate, and actionable.

# Tools
{tools}

# Skills
Use available tools and skills when they help. After receiving tool results, answer the user directly.

{skills}

# Retrieved Memory
{retrieved_memory}"""


class SkillRegistry:
    """技能注册表，管理可用技能并渲染技能描述到系统提示词。

    技能是纯粹的元数据（名称、描述、触发条件等），
    它们被注入到系统提示词中，让模型知道有哪些能力可用。

    使用示例：
        registry = SkillRegistry()
        registry.register(SkillManifest(
            name="weather",
            description="查询天气信息",
            triggers=["天气", "温度"],
        ))
        
        # 获取带有技能描述的系统提示词
        prompt = registry.apply_to_system_prompt()
    """

    def __init__(self, skills: list[SkillManifest] | None = None) -> None:
        self._skills: dict[str, SkillManifest] = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: SkillManifest) -> None:
        """注册一个技能。

        Args:
            skill: 技能元数据

        Raises:
            ValueError: 如果技能名称为空或已存在
        """
        if not skill.name:
            raise ValueError("Skill name is required.")
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def list(self) -> list[SkillManifest]:
        """列出所有已注册的技能。

        Returns:
            技能列表，按名称排序
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
        """将技能描述注入到系统提示词中。

        如果有注册的技能，会在 {skills} 占位符位置插入技能列表。
        如果没有注册的技能，会移除 {skills} 相关的内容。

        Args:
            system_prompt: 系统提示词模板，如果为 None 则使用默认模板

        Returns:
            注入技能描述后的系统提示词
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
