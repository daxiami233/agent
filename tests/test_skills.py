"""Skill registry tests."""

from __future__ import annotations

from agent_runtime.skills import SkillManifest, SkillRegistry, load_builtin_skills, load_skill, load_skills


def _test_skill():
    return SkillManifest(
        name="example",
        description="A tiny example skill used to verify skill registration.",
        triggers=["example", "test skill"],
    )


def test_example_skill_can_be_registered_and_rendered():
    registry = SkillRegistry([_test_skill()])

    prompt = registry.apply_to_system_prompt("System\n\n{skills}")

    assert registry.list()[0].name == "example"
    assert "- example: A tiny example skill used to verify skill registration." in prompt


def test_skill_prompt_rendering_is_idempotent():
    registry = SkillRegistry([_test_skill()])
    rendered = registry.apply_to_system_prompt("System")

    rendered_again = registry.apply_to_system_prompt(rendered)

    assert rendered_again.count("# Available Skills") == 1
    assert rendered_again.count("- example:") == 1


def test_builtin_skills_load():
    skills = load_builtin_skills()
    assert len(skills) >= 1
    assert any(s.name == "demo_echo_skill" for s in skills)


def test_load_skill_from_skill_md(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: Demo skill.
triggers: [demo, test]
required_tools: [shell_command]
---
# Demo
Use it.
""",
        encoding="utf-8",
    )

    skill = load_skill(skill_dir)

    assert skill.name == "demo"
    assert skill.description == "Demo skill."
    assert skill.triggers == ["demo", "test"]
    assert skill.required_tools == ["shell_command"]
    assert load_skills(tmp_path)[0].name == "demo"
