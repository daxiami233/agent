"""Load local skills from SKILL.md directories."""

from __future__ import annotations

from pathlib import Path

from .manifest import SkillManifest


def load_skill(path: Path | str) -> SkillManifest:
    """Load one skill directory containing a SKILL.md file."""

    skill_dir = Path(path).expanduser().resolve()
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill file not found: {skill_file}")
    text = skill_file.read_text(encoding="utf-8")
    metadata, body = _frontmatter(text)
    name = str(metadata.get("name") or skill_dir.name).strip()
    description = str(metadata.get("description") or _description_from_body(body)).strip()
    return SkillManifest(
        name=name,
        description=description or f"Skill from {skill_dir.name}.",
        triggers=_list_value(metadata.get("triggers")),
        context_files=_list_value(metadata.get("context_files")),
        required_tools=_list_value(metadata.get("required_tools")),
        skill_dir=skill_dir,
    )


def load_skills(path: Path | str) -> list[SkillManifest]:
    """Load all direct child skill directories under path."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        return []
    if (root / "SKILL.md").is_file():
        return [load_skill(root)]
    skills: list[SkillManifest] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            skills.append(load_skill(child))
    return skills


def _frontmatter(text: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    metadata: dict[str, object] = {}
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _metadata_value(value.strip())
    if end is None:
        return {}, text
    return metadata, "\n".join(lines[end + 1 :])


def _metadata_value(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
    return value.strip("\"'")


def _list_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _description_from_body(body: str) -> str:
    for line in body.splitlines():
        value = line.strip().lstrip("#").strip()
        if value:
            return value
    return ""
