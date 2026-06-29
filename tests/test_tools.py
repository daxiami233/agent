"""Tool registry tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from agent_runtime.tools import (
    ToolArgumentError,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
    apply_patch_tool,
    built_in_tool_registry,
    list_skills_tool,
    memory_tools,
    shell_command_tool,
    skill_tools,
    tool,
    tool_from_function,
)
from agent_runtime.memory import LongTermMemory
from agent_runtime.skills import SkillManifest, SkillRegistry


def echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo arguments.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda arguments: arguments["text"],
    )


def test_tool_registry_registers_and_executes_tools():
    registry = ToolRegistry()
    registry.register(echo_tool())

    assert registry.has("echo")
    assert registry.execute("echo", {"text": "hello"}) == "hello"
    assert [tool.name for tool in registry.list()] == ["echo"]


def test_tool_registry_rejects_duplicate_tools():
    registry = ToolRegistry([echo_tool()])

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(echo_tool())


def test_tool_registry_rejects_missing_tools():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.execute("missing", {})


def test_tool_registry_validates_tool_arguments():
    registry = ToolRegistry([echo_tool()])

    with pytest.raises(ToolArgumentError, match="missing required field: text"):
        registry.execute("echo", {})

    with pytest.raises(ToolArgumentError, match="text must be string"):
        registry.execute("echo", {"text": 1})


def test_tool_registry_validates_enum_array_and_extra_fields():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="search",
                description="Search.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["fast", "deep"]},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["mode", "tags"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: arguments,
            )
        ]
    )

    assert registry.execute("search", {"mode": "fast", "tags": ["a"]}) == {
        "mode": "fast",
        "tags": ["a"],
    }
    with pytest.raises(ToolArgumentError, match="must be one of"):
        registry.execute("search", {"mode": "slow", "tags": ["a"]})
    with pytest.raises(ToolArgumentError, match=r"tags\[0\] must be string"):
        registry.execute("search", {"mode": "fast", "tags": [1]})
    with pytest.raises(ToolArgumentError, match="unexpected field"):
        registry.execute("search", {"mode": "fast", "tags": [], "extra": True})


def test_tool_registry_exports_provider_schemas():
    registry = ToolRegistry([echo_tool()])

    assert registry.provider_schemas() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo arguments.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


def test_built_in_tool_registry_is_empty():
    registry = built_in_tool_registry()

    assert registry.list() == []


def test_memory_tools_read_append_and_replace():
    memory = LongTermMemory()
    registry = ToolRegistry(memory_tools(memory))

    assert registry.execute("memory_read", {}) == {"content": ""}
    assert registry.execute("memory_append", {"content": "remember tea"}) == {"ok": True}
    assert registry.execute("memory_read", {}) == {"content": "remember tea"}
    assert registry.execute("memory_search", {"query": "tea"}) == {
        "matches": [{"line_number": 1, "content": "remember tea"}]
    }
    assert registry.execute(
        "memory_replace",
        {"old": "tea", "new": "coffee"},
    ) == {"ok": True}
    assert registry.execute("memory_read", {}) == {"content": "remember coffee"}


def test_memory_append_requires_content():
    registry = ToolRegistry(memory_tools(LongTermMemory()))

    with pytest.raises(ValueError, match="content"):
        registry.execute("memory_append", {"content": "   "})


def test_list_skills_tool_returns_name_and_description_only():
    registry = ToolRegistry(
        [
            list_skills_tool(
                SkillRegistry(
                    [
                        SkillManifest(
                            name="example",
                            description="Example skill.",
                            source="system",
                        )
                    ]
                )
            )
        ]
    )

    assert registry.execute("list_skills", {}) == [
        {"name": "example", "description": "Example skill."}
    ]


def test_skill_tools_read_skill_and_resource(tmp_path):
    skill_dir = tmp_path / "example"
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Example\nUse carefully.", encoding="utf-8")
    (references_dir / "detail.md").write_text("deep detail", encoding="utf-8")
    registry = ToolRegistry(
        skill_tools(
            SkillRegistry(
                [
                    SkillManifest(
                        name="example",
                        description="Example skill.",
                        triggers=["example"],
                        context_files=["references/detail.md"],
                        required_tools=["shell_command"],
                        skill_dir=skill_dir,
                    )
                ]
            )
        )
    )

    skill = registry.execute("skill_read", {"name": "example"})
    resource = registry.execute(
        "skill_read_resource",
        {"name": "example", "path": "references/detail.md"},
    )

    assert skill["content"] == "# Example\nUse carefully."
    assert skill["triggers"] == ["example"]
    assert skill["required_tools"] == ["shell_command"]
    assert resource["content"] == "deep detail"
    assert resource["path"] == "references/detail.md"


def test_skill_read_resource_rejects_path_traversal(tmp_path):
    skill_dir = tmp_path / "example"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Example", encoding="utf-8")
    registry = ToolRegistry(
        skill_tools(
            SkillRegistry(
                [
                    SkillManifest(
                        name="example",
                        description="Example skill.",
                        skill_dir=skill_dir,
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError, match="inside the skill directory"):
        registry.execute(
            "skill_read_resource",
            {"name": "example", "path": "../secret.txt"},
        )


def test_shell_command_tool_executes_command(tmp_path):
    registry = ToolRegistry([shell_command_tool(default_cwd=tmp_path)])

    result = registry.execute("shell_command", {"command": "printf hello"})

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello"
    assert result["stderr"] == ""
    assert result["timed_out"] is False
    assert result["cwd"] == str(tmp_path)


def test_shell_command_tool_writes_large_output_to_artifact(tmp_path):
    long_text = "0123456789" * 30
    registry = ToolRegistry(
        [
            shell_command_tool(
                default_cwd=tmp_path,
                max_output_chars=120,
                artifact_dir=tmp_path / "artifacts",
            )
        ]
    )

    result = registry.execute("shell_command", {"command": f"printf '{long_text}'"})

    assert result["exit_code"] == 0
    assert result["stdout_truncated"] is True
    assert result["stdout_ref"]
    assert Path(result["stdout_ref"]).read_text(encoding="utf-8") == long_text
    assert "[output truncated: 300 chars total;" in result["stdout"]
    assert result["stdout_summary"] == "stdout: 300 chars, 1 lines"
    assert result["stdout_chars"] == 300
    assert result["stderr_ref"] is None
    assert result["stderr_summary"] == "stderr: empty"


def test_shell_command_tool_logs_large_output_artifact(tmp_path, monkeypatch):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))
    long_text = "x" * 300
    registry = ToolRegistry(
        [
            shell_command_tool(
                default_cwd=tmp_path,
                max_output_chars=120,
                artifact_dir=tmp_path / "artifacts",
            )
        ]
    )

    result = registry.execute("shell_command", {"command": f"printf '{long_text}'"})

    records = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    artifact_records = [
        record for record in records if record["event"] == "shell_output_artifact"
    ]
    assert artifact_records
    assert artifact_records[0]["payload"]["stream"] == "stdout"
    assert artifact_records[0]["payload"]["chars"] == 300
    assert artifact_records[0]["payload"]["artifact_ref"] == result["stdout_ref"]


def test_shell_command_tool_reports_timeout(tmp_path):
    registry = ToolRegistry(
        [shell_command_tool(default_cwd=tmp_path, timeout_seconds=1)]
    )

    result = registry.execute("shell_command", {"command": "sleep 2"})

    assert result["timed_out"] is True
    assert result["exit_code"] is None


def test_shell_command_tool_blocks_env_file_reads(tmp_path):
    registry = ToolRegistry([shell_command_tool(default_cwd=tmp_path)])

    with pytest.raises(ValueError, match="shell safety policy"):
        registry.execute("shell_command", {"command": "cat .env"})


def test_shell_command_tool_blocks_root_find(tmp_path):
    registry = ToolRegistry([shell_command_tool(default_cwd=tmp_path)])

    with pytest.raises(ValueError, match="filesystem root"):
        registry.execute("shell_command", {"command": "find / -name SKILL.md"})


def test_shell_command_tool_redacts_sensitive_output(tmp_path):
    registry = ToolRegistry([shell_command_tool(default_cwd=tmp_path)])

    result = registry.execute(
        "shell_command",
        {"command": "printf 'API_KEY=sk-1234567890abcdef'"},
    )

    assert result["stdout"] == "API_KEY=[redacted]"


def test_apply_patch_tool_applies_unified_diff(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    patch = """\
diff --git a/note.txt b/note.txt
--- a/note.txt
+++ b/note.txt
@@ -1 +1 @@
-hello
+hello world
"""
    registry = ToolRegistry([apply_patch_tool(default_cwd=tmp_path)])

    result = registry.execute("apply_patch", {"patch": patch})

    assert result["ok"] is True
    assert result["changed_files"] == ["note.txt"]
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_apply_patch_tool_dry_run_does_not_modify_file(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    patch = """\
diff --git a/note.txt b/note.txt
--- a/note.txt
+++ b/note.txt
@@ -1 +1 @@
-hello
+hello world
"""
    registry = ToolRegistry([apply_patch_tool(default_cwd=tmp_path)])

    result = registry.execute("apply_patch", {"patch": patch, "dry_run": True})

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_apply_patch_tool_blocks_env_files(tmp_path):
    (tmp_path / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    patch = """\
diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -1 +1 @@
-API_KEY=secret
+API_KEY=other
"""
    registry = ToolRegistry([apply_patch_tool(default_cwd=tmp_path)])

    with pytest.raises(ValueError, match="sensitive file"):
        registry.execute("apply_patch", {"patch": patch})


def test_tool_from_function_builds_schema_and_handler():
    def add(left: int, right: int = 1) -> int:
        """Add two numbers."""
        return left + right

    spec = tool_from_function(add)

    assert spec.name == "add"
    assert spec.description == "Add two numbers."
    assert spec.input_schema["properties"]["left"]["type"] == "integer"
    assert spec.input_schema["required"] == ["left"]
    assert spec.handler({"left": 2, "right": 3}) == 5


def test_tool_decorator_builds_tool_spec():
    @tool(name="shout")
    def make_upper(text: str) -> str:
        return text.upper()

    assert isinstance(make_upper, ToolSpec)
    assert make_upper.name == "shout"
    assert make_upper.handler({"text": "hi"}) == "HI"


def test_tool_from_function_supports_common_type_annotations():
    def search(
        query: str,
        tags: list[str],
        options: dict[str, int],
        limit: int | None = None,
        mode: Literal["fast", "deep"] = "fast",
    ):
        return {
            "query": query,
            "tags": tags,
            "options": options,
            "limit": limit,
            "mode": mode,
        }

    spec = tool_from_function(search)

    assert spec.input_schema["properties"]["limit"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert spec.input_schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert spec.input_schema["properties"]["options"]["type"] == "object"
    assert spec.input_schema["properties"]["mode"]["enum"] == ["fast", "deep"]
    assert spec.input_schema["required"] == ["query", "tags", "options"]


def test_tool_from_function_optional_schema_accepts_none():
    def search(query: str, limit: int | None = None):
        return {"query": query, "limit": limit}

    registry = ToolRegistry([tool_from_function(search)])

    assert registry.execute("search", {"query": "tea", "limit": None}) == {
        "query": "tea",
        "limit": None,
    }
    with pytest.raises(ToolArgumentError, match="limit must be integer or null"):
        registry.execute("search", {"query": "tea", "limit": "many"})
