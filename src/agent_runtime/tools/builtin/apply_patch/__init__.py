"""Patch application tool."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from agent_runtime.logging import runtime_log
from agent_runtime.tools.registry import ToolSpec


DEFAULT_MAX_PATCH_BYTES = 100_000
DEFAULT_TIMEOUT_SECONDS = 10

BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}
BLOCKED_SUFFIXES = {
    ".db",
    ".pem",
    ".sqlite",
    ".sqlite3",
}
BLOCKED_PARTS = {
    ".agent-runtime",
    ".git",
    ".mimocode",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}


def apply_patch_tool(
    *,
    default_cwd: Path | str | None = None,
    max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolSpec:
    """Build a tool that applies a unified diff patch."""

    base_cwd = Path(default_cwd or os.getcwd()).resolve()
    max_bytes = max(1, int(max_patch_bytes))
    timeout = max(1, int(timeout_seconds))

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        patch = str(arguments.get("patch", ""))
        if not patch.strip():
            raise ValueError("patch is required.")
        patch_size = len(patch.encode("utf-8"))
        if patch_size > max_bytes:
            raise ValueError(
                f"patch is too large: {patch_size} bytes, limit is {max_bytes}."
            )

        cwd_value = arguments.get("cwd")
        cwd = Path(str(cwd_value)).expanduser() if cwd_value else base_cwd
        if not cwd.is_absolute():
            cwd = base_cwd / cwd
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")
        if not _is_relative_to(cwd, base_cwd):
            raise ValueError("cwd must be inside the default project directory.")

        changed_files = _changed_files_from_patch(patch, cwd=cwd)
        dry_run = bool(arguments.get("dry_run", False))
        _run_git_apply(patch, cwd=cwd, check=True, timeout_seconds=timeout)

        runtime_log(
            "apply_patch_check",
            {
                "cwd": str(cwd),
                "changed_files": changed_files,
                "patch_bytes": patch_size,
                "dry_run": dry_run,
            },
        )

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "cwd": str(cwd),
                "changed_files": changed_files,
                "patch_bytes": patch_size,
            }

        _run_git_apply(patch, cwd=cwd, check=False, timeout_seconds=timeout)
        runtime_log(
            "apply_patch_done",
            {
                "cwd": str(cwd),
                "changed_files": changed_files,
                "patch_bytes": patch_size,
            },
        )
        return {
            "ok": True,
            "dry_run": False,
            "cwd": str(cwd),
            "changed_files": changed_files,
            "patch_bytes": patch_size,
        }

    return ToolSpec(
        name="apply_patch",
        description=(
            "Apply a unified diff patch to files in the current project. Use this "
            "for controlled file edits instead of shell redirection. The tool "
            "blocks patches that touch secrets, runtime data, databases, private "
            "keys, node_modules, caches, or paths outside the project."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch text to apply.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory inside the project.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate the patch without applying it.",
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["file", "write"],
    )


def _run_git_apply(
    patch: str,
    *,
    cwd: Path,
    check: bool,
    timeout_seconds: int,
) -> None:
    command = ["git", "apply", "--check", "-"] if check else ["git", "apply", "-"]
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=patch,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        action = "validate" if check else "apply"
        raise ValueError(f"failed to {action} patch: {detail}")


def _changed_files_from_patch(patch: str, *, cwd: Path) -> list[str]:
    files: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.add(_normalize_patch_path(parts[2]))
                files.add(_normalize_patch_path(parts[3]))
        elif line.startswith("--- ") or line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0].strip()
            files.add(_normalize_patch_path(path))

    files.discard("")
    files.discard("/dev/null")
    normalized = sorted(files)
    if not normalized:
        raise ValueError("patch does not contain any changed file paths.")
    for file_path in normalized:
        _validate_patch_path(file_path, cwd=cwd)
    return normalized


def _normalize_patch_path(path: str) -> str:
    value = path.strip()
    if value in {"", "/dev/null"}:
        return value
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return value


def _validate_patch_path(path: str, *, cwd: Path) -> None:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"patch path must be relative: {path}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"patch path may not contain '..': {path}")
    if candidate.name in BLOCKED_NAMES:
        raise ValueError(f"patch may not modify sensitive file: {path}")
    if candidate.suffix in BLOCKED_SUFFIXES:
        raise ValueError(f"patch may not modify blocked file type: {path}")
    if any(part in BLOCKED_PARTS for part in candidate.parts):
        raise ValueError(f"patch may not modify blocked path: {path}")

    resolved = (cwd / candidate).resolve()
    if not _is_relative_to(resolved, cwd):
        raise ValueError(f"patch path escapes cwd: {path}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
