"""Local shell execution tool."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from agent_runtime.logging import redact_sensitive_text, runtime_log

from agent_runtime.tools.registry import ToolSpec


DEFAULT_SHELL_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_CHARS = 20_000

BLOCKED_COMMAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(^|[\s;&|()<>])find\s+/(?:\s|$)"),
        "recursive search from filesystem root",
    ),
    (
        re.compile(r"(^|[\s;&|()<>])(env|printenv|set)(?:\s|$)"),
        "dumping process environment",
    ),
    (
        re.compile(r"\$(API[_-]?KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL)", re.I),
        "expanding sensitive environment variables",
    ),
    (
        re.compile(r"(^|[\s/\"'])\.env(?:[\s/\"']|$|[.*])"),
        "reading .env files",
    ),
    (
        re.compile(r"(^|[\s/\"'])(id_rsa|id_ed25519|.*\.pem)(?:[\s/\"']|$|\*)"),
        "reading private key files",
    ),
)


def shell_command_tool(
    *,
    default_cwd: Path | str | None = None,
    timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ToolSpec:
    """Build a tool that executes a local shell command."""

    base_cwd = Path(default_cwd or os.getcwd())

    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "")).strip()
        if not command:
            raise ValueError("command is required.")
        _validate_command(command)

        cwd_value = arguments.get("cwd")
        cwd = Path(str(cwd_value)).expanduser() if cwd_value else base_cwd
        if not cwd.is_absolute():
            cwd = base_cwd / cwd
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")

        requested_timeout = arguments.get("timeout_seconds", timeout_seconds)
        timeout = min(max(1, int(requested_timeout)), timeout_seconds)
        runtime_log(
            "shell_command_start",
            {
                "command": command,
                "cwd": str(cwd),
                "timeout_seconds": timeout,
            },
        )

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                shell=True,
                executable="/bin/zsh",
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout_text = redact_sensitive_text(completed.stdout)
            stderr_text = redact_sensitive_text(completed.stderr)
            stdout, stdout_truncated = _limit_output(
                stdout_text,
                max_output_chars=max_output_chars,
            )
            stderr, stderr_truncated = _limit_output(
                stderr_text,
                max_output_chars=max_output_chars,
            )
            runtime_log(
                "shell_command_complete",
                {
                    "command": command,
                    "cwd": str(cwd),
                    "exit_code": completed.returncode,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            )
            return {
                "command": command,
                "cwd": str(cwd),
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            stdout_text = redact_sensitive_text(_coerce_output(exc.stdout or ""))
            stderr_text = redact_sensitive_text(_coerce_output(exc.stderr or ""))
            stdout, stdout_truncated = _limit_output(
                stdout_text,
                max_output_chars=max_output_chars,
            )
            stderr, stderr_truncated = _limit_output(
                stderr_text,
                max_output_chars=max_output_chars,
            )
            runtime_log(
                "shell_command_timeout",
                {
                    "command": command,
                    "cwd": str(cwd),
                    "timeout_seconds": timeout,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            )
            return {
                "command": command,
                "cwd": str(cwd),
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "timed_out": True,
            }

    return ToolSpec(
        name="shell_command",
        description=(
            "Execute a local shell command in the current project and return "
            "stdout, stderr, and exit code. Commands that read secrets, dump "
            "environment variables, or recursively scan from filesystem root are "
            "blocked."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout capped by the tool configuration.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=handle,
        capabilities=["shell"],
    )


def _validate_command(command: str) -> None:
    for pattern, reason in BLOCKED_COMMAND_PATTERNS:
        if pattern.search(command):
            raise ValueError(f"Command blocked by shell safety policy: {reason}.")


def _limit_output(value: str | bytes, *, max_output_chars: int) -> tuple[str, bool]:
    value = _coerce_output(value)
    if len(value) <= max_output_chars:
        return value, False
    return value[:max_output_chars], True


def _coerce_output(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
