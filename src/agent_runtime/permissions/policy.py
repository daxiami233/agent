"""Default permission policy."""

from __future__ import annotations

import re
import shlex
from typing import Protocol

from .types import PermissionDecision, PermissionProfile, PermissionRequest


class PermissionPolicyProtocol(Protocol):
    """Interface implemented by runtime or business permission policies."""

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        """Return the permission decision for a tool call."""


class DefaultPermissionPolicy:
    """Default local development policy.

    The policy is intentionally conservative for file writes and unknown shell
    commands, but allows common read, test, and build shell commands in the
    workspace profile so the agent remains usable for development tasks.
    """

    def __init__(self, *, profile: PermissionProfile = "workspace") -> None:
        self.profile = profile

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        profile = request.profile or self.profile
        if request.risk_level == "blocked":
            return PermissionDecision("deny", "Tool is marked as blocked.")

        if request.tool_name == "shell_command":
            return self._evaluate_shell(request)

        if profile == "read_only" and _has_any(
            request,
            {"write", "execute", "network", "destructive"},
        ):
            return PermissionDecision(
                "deny",
                "Current permission profile is read-only.",
            )

        if "memory" in request.capabilities and not _has_any(
            request,
            {"execute", "network", "destructive"},
        ):
            return PermissionDecision(
                "allow",
                "Memory tool is allowed by default policy.",
            )

        if profile == "full_access":
            return PermissionDecision("allow", "Tool is allowed in full-access profile.")

        if request.risk_level == "confirm":
            return PermissionDecision("confirm", "Tool requires confirmation.")

        if profile != "full_access" and _has_any(
            request,
            {"write", "execute", "network", "destructive"},
        ):
            return PermissionDecision(
                "confirm",
                "Tool has side effects and requires confirmation.",
            )

        return PermissionDecision("allow", "Tool is allowed by default policy.")

    def _evaluate_shell(self, request: PermissionRequest) -> PermissionDecision:
        profile = request.profile or self.profile
        command = str(request.arguments.get("command", "")).strip()
        if not command:
            return PermissionDecision("deny", "Shell command is empty.")
        if _is_dangerous_shell(command):
            return PermissionDecision("deny", "Shell command is blocked by policy.")
        if profile == "read_only":
            if _is_safe_read_shell(command):
                return PermissionDecision("allow", "Read-only shell command is allowed.")
            return PermissionDecision(
                "deny",
                "Current permission profile is read-only.",
            )
        if profile == "full_access":
            return PermissionDecision(
                "allow",
                "Shell command is allowed in full-access profile.",
            )
        if _is_safe_read_shell(command) or _is_test_or_build_shell(command):
            return PermissionDecision(
                "allow",
                "Shell command is allowed by workspace policy.",
            )
        return PermissionDecision("confirm", "Shell command requires confirmation.")


def _has_any(request: PermissionRequest, effects: set[str]) -> bool:
    return any(effect in effects for effect in request.effects)


def _is_safe_read_shell(command: str) -> bool:
    if _has_shell_write_or_unsafe_chain_operator(command):
        return False
    parts_list = _split_safe_and_chain(command)
    if not parts_list:
        return False
    return all(_is_single_safe_read_shell(parts) for parts in parts_list)


def _is_single_safe_read_shell(parts: list[str]) -> bool:
    if not parts:
        return False
    first = parts[0].split("/")[-1]
    if first == "git":
        return len(parts) >= 2 and parts[1] in {
            "branch",
            "diff",
            "log",
            "rev-parse",
            "show",
            "status",
        }
    if first == "sed" and "-i" in parts:
        return False
    if first == "find" and "-delete" in parts:
        return False
    return first in {
        "cat",
        "echo",
        "find",
        "head",
        "ls",
        "nl",
        "printf",
        "pwd",
        "rg",
        "sed",
        "tail",
        "wc",
    }


def _split_safe_and_chain(command: str) -> list[list[str]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="&|;<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    groups: list[list[str]] = [[]]
    for token in tokens:
        if token == "&&":
            if not groups[-1]:
                return []
            groups.append([])
            continue
        groups[-1].append(token)
    if not groups or not groups[-1]:
        return []
    return groups


def _is_test_or_build_shell(command: str) -> bool:
    if _has_shell_write_or_chain_operator(command):
        return False
    normalized = command.strip()
    return bool(
        re.match(r"^(python|python3) -m pytest(?:\s|$)", normalized)
        or re.match(r"^pytest(?:\s|$)", normalized)
        or normalized == "npm run build"
    )


def _is_dangerous_shell(command: str) -> bool:
    return bool(
        re.search(r"(^|[\s;&|])sudo(?:\s|$)", command)
        or re.search(r"(^|[\s;&|])rm\s+-[^;&|]*r[^;&|]*f\s+/(?:\s|$)", command)
        or re.search(r"(^|[\s;&|])(dd|mkfs)(?:\s|$)", command)
        or re.search(r"(^|[\s;&|])chmod\s+-R\s+777(?:\s|$)", command)
    )


def _has_shell_write_or_unsafe_chain_operator(command: str) -> bool:
    return any(operator in command for operator in {">", ">>", "||", ";", "|"})


def _has_shell_write_or_chain_operator(command: str) -> bool:
    return any(operator in command for operator in {">", ">>", "&&", "||", ";", "|"})
