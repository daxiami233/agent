"""Permission types used by Agent Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RiskLevel = Literal["auto", "confirm", "blocked"]
Effect = Literal["read", "write", "execute", "network", "destructive"]
PermissionAction = Literal["allow", "confirm", "deny"]
PermissionProfile = Literal["read_only", "workspace", "full_access"]


@dataclass(slots=True)
class PermissionRequest:
    """A pending tool-call permission evaluation."""

    id: str
    conversation_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: RiskLevel = "auto"
    effects: list[Effect] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    profile: PermissionProfile | None = None


@dataclass(slots=True)
class PermissionDecision:
    """Decision returned by a permission policy."""

    action: PermissionAction
    reason: str = ""
