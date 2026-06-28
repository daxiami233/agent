"""Permission manager used by the agent loop."""

from __future__ import annotations

from .policy import DefaultPermissionPolicy, PermissionPolicyProtocol
from .types import PermissionDecision, PermissionRequest


class PermissionManager:
    """Thin runtime wrapper around an injected permission policy."""

    def __init__(self, policy: PermissionPolicyProtocol | None = None) -> None:
        self.policy = policy or DefaultPermissionPolicy()

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        """Evaluate one tool call permission request."""

        return self.policy.evaluate(request)
