"""Tool permission tests."""

from __future__ import annotations

from agent_runtime.context import ContextEngine
from agent_runtime.memory import MemoryStore
from agent_runtime.permissions import (
    DefaultPermissionPolicy,
    PermissionManager,
    PermissionRequest,
)
from agent_runtime.providers import ModelResponse, ModelStreamEvent, ToolCall
from agent_runtime.runtime import AgentLoop
from agent_runtime.tools import ToolRegistry, ToolSpec


class OneToolCallProvider:
    model = "fake-model"

    def __init__(self, tool_name: str, arguments: dict | None = None) -> None:
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.calls = 0

    def stream(self, input, **kwargs):
        self.calls += 1
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name=self.tool_name,
                        arguments=self.arguments,
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 4},
            ),
        )


class ToolThenAnswerProvider:
    model = "fake-model"

    def __init__(self, tool_name: str, arguments: dict | None = None) -> None:
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.calls = 0

    def stream(self, input, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield ModelStreamEvent(
                type="finish",
                response=ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name=self.tool_name,
                            arguments=self.arguments,
                        )
                    ],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 4},
                ),
            )
            return
        yield ModelStreamEvent(type="content_delta", delta="done")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 1},
            ),
        )


def _request(
    *,
    tool_name: str = "tool",
    arguments: dict | None = None,
    risk_level: str = "auto",
    effects: list[str] | None = None,
) -> PermissionRequest:
    return PermissionRequest(
        id="call-1",
        conversation_id="conversation-1",
        tool_name=tool_name,
        arguments=arguments or {},
        risk_level=risk_level,  # type: ignore[arg-type]
        effects=effects or [],  # type: ignore[arg-type]
    )


def test_default_permission_policy_allows_read_tools():
    policy = DefaultPermissionPolicy()

    decision = policy.evaluate(_request(effects=["read"]))

    assert decision.action == "allow"


def test_default_permission_policy_confirms_write_tools():
    policy = DefaultPermissionPolicy()

    decision = policy.evaluate(_request(effects=["write"]))

    assert decision.action == "confirm"


def test_permission_request_profile_overrides_default_profile():
    policy = DefaultPermissionPolicy(profile="workspace")

    decision = policy.evaluate(_request(effects=["write"]))
    overridden = policy.evaluate(
        PermissionRequest(
            id="call-1",
            conversation_id="conversation-1",
            tool_name="tool",
            arguments={},
            effects=["write"],
            profile="full_access",
        )
    )
    confirmed = policy.evaluate(
        PermissionRequest(
            id="call-2",
            conversation_id="conversation-1",
            tool_name="tool",
            arguments={},
            risk_level="confirm",
            effects=["write"],
            profile="full_access",
        )
    )
    blocked = policy.evaluate(
        PermissionRequest(
            id="call-3",
            conversation_id="conversation-1",
            tool_name="tool",
            arguments={},
            risk_level="blocked",
            effects=["destructive"],
            profile="full_access",
        )
    )

    assert decision.action == "confirm"
    assert overridden.action == "allow"
    assert confirmed.action == "allow"
    assert blocked.action == "deny"


def test_default_permission_policy_denies_blocked_tools():
    policy = DefaultPermissionPolicy()

    decision = policy.evaluate(_request(risk_level="blocked"))

    assert decision.action == "deny"


def test_default_permission_policy_allows_safe_shell_commands():
    policy = DefaultPermissionPolicy()

    for command in [
        "ls -la",
        "pwd && ls -la",
        "pwd && find . -maxdepth 1 -type f",
        "rg permission src",
        "git status --short",
        "python -m pytest tests/test_permissions.py -q",
    ]:
        decision = policy.evaluate(
            _request(
                tool_name="shell_command",
                arguments={"command": command},
                effects=["execute"],
            )
        )
        assert decision.action == "allow"


def test_default_permission_policy_confirms_shell_commands_with_side_effects():
    policy = DefaultPermissionPolicy()

    for command in [
        "echo hi > file.txt",
        "pwd && echo hi > file.txt",
        "pwd && rm file.txt",
        "pwd || ls -la",
        "pwd; ls -la",
        "ls -la | head",
        "git commit -m test",
        "sed -i '' 's/a/b/' file.txt",
    ]:
        decision = policy.evaluate(
            _request(
                tool_name="shell_command",
                arguments={"command": command},
                effects=["execute"],
            )
        )
        assert decision.action == "confirm"


def test_default_permission_policy_denies_dangerous_shell_commands():
    policy = DefaultPermissionPolicy()

    for command in ["sudo ls", "rm -rf /", "dd if=/dev/zero of=/tmp/x"]:
        decision = policy.evaluate(
            _request(
                tool_name="shell_command",
                arguments={"command": command},
                effects=["execute"],
            )
        )
        assert decision.action == "deny"


def test_agent_loop_stops_for_confirm_tool_without_executing_or_persisting_tool_call(tmp_path):
    executed: list[dict] = []
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "write")
    provider = OneToolCallProvider("write_file", {"path": "x.txt"})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments),
                risk_level="confirm",
                effects=["write"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    events = list(loop.run("conversation-1"))

    assert [event.type for event in events] == ["permission_request", "notice"]
    assert events[0].payload["tool_name"] == "write_file"
    assert events[0].payload["permission_id"]
    assert executed == []
    model_input = context.build_model_input("conversation-1")
    assert [message["role"] for message in model_input] == ["system", "user"]
    assert model_input[1] == {"role": "user", "content": "write"}


def test_agent_loop_stops_for_denied_tool_without_executing(tmp_path):
    executed: list[dict] = []
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "blocked")
    provider = OneToolCallProvider("danger", {})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="danger",
                description="Dangerous tool.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments),
                risk_level="blocked",
                effects=["destructive"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    events = list(loop.run("conversation-1"))

    assert [event.type for event in events] == ["permission_denied", "notice"]
    assert executed == []


def test_agent_loop_uses_permission_profile_for_current_run(tmp_path):
    executed: list[dict] = []
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "write")
    provider = ToolThenAnswerProvider("write_file", {"path": "x.txt"})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments) or {"ok": True},
                effects=["write"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    events = list(loop.run("conversation-1", permission_profile="full_access"))

    assert "permission_request" not in [event.type for event in events]
    assert [event.type for event in events][-3:] == [
        "assistant_start",
        "assistant_delta",
        "usage",
    ]
    assert executed == [{"path": "x.txt"}]


def test_agent_loop_resumes_approved_permission_without_regenerating_tool_call(tmp_path):
    executed: list[dict] = []
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "write")
    provider = ToolThenAnswerProvider("write_file", {"path": "x.txt"})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments) or {"ok": True},
                effects=["write"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    first_events = list(loop.run("conversation-1"))
    permission_id = first_events[0].payload["permission_id"]
    approved_events = list(
        loop.resume_permission(
            permission_id,
            approved=True,
        )
    )

    assert provider.calls == 1
    assert executed == [{"path": "x.txt"}]
    assert [event.type for event in approved_events] == [
        "tool_call_start",
        "tool_call_result",
        "assistant_start",
        "assistant_delta",
        "usage",
    ]
    assert approved_events[-2].payload["text"] == "操作已完成。"
    assert [
        message.role
        for message in store.list_messages("conversation-1")
    ] == ["user", "assistant", "tool", "assistant"]


def test_agent_loop_records_denied_permission_as_tool_result(tmp_path):
    executed: list[dict] = []
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "write")
    provider = OneToolCallProvider("write_file", {"path": "x.txt"})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments),
                effects=["write"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    first_events = list(loop.run("conversation-1"))
    permission_id = first_events[0].payload["permission_id"]
    denied_events = list(loop.resume_permission(permission_id, approved=False))

    assert provider.calls == 1
    assert executed == []
    assert [event.type for event in denied_events] == ["tool_call_result", "notice"]
    assert denied_events[0].payload["status"] == "denied"
    messages = store.list_messages("conversation-1")
    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert "User denied permission" in messages[-1].content


def test_apply_patch_error_result_has_clear_summary(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "patch")
    provider = ToolThenAnswerProvider("apply_patch", {"patch": "bad patch"})
    registry = ToolRegistry(
        [
            ToolSpec(
                name="apply_patch",
                description="Apply patch.",
                input_schema={"type": "object"},
                handler=lambda arguments: (_ for _ in ()).throw(
                    ValueError(
                        "failed to validate patch: error: corrupt patch at line 6"
                    )
                ),
                effects=["write"],
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=registry,
        permission_manager=PermissionManager(DefaultPermissionPolicy()),
    )

    events = list(loop.run("conversation-1", permission_profile="full_access"))
    result = next(event for event in events if event.type == "tool_call_result")

    assert result.payload["status"] == "error"
    assert result.payload["result"]["tool"] == "apply_patch"
    assert result.payload["result"]["phase"] == "validate"
    assert "not applied" in result.payload["result"]["message"]
    assert "valid unified diff" in result.payload["summary"]
