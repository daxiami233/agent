"""Web runtime tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agent_runtime.memory import LongTermMemory, MemoryStore
from agent_runtime.providers import ModelResponse, ModelStreamEvent, ToolCall
from agent_runtime.skills import SkillManifest
from agent_runtime.tools import ToolRegistry, ToolSpec
from web.server import create_app
from web.session import WebSession


class FakeProvider:
    model = "fake-model"
    base_url = "https://example.test/v1"
    context_window_tokens = 128000

    def current_api_mode(self):
        return "responses"

    def stream(self, input, **kwargs):
        yield ModelStreamEvent(type="reasoning_delta", delta="thinking")
        yield ModelStreamEvent(type="content_delta", delta="hello")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        )


class RecordingProvider(FakeProvider):
    def __init__(self):
        self.inputs = []
        self.configs = []
        self.tools = []

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.configs.append(kwargs.get("model_config"))
        self.tools.append(kwargs.get("tools"))
        yield from super().stream(input, **kwargs)


class ToolCallingProvider(FakeProvider):
    def __init__(self):
        self.inputs = []
        self.tools = []
        self.calls = 0

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.tools.append(kwargs.get("tools"))
        if self.calls == 0:
            self.calls += 1
            yield ModelStreamEvent(type="reasoning_delta", delta="需要查天气")
            yield ModelStreamEvent(
                type="finish",
                response=ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="weather",
                            arguments={"location": "Shanghai"},
                        )
                    ],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 12},
                ),
            )
            return

        self.calls += 1
        yield ModelStreamEvent(type="content_delta", delta="上海现在晴，25℃。")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 8},
            ),
        )


class RequestedToolProvider(FakeProvider):
    def __init__(self, tool_call: ToolCall):
        self.inputs = []
        self.tools = []
        self.calls = 0
        self.tool_call = tool_call

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.tools.append(kwargs.get("tools"))
        if self.calls == 0:
            self.calls += 1
            yield ModelStreamEvent(
                type="finish",
                response=ModelResponse(
                    content=None,
                    tool_calls=[self.tool_call],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 12},
                ),
            )
            return

        self.calls += 1
        yield ModelStreamEvent(type="content_delta", delta="done")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 1},
            ),
        )


def weather_registry(result=None, handler=None):
    return ToolRegistry(
        [
            ToolSpec(
                name="weather",
                description="查天气",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                handler=handler or (lambda arguments: result or {"temperature_c": 25}),
            )
        ]
    )


def make_session(tmp_path, provider=None, tool_registry=None):
    return WebSession(
        provider_factory=lambda: provider or FakeProvider(),
        memory_store=MemoryStore(tmp_path / "memory.sqlite3"),
        long_term_memory=LongTermMemory(),
        tool_registry=tool_registry,
    )


def test_web_session_bootstrap_status(tmp_path):
    session = make_session(tmp_path)

    events = session.boot_events()

    assert events[0].type == "status"
    assert events[0].payload["model"] == "fake-model"
    assert events[0].payload["context"].startswith("输入上下文：")
    assert events[0].payload["context"].endswith(" / 123.0k")
    assert events[0].payload["contextUsed"] > 0
    assert events[0].payload["inputBudgetTokens"] == 123000
    assert events[0].payload["contextWindow"] == 128000
    assert events[0].payload["logFile"]


def test_web_session_uses_configured_memory_store_as_sdk_component(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    session = WebSession(
        provider_factory=lambda: FakeProvider(),
        memory_store=store,
        long_term_memory=LongTermMemory(),
    )

    assert session.memory_store.path == store.path
    assert session.agent.memory_store.path == store.path


def test_web_session_streams_model_events(tmp_path):
    session = make_session(tmp_path)

    events = list(session.submit("conversation-1", "hi"))

    assert [event.type for event in events] == [
        "user_message",
        "reasoning_delta",
        "assistant_start",
        "assistant_delta",
        "status",
    ]
    assert events[3].payload["text"] == "hello"
    assert events[4].payload["context"].startswith("输入上下文：")
    assert events[4].payload["context"].endswith(" / 123.0k")
    assert events[4].payload["contextUsed"] > 0


def test_web_session_can_disable_reasoning(tmp_path):
    provider = RecordingProvider()
    session = make_session(tmp_path, provider=provider)

    events = list(
        session.submit(
            "conversation-1",
            "hi",
            reasoning_enabled=False,
        )
    )

    assert [event.type for event in events] == [
        "user_message",
        "assistant_start",
        "assistant_delta",
        "status",
    ]
    assert provider.configs[0].extra_body == {"enable_thinking": False}


def test_web_session_bootstrap_reuses_previous_usage(tmp_path):
    session = make_session(tmp_path)

    list(session.submit("conversation-1", "hi"))
    events = session.boot_events()

    assert events[0].type == "status"
    assert events[0].payload["context"].startswith("输入上下文：")
    assert events[0].payload["context"].endswith(" / 123.0k")
    assert events[0].payload["contextUsed"] > 0


def test_conversation_payload_uses_per_conversation_context_window(tmp_path):
    session = make_session(tmp_path)

    list(session.submit("conversation-1", "hi"))
    empty = session.create_conversation("conversation-2")
    conversations = {item["id"]: item for item in session.list_conversations()}

    assert conversations["conversation-1"]["context"].startswith("输入上下文：")
    assert conversations["conversation-1"]["context"].endswith(" / 123.0k")
    assert empty["context"].startswith("输入上下文：")
    assert empty["context"].endswith(" / 123.0k")
    assert conversations["conversation-1"]["contextUsed"] > empty["contextUsed"]
    assert conversations["conversation-2"]["contextUsed"] == empty["contextUsed"]


def test_web_session_reuses_slash_commands(tmp_path):
    session = make_session(tmp_path)

    events = list(session.submit("conversation-1", "/model"))

    assert [event.type for event in events] == ["notice", "notice", "notice"]
    assert events[0].payload["text"] == "模型：fake-model"
    assert events[1].payload["text"] == "接口地址：https://example.test/v1"


def test_removed_command_is_unknown(tmp_path):
    session = make_session(tmp_path)

    events = list(session.submit("conversation-1", "/clear"))

    assert [event.type for event in events] == ["notice", "notice"]
    assert events[0].payload["text"] == "未知命令：/clear"


def test_command_completion_rows_include_registered_commands_only(tmp_path):
    session = make_session(tmp_path)

    assert ("/model", "显示当前模型") in session.command_matches("/mo")
    assert session.command_matches(":") == []


def test_web_session_can_cancel_stream(tmp_path):
    session = make_session(tmp_path)
    stream = session.submit("conversation-1", "hi", request_id="request-1")

    assert next(stream).type == "user_message"
    session.cancel("request-1")
    events = list(stream)

    assert [event.type for event in events] == ["notice"]
    assert events[0].payload["text"] == "生成已停止"


def test_fastapi_bootstrap_and_commands(tmp_path):
    session = make_session(tmp_path)
    client = TestClient(create_app(session=session))

    bootstrap = client.get("/api/bootstrap")
    commands = client.get("/api/commands")

    assert bootstrap.status_code == 200
    assert bootstrap.json()["events"][0]["model"] == "fake-model"
    assert commands.status_code == 200
    assert {item["name"] for item in commands.json()["commands"]} == {
        "/help",
        "/model",
        "/status",
    }


def test_fastapi_runtime_logs_endpoint_returns_log_content(tmp_path, monkeypatch):
    log_path = tmp_path / "runtime.jsonl"
    log_path.write_text('{"event":"test","payload":{"ok":true}}\n', encoding="utf-8")
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_path))
    session = make_session(tmp_path)
    client = TestClient(create_app(session=session))

    response = client.get("/api/logs/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == str(log_path)
    assert payload["exists"] is True
    assert '{"event":"test","payload":{"ok":true}}\n' in payload["content"]
    assert payload["error"] == ""


def test_fastapi_cancel_marks_request_cancelled(tmp_path):
    session = make_session(tmp_path)
    client = TestClient(create_app(session=session))

    response = client.post("/api/cancel", json={"request_id": "request-1"})

    assert response.status_code == 200
    stream = session.submit("conversation-1", "hi", request_id="request-1")
    assert next(stream).type == "user_message"
    assert [event.type for event in stream] == ["notice"]


def test_web_session_sends_conversation_context_to_provider(tmp_path):
    provider = RecordingProvider()
    session = make_session(tmp_path, provider=provider)

    list(session.submit("conversation-1", "hi"))
    list(session.submit("conversation-1", "what did I say?"))

    assert provider.inputs[1][0]["role"] == "system"
    assert provider.inputs[1][1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what did I say?"},
    ]


def test_web_session_keeps_conversations_isolated(tmp_path):
    provider = RecordingProvider()
    session = make_session(tmp_path, provider=provider)

    list(session.submit("a", "first"))
    list(session.submit("b", "second"))

    assert provider.inputs[1][0]["role"] == "system"
    assert provider.inputs[1][1:] == [{"role": "user", "content": "second"}]


def test_web_session_does_not_store_slash_commands(tmp_path):
    session = make_session(tmp_path)

    list(session.submit("conversation-1", "/model"))

    assert session.context.build_model_input("conversation-1")[0]["role"] == "system"
    assert session.context.build_model_input("conversation-1")[1:] == []


def test_web_session_cancel_does_not_store_assistant(tmp_path):
    session = make_session(tmp_path)
    stream = session.submit("conversation-1", "hi", request_id="request-1")

    assert next(stream).type == "user_message"
    session.cancel("request-1")
    list(stream)

    assert session.context.build_model_input("conversation-1")[1:] == [
        {"role": "user", "content": "hi"},
    ]


def test_web_session_executes_model_requested_tools(tmp_path):
    provider = ToolCallingProvider()
    session = make_session(
        tmp_path,
        provider=provider,
        tool_registry=weather_registry(
            {"location": "Shanghai", "temperature_c": 25, "description": "Sunny"}
        ),
    )

    events = list(session.submit("conversation-1", "查上海天气"))

    assert [event.type for event in events] == [
        "user_message",
        "reasoning_delta",
        "tool_call_start",
        "tool_call_result",
        "assistant_start",
        "assistant_delta",
        "status",
    ]
    assert "weather" in {tool["function"]["name"] for tool in provider.tools[0]}
    assert events[2].payload["name"] == "weather"
    assert events[3].payload["status"] == "completed"
    assert provider.inputs[1][0]["role"] == "system"
    assert any(
        item["role"] == "assistant"
        and item.get("tool_calls")
        and "Shanghai" in item["tool_calls"][0]["function"]["arguments"]
        for item in provider.inputs[1]
    )
    assert any(
        item["role"] == "tool"
        and item.get("tool_call_id") == "call-1"
        and "Shanghai" in item["content"]
        for item in provider.inputs[1]
    )
    assert session.context.build_model_input("conversation-1")[-1] == {
        "role": "assistant",
        "content": "上海现在晴，25℃。",
    }


def test_web_session_records_tool_errors_in_context(tmp_path):
    provider = ToolCallingProvider()

    def broken_tool(arguments):
        raise RuntimeError("weather failed")

    session = make_session(
        tmp_path,
        provider=provider,
        tool_registry=weather_registry(handler=broken_tool),
    )

    events = list(session.submit("conversation-1", "查上海天气"))

    assert events[3].type == "tool_call_result"
    assert events[3].payload["status"] == "error"
    assert "weather failed" in events[3].payload["summary"]
    assert any("weather failed" in item["content"] for item in provider.inputs[1])


def test_web_session_exposes_agent_core_tools_by_default(tmp_path):
    provider = RecordingProvider()
    session = make_session(tmp_path, provider=provider)

    list(session.submit("conversation-1", "hi"))
    tool_names = {tool["function"]["name"] for tool in provider.tools[0]}

    assert {
        "memory_read",
        "memory_search",
        "memory_append",
        "memory_replace",
        "skill_read",
        "skill_read_resource",
        "shell_command",
    }.issubset(tool_names)


def test_web_session_executes_memory_tool_via_agent_core(tmp_path):
    memory = LongTermMemory()
    provider = RequestedToolProvider(
        ToolCall(
            id="call-memory",
            name="memory_append",
            arguments={"content": "user likes tea"},
        )
    )
    session = WebSession(
        provider_factory=lambda: provider,
        memory_store=MemoryStore(tmp_path / "memory.sqlite3"),
        long_term_memory=memory,
    )

    events = list(session.submit("conversation-1", "remember this"))

    result = next(event for event in events if event.type == "tool_call_result")
    assert result.payload["name"] == "memory_append"
    assert result.payload["status"] == "completed"
    assert memory.read() == "user likes tea"


def test_web_session_executes_skill_tool_via_agent_core(tmp_path):
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo Skill\nUse this.", encoding="utf-8")
    provider = RequestedToolProvider(
        ToolCall(
            id="call-skill",
            name="skill_read",
            arguments={"name": "demo"},
        )
    )
    session = WebSession(
        provider_factory=lambda: provider,
        memory_store=MemoryStore(tmp_path / "memory.sqlite3"),
        skills=[
            SkillManifest(
                name="demo",
                description="Demo skill.",
                skill_dir=skill_dir,
            )
        ],
    )

    events = list(session.submit("conversation-1", "use demo skill"))

    result = next(event for event in events if event.type == "tool_call_result")
    assert result.payload["name"] == "skill_read"
    assert result.payload["result"]["content"] == "# Demo Skill\nUse this."


def test_web_session_executes_shell_tool_via_agent_core(tmp_path):
    provider = RequestedToolProvider(
        ToolCall(
            id="call-shell",
            name="shell_command",
            arguments={"command": "printf web-core", "cwd": str(tmp_path)},
        )
    )
    session = make_session(tmp_path, provider=provider)

    events = list(session.submit("conversation-1", "run command"))

    result = next(event for event in events if event.type == "tool_call_result")
    assert result.payload["name"] == "shell_command"
    assert result.payload["result"]["exit_code"] == 0
    assert result.payload["result"]["stdout"] == "web-core"


def test_web_session_restores_assistant_execution_steps(tmp_path):
    session = make_session(tmp_path)
    session.memory_store.append_message("conversation-1", "user", "weather")
    session.memory_store.append_message(
        "conversation-1",
        "assistant",
        json.dumps(
            {
                "text": "done",
                "reasoning": "first\nsecond",
                "steps": [
                    {
                        "id": "r1",
                        "type": "reasoning",
                        "text": "first",
                        "complete": True,
                    },
                    {
                        "id": "t1",
                        "type": "tool",
                        "name": "weather",
                        "status": "completed",
                    },
                    {
                        "id": "r2",
                        "type": "reasoning",
                        "text": "second",
                        "complete": True,
                    },
                ],
            },
            ensure_ascii=False,
        ),
    )

    conversation = session.list_conversations()[0]

    assert conversation["messages"][1]["text"] == "done"
    assert [step["type"] for step in conversation["messages"][1]["steps"]] == [
        "reasoning",
        "tool",
        "reasoning",
    ]


def test_web_session_groups_one_user_turn_into_one_assistant_message(tmp_path):
    session = make_session(tmp_path)
    session.memory_store.append_message("conversation-1", "user", "skills?")
    session.memory_store.append_message(
        "conversation-1",
        "assistant",
        json.dumps(
            {
                "text": "I can check.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "skill_read",
                        "arguments": {"name": "skill-creator"},
                    }
                ],
                "steps": [
                    {
                        "id": "r1",
                        "type": "reasoning",
                        "text": "checking",
                        "complete": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )
    session.memory_store.append_message(
        "conversation-1",
        "tool",
        json.dumps(
            {
                "call_id": "call-1",
                "name": "skill_read",
                "arguments": {"name": "skill-creator"},
                "result": {"error": "missing"},
            },
            ensure_ascii=False,
        ),
    )
    session.memory_store.append_message(
        "conversation-1",
        "assistant",
        json.dumps(
            {
                "text": "The skill is not registered.",
                "steps": [
                    {
                        "id": "r2",
                        "type": "reasoning",
                        "text": "done",
                        "complete": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    conversation = session.list_conversations()[0]

    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assistant = conversation["messages"][1]
    assert assistant["text"] == "I can check.\n\nThe skill is not registered."
    assert assistant["tools"][0]["name"] == "skill_read"
    assert assistant["steps"] == [
        {
            "id": "r2",
            "type": "reasoning",
            "text": "done",
            "complete": True,
        }
    ]


def test_fastapi_conversation_crud_and_chat(tmp_path):
    session = make_session(tmp_path)
    client = TestClient(create_app(session=session))

    created = client.post("/api/conversations", json={"id": "conversation-1"}).json()
    chat = client.post(
        "/api/chat",
        json={
            "conversation_id": "conversation-1",
            "message": "hi",
            "request_id": "request-1",
        },
    )
    listed = client.get("/api/conversations").json()
    deleted = client.delete("/api/conversations/conversation-1")

    assert created["id"] == "conversation-1"
    assert chat.status_code == 200
    assert listed["items"][0]["messages"][0]["text"] == "hi"
    assert deleted.status_code == 200
    assert client.get("/api/conversations").json()["items"] == []
