"""Agent runtime loop tests."""

from __future__ import annotations

from agent_runtime.context import ContextEngine
from agent_runtime.memory import MemoryStore
from agent_runtime.providers import ModelResponse, ModelStreamEvent, ToolCall
from agent_runtime.runtime import AgentLoop
from agent_runtime.tools import ToolRegistry, ToolSpec


class ToolProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0
        self.inputs = []

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        if self.calls == 0:
            self.calls += 1
            yield ModelStreamEvent(type="reasoning_delta", delta="need tool")
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
                    usage={"prompt_tokens": 4},
                ),
            )
            return

        self.calls += 1
        yield ModelStreamEvent(type="content_delta", delta="sunny")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 6, "completion_tokens": 2},
            ),
        )


class RepeatingToolProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def stream(self, input, **kwargs):
        self.calls += 1
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call-{self.calls}",
                        name="weather",
                        arguments={"location": "Shanghai"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": self.calls},
            ),
        )


def test_agent_loop_runs_tools_and_writes_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "weather")
    provider = ToolProvider()
    registry = ToolRegistry(
        [
            ToolSpec(
                name="weather",
                description="查天气",
                input_schema={"type": "object"},
                handler=lambda arguments: {"temperature_c": 25},
            )
        ]
    )
    loop = AgentLoop(provider=provider, context=context, tool_registry=registry)

    events = list(loop.run("conversation-1"))

    assert [event.type for event in events] == [
        "reasoning_delta",
        "tool_call_start",
        "tool_call_result",
        "assistant_start",
        "assistant_delta",
        "usage",
    ]
    assert events[-1].payload["usage"] == {"prompt_tokens": 6, "completion_tokens": 2}
    assert any(item["role"] == "tool" for item in provider.inputs[1])
    assert any(
        item["role"] == "assistant" and item.get("tool_calls")
        for item in provider.inputs[1]
    )
    assert context.build_model_input("conversation-1")[-1] == {
        "role": "assistant",
        "content": "sunny",
    }


def test_agent_loop_default_context_log_does_not_print_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(tmp_path / "runtime.jsonl"))
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "weather")
    provider = ToolProvider()
    registry = ToolRegistry(
        [
            ToolSpec(
                name="weather",
                description="查天气",
                input_schema={"type": "object"},
                handler=lambda arguments: {"temperature_c": 25},
            )
        ]
    )
    loop = AgentLoop(provider=provider, context=context, tool_registry=registry)

    list(loop.run("conversation-1"))

    assert "Agent Runtime Model Context" not in capsys.readouterr().out


def test_agent_loop_run_user_turn_writes_user_message(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    provider = ToolProvider()
    registry = ToolRegistry(
        [
            ToolSpec(
                name="weather",
                description="查天气",
                input_schema={"type": "object"},
                handler=lambda arguments: {"temperature_c": 25},
            )
        ]
    )
    loop = AgentLoop(provider=provider, context=context, tool_registry=registry)

    list(loop.run_user_turn("conversation-1", "weather"))

    assert context.build_model_input("conversation-1")[1] == {
        "role": "user",
        "content": "weather",
    }


def test_agent_loop_stops_on_third_identical_tool_call(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)
    context.add_user_message("conversation-1", "weather")
    provider = RepeatingToolProvider()
    registry = ToolRegistry(
        [
            ToolSpec(
                name="weather",
                description="查天气",
                input_schema={"type": "object"},
                handler=lambda arguments: {"temperature_c": 25},
            )
        ]
    )
    loop = AgentLoop(provider=provider, context=context, tool_registry=registry)

    events = list(loop.run("conversation-1"))

    assert [event.type for event in events] == [
        "tool_call_start",
        "tool_call_result",
        "tool_call_start",
        "tool_call_result",
        "notice",
    ]
    assert events[-1].payload == {
        "tone": "error",
        "text": "检测到重复工具调用，已停止继续调用。",
    }
    assert provider.calls == 3
    assert store.message_count("conversation-1") == 5
