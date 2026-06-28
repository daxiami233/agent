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
        self.inputs = []
        self.tools = []

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.tools.append(kwargs.get("tools") or [])
        self.calls += 1
        if not self.tools[-1]:
            yield ModelStreamEvent(type="content_delta", delta="based on existing results")
            yield ModelStreamEvent(
                type="finish",
                response=ModelResponse(
                    content=None,
                    finish_reason="stop",
                    usage={"prompt_tokens": self.calls, "completion_tokens": 4},
                ),
            )
            return
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


class CountingProvider(ToolProvider):
    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.calls += 1
        raise AssertionError("provider should not be called when context overflows")


class SimpleTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)

    def count_message(self, message) -> int:
        return len(message.content)


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


def test_agent_loop_stops_before_provider_when_context_overflows(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        context_window_tokens=20,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
    )
    context.add_user_message("conversation-1", "x" * 100)
    provider = CountingProvider()
    loop = AgentLoop(
        provider=provider,
        context=context,
        tool_registry=ToolRegistry(),
    )

    events = list(loop.run("conversation-1"))

    assert provider.calls == 0
    assert provider.inputs == []
    assert events[-1].type == "notice"
    assert events[-1].payload["tone"] == "error"
    assert "超过模型输入上下文预算" in events[-1].payload["text"]


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


def test_agent_loop_skips_repeated_tool_call_and_forces_final_answer(tmp_path):
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

    event_types = [event.type for event in events]
    assert event_types.count("tool_call_start") == 2
    assert event_types.count("tool_call_result") == 2
    assert event_types[-3:] == ["assistant_start", "assistant_delta", "usage"]
    assert events[-2].payload == {"text": "based on existing results"}
    assert provider.calls >= 6
    assert any(
        any(
            message.get("role") == "system"
            and "Duplicate call skipped for weather" in str(message.get("content", ""))
            for message in model_input
        )
        for model_input in provider.inputs
    )
    assert provider.tools[-1] == []
    assert all(
        "weather" in {tool["function"]["name"] for tool in tools}
        for tools in provider.tools[:-1]
    )


def test_agent_loop_repeated_tool_call_requires_exact_arguments(tmp_path):
    loop = AgentLoop(
        provider=RepeatingToolProvider(),
        context=ContextEngine(MemoryStore(tmp_path / "memory.sqlite3")),
        tool_registry=ToolRegistry(),
    )
    last_key = None
    streak = 0

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [ToolCall(id="call-1", name="weather", arguments={"location": "Shanghai"})],
        last_key,
        streak,
    )
    assert [call.id for call in executable] == ["call-1"]
    assert warnings == []
    assert skipped == []

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [ToolCall(id="call-2", name="weather", arguments={"location": "Beijing"})],
        last_key,
        streak,
    )
    assert [call.id for call in executable] == ["call-2"]
    assert warnings == []
    assert skipped == []

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [ToolCall(id="call-3", name="weather", arguments={"location": "Shanghai"})],
        last_key,
        streak,
    )
    assert [call.id for call in executable] == ["call-3"]
    assert warnings == []
    assert skipped == []

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [ToolCall(id="call-4", name="weather", arguments={"location": "Shanghai"})],
        last_key,
        streak,
    )
    assert [call.id for call in executable] == ["call-4"]
    assert [call.id for call in warnings] == ["call-4"]
    assert skipped == []

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [ToolCall(id="call-5", name="weather", arguments={"location": "Shanghai"})],
        last_key,
        streak,
    )
    assert executable == []
    assert warnings == []
    assert [call.id for call in skipped] == ["call-5"]


def test_agent_loop_repeated_tool_call_detects_consecutive_calls_in_same_batch(tmp_path):
    loop = AgentLoop(
        provider=RepeatingToolProvider(),
        context=ContextEngine(MemoryStore(tmp_path / "memory.sqlite3")),
        tool_registry=ToolRegistry(),
    )

    executable, warnings, skipped, last_key, streak = loop._split_repeated_tool_calls(
        [
            ToolCall(id="call-1", name="memory_read", arguments={}),
            ToolCall(id="call-2", name="memory_read", arguments={}),
            ToolCall(id="call-3", name="memory_read", arguments={}),
        ],
        None,
        0,
    )
    assert [call.id for call in executable] == ["call-1", "call-2"]
    assert [call.id for call in warnings] == ["call-2"]
    assert [call.id for call in skipped] == ["call-3"]
    assert last_key == ("memory_read", "{}")
    assert streak == 3
