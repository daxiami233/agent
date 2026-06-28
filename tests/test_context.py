"""Context and memory store tests."""

from __future__ import annotations

import json

import pytest

from agent_runtime.context import (
    CompressionResult,
    ContextCompressor,
    ContextEngine,
    ContextMessage,
    ContextOverflowError,
    TokenCounter,
)
from agent_runtime.memory import LongTermMemory, MemoryStore


class SimpleTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)

    def count_message(self, message) -> int:
        return len(message.content)


class RecordingCompressor(ContextCompressor):
    def __init__(self, result: CompressionResult) -> None:
        self.result = result
        self.calls = []

    def compress(self, *, conversation_id, messages, target_tokens):
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "messages": list(messages),
                "target_tokens": target_tokens,
            }
        )
        return self.result


def without_system(messages):
    return messages[1:]


def test_memory_store_persists_conversations_across_instances(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = MemoryStore(db_path)
    store.create_conversation("conversation-1", "Hello")
    store.append_message("conversation-1", "user", "hi")

    restored = MemoryStore(db_path)

    assert restored.list_conversations()[0].id == "conversation-1"
    assert restored.list_messages("conversation-1")[0].content == "hi"


def test_memory_store_keeps_conversations_isolated(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.append_message("a", "user", "message a")
    store.append_message("b", "user", "message b")

    assert [message.content for message in store.list_messages("a")] == ["message a"]
    assert [message.content for message in store.list_messages("b")] == ["message b"]


def test_context_engine_builds_ordered_model_input(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "hi")
    context.add_assistant_message("conversation-1", "hello")
    context.add_user_message("conversation-1", "what did I say?")

    model_input = context.build_model_input("conversation-1")

    assert model_input[0]["role"] == "system"
    assert "Agent Runtime" in model_input[0]["content"]
    assert "# Skills" in model_input[0]["content"]
    assert "# Retrieved Memory" in model_input[0]["content"]
    assert without_system(model_input) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what did I say?"},
    ]


def test_context_engine_returns_all_messages_without_compression(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        context_window_tokens=100,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
    )

    context.add_user_message("conversation-1", "one")
    context.add_assistant_message("conversation-1", "two")
    context.add_user_message("conversation-1", "three")

    assert without_system(context.build_model_input("conversation-1")) == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_context_engine_estimates_model_input_tokens(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        token_counter=SimpleTokenCounter(),
    )

    context.add_user_message("conversation-1", "hello")

    assert context.estimate_model_input_tokens("conversation-1") == len("syshello")


def test_context_engine_uses_one_consolidated_system_message(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store, system_prompt="compact system")

    context.add_user_message("conversation-1", "hi")

    assert context.build_model_input("conversation-1") == [
        {"role": "system", "content": "compact system"},
        {"role": "user", "content": "hi"},
    ]


def test_context_engine_fills_system_prompt_placeholders(tmp_path):
    memory = LongTermMemory()
    memory.write("remember this\nand this")
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="Skills:\n{skills}\n\nMemory:\n{retrieved_memory}",
        long_term_memory=memory,
    )

    system = context.build_model_input("conversation-1")[0]

    assert system["content"] == "Skills:\n\nMemory:\nremember this\nand this"


def test_context_engine_uses_empty_memory_text_when_memory_is_empty(tmp_path):
    memory = LongTermMemory()
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="Memory: {retrieved_memory}",
        long_term_memory=memory,
    )

    assert context.build_model_input("conversation-1")[0] == {
        "role": "system",
        "content": "Memory: No memories stored yet.",
    }


def test_context_engine_freezes_retrieved_memory_per_conversation(tmp_path):
    memory = LongTermMemory()
    memory.write("initial memory")
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="Memory: {retrieved_memory}",
        long_term_memory=memory,
    )

    assert context.build_model_input("conversation-1")[0]["content"] == "Memory: initial memory"
    memory.write("updated memory")

    assert context.build_model_input("conversation-1")[0]["content"] == "Memory: initial memory"
    assert context.build_model_input("conversation-2")[0]["content"] == "Memory: updated memory"


def test_context_engine_includes_tool_results(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "上海天气")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[{"id": "call_1", "name": "weather", "arguments": {"location": "Shanghai"}}],
    )
    context.add_tool_result(
        "conversation-1",
        "weather",
        {"location": "Shanghai"},
        {"temperature_c": 25, "description": "Sunny"},
        call_id="call_1",
    )

    model_input = context.build_model_input("conversation-1")

    assert without_system(model_input)[0] == {"role": "user", "content": "上海天气"}
    assert without_system(model_input)[1]["role"] == "assistant"
    assert without_system(model_input)[1]["tool_calls"][0]["id"] == "call_1"
    assert without_system(model_input)[1]["tool_calls"][0]["function"]["name"] == "weather"
    assert "Shanghai" in without_system(model_input)[1]["tool_calls"][0]["function"]["arguments"]
    assert without_system(model_input)[2]["role"] == "tool"
    assert without_system(model_input)[2]["tool_call_id"] == "call_1"
    assert "Sunny" in without_system(model_input)[2]["content"]


def test_context_engine_accepts_native_tool_call_shape(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    store.append_message("conversation-1", "user", "weather")
    store.append_message(
        "conversation-1",
        "assistant",
        context._json_dumps(
            {
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"location": "Shanghai"},
                        },
                    }
                ],
            }
        ),
    )

    tool_call = without_system(context.build_model_input("conversation-1"))[1]["tool_calls"][0]

    assert tool_call == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "weather",
            "arguments": '{"location": "Shanghai"}',
        },
    }


def test_context_engine_converts_orphan_legacy_tool_to_system_summary(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    store.append_message("conversation-1", "tool", "not json")

    messages = without_system(context.build_model_input("conversation-1"))

    assert messages[0]["role"] == "system"
    assert "孤立工具结果" in messages[0]["content"]
    assert "not json" in messages[0]["content"]


def test_context_engine_assistant_text_with_content_and_tool_calls(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "查天气")
    context.add_assistant_message(
        "conversation-1",
        "我来帮你查一下",
        tool_calls=[{"id": "call_1", "name": "weather", "arguments": {"city": "北京"}}],
    )
    context.add_tool_result(
        "conversation-1",
        "weather",
        {"city": "北京"},
        {"temp": 20, "condition": "晴"},
        call_id="call_1",
    )

    model_input = context.build_model_input("conversation-1")
    assistant_msg = without_system(model_input)[1]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "我来帮你查一下"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"


def test_context_engine_assistant_pure_text_no_json(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "你好")
    context.add_assistant_message("conversation-1", "你好！有什么可以帮你的？")

    model_input = context.build_model_input("conversation-1")

    assert without_system(model_input) == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    ]


def test_context_engine_assistant_with_reasoning(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "1+1=?")
    context.add_assistant_message(
        "conversation-1",
        "等于2",
        reasoning="用户问的是简单加法",
    )

    model_input = context.build_model_input("conversation-1")
    assistant_msg = without_system(model_input)[1]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "等于2"


def test_context_engine_multiple_tool_calls(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "北京和上海天气")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[
            {"id": "call_1", "name": "weather", "arguments": {"city": "北京"}},
            {"id": "call_2", "name": "weather", "arguments": {"city": "上海"}},
        ],
    )
    context.add_tool_result("conversation-1", "weather", {"city": "北京"}, {"temp": 20}, call_id="call_1")
    context.add_tool_result("conversation-1", "weather", {"city": "上海"}, {"temp": 25}, call_id="call_2")

    model_input = context.build_model_input("conversation-1")
    assistant_msg = without_system(model_input)[1]

    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 2
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][1]["id"] == "call_2"

    tool_msg_1 = without_system(model_input)[2]
    tool_msg_2 = without_system(model_input)[3]
    assert tool_msg_1["tool_call_id"] == "call_1"
    assert tool_msg_2["tool_call_id"] == "call_2"


def test_context_engine_tool_result_only_without_assistant(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    context.add_user_message("conversation-1", "天气")
    context.add_tool_result(
        "conversation-1",
        "weather",
        {"city": "北京"},
        {"temp": 20},
        call_id="call_1",
    )

    model_input = context.build_model_input("conversation-1")
    messages = without_system(model_input)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "system"
    assert "孤立工具结果" in messages[1]["content"]
    assert "20" in messages[1]["content"]


def test_context_engine_preserves_all_messages(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        context_window_tokens=80,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
    )

    context.add_user_message("conversation-1", "a" * 10)
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[{"id": "call_1", "name": "tool", "arguments": {}}],
    )
    context.add_tool_result("conversation-1", "tool", {}, {"result": "ok"}, call_id="call_1")
    context.add_user_message("conversation-1", "b" * 10)

    model_input = context.build_model_input("conversation-1")
    messages = without_system(model_input)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "a" * 10
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "tool"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "b" * 10


def test_context_engine_repairs_assistant_tool_calls_without_results(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store, system_prompt="sys")
    store.append_message(
        "conversation-1",
        "assistant",
        json.dumps(
            {
                "text": "I looked.",
                "tool_calls": [
                    {"id": "call_missing", "name": "tool", "arguments": {}}
                ],
            }
        ),
    )
    store.append_message("conversation-1", "assistant", "final answer")

    messages = without_system(context.build_model_input("conversation-1"))

    assert messages == [
        {"role": "assistant", "content": "I looked."},
        {"role": "assistant", "content": "final answer"},
    ]
    assert [
        message.role
        for message in store.list_messages("conversation-1")
    ] == ["assistant", "assistant"]


def test_context_engine_does_not_repair_incomplete_tool_group_at_tail(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store, system_prompt="sys")
    context.add_user_message("conversation-1", "inspect")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[
            {"id": "call_1", "name": "tool", "arguments": {"i": 1}},
            {"id": "call_2", "name": "tool", "arguments": {"i": 2}},
        ],
    )
    context.add_tool_result(
        "conversation-1",
        "tool",
        {"i": 1},
        {"result": "one"},
        call_id="call_1",
    )

    partial = without_system(context.build_model_input("conversation-1"))

    assert [message.role for message in store.list_messages("conversation-1")] == [
        "user",
        "assistant",
        "tool",
    ]
    assert partial[1]["role"] == "assistant"
    assert len(partial[1]["tool_calls"]) == 2
    assert partial[2]["role"] == "tool"
    assert partial[2]["tool_call_id"] == "call_1"

    context.add_tool_result(
        "conversation-1",
        "tool",
        {"i": 2},
        {"result": "two"},
        call_id="call_2",
    )
    complete = without_system(context.build_model_input("conversation-1"))

    assert [message["role"] for message in complete] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert complete[3]["tool_call_id"] == "call_2"


def test_context_budget_does_not_repair_incomplete_tool_group(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store, system_prompt="sys")
    context.add_user_message("conversation-1", "inspect")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[
            {"id": "call_1", "name": "tool", "arguments": {"i": 1}},
            {"id": "call_2", "name": "tool", "arguments": {"i": 2}},
        ],
    )
    context.add_tool_result(
        "conversation-1",
        "tool",
        {"i": 1},
        {"result": "one"},
        call_id="call_1",
    )

    budget = context.context_budget("conversation-1")

    assert budget.used_input_tokens > 0
    assert [
        message.role
        for message in store.list_messages("conversation-1")
    ] == ["user", "assistant", "tool"]


def test_context_engine_rejects_oversized_latest_tool_group_after_truncation(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        context_window_tokens=20,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
    )
    context.add_user_message("conversation-1", "x" * 80)
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[{"id": "call_1", "name": "tool", "arguments": {}}],
    )
    context.add_tool_result(
        "conversation-1",
        "tool",
        {},
        {"result": "y" * 80},
        call_id="call_1",
    )

    with pytest.raises(ContextOverflowError, match="超过模型输入上下文预算"):
        context.build_model_input("conversation-1")


def test_context_engine_rejects_oversized_latest_user_message(tmp_path):
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys",
        context_window_tokens=20,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
    )

    context.add_user_message("conversation-1", "x" * 100)

    with pytest.raises(ContextOverflowError, match="超过模型输入上下文预算"):
        context.build_model_input("conversation-1")


def test_context_engine_calls_compressor_when_over_budget(tmp_path):
    compressor = RecordingCompressor(
        CompressionResult(summary="older turns summarized", compressed=True)
    )
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys",
        context_window_tokens=500,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        compact_threshold_ratio=0.1,
        recent_turns=1,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "old message " + ("x" * 80))
    context.add_assistant_message("conversation-1", "assistant message " + ("y" * 80))
    context.add_user_message("conversation-1", "new message")

    model_input = context.build_model_input("conversation-1")
    assert model_input[0] == {"role": "system", "content": "sys"}
    assert model_input[1]["role"] == "system"
    assert "较早对话摘要" in model_input[1]["content"]
    assert "older turns summarized" in model_input[1]["content"]
    assert model_input[2] == {"role": "user", "content": "new message"}
    assert compressor.calls[0]["conversation_id"] == "conversation-1"
    assert [message.role for message in compressor.calls[0]["messages"]] == [
        "user",
        "assistant",
    ]
    assert compressor.calls[0]["messages"][0].content.startswith("old message")
    assert compressor.calls[0]["messages"][1].content.startswith("assistant message")


def test_context_engine_logs_compaction_events(tmp_path, monkeypatch):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))
    compressor = RecordingCompressor(
        CompressionResult(summary="older turns summarized", compressed=True)
    )
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys",
        context_window_tokens=50,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        recent_turns=1,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "x" * 20)
    context.add_assistant_message("conversation-1", "y" * 20)
    context.add_user_message("conversation-1", "new")
    context.build_model_input("conversation-1")

    events = [
        json.loads(line)["event"]
        for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert "context_compact_check" in events
    assert "context_compact_split" in events
    assert "context_turns_compress_start" in events
    assert "context_turns_compress_done" in events
    assert "context_compact_persist" in events
    assert "context_budget_final" in events


def test_context_engine_ratio_compresses_recent_raw_into_single_summary(tmp_path):
    compressor = RecordingCompressor(
        CompressionResult(summary="recent raw summarized", compressed=True)
    )
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys",
        context_window_tokens=500,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        compact_threshold_ratio=0.1,
        recent_turns=6,
        raw_keep_ratio=0.3,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "inspect project")
    for index in range(3):
        call_id = f"call_{index}"
        context.add_assistant_message(
            "conversation-1",
            "",
            tool_calls=[{"id": call_id, "name": "shell", "arguments": {"i": index}}],
        )
        context.add_tool_result(
            "conversation-1",
            "shell",
            {"i": index},
            {"result": f"tool output {index} " + ("x" * 250)},
            call_id=call_id,
        )

    model_input = without_system(context.build_model_input("conversation-1"))

    assert len(compressor.calls) == 1
    assert [message.role for message in compressor.calls[0]["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [message["role"] for message in model_input] == ["system", "assistant", "tool"]
    assert "较早对话摘要" in model_input[0]["content"]
    assert "recent raw summarized" in model_input[0]["content"]
    assert model_input[1]["tool_calls"][0]["id"] == "call_2"
    assert model_input[2]["tool_call_id"] == "call_2"


def test_context_engine_does_not_call_compressor_when_within_budget(tmp_path):
    compressor = RecordingCompressor(CompressionResult())
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys",
        context_window_tokens=100,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "short")

    assert without_system(context.build_model_input("conversation-1")) == [
        {"role": "user", "content": "short"}
    ]
    assert compressor.calls == []


def test_context_engine_writes_compressed_summary_back_to_store(tmp_path):
    native_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "weather",
            "arguments": '{"location": "Shanghai"}',
        },
    }
    compressor = RecordingCompressor(
        CompressionResult(
            summary="older turns summarized",
            compressed=True,
        )
    )
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(
        store,
        system_prompt="sys",
        context_window_tokens=500,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        compact_threshold_ratio=0.1,
        recent_turns=1,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "old message " + ("x" * 80))
    context.add_assistant_message("conversation-1", "old answer " + ("y" * 80))
    context.add_user_message("conversation-1", "new weather")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[{"id": "call_1", "name": "weather", "arguments": {"location": "Shanghai"}}],
    )
    context.add_tool_result("conversation-1", "weather", {}, {"temperature_c": 25}, call_id="call_1")
    model_input = context.build_model_input("conversation-1")
    stored_messages = store.list_messages("conversation-1")

    assert model_input[0] == {"role": "system", "content": "sys"}
    assert model_input[1]["role"] == "system"
    assert "older turns summarized" in model_input[1]["content"]
    assert without_system(context.build_model_input("conversation-1"))[2]["tool_calls"] == [
        native_tool_call
    ]
    assert without_system(context.build_model_input("conversation-1"))[3]["tool_call_id"] == "call_1"
    assert [message.role for message in stored_messages] == ["system", "user", "assistant", "tool"]
    assert "older turns summarized" in stored_messages[0].content


def test_context_engine_clear_removes_conversation_messages(tmp_path):
    context = ContextEngine(MemoryStore(tmp_path / "memory.sqlite3"))

    context.add_user_message("conversation-1", "hello")
    context.clear("conversation-1")

    assert without_system(context.build_model_input("conversation-1")) == []


def test_token_counter_falls_back_to_byte_estimation_for_unknown_encoding():
    counter = TokenCounter(model="unknown-model", encoding_name="unknown-encoding")

    assert counter.count_text("abcd") == 1
    assert counter.count_text("abcde") == 2
    assert counter.count_text("") == 0
