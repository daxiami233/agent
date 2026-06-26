"""Context and memory store tests."""

from __future__ import annotations

import json

from agent_runtime.context import (
    CompressionResult,
    ContextCompressor,
    ContextEngine,
    ContextMessage,
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

    def compress(self, *, conversation_id, messages, target_tokens, previous_summary=""):
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "messages": list(messages),
                "target_tokens": target_tokens,
                "previous_summary": previous_summary,
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
    assert "# Conversation Summary" in model_input[0]["content"]
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
        system_prompt=(
            "Skills:\n{skills}\n\nMemory:\n{retrieved_memory}\n\n"
            "Summary:\n{conversation_summary}"
        ),
        long_term_memory=memory,
    )

    system = context.build_model_input("conversation-1")[0]

    assert system["content"] == (
        "Skills:\n\nMemory:\nremember this\nand this\n\nSummary:\nNo memories stored yet."
    )


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


def test_context_engine_keeps_legacy_tool_content_with_legacy_call_id(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    context = ContextEngine(store)

    store.append_message("conversation-1", "tool", "not json")

    assert without_system(context.build_model_input("conversation-1")) == [
        {"role": "tool", "content": "not json", "tool_call_id": "legacy"}
    ]


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
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_1"


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


def test_context_engine_force_truncates_tool_call_groups_atomically(tmp_path):
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

    messages = without_system(context.build_model_input("conversation-1"))

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_1"


def test_context_engine_calls_compressor_when_over_budget(tmp_path):
    compressor = RecordingCompressor(
        CompressionResult(summary="older turns summarized", compressed=True)
    )
    context = ContextEngine(
        MemoryStore(tmp_path / "memory.sqlite3"),
        system_prompt="sys {conversation_summary}",
        context_window_tokens=50,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        recent_turns=1,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "old message")
    context.add_assistant_message("conversation-1", "assistant message")
    context.add_user_message("conversation-1", "new message")

    assert context.build_model_input("conversation-1") == [
        {"role": "system", "content": "sys older turns summarized"},
        {"role": "user", "content": "new message"},
    ]
    assert compressor.calls[0]["conversation_id"] == "conversation-1"
    assert [message.content for message in compressor.calls[0]["messages"]] == [
        "old message",
        "assistant message",
    ]


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
        system_prompt="sys {conversation_summary}",
        context_window_tokens=80,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        recent_turns=1,
        token_counter=SimpleTokenCounter(),
        compressor=compressor,
    )

    context.add_user_message("conversation-1", "old message")
    context.add_assistant_message("conversation-1", "old answer")
    context.add_user_message("conversation-1", "new weather")
    context.add_assistant_message(
        "conversation-1",
        "",
        tool_calls=[{"id": "call_1", "name": "weather", "arguments": {"location": "Shanghai"}}],
    )
    context.add_tool_result("conversation-1", "weather", {}, {"temperature_c": 25}, call_id="call_1")
    model_input = context.build_model_input("conversation-1")
    stored_messages = store.list_messages("conversation-1")

    assert model_input[0] == {"role": "system", "content": "sys older turns summarized"}
    assert without_system(context.build_model_input("conversation-1"))[1]["tool_calls"] == [
        native_tool_call
    ]
    assert without_system(context.build_model_input("conversation-1"))[2]["tool_call_id"] == "call_1"
    assert [message.role for message in stored_messages] == ["user", "assistant", "tool"]
    assert store.get_conversation("conversation-1").summary == "older turns summarized"


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
