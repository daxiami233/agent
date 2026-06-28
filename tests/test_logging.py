"""Runtime logging tests."""

from __future__ import annotations

import json

from agent_runtime.logging import format_runtime_log_content, runtime_log, runtime_log_path


def test_runtime_log_writes_jsonl(monkeypatch, tmp_path):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))

    runtime_log("test_event", {"value": "ok"})

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["event"] == "test_event"
    assert record["message"]
    assert record["payload"] == {"value": "ok"}
    assert runtime_log_path() == log_file


def test_runtime_log_redacts_sensitive_values(monkeypatch, tmp_path):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))

    runtime_log(
        "test_event",
        {
            "api_key": "sk-1234567890abcdef",
            "text": "API_KEY=sk-abcdef1234567890 token=secret-value",
        },
    )

    raw = log_file.read_text(encoding="utf-8")
    record = json.loads(raw.strip())
    assert "sk-1234567890abcdef" not in raw
    assert "secret-value" not in raw
    assert record["payload"]["api_key"] == "[redacted]"
    assert record["payload"]["text"] == "API_KEY=[redacted] token=[redacted]"


def test_runtime_log_keeps_usage_token_counts(monkeypatch, tmp_path):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))

    runtime_log(
        "web_stream_usage",
        {"usage": {"prompt_tokens": 12, "completion_tokens": 3}},
    )

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["payload"]["usage"]["prompt_tokens"] == 12
    assert record["payload"]["usage"]["completion_tokens"] == 3


def test_runtime_log_keeps_context_budget_token_counts(monkeypatch, tmp_path):
    log_file = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("AGENT_RUNTIME_LOG_FILE", str(log_file))

    runtime_log(
        "context_budget_final",
        {
            "used_tokens": 20,
            "input_budget_tokens": 100,
            "compact_threshold_tokens": 80,
            "target_tokens": 50,
        },
    )

    raw = log_file.read_text(encoding="utf-8")
    record = json.loads(raw.strip())
    assert "[redacted]" not in raw
    assert record["payload"]["used_tokens"] == 20
    assert record["payload"]["compact_threshold_tokens"] == 80


def test_format_runtime_log_content_renders_readable_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"agent_round_start",'
        '"payload":{"conversation_id":"conversation-1","round_index":2}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "16:30:52 [conversa] 第 2 轮：准备上下文" in content


def test_format_runtime_log_content_renders_web_stream_start():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"web_stream_start",'
        '"payload":{"conversation_id":"conversation-1",'
        '"request_id":"request-1234567890"}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "16:30:52 [conversa] Web 流开始：请求 request-" in content


def test_format_runtime_log_content_renders_context_compaction_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"context_compact_check",'
        '"payload":{"conversation_id":"conversation-1","request_tokens":120,'
        '"compact_threshold_tokens":80,"input_budget_tokens":100,'
        '"over_threshold":true}}\n'
        '{"ts":"2026-06-26T16:30:53+0800","event":"context_force_truncate_done",'
        '"payload":{"conversation_id":"conversation-1","before_messages":8,'
        '"after_messages":4,"before_tokens":120,"after_tokens":60}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "上下文预算检查：120 / 80 tokens (超过阈值)" in content
    assert "强制截断完成：8 -> 4 条，120 -> 60 tokens" in content


def test_format_runtime_log_content_renders_turns_compaction_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"context_turns_compress_start",'
        '"payload":{"conversation_id":"conversation-1","summary_messages":6,'
        '"target_tokens":800}}\n'
        '{"ts":"2026-06-26T16:30:53+0800","event":"context_turns_compress_done",'
        '"payload":{"conversation_id":"conversation-1","compressed":true,'
        '"summary_chars":700}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "开始压缩旧对话：6 条消息" in content
    assert "旧对话压缩完成：摘要 700 字" in content


def test_format_runtime_log_content_renders_ratio_compaction_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"context_ratio_compress_start",'
        '"payload":{"conversation_id":"conversation-1","summary_messages":4,'
        '"target_tokens":800}}\n'
        '{"ts":"2026-06-26T16:30:53+0800","event":"context_ratio_compress_done",'
        '"payload":{"conversation_id":"conversation-1","compressed":true,'
        '"summary_chars":500}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "开始压缩近期原文：4 条消息" in content
    assert "近期原文压缩完成：摘要 500 字" in content


def test_format_runtime_log_content_renders_shell_artifact_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"shell_command_complete",'
        '"payload":{"exit_code":0,"stdout_chars":211127,"stderr_chars":0,'
        '"stdout_ref":"/tmp/artifacts/stdout.txt"}}\n'
        '{"ts":"2026-06-26T16:30:53+0800","event":"shell_output_artifact",'
        '"payload":{"stream":"stdout","chars":211127,'
        '"artifact_ref":"/tmp/artifacts/stdout.txt"}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "stdout 211127 字（已截断，artifact=/tmp/artifacts/stdout.txt）" in content
    assert "Shell 输出落盘：stdout 211127 字 -> /tmp/artifacts/stdout.txt" in content


def test_format_runtime_log_content_renders_tool_error_detail():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"tool_call_result",'
        '"payload":{"conversation_id":"conversation-1","name":"shell_command",'
        '"status":"error","error":"Command blocked"}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "工具结束：shell_command (error: Command blocked)" in content
