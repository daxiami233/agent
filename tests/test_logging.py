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


def test_format_runtime_log_content_renders_readable_trace():
    raw = (
        '{"ts":"2026-06-26T16:30:52+0800","event":"agent_round_start",'
        '"payload":{"conversation_id":"conversation-1","round_index":2}}\n'
    )

    content = format_runtime_log_content(raw)

    assert "16:30:52 [conversa] 第 2 轮：准备上下文" in content
