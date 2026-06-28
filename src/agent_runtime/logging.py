"""Runtime JSONL trace logging."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


STATE_DIR = Path.home() / ".agent-runtime"
DEFAULT_LOG_FILE = STATE_DIR / "logs" / "runtime.jsonl"
MAX_STRING_LENGTH = 2_000
MAX_LOG_BYTES = 10 * 1024 * 1024
REDACTED = "[redacted]"
NOISY_EVENTS = {
    "web_stream_event",
    "model_tool_call_delta",
}
TOKEN_COUNT_KEYS = {
    "after_tokens",
    "before_tokens",
    "budget_tokens",
    "completion_tokens",
    "compact_threshold_tokens",
    "context_window_tokens",
    "extra_input_tokens",
    "input_budget_tokens",
    "input_tokens",
    "message_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "recent_tokens",
    "older_tokens",
    "request_tokens",
    "reserved_output_tokens",
    "safety_margin_tokens",
    "summary_tokens",
    "system_tokens",
    "target_tokens",
    "total_tokens",
    "used_tokens",
}

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)

SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|authorization|credential)"
        r"\s*=\s*([^\s\"']+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{12,}\b"),
)

_LOCK = threading.Lock()


def runtime_log_path() -> Path:
    """Return the active runtime log path."""
    configured = os.getenv("AGENT_RUNTIME_LOG_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_LOG_FILE


def runtime_log(event: str, payload: dict[str, Any] | None = None) -> None:
    """Append one JSONL runtime trace event.

    Logging must never break agent execution, so all filesystem and serialization
    errors are intentionally swallowed.
    """
    if os.getenv("AGENT_RUNTIME_LOG_DISABLED") == "1":
        return
    if event in NOISY_EVENTS:
        return

    safe_payload = _safe_value(payload or {})
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "message": runtime_log_message(event, safe_payload),
        "payload": safe_payload,
    }
    try:
        path = runtime_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except Exception:
        return


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            tail_bytes = MAX_LOG_BYTES // 2
            with path.open("rb") as f:
                f.seek(-tail_bytes, 2)
                f.readline()
                data = f.read()
            with path.open("wb") as f:
                f.write(data)
    except OSError:
        pass


def redact_sensitive_data(value: Any) -> Any:
    """Return value with obvious secrets removed for logs and tool output."""

    return _safe_value(value)


def format_runtime_log_content(raw: str) -> str:
    """Render JSONL runtime logs into a compact human-readable trace."""

    lines: list[str] = []
    for raw_line in raw.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            lines.append(raw_line)
            continue
        if not isinstance(record, dict):
            lines.append(str(record))
            continue
        event = str(record.get("event", "runtime"))
        if event in NOISY_EVENTS:
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        message = str(record.get("message") or runtime_log_message(event, payload))
        ts = _format_ts(str(record.get("ts", "")))
        lines.append(f"{ts} {message}".strip())
    return "\n".join(lines) + ("\n" if lines else "")


def runtime_log_message(event: str, payload: dict[str, Any]) -> str:
    """Return a readable one-line summary for a runtime event."""

    conversation = _short_id(payload.get("conversation_id"))
    prefix = f"[{conversation}] " if conversation else ""
    if event == "web_session_init":
        return f"Web 会话初始化：默认项目 {payload.get('default_project', '')}"
    if event == "web_session_agent_ready":
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        return (
            f"模型就绪：{payload.get('model', '')} "
            f"({payload.get('provider', '')})，工具 {len(tools)} 个"
        )
    if event == "web_session_provider_error":
        return f"模型服务初始化失败：{payload.get('error', '')}"
    if event in {"web_submit", "user_turn_start"}:
        text = payload.get("text_preview") or payload.get("input_preview") or ""
        return f"{prefix}用户输入：{text}"
    if event == "web_command":
        return f"{prefix}执行命令：{payload.get('command', '')}"
    if event == "web_cancel":
        return f"取消请求：{_short_id(payload.get('request_id'))}"
    if event == "web_finish_request":
        return f"请求结束：{_short_id(payload.get('request_id'))}"
    if event == "web_stream_start":
        return (
            f"{prefix}Web 流开始：请求 "
            f"{_short_id(payload.get('request_id'))}"
        )
    if event == "agent_loop_init":
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        return (
            f"Agent 循环初始化：{payload.get('model', '')}，工具 {len(tools)} 个"
        )
    if event == "agent_run_start":
        return f"{prefix}开始智能体闭环：可用工具 {payload.get('tool_count', 0)} 个"
    if event == "agent_round_start":
        return f"{prefix}第 {payload.get('round_index', 0)} 轮：准备上下文"
    if event == "model_context":
        roles = ", ".join(str(role) for role in payload.get("roles", []))
        return (
            f"{prefix}上下文：{payload.get('message_count', 0)} 条消息"
            + (f" ({roles})" if roles else "")
        )
    if event == "model_request":
        tool_names = payload.get("tool_names") if isinstance(payload.get("tool_names"), list) else []
        return (
            f"{prefix}请求模型：{payload.get('message_count', 0)} 条消息，"
            f"工具 {len(tool_names)} 个，思考={'开' if payload.get('reasoning_enabled') else '关'}"
        )
    if event == "model_response_complete":
        return (
            f"{prefix}模型返回：状态 {payload.get('finish_status') or 'unknown'}，"
            f"工具调用 {payload.get('tool_call_count', 0)} 个，"
            f"推理 {payload.get('reasoning_chars', 0)} 字"
        )
    if event == "tool_call_start":
        return f"{prefix}调用工具：{payload.get('name', '')}"
    if event == "tool_call_result":
        if payload.get("status") == "error":
            detail = payload.get("error") or payload.get("error_type") or "unknown"
            return f"{prefix}工具结束：{payload.get('name', '')} (error: {detail})"
        return f"{prefix}工具结束：{payload.get('name', '')} ({payload.get('status', '')})"
    if event == "shell_command_start":
        return f"Shell 开始：{payload.get('command', '')} @ {payload.get('cwd', '')}"
    if event == "shell_command_complete":
        stdout_extra = _stream_log_suffix(payload, "stdout")
        stderr_extra = _stream_log_suffix(payload, "stderr")
        return (
            f"Shell 完成：退出码 {payload.get('exit_code')}，"
            f"stdout {payload.get('stdout_chars', 0)} 字{stdout_extra}，"
            f"stderr {payload.get('stderr_chars', 0)} 字{stderr_extra}"
        )
    if event == "shell_command_timeout":
        stdout_extra = _stream_log_suffix(payload, "stdout")
        stderr_extra = _stream_log_suffix(payload, "stderr")
        return (
            f"Shell 超时：{payload.get('command', '')}，"
            f"stdout {payload.get('stdout_chars', 0)} 字{stdout_extra}，"
            f"stderr {payload.get('stderr_chars', 0)} 字{stderr_extra}"
        )
    if event == "shell_output_artifact":
        return (
            f"Shell 输出落盘：{payload.get('stream', '')} "
            f"{payload.get('chars', 0)} 字 -> {payload.get('artifact_ref', '')}"
        )
    if event == "context_compact_check":
        status = "超过阈值" if payload.get("over_threshold") else "未超过阈值"
        return (
            f"{prefix}上下文预算检查：{payload.get('request_tokens', 0)} / "
            f"{payload.get('compact_threshold_tokens', 0)} tokens ({status})，"
            f"输入预算 {payload.get('input_budget_tokens', 0)}"
        )
    if event == "context_compact_split":
        return (
            f"{prefix}上下文分组：旧消息 {payload.get('older_messages', 0)} 条，"
            f"近期消息 {payload.get('recent_messages', 0)} 条，"
            f"保留最近 {payload.get('recent_turns', 0)} 轮"
        )
    if event in {"context_turns_compress_start", "context_ratio_compress_start"}:
        label = "旧对话" if event == "context_turns_compress_start" else "近期原文"
        return (
            f"{prefix}开始压缩{label}："
            f"{payload.get('summary_messages', 0)} 条消息，"
            f"目标 {payload.get('target_tokens', 0)} tokens"
        )
    if event in {"context_turns_compress_done", "context_ratio_compress_done"}:
        label = "旧对话" if event == "context_turns_compress_done" else "近期原文"
        if not payload.get("compressed"):
            return (
                f"{prefix}{label}压缩完成：未生成摘要，"
                f"摘要 {payload.get('summary_chars', 0)} 字"
            )
        return (
            f"{prefix}{label}压缩完成："
            f"摘要 {payload.get('summary_chars', 0)} 字"
        )
    if event in {"context_turns_compress_error", "context_ratio_compress_error"}:
        label = "旧对话" if event == "context_turns_compress_error" else "近期原文"
        return f"{prefix}{label}压缩失败：{payload.get('error', '')}"
    if event in {"context_turns_compress_skipped", "context_ratio_compress_skipped"}:
        label = "旧对话" if event == "context_turns_compress_skipped" else "近期原文"
        return (
            f"{prefix}跳过{label}压缩：{payload.get('reason', '')}"
        )
    if event == "context_force_truncate_start":
        return (
            f"{prefix}开始强制截断上下文：{payload.get('message_count', 0)} 条消息，"
            f"{payload.get('request_tokens', 0)} tokens"
        )
    if event == "context_force_truncate_done":
        return (
            f"{prefix}强制截断完成：{payload.get('before_messages', 0)} -> "
            f"{payload.get('after_messages', 0)} 条，"
            f"{payload.get('before_tokens', 0)} -> {payload.get('after_tokens', 0)} tokens"
        )
    if event == "context_compact_persist":
        return (
            f"{prefix}上下文已写回：{payload.get('before_messages', 0)} -> "
            f"{payload.get('after_messages', 0)} 条，"
            f"{payload.get('before_tokens', 0)} -> {payload.get('after_tokens', 0)} tokens"
        )
    if event == "context_budget_final":
        status = "超出预算" if payload.get("overflow") else "通过"
        return (
            f"{prefix}最终上下文预算：{payload.get('used_tokens', 0)} / "
            f"{payload.get('input_budget_tokens', 0)} tokens ({status})"
        )
    if event == "context_overflow":
        return f"{prefix}上下文超预算：{payload.get('error', '')}"
    if event == "agent_run_complete":
        return (
            f"{prefix}闭环完成：第 {payload.get('round_index', 0)} 轮，"
            f"状态 {payload.get('finish_status') or 'unknown'}，"
            f"回复预览：{payload.get('assistant_text_preview', '')}"
        )
    if event == "agent_run_cancelled":
        return f"{prefix}闭环已取消"
    if event == "agent_run_failed":
        return f"{prefix}闭环失败"
    if event == "agent_repeated_tool_call_warning":
        tool_call = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else {}
        return f"{prefix}重复工具调用提醒：{tool_call.get('name', '')}"
    if event == "agent_repeated_tool_call_blocked":
        tool_call = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else {}
        return f"{prefix}跳过重复工具调用：{tool_call.get('name', '')}"
    if event == "agent_final_answer_tool_calls_ignored":
        return (
            f"{prefix}最终回答模式忽略工具调用："
            f"{payload.get('tool_count', 0)} 个"
        )
    if event == "provider_error":
        return f"{prefix}模型请求失败：{payload.get('error', '')}"
    if event == "web_stream_usage":
        return f"{prefix}用量更新：{_usage_text(payload.get('usage'))}"
    if event == "skills_loaded":
        skills = payload.get("skills") if isinstance(payload.get("skills"), list) else []
        return f"加载技能：{len(skills)} 个 ({payload.get('source', '')})"
    return f"{prefix}{event}: {_compact_payload(payload)}"


def redact_sensitive_text(value: str) -> str:
    """Redact obvious secret values in free-form text."""

    text = value
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(api"):
            text = pattern.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _safe_value(value: Any, *, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {
            str(item_key): _safe_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        value = redact_sensitive_text(value)
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return value[:MAX_STRING_LENGTH] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in TOKEN_COUNT_KEYS:
        return False
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _format_ts(value: str) -> str:
    if len(value) >= 19 and "T" in value:
        return value[11:19]
    return value


def _short_id(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 12:
        return text
    return text[:8]


def _usage_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "无"
    parts = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in value:
            parts.append(f"{key}={value[key]}")
    return ", ".join(parts) if parts else _compact_payload(value)


def _stream_log_suffix(payload: dict[str, Any], stream: str) -> str:
    ref = payload.get(f"{stream}_ref")
    if not ref:
        return ""
    return f"（已截断，artifact={ref}）"


def _compact_payload(payload: dict[str, Any]) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(payload)
    text = " ".join(text.split())
    return text if len(text) <= 240 else f"{text[:237]}..."
