"""Core agent loop orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.context import ContextEngine
from agent_runtime.logging import runtime_log
from agent_runtime.providers import ModelConfig, Provider, ProviderError, ToolCall
from agent_runtime.tools import ToolRegistry


MAX_IDENTICAL_TOOL_CALLS = 2
DUPLICATE_WARNING_THRESHOLD = 1
MAX_DUPLICATE_BLOCKS = 3


@dataclass(slots=True)
class AgentEvent:
    """Provider-neutral event emitted by the agent loop."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ModelTurnResult:
    """Result collected from one streamed model request."""

    assistant_text: str = ""
    reasoning_text: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    finish_status: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    was_cancelled: bool = False
    failed: bool = False


class AgentLoop:
    """Coordinates context building, model calls, tool calls, and memory writes."""

    def __init__(
        self,
        *,
        provider: Provider,
        context: ContextEngine,
        tool_registry: ToolRegistry,
        log_context: Callable[[str, list[dict[str, Any]]], None] | None = None,
        model_timeout_seconds: float = 60,
    ) -> None:
        self.provider = provider
        self.context = context
        self.tool_registry = tool_registry
        self.log_context = log_context or self._log_model_context
        self.model_timeout_seconds = model_timeout_seconds
        self._compress_events: list[str] = []
        self.context.on_compress = self._on_compress
        runtime_log(
            "agent_loop_init",
            {
                "provider": provider.__class__.__name__,
                "model": getattr(provider, "model", ""),
                "tools": [tool.name for tool in tool_registry.list()],
            },
        )

    def run(
        self,
        conversation_id: str,
        *,
        reasoning_enabled: bool = True,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[AgentEvent]:
        """Run the model/tool loop for one persisted conversation."""

        cancelled = is_cancelled or (lambda: False)
        tools = self.tool_registry.provider_schemas()
        execution_steps: list[dict[str, Any]] = []
        all_reasoning_parts: list[str] = []
        final_assistant_text = ""
        tool_call_counts: dict[tuple[str, str], int] = {}
        next_round_tool_blocks: set[str] = set()
        next_round_reminders: list[str] = []
        duplicate_block_count = 0
        runtime_log(
            "agent_run_start",
            {
                "conversation_id": conversation_id,
                "reasoning_enabled": reasoning_enabled,
                "tool_count": len(tools),
            },
        )

        round_index = 0
        while True:
            runtime_log(
                "agent_round_start",
                {
                    "conversation_id": conversation_id,
                    "round_index": round_index,
                },
            )
            blocked_tools = next_round_tool_blocks
            reminders = next_round_reminders
            next_round_tool_blocks = set()
            next_round_reminders = []
            active_tools = self._filter_tools(tools, blocked_tools)
            result = yield from self._stream_model_once(
                conversation_id,
                reasoning_enabled=reasoning_enabled,
                tools=active_tools,
                extra_messages=self._reminder_messages(reminders),
                is_cancelled=cancelled,
            )

            if result.failed:
                runtime_log(
                    "agent_run_failed",
                    {
                        "conversation_id": conversation_id,
                        "round_index": round_index,
                    },
                )
                return
            execution_steps.extend(result.steps)
            if result.reasoning_text:
                all_reasoning_parts.append(result.reasoning_text)
            if result.assistant_text.strip():
                final_assistant_text = result.assistant_text.strip()

            if result.was_cancelled:
                if final_assistant_text:
                    self.context.add_assistant_message(
                        conversation_id,
                        final_assistant_text,
                        "\n".join(all_reasoning_parts),
                        execution_steps,
                    )
                yield AgentEvent("notice", {"tone": "muted", "text": "生成已停止"})
                runtime_log(
                    "agent_run_cancelled",
                    {
                        "conversation_id": conversation_id,
                        "round_index": round_index,
                    },
                )
                return

            if result.tool_calls:
                repeated_call, warning_calls = self._check_repeated_tool_calls(
                    result.tool_calls,
                    tool_call_counts,
                )
                if repeated_call is not None:
                    runtime_log(
                        "agent_repeated_tool_call_blocked",
                        {
                            "conversation_id": conversation_id,
                            "round_index": round_index,
                            "tool_call": {
                                "id": repeated_call.id,
                                "name": repeated_call.name,
                                "arguments": repeated_call.arguments,
                            },
                        },
                    )
                    duplicate_block_count += 1
                    if duplicate_block_count >= MAX_DUPLICATE_BLOCKS:
                        yield AgentEvent(
                            "notice",
                            {
                                "tone": "error",
                                "text": "检测到重复工具调用，已停止继续调用。",
                            },
                        )
                        return
                    next_round_tool_blocks.add(repeated_call.name)
                    next_round_reminders.append(
                        self._duplicate_call_reminder(repeated_call.name)
                    )
                    round_index += 1
                    continue
                if warning_calls:
                    for wc in warning_calls:
                        next_round_reminders.append(self._duplicate_call_reminder(wc.name))
                        runtime_log(
                            "agent_repeated_tool_call_warning",
                            {
                                "conversation_id": conversation_id,
                                "round_index": round_index,
                                "tool_name": wc.name,
                            },
                        )
                self.context.add_assistant_message(
                    conversation_id,
                    final_assistant_text,
                    "\n".join(all_reasoning_parts),
                    execution_steps,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in result.tool_calls
                    ],
                )
                tool_steps = yield from self._execute_tool_calls(conversation_id, result.tool_calls)
                execution_steps.extend(tool_steps)
                duplicate_block_count = 0
                round_index += 1
                continue

            if result.finish_status and result.finish_status not in {
                "completed",
                "succeeded",
                "stop",
            }:
                yield AgentEvent(
                    "notice",
                    {"tone": "muted", "text": f"响应状态：{result.finish_status}"},
                )
            if final_assistant_text:
                self.context.add_assistant_message(
                    conversation_id,
                    final_assistant_text,
                    "\n".join(all_reasoning_parts),
                    execution_steps,
                )
            yield AgentEvent("usage", {"usage": result.usage})
            runtime_log(
                "agent_run_complete",
                {
                    "conversation_id": conversation_id,
                    "round_index": round_index,
                    "finish_status": result.finish_status,
                    "usage": result.usage,
                    "assistant_text_preview": self._summarize_value(final_assistant_text),
                },
            )
            return

    def run_user_turn(
        self,
        conversation_id: str,
        user_input: str,
        *,
        reasoning_enabled: bool = True,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[AgentEvent]:
        """Persist a user message and run the agent loop for that turn."""

        value = user_input.strip()
        if not value:
            return
        runtime_log(
            "user_turn_start",
            {
                "conversation_id": conversation_id,
                "input_preview": self._summarize_value(value),
                "reasoning_enabled": reasoning_enabled,
            },
        )
        self.context.add_user_message(conversation_id, value)
        yield from self.run(
            conversation_id,
            reasoning_enabled=reasoning_enabled,
            is_cancelled=is_cancelled,
        )

    def _stream_model_once(
        self,
        conversation_id: str,
        *,
        reasoning_enabled: bool,
        tools: list[dict[str, Any]],
        extra_messages: list[dict[str, Any]] | None = None,
        is_cancelled: Callable[[], bool],
    ) -> Iterator[AgentEvent]:
        assistant_started = False
        assistant_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_status = None
        usage: dict[str, Any] = {}
        tool_calls: list[ToolCall] = []
        was_cancelled = False
        model_input = self.context.build_model_input(
            conversation_id,
            extra_input_tokens=(
                self._tools_token_count(tools)
                + self._model_messages_token_count(extra_messages or [])
            ),
        )
        if extra_messages:
            model_input = [*model_input, *extra_messages]
        yield from self._drain_compress_events()
        self.log_context(conversation_id, model_input)
        runtime_log(
            "model_request",
            {
                "conversation_id": conversation_id,
                "message_count": len(model_input),
                "roles": [str(message.get("role", "")) for message in model_input],
                "tool_names": [
                    str(tool.get("function", {}).get("name", ""))
                    for tool in tools
                    if isinstance(tool, dict)
                ],
                "reasoning_enabled": reasoning_enabled,
            },
        )

        try:
            for event in self.provider.stream(
                model_input,
                tools=tools,
                model_config=self._model_config(reasoning_enabled),
            ):
                if is_cancelled():
                    was_cancelled = True
                    break
                if event.tool_call is not None:
                    tool_calls.append(event.tool_call)
                    continue
                if self._is_reasoning_event(event) and event.delta:
                    if not reasoning_enabled:
                        continue
                    reasoning_parts.append(event.delta)
                    yield AgentEvent("reasoning_delta", {"text": event.delta})
                    continue
                if event.type == "content_delta" and event.delta:
                    if not assistant_started:
                        yield AgentEvent("assistant_start", {})
                        assistant_started = True
                    assistant_parts.append(event.delta)
                    yield AgentEvent("assistant_delta", {"text": event.delta})
                    continue
                if event.type == "finish" and event.response is not None:
                    finish_status = event.response.finish_reason
                    usage = event.response.usage
                    tool_calls.extend(event.response.tool_calls)
                    if event.response.content and not assistant_started:
                        yield AgentEvent("assistant_start", {})
                        assistant_started = True
                        assistant_parts.append(event.response.content)
                        yield AgentEvent(
                            "assistant_delta",
                            {"text": event.response.content},
                        )
        except ProviderError as exc:
            runtime_log(
                "provider_error",
                {
                    "conversation_id": conversation_id,
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                },
            )
            yield AgentEvent("notice", {"tone": "error", "text": str(exc)})
            return _ModelTurnResult(failed=True)

        reasoning_text = "".join(reasoning_parts)
        runtime_log(
            "model_response_complete",
            {
                "conversation_id": conversation_id,
                "finish_status": finish_status,
                "reasoning_chars": len(reasoning_text),
                "tool_call_count": len(tool_calls),
                "was_cancelled": was_cancelled,
                "usage": usage,
            },
        )
        return _ModelTurnResult(
            assistant_text="".join(assistant_parts),
            reasoning_text=reasoning_text,
            steps=(
                [
                    {
                        "id": f"reasoning-{int(time.time() * 1000)}",
                        "type": "reasoning",
                        "text": reasoning_text,
                        "open": False,
                        "complete": True,
                    }
                ]
                if reasoning_text
                else []
            ),
            finish_status=finish_status,
            tool_calls=tool_calls,
            usage=usage,
            was_cancelled=was_cancelled,
        )

    def _execute_tool_calls(
        self,
        conversation_id: str,
        tool_calls: list[ToolCall],
    ) -> Iterator[AgentEvent]:
        steps: list[dict[str, Any]] = []
        for call in tool_calls:
            arguments = call.arguments or {}
            arguments_summary = self._summarize_value(arguments)
            runtime_log(
                "tool_call_start",
                {
                    "conversation_id": conversation_id,
                    "id": call.id,
                    "name": call.name,
                },
            )
            yield AgentEvent(
                "tool_call_start",
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": arguments,
                    "argumentsSummary": arguments_summary,
                    "status": "running",
                },
            )
            try:
                result = self.tool_registry.execute(call.name, arguments)
                status = "completed"
            except Exception as exc:
                result = {
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                }
                status = "error"

            result_summary = self._summarize_value(result)
            runtime_log(
                "tool_call_result",
                {
                    "conversation_id": conversation_id,
                    "id": call.id,
                    "name": call.name,
                    "status": status,
                },
            )
            self.context.add_tool_result(
                conversation_id,
                call.name,
                arguments,
                result,
                call_id=call.id,
            )
            steps.append(
                {
                    "id": call.id,
                    "type": "tool",
                    "name": call.name,
                    "arguments": arguments,
                    "argumentsSummary": arguments_summary,
                    "status": status,
                    "result": result,
                    "summary": result_summary,
                }
            )
            yield AgentEvent(
                "tool_call_result",
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": arguments,
                    "argumentsSummary": arguments_summary,
                    "status": status,
                    "result": result,
                    "summary": result_summary,
                },
            )
        return steps

    def _check_repeated_tool_calls(
        self,
        tool_calls: list[ToolCall],
        counts: dict[tuple[str, str], int],
    ) -> tuple[ToolCall | None, list[ToolCall]]:
        """Two-level duplicate detection.

        Returns (blocked_call, warning_calls):
        - blocked_call: the call that exceeded MAX_IDENTICAL_TOOL_CALLS (should block tool)
        - warning_calls: calls that exceeded DUPLICATE_WARNING_THRESHOLD but not max (inject warning)
        """
        blocked: ToolCall | None = None
        warnings: list[ToolCall] = []
        for call in tool_calls:
            key = self._tool_call_key(call)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > MAX_IDENTICAL_TOOL_CALLS and blocked is None:
                blocked = call
            elif counts[key] > DUPLICATE_WARNING_THRESHOLD and counts[key] <= MAX_IDENTICAL_TOOL_CALLS:
                warnings.append(call)
        return blocked, warnings

    def _tool_call_key(self, call: ToolCall) -> tuple[str, str]:
        try:
            arguments = json.dumps(
                call.arguments or {},
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            arguments = str(call.arguments or {})
        return call.name, arguments

    def _model_config(self, reasoning_enabled: bool) -> ModelConfig:
        # DeepSeek-style compatible endpoints use enable_thinking. Providers that
        # do not understand it should ignore extra_body or reject during tests.
        extra_body = {} if reasoning_enabled else {"enable_thinking": False}
        return ModelConfig(
            timeout_seconds=self.model_timeout_seconds,
            extra_body=extra_body,
        )

    def _is_reasoning_event(self, event: Any) -> bool:
        event_type = getattr(event, "type", "")
        return (
            isinstance(event_type, str)
            and event_type != "content_delta"
            and ("reasoning" in event_type or "thinking" in event_type)
        )

    def _tools_token_count(self, tools: list[dict[str, Any]]) -> int:
        if not tools:
            return 0
        try:
            text = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(tools)
        return self.context.token_counter.count_text(text)

    def _model_messages_token_count(self, messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        try:
            text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(messages)
        return self.context.token_counter.count_text(text)

    def _filter_tools(
        self,
        tools: list[dict[str, Any]],
        blocked_names: set[str],
    ) -> list[dict[str, Any]]:
        if not blocked_names:
            return tools
        return [
            tool
            for tool in tools
            if str(tool.get("function", {}).get("name", "")) not in blocked_names
        ]

    def _reminder_messages(self, reminders: list[str]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": reminder}
            for reminder in reminders
            if reminder.strip()
        ]

    def _duplicate_call_reminder(self, tool_name: str) -> str:
        return (
            f"Duplicate call detected for {tool_name} with the same arguments. "
            "Do not call it again. Continue using the existing result."
        )

    def _summarize_value(self, value: Any, limit: int = 220) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                text = str(value)
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _on_compress(self, stage: str) -> None:
        self._compress_events.append(stage)

    def _drain_compress_events(self) -> Iterator[AgentEvent]:
        while self._compress_events:
            stage = self._compress_events.pop(0)
            if stage == "start":
                yield AgentEvent(
                    "notice",
                    {"tone": "muted", "text": "正在压缩上下文..."},
                )
            elif stage == "done":
                yield AgentEvent(
                    "notice",
                    {"tone": "muted", "text": "上下文压缩完成"},
                )

    def _log_model_context(
        self,
        conversation_id: str,
        model_input: list[dict[str, Any]],
    ) -> None:
        runtime_log(
            "model_context",
            {
                "conversation_id": conversation_id,
                "message_count": len(model_input),
                "roles": [str(m.get("role", "")) for m in model_input],
            },
        )
