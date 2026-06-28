"""Core agent loop orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.context import ContextEngine, ContextOverflowError
from agent_runtime.logging import runtime_log
from agent_runtime.providers import ModelConfig, Provider, ProviderError, ToolCall
from agent_runtime.tools import ToolRegistry


MAX_IDENTICAL_TOOL_CALLS = 2
DUPLICATE_WARNING_THRESHOLD = 1
MAX_DUPLICATE_SKIP_ROUNDS = 3


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
        last_tool_call_key: tuple[str, str] | None = None
        same_tool_call_streak = 0
        next_round_reminders: list[str] = []
        duplicate_skip_rounds = 0
        force_final_answer_next = False
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
            reminders = next_round_reminders
            next_round_reminders = []
            final_answer_mode = force_final_answer_next
            force_final_answer_next = False
            if final_answer_mode:
                reminders = [*reminders, self._final_answer_reminder()]
            active_tools = [] if final_answer_mode else tools
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

            if result.tool_calls and final_answer_mode:
                runtime_log(
                    "agent_final_answer_tool_calls_ignored",
                    {
                        "conversation_id": conversation_id,
                        "round_index": round_index,
                        "tool_count": len(result.tool_calls),
                    },
                )
                if not final_assistant_text:
                    yield AgentEvent(
                        "notice",
                        {
                            "tone": "error",
                            "text": "模型在最终回答模式下仍请求工具，已停止。",
                        },
                    )
                    return
                result.tool_calls = []

            if result.tool_calls:
                (
                    executable_calls,
                    warning_calls,
                    skipped_calls,
                    last_tool_call_key,
                    same_tool_call_streak,
                ) = self._split_repeated_tool_calls(
                    result.tool_calls,
                    last_tool_call_key,
                    same_tool_call_streak,
                )
                for skipped_call in skipped_calls:
                    runtime_log(
                        "agent_repeated_tool_call_blocked",
                        {
                            "conversation_id": conversation_id,
                            "round_index": round_index,
                            "tool_call": {
                                "id": skipped_call.id,
                                "name": skipped_call.name,
                                "arguments": skipped_call.arguments,
                            },
                        },
                    )
                    next_round_reminders.append(
                        self._duplicate_call_reminder(skipped_call)
                    )
                if skipped_calls and not executable_calls:
                    duplicate_skip_rounds += 1
                    if duplicate_skip_rounds >= MAX_DUPLICATE_SKIP_ROUNDS:
                        force_final_answer_next = True
                    round_index += 1
                    continue
                if warning_calls:
                    for wc in warning_calls:
                        next_round_reminders.append(self._duplicate_call_reminder(wc))
                        runtime_log(
                            "agent_repeated_tool_call_warning",
                            {
                                "conversation_id": conversation_id,
                                "round_index": round_index,
                                "tool_call": {
                                    "id": wc.id,
                                    "name": wc.name,
                                    "arguments": wc.arguments,
                                },
                            },
                        )
                self.context.add_assistant_message(
                    conversation_id,
                    final_assistant_text,
                    "\n".join(all_reasoning_parts),
                    execution_steps,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in executable_calls
                    ],
                )
                tool_steps = yield from self._execute_tool_calls(conversation_id, executable_calls)
                execution_steps.extend(tool_steps)
                duplicate_skip_rounds = 0
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
        extra_input_tokens = (
            self._tools_token_count(tools)
            + self._model_messages_token_count(extra_messages or [])
        )
        try:
            model_input = self.context.build_model_input(
                conversation_id,
                extra_input_tokens=extra_input_tokens,
            )
        except ContextOverflowError as exc:
            runtime_log(
                "context_overflow",
                {
                    "conversation_id": conversation_id,
                    "error": str(exc),
                },
            )
            yield AgentEvent("notice", {"tone": "error", "text": str(exc)})
            return _ModelTurnResult(failed=True)
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
                    "error": result.get("error") if status == "error" else "",
                    "error_type": result.get("type") if status == "error" else "",
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

    def _split_repeated_tool_calls(
        self,
        tool_calls: list[ToolCall],
        last_key: tuple[str, str] | None,
        streak: int,
    ) -> tuple[list[ToolCall], list[ToolCall], list[ToolCall], tuple[str, str] | None, int]:
        """Split consecutive exact duplicate calls: tool name + canonical args."""
        executable: list[ToolCall] = []
        warnings: list[ToolCall] = []
        skipped: list[ToolCall] = []
        for call in tool_calls:
            key = self._tool_call_key(call)
            if key == last_key:
                streak += 1
            else:
                last_key = key
                streak = 1
            if streak > MAX_IDENTICAL_TOOL_CALLS:
                skipped.append(call)
                continue
            executable.append(call)
            if streak > DUPLICATE_WARNING_THRESHOLD:
                warnings.append(call)
        return executable, warnings, skipped, last_key, streak

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

    def _reminder_messages(self, reminders: list[str]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": reminder}
            for reminder in reminders
            if reminder.strip()
        ]

    def _duplicate_call_reminder(self, call: ToolCall) -> str:
        return (
            f"Duplicate call skipped for {call.name} with the exact same arguments: "
            f"{self._summarize_value(call.arguments)}. Do not call it again with "
            "the same arguments. Use the existing result, use different arguments, "
            "or answer directly."
        )

    def _final_answer_reminder(self) -> str:
        return (
            "Repeated identical tool calls were skipped. Do not call any tools now. "
            "Based only on the existing conversation and tool results, directly "
            "answer the user's original request in Chinese."
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
