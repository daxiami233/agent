"""Web chat session state and streaming orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime import Agent, AgentRuntimeConfig, create_agent
from agent_runtime.mcp import MCPClientHost
from agent_runtime.memory import LongTermMemory, MemoryStore
from agent_runtime.providers import OpenAIProvider, ProviderError
from agent_runtime.logging import runtime_log, runtime_log_path
from agent_runtime.tools import ToolRegistry
from agent_runtime.skills import SkillManifest, SkillRegistry

from .commands import (
    CommandContext,
    CommandRegistry,
    help_command,
    model_command,
    status_command,
)


STATE_DIR = Path.home() / ".agent-runtime"
HISTORY_DIR = STATE_DIR / "history"


@dataclass(slots=True)
class WebEvent:
    """Event sent to the React client."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class WebSession:
    """One local browser session backed by a model provider."""

    def __init__(
        self,
        provider_factory: Callable[[], OpenAIProvider] = OpenAIProvider.from_env,
        memory_store: MemoryStore | None = None,
        long_term_memory: LongTermMemory | None = None,
        tool_registry: ToolRegistry | None = None,
        skills: list[SkillManifest] | None = None,
        skill_registry: SkillRegistry | None = None,
        mcp_host: MCPClientHost | None = None,
        include_memory_tools: bool = True,
        include_skill_tools: bool = True,
        include_shell_tool: bool = True,
    ) -> None:
        self.provider_factory = provider_factory
        self.provider: OpenAIProvider | None = None
        self.history = self._load_history()
        self.memory_store = memory_store or MemoryStore()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.context = None
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry(skills or [])
        self.mcp_host = mcp_host or MCPClientHost()
        self.agent: Agent | None = None
        self.commands = self._build_commands()
        self.should_shutdown = False
        self._cancelled_requests: set[str] = set()
        self._output_events: list[WebEvent] = []
        runtime_log(
            "web_session_init",
            {
                "include_memory_tools": include_memory_tools,
                "include_skill_tools": include_skill_tools,
                "include_shell_tool": include_shell_tool,
            },
        )
        try:
            self.provider = self.provider_factory()
            self.agent = create_agent(
                config=AgentRuntimeConfig(
                    context_window_tokens=int(
                        getattr(self.provider, "context_window_tokens", None)
                        or AgentRuntimeConfig().context_window_tokens
                    ),
                    include_memory_tools=include_memory_tools,
                    include_skill_tools=include_skill_tools,
                    include_shell_tool=include_shell_tool,
                ),
                provider=self.provider,
                tools=self.tool_registry.list(),
                skills=self.skill_registry.list(),
                memory_store=self.memory_store,
                long_term_memory=self.long_term_memory,
            )
            self.context = self.agent.context
            self.memory_store = self.agent.memory_store
            self.tool_registry = self.agent.tool_registry
            self.skill_registry = self.agent.skill_registry
            self.mcp_host = self.agent.mcp_host
            self.long_term_memory = self.context.long_term_memory
            runtime_log(
                "web_session_agent_ready",
                {
                    "provider": self.provider.__class__.__name__,
                    "model": getattr(self.provider, "model", ""),
                    "tools": [tool.name for tool in self.tool_registry.list()],
                    "log_file": str(runtime_log_path()),
                },
            )
        except ProviderError as exc:
            runtime_log(
                "web_session_provider_error",
                {"error": str(exc), "type": exc.__class__.__name__},
            )
            self._append_error(str(exc))
            self._append_hint("请在 .env 中设置 API_KEY、BASE_URL 和 MODEL。")

    def boot_events(self) -> list[WebEvent]:
        return [
            WebEvent(
                "status",
                {
                    "cwd": Path.cwd().name or str(Path.cwd()),
                    "cwdPath": str(Path.cwd()),
                    "logFile": str(runtime_log_path()),
                    "model": self._model_display_text(),
                    **self._context_window_payload(),
                    "apiMode": (
                        self.provider.current_api_mode()
                        if self.provider is not None
                        else "未配置"
                    ),
                },
            ),
            *self._drain_output_events(),
        ]

    def list_conversations(self) -> list[dict[str, Any]]:
        return [self._conversation_payload(record.id) for record in self.memory_store.list_conversations()]

    def create_conversation(
        self,
        conversation_id: str | None = None,
        title: str = "新对话",
    ) -> dict[str, Any]:
        record = self.memory_store.create_conversation(conversation_id or str(uuid4()), title)
        return self._conversation_payload(record.id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.memory_store.delete_conversation(conversation_id)

    def submit(
        self,
        conversation_id: str,
        text: str,
        request_id: str = "",
        reasoning_enabled: bool = True,
    ) -> Iterator[WebEvent]:
        value = text.strip()
        if not value:
            return

        runtime_log(
            "web_submit",
            {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "text_preview": self._summarize_value(value),
                "reasoning_enabled": reasoning_enabled,
            },
        )
        self._append_history(value)
        if value.startswith("/") or value.startswith(":"):
            runtime_log(
                "web_command",
                {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "command": value,
                },
            )
            yield from self._handle_command(value)
            return

        self._ensure_conversation_title(conversation_id, value)
        yield WebEvent("user_message", {"text": value})
        yield from self._stream_model(
            conversation_id,
            user_input=value,
            request_id=request_id,
            reasoning_enabled=reasoning_enabled,
        )

    def cancel(self, request_id: str) -> None:
        if request_id:
            self._cancelled_requests.add(request_id)
            runtime_log("web_cancel", {"request_id": request_id})

    def finish_request(self, request_id: str) -> None:
        if request_id:
            self._cancelled_requests.discard(request_id)
            runtime_log("web_finish_request", {"request_id": request_id})

    def command_matches(self, text: str) -> list[tuple[str, str]]:
        if not text.startswith("/") and not text.startswith(":"):
            return []
        if any(char.isspace() for char in text):
            return []
        return [
            (name, description)
            for name, description in self.commands.completion_rows()
            if name.startswith(text)
        ]

    def _handle_command(self, command: str) -> Iterator[WebEvent]:
        result = self.commands.execute(command, self._command_context())
        if result is None:
            self._append_error(f"未知命令：{command}")
            self._append_hint("输入 /help 查看可用命令。")
        elif result:
            self.should_shutdown = True
        yield from self._drain_output_events()
        if self.should_shutdown:
            yield WebEvent("shutdown", {})

    def _stream_model(
        self,
        conversation_id: str,
        user_input: str,
        request_id: str,
        reasoning_enabled: bool,
    ) -> Iterator[WebEvent]:
        if self.agent is None:
            self._append_error("模型服务尚未初始化。")
            yield from self._drain_output_events()
            return

        runtime_log(
            "web_stream_start",
            {
                "conversation_id": conversation_id,
                "request_id": request_id,
            },
        )
        for event in self.agent.stream(
            user_input,
            conversation_id=conversation_id,
            reasoning_enabled=reasoning_enabled,
            is_cancelled=lambda: bool(request_id and request_id in self._cancelled_requests),
        ):
            if event.type == "usage":
                usage = event.payload.get("usage")
                runtime_log(
                    "web_stream_usage",
                    {
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                        "usage": usage if isinstance(usage, dict) else {},
                    },
                )
                yield WebEvent("status", self._context_window_payload(conversation_id))
                continue
            if event.type in {"notice", "tool_call_start", "tool_call_result"}:
                runtime_log(
                    "web_stream_event",
                    {
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                        "event_type": event.type,
                        "payload_summary": self._summarize_value(event.payload),
                    },
                )
            yield WebEvent(event.type, event.payload)

        yield from self._drain_output_events()

    def _build_commands(self) -> CommandRegistry:
        registry = CommandRegistry()
        registry.register(help_command())
        registry.register(model_command())
        registry.register(status_command())
        return registry

    def _command_context(self) -> CommandContext:
        return CommandContext(
            provider=self.provider,
            history_path=self._history_path(),
            print_error=self._append_error,
            print_help=self._append_help,
            print_hint=self._append_hint,
            print_info=self._append_info,
            print_model=self._append_model,
            print_status=self._append_status,
            clear_screen=self._clear_screen,
        )

    def _append_error(self, message: str) -> None:
        self._output_events.append(WebEvent("notice", {"tone": "error", "text": message}))

    def _append_info(self, message: str) -> None:
        self._output_events.append(WebEvent("notice", {"tone": "info", "text": message}))

    def _append_hint(self, message: str) -> None:
        self._output_events.append(WebEvent("notice", {"tone": "muted", "text": message}))

    def _append_help(self) -> None:
        lines = ["命令"]
        lines.extend(
            f"{name:<10} {description}" for name, description in self.commands.help_rows()
        )
        aliases = ", ".join(
            f"{alias} -> {canonical}"
            for alias, canonical in sorted(self.commands.aliases().items())
        )
        if aliases:
            lines.append(f"别名：{aliases}")
        lines.extend(["", "编辑", "回车键发送", "Shift+回车键换行"])
        self._append_info("\n".join(lines))

    def _append_model(self) -> None:
        if self.provider is None:
            self._append_error("模型服务尚未初始化。")
            return
        self._append_info(f"模型：{self.provider.model}")
        self._append_hint(f"接口地址：{self.provider.base_url}")
        if self.provider.context_window_tokens is None:
            self._append_hint("上下文窗口：未配置")
        else:
            self._append_hint(
                f"上下文窗口：{self._format_tokens(self.provider.context_window_tokens)}"
            )

    def _append_status(self) -> None:
        self._append_info("状态")
        self._append_hint(f"工作目录：{Path.cwd()}")
        self._append_hint(f"历史记录：{self._history_path()}")
        if self.provider is None:
            self._append_hint("模型服务：未配置")
        else:
            self._append_hint(
                f"模型服务：{self.provider.current_api_mode()} ({self.provider.model})"
            )
            self._append_hint(f"上下文：{self._context_window_text()}")

    def _clear_screen(self) -> None:
        self._output_events.append(WebEvent("clear", {}))

    def _drain_output_events(self) -> list[WebEvent]:
        events = self._output_events
        self._output_events = []
        return events

    def _conversation_payload(self, conversation_id: str) -> dict[str, Any]:
        record = self.memory_store.get_conversation(conversation_id)
        if record is None:
            record = self.memory_store.create_conversation(conversation_id)
        messages = []
        current_assistant: dict[str, Any] | None = None
        pending_tools = []
        for message in self.memory_store.list_messages(record.id):
            if message.role == "user":
                if pending_tools and current_assistant is not None:
                    current_assistant["tools"] = [
                        *current_assistant.get("tools", []),
                        *pending_tools,
                    ]
                    pending_tools = []
                current_assistant = None
                messages.append({
                    "id": str(message.id),
                    "role": "user",
                    "text": message.content,
                    "createdAt": message.created_at,
                })
            elif message.role == "assistant":
                if current_assistant is None:
                    current_assistant = {
                        "id": str(message.id),
                        "role": "assistant",
                        "tools": [],
                        "createdAt": message.created_at,
                    }
                    messages.append(current_assistant)
                if pending_tools:
                    current_assistant["tools"] = [
                        *current_assistant.get("tools", []),
                        *pending_tools,
                    ]
                    pending_tools = []
                text, reasoning, steps = self._parse_assistant_content(message.content)
                if text:
                    existing_text = str(current_assistant.get("text", ""))
                    current_assistant["text"] = (
                        f"{existing_text}\n\n{text}" if existing_text else text
                    )
                else:
                    current_assistant.setdefault("text", "")
                if reasoning:
                    current_assistant["reasoning"] = reasoning
                    current_assistant["reasoningComplete"] = True
                if steps:
                    current_assistant["steps"] = steps
            elif message.role == "tool":
                try:
                    payload = json.loads(message.content)
                    tool_name = payload.get("name", "unknown")
                    tool_call_id = payload.get("call_id") or f"{tool_name}-{message.id}"
                    tool_arguments = payload.get("arguments", {})
                    tool_result = payload.get("result")
                    pending_tools.append({
                        "id": tool_call_id,
                        "name": tool_name,
                        "arguments": tool_arguments,
                        "argumentsSummary": self._summarize_value(tool_arguments),
                        "status": "completed",
                        "result": tool_result,
                        "summary": self._summarize_value(tool_result),
                    })
                except json.JSONDecodeError:
                    pass
        if pending_tools and current_assistant is not None:
            current_assistant["tools"] = [
                *current_assistant.get("tools", []),
                *pending_tools,
            ]
        return {
            "id": record.id,
            "title": record.title,
            "createdAt": record.created_at,
            "updatedAt": record.updated_at,
            **self._context_window_payload(record.id),
            "messages": messages,
        }

    def _parse_assistant_content(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "text" in payload:
                raw_steps = payload.get("steps", [])
                steps = raw_steps if isinstance(raw_steps, list) else []
                normalized_steps = [
                    step
                    for step in steps
                    if isinstance(step, dict) and step.get("type") in {"reasoning", "tool"}
                ]
                return str(payload["text"]), str(payload.get("reasoning", "")), normalized_steps
        except (json.JSONDecodeError, TypeError):
            pass
        return content, "", []

    def _ensure_conversation_title(self, conversation_id: str, prompt: str) -> None:
        record = self.memory_store.create_conversation(conversation_id)
        if record.title in {"New chat", "新对话"}:
            self.memory_store.update_conversation_title(
                conversation_id,
                self._prompt_title(prompt),
            )

    def _prompt_title(self, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            return "新对话"
        return f"{normalized[:31]}..." if len(normalized) > 34 else normalized

    def _model_display_text(self) -> str:
        return self.provider.model if self.provider is not None else "未配置"

    def _context_window_text(self) -> str:
        payload = self._context_window_payload()
        return str(payload["context"])

    def _context_window_payload(
        self,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        if self.context is None:
            return {
                "context": "输入上下文：未配置",
                "contextUsed": 0,
                "inputBudgetTokens": None,
                "contextWindow": None,
            }

        budget = self.context.context_budget(
            conversation_id or "",
            extra_input_tokens=self._tool_schema_tokens(),
        )
        used = budget.used_input_tokens
        input_budget = budget.input_budget_tokens
        return {
            "context": f"输入上下文：{self._format_tokens(used)} / {self._format_tokens(input_budget)}",
            "contextUsed": used,
            **budget.to_payload(),
        }

    def _tool_schema_tokens(self) -> int:
        if self.context is None:
            return 0
        try:
            text = json.dumps(
                self.tool_registry.provider_schemas(),
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            text = str(self.tool_registry.provider_schemas())
        return self.context.token_counter.count_text(text)

    def _format_tokens(self, value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

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

    def _load_history(self) -> list[str]:
        path = self._history_path()
        if not path.exists():
            return []
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def _append_history(self, value: str) -> None:
        self.history.append(value)
        try:
            with self._history_path().open("a", encoding="utf-8") as file:
                file.write(value.replace("\n", "\\n") + "\n")
        except OSError:
            pass

    def _history_path(self) -> Path:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        cwd_hash = sha1(str(Path.cwd()).encode("utf-8")).hexdigest()[:12]
        return HISTORY_DIR / f"{cwd_hash}.history"
