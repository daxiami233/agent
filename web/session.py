"""Web chat session state and streaming orchestration."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
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
PROJECT_DB = STATE_DIR / "web_projects.sqlite3"


@dataclass(slots=True)
class WebEvent:
    """Event sent to the React client."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectRecord:
    """Web-layer project metadata. Agent core only sees conversation ids."""

    id: str
    name: str
    path: str
    created_at: float
    updated_at: float
    last_opened_at: float


class WebProjectStore:
    """Persist project and project-conversation mapping outside agent core."""

    def __init__(self, path: Path | str = PROJECT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def ensure_project(self, path: Path | str, name: str | None = None) -> ProjectRecord:
        project_path = str(Path(path).expanduser().resolve())
        project_id = sha1(project_path.encode("utf-8")).hexdigest()[:16]
        now = time.time()
        project_name = name or Path(project_path).name or project_path
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, name, path, created_at, updated_at, last_opened_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    path = excluded.path,
                    updated_at = excluded.updated_at
                """,
                (project_id, project_name, project_path, now, now, 0.0),
            )
            row = conn.execute(
                """
                SELECT id, name, path, created_at, updated_at, last_opened_at
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
        return self._project_from_row(row)

    def list_projects(self) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, path, created_at, updated_at, last_opened_at
                FROM projects
                ORDER BY last_opened_at DESC, updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, path, created_at, updated_at, last_opened_at
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
        return self._project_from_row(row) if row is not None else None

    def touch_project(self, project_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET last_opened_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, project_id),
            )

    def link_conversation(self, project_id: str, conversation_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO project_conversations
                    (project_id, conversation_id, created_at)
                VALUES (?, ?, ?)
                """,
                (project_id, conversation_id, now),
            )

    def unlink_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM project_conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0

    def conversation_ids(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id
                FROM project_conversations
                WHERE project_id = ?
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def conversation_count(self, project_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM project_conversations
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_opened_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_conversations (
                    project_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (project_id, conversation_id),
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _project_from_row(self, row: sqlite3.Row | tuple[object, ...]) -> ProjectRecord:
        return ProjectRecord(
            id=str(row[0]),
            name=str(row[1]),
            path=str(row[2]),
            created_at=float(row[3]),
            updated_at=float(row[4]),
            last_opened_at=float(row[5]),
        )


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
        project_store: WebProjectStore | None = None,
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
        self.project_store = project_store or WebProjectStore(
            self._default_project_store_path(self.memory_store)
        )
        self.default_project = self.project_store.ensure_project(Path.cwd())
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
                    "default_project": self.default_project.path,
                },
        )
        try:
            self.provider = self.provider_factory()
            runtime_config = AgentRuntimeConfig.from_env()
            provider_context_window = getattr(self.provider, "context_window_tokens", None)
            if provider_context_window:
                runtime_config = replace(
                    runtime_config,
                    context_window_tokens=int(provider_context_window),
                )
            runtime_config = replace(
                runtime_config,
                include_memory_tools=include_memory_tools,
                include_skill_tools=include_skill_tools,
                include_shell_tool=include_shell_tool,
            )
            self.agent = create_agent(
                config=runtime_config,
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

    def list_projects(self) -> list[dict[str, Any]]:
        return [self._project_payload(project) for project in self.project_store.list_projects()]

    def add_project(self, path: Path | str, name: str | None = None) -> dict[str, Any]:
        raw_path = str(path).strip()
        if not raw_path:
            raise ValueError("Project path is required")
        project_path = Path(raw_path).expanduser().resolve()
        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")
        project = self.project_store.ensure_project(project_path, name)
        self.project_store.touch_project(project.id)
        project = self.project_store.get_project(project.id) or project
        return self._project_payload(project)

    def select_project(self, project_id: str) -> dict[str, Any]:
        project = self.project_store.get_project(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        self.project_store.touch_project(project_id)
        project = self.project_store.get_project(project_id) or project
        return {
            "project": self._project_payload(project),
            "conversations": self.list_conversations(project_id),
        }

    def delete_project(self, project_id: str) -> None:
        if not self.project_store.delete_project(project_id):
            raise ValueError(f"Project not found: {project_id}")

    def list_conversations(self, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            ids = set(self.project_store.conversation_ids(project_id))
            return [
                self._conversation_payload(record.id)
                for record in self.memory_store.list_conversations()
                if record.id in ids
            ]
        return [
            self._conversation_payload(record.id)
            for record in self.memory_store.list_conversations()
        ]

    def create_conversation(
        self,
        conversation_id: str | None = None,
        title: str = "新对话",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        record = self.memory_store.create_conversation(conversation_id or str(uuid4()), title)
        if project_id:
            self.project_store.link_conversation(project_id, record.id)
        return self._conversation_payload(record.id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.project_store.unlink_conversation(conversation_id)
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
            yield WebEvent(event.type, event.payload)
            if event.type == "tool_call_result" or self._is_context_notice(event):
                yield WebEvent("status", self._context_window_payload(conversation_id))

        yield from self._drain_output_events()

    def _is_context_notice(self, event: Any) -> bool:
        if event.type != "notice":
            return False
        text = str(event.payload.get("text", ""))
        return "上下文" in text

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
                    current_assistant["text"] = text
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

    def _project_payload(self, project: ProjectRecord) -> dict[str, Any]:
        return {
            "id": project.id,
            "name": project.name,
            "path": project.path,
            "createdAt": project.created_at,
            "updatedAt": project.updated_at,
            "lastOpenedAt": project.last_opened_at,
            "conversationCount": self.project_store.conversation_count(project.id),
        }

    def _default_project_store_path(self, memory_store: Any) -> Path:
        path = getattr(memory_store, "path", None)
        if path is None:
            return PROJECT_DB
        return Path(path).with_name("web_projects.sqlite3")

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

        has_messages = bool(conversation_id and self.memory_store.message_count(conversation_id) > 0)
        budget = self.context.context_budget(
            conversation_id or "",
            extra_input_tokens=self._tool_schema_tokens(),
        )
        input_budget = budget.input_budget_tokens
        if not has_messages:
            return {
                **budget.to_payload(),
                "context": f"输入上下文：0 / {self._format_tokens(input_budget)}",
                "contextUsed": 0,
            }
        used = self.context.conversation_tokens(conversation_id)
        return {
            **budget.to_payload(),
            "context": f"输入上下文：{self._format_tokens(used)} / {self._format_tokens(input_budget)}",
            "contextUsed": used,
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
