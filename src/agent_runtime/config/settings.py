"""Runtime configuration loaded from explicit values or environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

from agent_runtime.context import (
    DEFAULT_COMPACT_THRESHOLD_RATIO,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
)
from agent_runtime.memory.long_term import MAX_LINES as DEFAULT_LONG_TERM_MEMORY_LINES
from agent_runtime.tools.skills import DEFAULT_MAX_RESOURCE_BYTES


DEFAULT_DATA_DIR = Path.home() / ".agent-runtime"
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 60.0
DEFAULT_PROVIDER_MAX_RETRIES = 2
DEFAULT_COMPACT_SUMMARY_TOKENS = 1_200
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
DEFAULT_SHELL_MAX_OUTPUT_CHARS = 20_000

MemoryBackend = Literal["sqlite", "memory"]


@dataclass(slots=True)
class AgentRuntimeConfig:
    """Stable SDK-facing runtime configuration.

    Function:
        Describes normal runtime policy for ``create_agent``: provider settings,
        context-window budgeting, memory backend selection, default tool
        switches, skill paths, and local shell limits.

    Parameter guide:
        data_dir: Directory for runtime data such as the SQLite memory database.
        model/api_key/base_url/api_mode: Provider connection settings.
        provider_timeout_seconds: Default timeout for model requests.
        provider_max_retries: Default provider retry count.
        context_window_tokens: Full model context window.
        reserved_output_tokens: Tokens reserved for model output.
        safety_margin_tokens: Buffer for tokenizer/provider overhead.
        compact_threshold_ratio: Ratio of input budget that triggers compaction.
        recent_turns: Recent conversation turns kept verbatim during compaction.
        compact_summary_tokens: Maximum tokens used when creating summaries.
        memory_backend: ``"sqlite"`` for persistence or ``"memory"`` for
            process-local storage.
        long_term_memory_max_lines: Lines injected from long-term memory.
        include_*_tool: Whether to register built-in local tool groups.
        shell_*: Limits for the local shell tool.
        skill_paths: Directories scanned for ``SKILL.md`` manifests.
        skill_resource_max_bytes: Maximum bytes returned by skill resource tools.

    Usage:
        agent = create_agent()

        agent = create_agent(
            config=AgentRuntimeConfig(
                model="deepseek-v4-pro",
                base_url="https://example.test/v1",
                api_key="...",
                memory_backend="sqlite",
                context_window_tokens=128_000,
            )
        )

        test_agent = create_agent(
            config=AgentRuntimeConfig(memory_backend="memory"),
            provider=fake_provider,
        )
    """

    # Runtime data directory. SQLite memory uses data_dir / "memory.sqlite3".
    data_dir: Path = DEFAULT_DATA_DIR

    # Model name, for example deepseek-v4-pro or gpt-4.1.
    model: str | None = None

    # Model API key. Usually read from API_KEY in .env.
    api_key: str | None = None

    # OpenAI-compatible API base URL, for example https://api.openai.com/v1.
    base_url: str | None = None

    # Provider API mode: auto, responses, or chat_completions.
    api_mode: str | None = None

    # Default timeout for provider requests, in seconds.
    provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS

    # Default retry count passed to the provider client.
    provider_max_retries: int = DEFAULT_PROVIDER_MAX_RETRIES

    # Total model context window. Input budget subtracts output reserve and margin.
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS

    # Tokens reserved for model output and reasoning, not for input context.
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS

    # Safety margin for tokenizer differences, provider overhead, and tool schemas.
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS

    # Automatic compaction threshold as a ratio of the input budget.
    compact_threshold_ratio: float = DEFAULT_COMPACT_THRESHOLD_RATIO

    # Number of recent conversation turns kept verbatim during compaction.
    recent_turns: int = 6

    # Maximum tokens requested when summarizing older context.
    compact_summary_tokens: int = DEFAULT_COMPACT_SUMMARY_TOKENS

    # Memory backend: "sqlite" persists under data_dir, "memory" is process-local.
    memory_backend: MemoryBackend = "sqlite"

    # Number of long-term memory lines injected into the system prompt.
    long_term_memory_max_lines: int = DEFAULT_LONG_TERM_MEMORY_LINES

    # Whether to register memory_read/search/append/replace tools by default.
    include_memory_tools: bool = True

    # Whether to register skill_read and skill_read_resource tools by default.
    include_skill_tools: bool = True

    # Whether to register the local shell execution tool by default.
    include_shell_tool: bool = True

    # Skill directories. In .env, multiple paths use the OS path separator.
    skill_paths: list[str] = field(default_factory=list)

    # Maximum timeout for one shell command, in seconds.
    shell_timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS

    # Maximum stdout/stderr characters returned to the model from shell commands.
    shell_max_output_chars: int = DEFAULT_SHELL_MAX_OUTPUT_CHARS

    # Maximum bytes returned by skill_read and skill_read_resource.
    skill_resource_max_bytes: int = DEFAULT_MAX_RESOURCE_BYTES

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        if self.memory_backend not in {"sqlite", "memory"}:
            raise ValueError(
                f'Invalid memory_backend="{self.memory_backend}". '
                'Must be "sqlite" (persistent) or "memory" (in-process). '
                'Example: AgentRuntimeConfig(memory_backend="sqlite")'
            )
        self.provider_timeout_seconds = max(1.0, float(self.provider_timeout_seconds))
        self.provider_max_retries = max(0, int(self.provider_max_retries))
        self.context_window_tokens = max(1, int(self.context_window_tokens))
        self.reserved_output_tokens = max(0, int(self.reserved_output_tokens))
        self.safety_margin_tokens = max(0, int(self.safety_margin_tokens))
        self.compact_threshold_ratio = min(
            1.0,
            max(0.1, float(self.compact_threshold_ratio)),
        )
        self.recent_turns = max(1, int(self.recent_turns))
        self.compact_summary_tokens = max(256, int(self.compact_summary_tokens))
        self.long_term_memory_max_lines = max(1, int(self.long_term_memory_max_lines))
        self.shell_timeout_seconds = max(1, int(self.shell_timeout_seconds))
        self.shell_max_output_chars = max(1, int(self.shell_max_output_chars))
        self.skill_resource_max_bytes = max(1, int(self.skill_resource_max_bytes))

    @classmethod
    def from_env(cls) -> "AgentRuntimeConfig":
        """Build config from .env/process environment variables.

        Example:
            API_KEY=...
            BASE_URL=https://example.test/v1
            MODEL=deepseek-v4-pro
            MEMORY_BACKEND=sqlite
            CONTEXT_WINDOW=128000
        """

        return cls(
            data_dir=Path(os.getenv("AGENT_RUNTIME_DATA_DIR", DEFAULT_DATA_DIR)).expanduser(),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model=os.getenv("MODEL"),
            api_mode=os.getenv("PROVIDER_API"),
            provider_timeout_seconds=_env_float(
                "PROVIDER_TIMEOUT_SECONDS",
                DEFAULT_PROVIDER_TIMEOUT_SECONDS,
            ),
            provider_max_retries=_env_int("MAX_RETRIES", DEFAULT_PROVIDER_MAX_RETRIES),
            context_window_tokens=_env_int("CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW_TOKENS),
            reserved_output_tokens=_env_int("RESERVED_OUTPUT_TOKENS", DEFAULT_RESERVED_OUTPUT_TOKENS),
            safety_margin_tokens=_env_int("CONTEXT_SAFETY_MARGIN", DEFAULT_SAFETY_MARGIN_TOKENS),
            compact_threshold_ratio=_env_float(
                "COMPACT_THRESHOLD_RATIO",
                DEFAULT_COMPACT_THRESHOLD_RATIO,
            ),
            recent_turns=_env_int("RECENT_TURNS", 6),
            compact_summary_tokens=_env_int(
                "COMPACT_SUMMARY_TOKENS",
                DEFAULT_COMPACT_SUMMARY_TOKENS,
            ),
            memory_backend=_env_memory_backend("MEMORY_BACKEND", "sqlite"),
            long_term_memory_max_lines=_env_int(
                "LONG_TERM_MEMORY_MAX_LINES",
                DEFAULT_LONG_TERM_MEMORY_LINES,
            ),
            include_memory_tools=_env_bool("ENABLE_MEMORY_TOOLS", True),
            include_skill_tools=_env_bool("ENABLE_SKILL_TOOLS", True),
            include_shell_tool=_env_bool("ENABLE_SHELL_TOOL", True),
            skill_paths=_env_list("SKILL_PATHS"),
            shell_timeout_seconds=_env_int(
                "SHELL_TIMEOUT_SECONDS",
                DEFAULT_SHELL_TIMEOUT_SECONDS,
            ),
            shell_max_output_chars=_env_int(
                "SHELL_MAX_OUTPUT_CHARS",
                DEFAULT_SHELL_MAX_OUTPUT_CHARS,
            ),
            skill_resource_max_bytes=_env_int(
                "SKILL_RESOURCE_MAX_BYTES",
                DEFAULT_MAX_RESOURCE_BYTES,
            ),
        )


RuntimeSettings = AgentRuntimeConfig


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return []
    return [item.strip() for item in value.split(os.pathsep) if item.strip()]


def _env_memory_backend(name: str, default: MemoryBackend) -> MemoryBackend:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"sqlite", "memory"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(
        f'Invalid {name}="{value}". '
        'Must be "sqlite" (persistent) or "memory" (in-process). '
        'Example: MEMORY_BACKEND=sqlite'
    )
