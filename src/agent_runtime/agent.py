"""Public Agent facade for SDK users."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agent_runtime.context import (
    ContextCompressor,
    ContextEngine,
    ModelContextCompressor,
    TokenCounter,
    TokenCounterProtocol,
)
from agent_runtime.config.settings import AgentRuntimeConfig
from agent_runtime.mcp import MCPClientHost
from agent_runtime.memory import (
    ConversationRecord,
    InMemoryMemoryStore,
    LongTermMemory,
    LongTermMemoryProtocol,
    MemoryStore,
    MemoryStoreProtocol,
    SQLiteLongTermMemory,
    SQLiteMemoryStore,
)
from agent_runtime.providers import OpenAIProvider, Provider
from agent_runtime.runtime import AgentEvent, AgentLoop
from agent_runtime.skills import SkillManifest, SkillRegistry, load_skills
from agent_runtime.tools import (
    ToolRegistry,
    ToolSpec,
    memory_tools,
    shell_command_tool,
    skill_tools,
)


@dataclass(slots=True)
class AgentResponse:
    """Non-streaming response returned by ``Agent.run``."""

    text: str
    conversation_id: str
    reasoning: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    notices: list[dict[str, Any]] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)


class Agent:
    """SDK-facing agent composed from provider, context, memory, tools, skills, and MCP."""

    def __init__(
        self,
        *,
        provider: Provider,
        context: ContextEngine | None = None,
        memory_store: MemoryStoreProtocol | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        mcp_host: MCPClientHost | None = None,
        log_context: Callable[[str, list[dict[str, Any]]], None] | None = None,
        base_system_prompt: str | None = None,
        model_timeout_seconds: float = 60,
    ) -> None:
        """Create an agent from already-built runtime components.

        Args:
            provider: Model provider used to generate responses. This is the only
                required dependency and can be any implementation of ``Provider``.
            context: Optional context builder. Pass this when callers need full
                control over system prompt assembly, history trimming, summaries,
                retrieved memory, or tool result formatting.
            memory_store: Optional conversation storage. Used to create a default
                context when ``context`` is not provided. If both are provided,
                ``memory_store`` must be the same object as ``context.store``.
            tool_registry: Optional registry containing callable tools. If omitted,
                the agent starts with built-in tools.
            skill_registry: Optional registry containing skill metadata. Skills are
                currently rendered into the system prompt and can later drive more
                advanced skill selection.
            mcp_host: Optional MCP host. Any tools exposed by this host are
                registered into ``tool_registry`` during initialization.
            log_context: Optional callback that receives the final model input for
                debugging or observability. Pass a no-op to suppress default logs.
            base_system_prompt: Raw prompt template used when skills are re-rendered
                after ``add_skill``. This should not include already-rendered skill
                metadata.
        """

        if (
            context is not None
            and memory_store is not None
            and context.store is not memory_store
        ):
            raise ValueError(
                "Pass either context or memory_store, or pass a context "
                "that uses the same memory_store."
            )

        self.provider = provider
        self.skill_registry = skill_registry or SkillRegistry()
        if base_system_prompt is not None:
            self._base_system_prompt = base_system_prompt
        elif context is not None:
            self._base_system_prompt = context.system_prompt
        else:
            self._base_system_prompt = None
        self.mcp_host = mcp_host or MCPClientHost()
        self.tool_registry = tool_registry or ToolRegistry()
        self.mcp_host.register_tools(self.tool_registry)
        self.context = context or ContextEngine(
            memory_store or MemoryStore(),
            system_prompt=self.skill_registry.apply_to_system_prompt(),
        )
        self.memory_store = self.context.store
        self.loop = AgentLoop(
            provider=self.provider,
            context=self.context,
            tool_registry=self.tool_registry,
            log_context=log_context,
            model_timeout_seconds=model_timeout_seconds,
        )

    def stream(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        reasoning_enabled: bool = True,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[AgentEvent]:
        """Stream one user turn as provider-neutral agent events."""

        yield from self.loop.run_user_turn(
            conversation_id or self.new_conversation_id(),
            message,
            reasoning_enabled=reasoning_enabled,
            is_cancelled=is_cancelled,
        )

    def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        reasoning_enabled: bool = True,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AgentResponse:
        """Run one user turn and collect streamed events into a response object."""

        resolved_conversation_id = conversation_id or self.new_conversation_id()
        events: list[AgentEvent] = []
        assistant_parts: list[str] = []
        reasoning_parts: list[str] = []
        steps: dict[str, dict[str, Any]] = {}
        notices: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}

        for event in self.stream(
            message,
            conversation_id=resolved_conversation_id,
            reasoning_enabled=reasoning_enabled,
            is_cancelled=is_cancelled,
        ):
            events.append(event)
            if event.type == "assistant_delta":
                assistant_parts.append(str(event.payload.get("text", "")))
            elif event.type == "reasoning_delta":
                reasoning_parts.append(str(event.payload.get("text", "")))
            elif event.type == "tool_call_start":
                step_id = str(event.payload.get("id", ""))
                if step_id:
                    steps[step_id] = {"type": "tool", **event.payload}
            elif event.type == "tool_call_result":
                step_id = str(event.payload.get("id", ""))
                if step_id:
                    steps[step_id] = {**steps.get(step_id, {"type": "tool"}), **event.payload}
            elif event.type == "notice":
                notices.append(event.payload)
            elif event.type == "usage":
                payload_usage = event.payload.get("usage")
                usage = payload_usage if isinstance(payload_usage, dict) else {}

        return AgentResponse(
            text="".join(assistant_parts),
            reasoning="".join(reasoning_parts),
            steps=list(steps.values()),
            usage=usage,
            notices=notices,
            events=events,
            conversation_id=resolved_conversation_id,
        )

    def add_tool(self, tool: ToolSpec) -> None:
        """Register a tool after the agent has been created."""

        self.tool_registry.register(tool)

    def add_skill(self, skill: SkillManifest) -> None:
        """Register skill metadata for future context construction."""

        self.skill_registry.register(skill)
        self.context.system_prompt = self.skill_registry.apply_to_system_prompt(
            self._base_system_prompt,
        )

    def new_conversation_id(self) -> str:
        """Return a new conversation id suitable for callers that do not manage ids."""

        return str(uuid4())

    def clear_conversation(self, conversation_id: str) -> None:
        self.context.clear(conversation_id)

    def list_conversations(self) -> list[ConversationRecord]:
        return self.memory_store.list_conversations()


def create_agent(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config: AgentRuntimeConfig | None = None,
    provider: Provider | None = None,
    tools: list[ToolSpec] | None = None,
    skills: list[SkillManifest] | None = None,
    system_prompt: str | None = None,
    log_context: Callable[[str, list[dict[str, Any]]], None] | None = None,
    memory_store: MemoryStoreProtocol | None = None,
    long_term_memory: LongTermMemoryProtocol | None = None,
    token_counter: TokenCounterProtocol | None = None,
    compressor: ContextCompressor | None = None,
    mcp_host: MCPClientHost | None = None,
) -> Agent:
    """Create a local agent from config, optional provider, tools, and skills.

    Function:
        Build all runtime components needed by ``Agent``: provider, memory
        stores, skill registry, tool registry, context engine, and compressor.

    Parameters:
        model/api_key/base_url: Convenience provider settings. These override
            values from config/env and are enough for the common direct setup.
        config: Stable runtime strategy. If omitted, values are read from env.
        provider: Optional model provider. If omitted, OpenAIProvider is built
            from config.
        tools: User-defined tools appended to built-in local tools.
        skills: User-defined skill manifests appended to config.skill_paths.
        system_prompt: Optional base system prompt template.
        log_context: Optional callback receiving final model input.
        memory_store: Advanced override for conversation storage.
        long_term_memory: Advanced override for cross-conversation memory.
        token_counter: Advanced override for token budgeting.
        compressor: Advanced override for context summarization.
        mcp_host: Optional MCP host that contributes tools to the registry.

    Example:
        agent = create_agent()
        agent = create_agent(
            model="deepseek-v4-pro",
            api_key="...",
            base_url="https://example.test/v1",
        )
        agent = create_agent(config=AgentRuntimeConfig(memory_backend="memory"))
        agent = create_agent(provider=fake_provider, tools=[my_tool])
    """

    runtime_config = _config_with_provider_overrides(
        config or AgentRuntimeConfig.from_env(),
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    resolved_provider = provider or _provider_from_config(runtime_config)
    try:
        setattr(
            resolved_provider,
            "context_window_tokens",
            runtime_config.context_window_tokens,
        )
    except AttributeError:
        pass

    loaded_skills = _load_config_skills(runtime_config)
    resolved_skill_registry = SkillRegistry([*(skills or []), *loaded_skills])
    resolved_memory_store = memory_store or _memory_store_from_config(runtime_config)
    resolved_long_term_memory = long_term_memory or _long_term_memory_from_config(
        runtime_config,
        resolved_memory_store,
    )
    resolved_tool_registry = ToolRegistry()
    if runtime_config.include_memory_tools:
        for tool in memory_tools(resolved_long_term_memory):
            resolved_tool_registry.register(tool)
    if runtime_config.include_skill_tools:
        for tool in skill_tools(
            resolved_skill_registry,
            max_bytes=runtime_config.skill_resource_max_bytes,
        ):
            resolved_tool_registry.register(tool)
    if runtime_config.include_shell_tool:
        resolved_tool_registry.register(
            shell_command_tool(
                timeout_seconds=runtime_config.shell_timeout_seconds,
                max_output_chars=runtime_config.shell_max_output_chars,
            )
        )
    for tool in tools or []:
        _register_if_absent(resolved_tool_registry, tool)

    counter_model = str(
        getattr(resolved_provider, "model", runtime_config.model or "")
    ) or None
    resolved_token_counter = token_counter or TokenCounter(model=counter_model)
    resolved_compressor = compressor or ModelContextCompressor(
        resolved_provider,
        max_summary_tokens=runtime_config.compact_summary_tokens,
        timeout_seconds=runtime_config.provider_timeout_seconds,
    )
    resolved_context = ContextEngine(
        resolved_memory_store,
        system_prompt=resolved_skill_registry.apply_to_system_prompt(system_prompt),
        long_term_memory=resolved_long_term_memory,
        context_window_tokens=runtime_config.context_window_tokens,
        reserved_output_tokens=runtime_config.reserved_output_tokens,
        safety_margin_tokens=runtime_config.safety_margin_tokens,
        compact_threshold_ratio=runtime_config.compact_threshold_ratio,
        recent_turns=runtime_config.recent_turns,
        token_counter=resolved_token_counter,
        compressor=resolved_compressor,
    )

    return Agent(
        provider=resolved_provider,
        context=resolved_context,
        tool_registry=resolved_tool_registry,
        skill_registry=resolved_skill_registry,
        mcp_host=mcp_host,
        log_context=log_context,
        base_system_prompt=system_prompt or "",
        model_timeout_seconds=runtime_config.provider_timeout_seconds,
    )


def _provider_from_config(config: AgentRuntimeConfig) -> Provider:
    if config.api_key and config.base_url and config.model:
        return OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            api_mode=config.api_mode or "auto",
            context_window_tokens=config.context_window_tokens,
            timeout_seconds=config.provider_timeout_seconds,
            max_retries=config.provider_max_retries,
        )
    return OpenAIProvider.from_env()


def _config_with_provider_overrides(
    config: AgentRuntimeConfig,
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> AgentRuntimeConfig:
    overrides = {
        key: value
        for key, value in {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }.items()
        if value is not None
    }
    return replace(config, **overrides) if overrides else config


def _memory_store_from_config(config: AgentRuntimeConfig) -> MemoryStoreProtocol:
    if config.memory_backend == "memory":
        return InMemoryMemoryStore()
    return SQLiteMemoryStore(config.data_dir / "memory.sqlite3")


def _long_term_memory_from_config(
    config: AgentRuntimeConfig,
    memory_store: MemoryStoreProtocol,
) -> LongTermMemoryProtocol:
    if config.memory_backend == "memory":
        return LongTermMemory(max_lines=config.long_term_memory_max_lines)
    if isinstance(memory_store, SQLiteMemoryStore):
        return SQLiteLongTermMemory(
            memory_store.path,
            max_lines=config.long_term_memory_max_lines,
        )
    return LongTermMemory(max_lines=config.long_term_memory_max_lines)


def _load_config_skills(config: AgentRuntimeConfig) -> list[SkillManifest]:
    return [
        skill
        for path in config.skill_paths
        for skill in load_skills(path)
    ]


def _register_if_absent(registry: ToolRegistry, tool: ToolSpec) -> None:
    if not registry.has(tool.name):
        registry.register(tool)
