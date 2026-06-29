"""Public Agent SDK tests."""

from __future__ import annotations

import pytest

import agent_runtime
from agent_runtime import AgentResponse, AgentRuntimeConfig, create_agent
from agent_runtime.context import CompressionResult, ContextCompressor
from agent_runtime.mcp import MCPClientHost
from agent_runtime.memory import InMemoryMemoryStore, LongTermMemory, MemoryStore, SQLiteLongTermMemory
from agent_runtime.providers import ModelResponse, ModelStreamEvent, OpenAIProvider, ToolCall
from agent_runtime.skills import SkillManifest
from agent_runtime.tools import ToolSpec


class StreamingProvider:
    model = "fake-model"
    base_url = "https://example.test/v1"
    context_window_tokens = 128000

    def __init__(self) -> None:
        self.inputs = []
        self.tools = []

    def stream(self, input, **kwargs):
        self.inputs.append(input)
        self.tools.append(kwargs.get("tools"))
        yield ModelStreamEvent(type="reasoning_delta", delta="thinking")
        yield ModelStreamEvent(type="content_delta", delta="hello")
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                finish_reason="stop",
                usage={"prompt_tokens": 3, "completion_tokens": 2},
            ),
        )


class PermissionToolProvider:
    model = "fake-model"
    base_url = "https://example.test/v1"
    context_window_tokens = 128000

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, input, **kwargs):
        self.calls += 1
        yield ModelStreamEvent(
            type="finish",
            response=ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="write_file",
                        arguments={"path": "x.txt"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 4},
            ),
        )


class FixedTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)

    def count_message(self, message) -> int:
        return len(message.content)


class FixedCompressor(ContextCompressor):
    def compress(self, *, conversation_id, messages, target_tokens):
        return CompressionResult(summary="compressed", compressed=True)


def echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo arguments.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda arguments: arguments["text"],
    )


def test_create_agent_run_collects_response_and_writes_context(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
        ),
        provider=provider,
        tools=[echo_tool()],
        log_context=lambda _conversation_id, _model_input: None,
    )

    response = agent.run("hi", conversation_id="conversation-1")

    assert isinstance(response, AgentResponse)
    assert response.text == "hello"
    assert response.reasoning == "thinking"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert "echo" in {tool["function"]["name"] for tool in provider.tools[0]}
    assert agent.context.build_model_input("conversation-1")[1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_create_agent_accepts_direct_provider_settings(tmp_path):
    agent = create_agent(
        model="direct-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        config=AgentRuntimeConfig(data_dir=tmp_path, memory_backend="memory"),
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert isinstance(agent.provider, OpenAIProvider)
    assert agent.provider.model == "direct-model"
    assert agent.provider.api_key == "test-key"
    assert agent.provider.base_url == "https://example.test/v1"


def test_create_agent_direct_provider_settings_override_config(tmp_path):
    agent = create_agent(
        model="direct-model",
        api_key="direct-key",
        base_url="https://direct.example/v1",
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
            memory_backend="memory",
            model="config-model",
            api_key="config-key",
            base_url="https://config.example/v1",
        ),
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert isinstance(agent.provider, OpenAIProvider)
    assert agent.provider.model == "direct-model"
    assert agent.provider.api_key == "direct-key"
    assert agent.provider.base_url == "https://direct.example/v1"


def test_root_package_exports_common_sdk_types():
    assert agent_runtime.ToolSpec is ToolSpec
    assert agent_runtime.SkillManifest is SkillManifest
    assert agent_runtime.Provider is not None
    assert agent_runtime.PermissionPolicyProtocol is not None
    assert agent_runtime.PermissionRequest is not None
    assert callable(agent_runtime.tool)


def test_agent_runtime_config_rejects_invalid_memory_backend():
    with pytest.raises(ValueError, match='Invalid memory_backend="bad"'):
        AgentRuntimeConfig(memory_backend="bad")  # type: ignore[arg-type]


def test_agent_runtime_config_rejects_invalid_env_memory_backend(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "bad")

    with pytest.raises(ValueError, match='Invalid MEMORY_BACKEND="bad"'):
        AgentRuntimeConfig.from_env()


def test_create_agent_stream_emits_agent_events(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
        ),
        provider=provider,
        log_context=lambda _conversation_id, _model_input: None,
    )

    events = list(agent.stream("hi", conversation_id="conversation-1"))

    assert [event.type for event in events] == [
        "reasoning_delta",
        "assistant_start",
        "assistant_delta",
        "usage",
    ]


def test_agent_facade_resumes_pending_permission(tmp_path):
    executed = []
    provider = PermissionToolProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
            memory_backend="memory",
            include_memory_tools=False,
            include_skill_tools=False,
            include_shell_tool=False,
            include_apply_patch_tool=False,
        ),
        provider=provider,
        tools=[
            ToolSpec(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                handler=lambda arguments: executed.append(arguments) or {"ok": True},
                effects=["write"],
            )
        ],
        log_context=lambda _conversation_id, _model_input: None,
    )

    first_events = list(agent.stream("write", conversation_id="conversation-1"))
    permission_id = first_events[0].payload["permission_id"]
    approved_events = list(agent.resume_permission(permission_id, approved=True))

    assert provider.calls == 1
    assert executed == [{"path": "x.txt"}]
    assert [event.type for event in approved_events] == [
        "tool_call_start",
        "tool_call_result",
        "assistant_start",
        "assistant_delta",
        "usage",
    ]
    assert approved_events[-2].payload["text"] == "操作已完成。"


def test_create_agent_uses_configured_memory_store(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
        ),
        provider=provider,
        log_context=lambda _conversation_id, _model_input: None,
    )

    agent.run("hi", conversation_id="conversation-1")

    assert MemoryStore(tmp_path / "memory.sqlite3").message_count("conversation-1") == 2


def test_create_agent_accepts_advanced_runtime_injections(tmp_path):
    memory_store = InMemoryMemoryStore()
    long_term_memory = LongTermMemory()
    token_counter = FixedTokenCounter()
    compressor = FixedCompressor()

    agent = create_agent(
        config=AgentRuntimeConfig(data_dir=tmp_path, memory_backend="sqlite"),
        provider=StreamingProvider(),
        memory_store=memory_store,
        long_term_memory=long_term_memory,
        token_counter=token_counter,
        compressor=compressor,
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert agent.memory_store is memory_store
    assert agent.context.long_term_memory is long_term_memory
    assert agent.context.token_counter is token_counter
    assert agent.context.compressor is compressor


def test_create_agent_registers_mcp_host_tools(tmp_path):
    provider = StreamingProvider()
    mcp_tool = ToolSpec(
        name="mcp_echo",
        description="Echo through MCP.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda arguments: arguments["text"],
    )

    agent = create_agent(
        config=AgentRuntimeConfig(data_dir=tmp_path, memory_backend="memory"),
        provider=provider,
        mcp_host=MCPClientHost([mcp_tool]),
        log_context=lambda _conversation_id, _model_input: None,
    )
    agent.run("hi", conversation_id="conversation-1")

    tool_names = {schema["function"]["name"] for schema in provider.tools[0]}
    assert "mcp_echo" in tool_names
    assert agent.tool_registry.execute("mcp_echo", {"text": "ok"}) == "ok"


def test_create_agent_sqlite_backend_persists_long_term_memory(tmp_path):
    provider = StreamingProvider()
    config = AgentRuntimeConfig(data_dir=tmp_path, memory_backend="sqlite")
    agent = create_agent(
        config=config,
        provider=provider,
        log_context=lambda _conversation_id, _model_input: None,
    )
    agent.context.long_term_memory.append("remember tea")

    restored = create_agent(
        config=config,
        provider=StreamingProvider(),
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert restored.context.long_term_memory.read() == "remember tea"


def test_sqlite_long_term_memory_version_is_deterministic(tmp_path):
    memory = SQLiteLongTermMemory(tmp_path / "memory.sqlite3")
    memory.append("remember tea")

    assert isinstance(memory.version, int)
    assert memory.version == SQLiteLongTermMemory(tmp_path / "memory.sqlite3").version


def test_create_agent_memory_backend_is_process_local(tmp_path):
    config = AgentRuntimeConfig(data_dir=tmp_path, memory_backend="memory")
    agent = create_agent(
        config=config,
        provider=StreamingProvider(),
        log_context=lambda _conversation_id, _model_input: None,
    )
    agent.context.long_term_memory.append("remember tea")

    fresh = create_agent(
        config=config,
        provider=StreamingProvider(),
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert fresh.context.long_term_memory.read() == ""


def test_create_agent_passes_context_window_tokens_to_context_engine(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            context_window_tokens=12_000,
            data_dir=tmp_path,
        ),
        provider=provider,
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert agent.context.context_window_tokens == 12_000


def test_create_agent_passes_provider_timeout_to_agent_loop(tmp_path):
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
            provider_timeout_seconds=12,
        ),
        provider=StreamingProvider(),
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert agent.loop.model_timeout_seconds == 12


def test_create_agent_includes_working_directory_in_system_prompt(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    agent = create_agent(
        config=AgentRuntimeConfig(data_dir=tmp_path),
        provider=StreamingProvider(),
        working_directory=project_dir,
        log_context=lambda _conversation_id, _model_input: None,
    )

    assert "# Current Workspace" in agent.context.system_prompt
    assert f"Current working directory: {project_dir.resolve()}" in agent.context.system_prompt


def test_create_agent_registers_optional_runtime_tools(tmp_path):
    provider = StreamingProvider()
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")

    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
            include_memory_tools=True,
            include_skill_tools=True,
            include_shell_tool=True,
        ),
        provider=provider,
        skills=[
            SkillManifest(
                name="demo",
                description="Demo skill.",
                skill_dir=skill_dir,
            )
        ],
        log_context=lambda _conversation_id, _model_input: None,
    )

    agent.run("hi", conversation_id="conversation-1")
    tool_names = {
        schema["function"]["name"]
        for schema in provider.tools[0]
    }

    assert {
        "memory_read",
        "memory_search",
        "memory_append",
        "memory_replace",
        "skill_read",
        "skill_read_resource",
        "shell_command",
        "apply_patch",
    }.issubset(tool_names)


def test_create_agent_enables_core_local_tools_by_default(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        provider=provider,
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
        ),
        log_context=lambda _conversation_id, _model_input: None,
    )

    agent.run("hi", conversation_id="conversation-1")
    tool_names = {
        schema["function"]["name"]
        for schema in provider.tools[0]
    }

    assert {
        "memory_read",
        "memory_search",
        "memory_append",
        "memory_replace",
        "skill_read",
        "skill_read_resource",
        "shell_command",
        "apply_patch",
    }.issubset(tool_names)
    assert "weather" not in tool_names


def test_agent_add_skill_rerenders_prompt_without_duplicate_skill_sections(tmp_path):
    provider = StreamingProvider()
    agent = create_agent(
        config=AgentRuntimeConfig(
            data_dir=tmp_path,
        ),
        provider=provider,
        skills=[SkillManifest(name="first", description="First skill.")],
        log_context=lambda _conversation_id, _model_input: None,
    )

    agent.add_skill(SkillManifest(name="second", description="Second skill."))

    assert agent.context.system_prompt.count("# Available Skills") == 1
    assert "- first: First skill." in agent.context.system_prompt
    assert "- second: Second skill." in agent.context.system_prompt
