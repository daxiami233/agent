"""Provider tests using real Responses API calls."""

import pytest
from dotenv import load_dotenv
from openai import APIStatusError, InternalServerError
from httpx import Request, Response

from agent_runtime.providers import ModelConfig, OpenAIProvider, ProviderError


def provider_from_real_env() -> OpenAIProvider:
    load_dotenv()
    try:
        return OpenAIProvider.from_env()
    except ProviderError as exc:
        pytest.skip(f"API_KEY, BASE_URL, and MODEL are not configured: {exc}")


def test_provider_stores_required_settings():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1/",
        model="model-a",
        context_window_tokens=128000,
    )

    assert provider.api_key == "secret"
    assert provider.base_url == "https://example.test/v1"
    assert provider.model == "model-a"
    assert provider.api_mode == "auto"
    assert provider.context_window_tokens == 128000


def test_provider_loads_required_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")

    provider = OpenAIProvider.from_env()

    assert provider.api_key == "secret"
    assert provider.base_url == "https://example.test/v1"
    assert provider.model == "model-a"


def test_provider_loads_max_retries_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")
    monkeypatch.setenv("MAX_RETRIES", "4")

    provider = OpenAIProvider.from_env()

    assert provider.max_retries == 4


def test_provider_loads_context_window_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")
    monkeypatch.setenv("CONTEXT_WINDOW", "128000")

    provider = OpenAIProvider.from_env()

    assert provider.context_window_tokens == 128000


def test_provider_loads_api_mode_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")
    monkeypatch.setenv("PROVIDER_API", "chat")

    provider = OpenAIProvider.from_env()

    assert provider.api_mode == "chat_completions"


def test_provider_accepts_chatcompletions_alias(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")
    monkeypatch.setenv("PROVIDER_API", "chatcompletions")

    provider = OpenAIProvider.from_env()

    assert provider.api_mode == "chat_completions"


def test_provider_rejects_invalid_context_window(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL", "model-a")
    monkeypatch.setenv("CONTEXT_WINDOW", "0")

    try:
        OpenAIProvider.from_env()
    except ProviderError as exc:
        assert "CONTEXT_WINDOW" in str(exc)
    else:
        raise AssertionError("ProviderError was not raised")


def test_provider_extracts_reasoning_delta_variants():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )

    events = list(
        provider._parse_stream_event(
            {
                "type": "response.reasoning.delta",
                "reasoning_content": "thinking text",
            }
        )
    )

    assert len(events) == 1
    assert events[0].type == "response.reasoning.delta"
    assert events[0].delta == "thinking text"


def test_chat_backend_extracts_reasoning_and_content_deltas():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )

    assert provider._chat_backend.reasoning_delta(
        {"reasoning_content": "thinking text"}
    ) == "thinking text"
    messages = provider._chat_backend.input_to_messages("hello", "be brief")

    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_auto_mode_falls_back_to_chat_completions_on_missing_responses_endpoint():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )
    calls = []

    def fake_generate_with_mode(
        mode,
        input,
        *,
        instructions,
        tools,
        model_config,
    ):
        calls.append(mode)
        if mode == "responses":
            raise APIStatusError(
                "Not Found",
                response=Response(404, request=Request("POST", "https://example.test")),
                body={"error": "Not Found"},
            )
        return provider._chat_backend.parse_response(
            {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 3},
            }
        )

    provider._generate_with_mode = fake_generate_with_mode

    response = provider.generate("hi")

    assert response.content == "ok"
    assert calls == ["responses", "chat_completions"]
    assert provider._resolved_api_mode == "chat_completions"
    assert provider.current_api_mode() == "chat_completions"


def test_backend_payloads_preserve_zero_timeout_and_use_extra_body():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )
    config = ModelConfig(timeout_seconds=0, extra_body={"top_k": 20})

    chat_payload = provider._chat_backend.build_payload(
        "hi",
        model=provider.model,
        instructions=None,
        tools=[],
        model_config=config,
        timeout=provider._request_timeout(config.timeout_seconds),
    )
    responses_payload = provider._responses_backend.build_payload(
        "hi",
        model=provider.model,
        instructions=None,
        tools=[],
        model_config=config,
        timeout=provider._request_timeout(config.timeout_seconds),
    )

    assert chat_payload["timeout"] == 0
    assert responses_payload["timeout"] == 0
    assert chat_payload["extra_body"] == {"top_k": 20}
    assert responses_payload["extra_body"] == {"top_k": 20}


def test_responses_backend_flattens_chat_style_tool_schema():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )
    config = ModelConfig()

    payload = provider._responses_backend.build_payload(
        "hi",
        model=provider.model,
        instructions=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        model_config=config,
        timeout=provider._request_timeout(config.timeout_seconds),
    )

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "查天气",
            "parameters": {"type": "object"},
        }
    ]


def test_chat_backend_preserves_native_tool_messages():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )

    messages = provider._chat_backend.input_to_messages(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"location": "Shanghai"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "weather",
                "content": '{"temperature_c": 25}',
            },
        ],
        None,
    )

    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"location": "Shanghai"}',
                },
            }
        ],
    }
    assert messages[2] == {
        "role": "tool",
        "content": '{"temperature_c": 25}',
        "tool_call_id": "call-1",
    }


def test_responses_backend_converts_native_tool_messages():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )

    payload = provider._responses_backend.build_payload(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"location": "Shanghai"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"temperature_c": 25}',
            },
        ],
        model=provider.model,
        instructions=None,
        tools=[],
        model_config=ModelConfig(),
        timeout=None,
    )

    assert payload["input"] == [
        {"role": "user", "content": "weather"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "weather",
            "arguments": '{"location": "Shanghai"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"temperature_c": 25}',
        },
    ]


def test_chat_backend_collects_streaming_tool_calls():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
    )
    collected = {}

    provider._chat_backend.collect_tool_call_deltas(
        collected,
        [
            {
                "index": 0,
                "id": "call-1",
                "function": {"name": "weather", "arguments": '{"location"'},
            }
        ],
    )
    provider._chat_backend.collect_tool_call_deltas(
        collected,
        [{"index": 0, "function": {"arguments": ': "Shanghai"}'}}],
    )

    calls = provider._chat_backend.tool_calls_from_deltas(collected)

    assert len(calls) == 1
    assert calls[0].id == "call-1"
    assert calls[0].name == "weather"
    assert calls[0].arguments == {"location": "Shanghai"}


def test_auto_mode_falls_back_to_chat_for_local_responses_502():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="http://127.0.0.1:8000/v1",
        model="model-a",
    )

    error = InternalServerError(
        "Error code: 502",
        response=Response(502, request=Request("POST", "http://127.0.0.1:8000/v1/responses")),
        body=None,
    )

    assert provider._should_fallback_to_chat(error)


def test_local_provider_client_ignores_environment_proxies():
    provider = OpenAIProvider(
        api_key="secret",
        base_url="http://127.0.0.1:8000/v1",
        model="model-a",
    )

    client = provider._get_client()

    assert client._client.trust_env is False


def test_generate_uses_responses_api_real_network():
    provider = provider_from_real_env()

    response = provider.generate(
        "Reply with exactly: agent-runtime-ok",
        model_config=ModelConfig(temperature=0, max_tokens=200, timeout_seconds=30),
    )

    assert response.content
    assert isinstance(response.content, str)
    assert response.raw


def test_stream_uses_responses_api_real_network():
    provider = provider_from_real_env()

    events = list(
        provider.stream(
            "Reply with exactly: stream-ok",
            model_config=ModelConfig(temperature=0, max_tokens=200, timeout_seconds=30),
        )
    )

    content = "".join(event.delta or "" for event in events if event.type == "content_delta")
    assert content or any(event.type == "finish" for event in events)


def test_multimodal_input_is_accepted_by_responses_shape_real_network():
    provider = provider_from_real_env()
    input_payload = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Reply with exactly: multimodal-shape-ok"}
            ],
        }
    ]

    response = provider.generate(
        input_payload,
        model_config=ModelConfig(temperature=0, max_tokens=200, timeout_seconds=30),
    )

    assert response.content


def test_from_env_requires_exact_three_variables(monkeypatch):
    for key in ("API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("agent_runtime.providers.openai.load_dotenv", lambda: None)

    try:
        OpenAIProvider.from_env()
    except ProviderError as exc:
        message = str(exc)
        assert "API_KEY" in message
        assert "BASE_URL" in message
        assert "MODEL" in message
    else:
        raise AssertionError("ProviderError was not raised")


def test_generate_requires_model():
    provider = OpenAIProvider(api_key="secret", base_url="https://example.test/v1", model="")

    try:
        provider.generate("hi")
    except ProviderError as exc:
        assert "Model name is required" in str(exc)
    else:
        raise AssertionError("ProviderError was not raised")
