"""Context engine for building model input from conversation history.

This module is the core of the context system, responsible for:
1. Persisting messages via the configured MemoryStore backend
2. Converting messages to model input format
3. Token budget estimation
4. Context compression (extensible)
5. Dynamic system prompt (Skills, Retrieved Memory, Conversation Summary)

Data flow:
    Write: User/Assistant messages -> add_*_message() -> MemoryStore
    Read:  MemoryStore -> build_model_input() -> Model input format

Message format conversion:
    StoredMessage (memory backend)
        -> _context_messages()
    ContextMessage (internal, with extra field)
        -> to_model_input()
    dict (model input, e.g. {"role": "user", "content": "..."})

Usage:
    from agent_runtime.context import ContextEngine
    from agent_runtime.memory import MemoryStore

    store = MemoryStore()
    context = ContextEngine(store)

    context.add_user_message("conv-1", "Hello")
    context.add_assistant_message("conv-1", "Hi!")

    model_input = context.build_model_input("conv-1")
    # [
    #   {"role": "system", "content": "You are Agent Runtime..."},
    #   {"role": "user", "content": "Hello"},
    #   {"role": "assistant", "content": "Hi!"}
    # ]
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.context.compression import ContextCompressor
from agent_runtime.context.tokens import (
    ContextBudget,
    DEFAULT_COMPACT_THRESHOLD_RATIO,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    TokenCounter,
    TokenCounterProtocol,
)
from agent_runtime.memory.long_term import LongTermMemory, LongTermMemoryProtocol
from agent_runtime.memory.store import MemoryStore, MemoryStoreProtocol, StoredMessage


CONTEXT_MESSAGE_ROLES = {"user", "assistant", "tool", "system"}
EMPTY_MEMORY_TEXT = "No memories stored yet."


# System prompt template with dynamic placeholders:
# - {tools}: Filled by create_agent() or defaulted by _system_message()
# - {skills}: Injected by SkillRegistry.apply_to_system_prompt()
# - {retrieved_memory}: Filled by LongTermMemory.read()
# - {conversation_summary}: Optional inline summary placeholder
SYSTEM_PROMPT_TEMPLATE = """You are Agent Runtime, a local web coding agent. Answer in Chinese. Be concise, accurate, and actionable.

# Tools
{tools}

# Skills
Use available tools when they help. After receiving tool results, answer the user directly.

{skills}

# Retrieved Memory
{retrieved_memory}

# Conversation Summary
{conversation_summary}"""


@dataclass(slots=True)
class ContextMessage:
    """Internal message representation with support for OpenAI native fields.

    The extra field carries additional keys like tool_calls, tool_call_id, etc.
    to_model_input() expands extra into the top-level dict.

    Attributes:
        role: Message role ("user" / "assistant" / "tool" / "system")
        content: Message content text
        extra: Additional fields (e.g. tool_calls, tool_call_id)

    Example:
        msg = ContextMessage(
            role="assistant",
            content="",
            extra={"tool_calls": [{"id": "call_1", "type": "function", ...}]}
        )
        msg.to_model_input()
        # {"role": "assistant", "content": "", "tool_calls": [...]}
    """

    role: str
    content: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_model_input(self) -> dict[str, Any]:
        """Convert to model input format by expanding extra fields."""
        return {"role": self.role, "content": self.content, **self.extra}


class ContextEngine:
    """Context engine that builds model input from conversation history.

    Args:
        store: Message storage backend
        system_prompt: System prompt template with {skills}, {retrieved_memory}, {conversation_summary} placeholders
        long_term_memory: Long-term memory for cross-conversation knowledge retrieval
        context_window_tokens: Model context window size in tokens
        reserved_output_tokens: Tokens reserved for model output
        token_counter: Token counter implementation
        compressor: Context compressor implementation
    """

    def __init__(
        self,
        store: MemoryStoreProtocol | None = None,
        *,
        system_prompt: str | None = None,
        long_term_memory: LongTermMemoryProtocol | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
        safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
        compact_threshold_ratio: float = DEFAULT_COMPACT_THRESHOLD_RATIO,
        recent_turns: int = 6,
        token_counter: TokenCounterProtocol | None = None,
        compressor: ContextCompressor | None = None,
        on_compress: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.system_prompt = system_prompt or SYSTEM_PROMPT_TEMPLATE
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.context_window_tokens = context_window_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.compact_threshold_ratio = compact_threshold_ratio
        self.recent_turns = max(1, recent_turns)
        self.token_counter = token_counter or TokenCounter()
        self.compressor = compressor
        self.on_compress = on_compress
        self._message_cache: dict[
            str,
            tuple[tuple[int, int], list[ContextMessage]],
        ] = {}
        self._memory_cache: tuple[object | None, str] | None = None

    def add_user_message(self, conversation_id: str, text: str) -> None:
        """Add a user message to storage."""
        self._append(conversation_id, "user", text)

    def add_assistant_message(
        self,
        conversation_id: str,
        text: str,
        reasoning: str = "",
        steps: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add an assistant message to storage.

        Stores as JSON if reasoning/steps/tool_calls are present,
        otherwise stores as plain text.

        Args:
            conversation_id: Conversation ID
            text: Assistant reply text (can be empty for tool-only responses)
            reasoning: Chain-of-thought reasoning
            steps: Execution steps
            tool_calls: Tool calls in format [{"id": "...", "name": "...", "arguments": {...}}]
        """
        if not (reasoning or steps or tool_calls):
            self._append(conversation_id, "assistant", text)
            return

        self._append(
            conversation_id,
            "assistant",
            self._json_dumps(
                {
                    "text": text,
                    "reasoning": reasoning,
                    "steps": steps or [],
                    "tool_calls": tool_calls or [],
                }
            ),
        )

    def add_tool_result(
        self,
        conversation_id: str,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        call_id: str = "",
    ) -> None:
        """Add a tool execution result to storage.

        Args:
            conversation_id: Conversation ID
            name: Tool name
            arguments: Tool call arguments
            result: Tool execution result
            call_id: Tool call ID for linking to assistant's tool_calls
        """
        self._append(
            conversation_id,
            "tool",
            self._json_dumps(
                {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                }
            ),
        )

    def build_model_input(
        self,
        conversation_id: str,
        *,
        extra_input_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for model input.

        Pipeline:
        1. Load all messages from SQLite
        2. Convert to ContextMessage (handle tool call format)
        3. Compress if over budget
        4. Serialize + insert system message

        Args:
            conversation_id: Conversation ID

        Returns:
            Model input messages, e.g.:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool", "content": "...", "tool_call_id": "..."}
            ]
        """
        messages = self._load_context_messages(conversation_id)
        # Step 3: Compress if over budget
        messages = self._compress_if_needed(
            conversation_id,
            messages,
            extra_input_tokens=extra_input_tokens,
        )
        messages = self._repair_tool_call_sequence(conversation_id, messages)
        # Step 4: Serialize + insert system message
        return [
            self._system_message(conversation_id),
            *[message.to_model_input() for message in messages],
        ]

    def estimate_model_input_tokens(self, conversation_id: str) -> int:
        """Estimate token count for the current model input."""
        return self.context_budget(conversation_id).used_input_tokens

    def conversation_tokens(self, conversation_id: str) -> int:
        """Return token count for conversation messages only (excluding system prompt)."""
        messages = self._load_context_messages(conversation_id)
        return self._message_token_count(messages)

    def context_budget(
        self,
        conversation_id: str,
        *,
        extra_input_tokens: int = 0,
    ) -> ContextBudget:
        """Return the current input-budget usage for a conversation."""
        messages = [
            self._coerce_context_message(message)
            for message in self.build_model_input(
                conversation_id,
                extra_input_tokens=extra_input_tokens,
            )
        ]
        return self._base_budget().with_used(
            self._message_token_count(messages) + extra_input_tokens
        )

    def clear(self, conversation_id: str) -> None:
        """Clear all messages for a conversation."""
        self.store.clear_conversation(conversation_id)
        self._message_cache.pop(conversation_id, None)

    def _append(self, conversation_id: str, role: str, text: str) -> None:
        """Append a message to storage."""
        value = text.strip()
        if value:
            self.store.append_message(conversation_id, role, value)
            self._message_cache.pop(conversation_id, None)

    def _system_message(self, conversation_id: str) -> dict[str, str]:
        """Build the system message with dynamic placeholders filled.

        Dynamically fills:
        - {skills}: Pre-processed by SkillRegistry.apply_to_system_prompt()
        - {retrieved_memory}: From LongTermMemory.read() (first 200 lines)
        - {conversation_summary}: Cleared unless a custom system prompt pre-fills it
        """
        content = self.system_prompt
        record = self.store.get_conversation(conversation_id) if conversation_id else None
        has_summary_placeholder = "{conversation_summary}" in content
        summary = record.summary if record and record.summary else EMPTY_MEMORY_TEXT
        replacements = {
            "{tools}": "No tools are currently available.",
            "{retrieved_memory}": self._retrieved_memory_text(conversation_id),
            "{conversation_summary}": summary,
            "{skills}": "",
        }
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        if record and record.summary and not has_summary_placeholder:
            content = f"{content}\n\n# Conversation Summary\n{record.summary}"

        # Collapse 3+ consecutive newlines into 2
        content = re.sub(r'\n{3,}', '\n\n', content)

        return {"role": "system", "content": content.strip()}

    def _retrieved_memory_text(self, conversation_id: str) -> str:
        if conversation_id:
            snapshot = self.store.ensure_memory_snapshot(
                conversation_id,
                self._long_term_memory_text(),
            )
            return snapshot or EMPTY_MEMORY_TEXT
        return self._long_term_memory_text()

    def _long_term_memory_text(self) -> str:
        if not self.long_term_memory:
            return EMPTY_MEMORY_TEXT
        version = getattr(self.long_term_memory, "version", None)
        if version is not None and self._memory_cache is not None and self._memory_cache[0] == version:
            return self._memory_cache[1]
        value = self.long_term_memory.read() or EMPTY_MEMORY_TEXT
        if version is not None:
            self._memory_cache = (version, value)
        return value

    def _load_context_messages(self, conversation_id: str) -> list[ContextMessage]:
        version = self.store.conversation_version(conversation_id)
        cached = self._message_cache.get(conversation_id)
        if cached is not None and cached[0] == version:
            return list(cached[1])

        messages: list[ContextMessage] = []
        for message in self.store.list_messages(conversation_id):
            if message.role in CONTEXT_MESSAGE_ROLES and message.content:
                messages.extend(self._context_messages(message))
        self._message_cache[conversation_id] = (version, list(messages))
        return messages

    def _context_messages(self, message: StoredMessage) -> list[ContextMessage]:
        """Convert a StoredMessage to ContextMessage list.

        Dispatches to role-specific handlers:
        - tool -> _tool_messages()
        - assistant -> _assistant_messages()
        - user -> direct conversion
        """
        if message.role == "tool":
            return self._tool_messages(message.content)
        if message.role == "assistant":
            return self._assistant_messages(message.content)
        return [ContextMessage(role=message.role, content=message.content)]

    def _assistant_messages(self, content: str) -> list[ContextMessage]:
        """Parse assistant message content.

        Supports two formats:
        1. Plain text -> ContextMessage with content
        2. JSON with text + tool_calls -> ContextMessage with extra
        """
        payload = self._json_object(content)
        if payload is None:
            return [ContextMessage(role="assistant", content=content)]

        text = str(payload.get("text", ""))
        tool_calls = payload.get("tool_calls")

        if tool_calls:
            return [ContextMessage(
                role="assistant",
                content=text,
                extra={"tool_calls": self._normalize_tool_calls(tool_calls)},
            )]

        return [ContextMessage(role="assistant", content=text)]

    def _normalize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        """Convert stored tool_calls to OpenAI native format.

        Storage format:
            [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Beijing"}}]
        Native format:
            [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'}}]
        """
        if not isinstance(tool_calls, list):
            return []
        return [
            self._normalize_tool_call(tool_call)
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ]

    def _normalize_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            tool_type = tool_call.get("type", "function")
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            tool_type = "function"

        return {
            "id": tool_call.get("id", ""),
            "type": tool_type,
            "function": {
                "name": name,
                "arguments": self._arguments_json(arguments),
            },
        }

    def _tool_messages(self, content: str) -> list[ContextMessage]:
        """Parse tool result message content.

        Extracts call_id and result from stored JSON,
        generates native tool message format.
        """
        payload = self._json_object(content)
        if payload is None:
            return [ContextMessage(role="tool", content=content, extra={"tool_call_id": "legacy"})]

        call_id = str(payload.get("call_id") or "tool-call")
        result = payload.get("result")

        return [ContextMessage(
            role="tool",
            content=self._json_dumps(result, indent=2),
            extra={"tool_call_id": call_id},
        )]

    def _compress_if_needed(
        self,
        conversation_id: str,
        messages: list[ContextMessage],
        *,
        extra_input_tokens: int = 0,
    ) -> list[ContextMessage]:
        """Compress messages if they exceed the token budget.

        If the compressor emits a summary, replace stored history with the
        summary plus the returned recent messages.
        """
        if (
            self._request_token_count(
                conversation_id,
                messages,
                extra_input_tokens,
            )
            <= self._compact_threshold_tokens()
        ):
            return messages

        older_messages, recent_messages = self._split_for_compaction(messages)
        if older_messages and self.compressor is not None:
            if self.on_compress:
                self.on_compress("start")
            record = self.store.get_conversation(conversation_id)
            summary_budget = max(256, self._input_token_budget() // 4)
            result = self.compressor.compress(
                conversation_id=conversation_id,
                messages=older_messages,
                target_tokens=summary_budget,
                previous_summary=record.summary if record else "",
            )
            if result.compressed and result.summary:
                self.store.update_conversation_summary(
                    conversation_id,
                    result.summary,
                )
                self._message_cache.pop(conversation_id, None)
            if self.on_compress:
                self.on_compress("done")

        compacted_messages = recent_messages
        if (
            self._request_token_count(
                conversation_id,
                compacted_messages,
                extra_input_tokens,
            )
            > self._compact_threshold_tokens()
        ):
            compacted_messages = self._force_truncate_messages(
                conversation_id,
                compacted_messages,
                extra_input_tokens=extra_input_tokens,
            )

        if compacted_messages != messages:
            self.store.replace_messages(
                conversation_id,
                self._storage_messages(compacted_messages),
            )
            self._message_cache.pop(conversation_id, None)
            return compacted_messages

        return compacted_messages

    def _coerce_context_message(self, message: Any) -> ContextMessage:
        """Accept ContextMessage-like objects or dicts from custom compressors."""
        if isinstance(message, ContextMessage):
            return message
        if isinstance(message, dict):
            return ContextMessage(
                role=str(message.get("role", "user")),
                content=str(message.get("content", "")),
                extra={
                    str(key): value
                    for key, value in message.items()
                    if key not in {"role", "content"}
                },
            )
        return ContextMessage(
            role=str(getattr(message, "role", "user")),
            content=str(getattr(message, "content", "")),
            extra=dict(getattr(message, "extra", {}) or {}),
        )

    def _storage_messages(
        self,
        messages: list[ContextMessage],
    ) -> list[tuple[str, str]]:
        """Serialize context messages back to MemoryStore's storage format."""
        return [(message.role, self._storage_content(message)) for message in messages]

    def _storage_content(self, message: ContextMessage) -> str:
        if message.role == "assistant" and message.extra.get("tool_calls"):
            return self._json_dumps(
                {
                    "text": message.content,
                    "reasoning": "",
                    "steps": [],
                    "tool_calls": self._denormalize_tool_calls(
                        message.extra.get("tool_calls"),
                    ),
                }
            )

        if message.role == "tool":
            return self._json_dumps(
                {
                    "call_id": message.extra.get("tool_call_id", ""),
                    "name": message.extra.get("name", ""),
                    "arguments": message.extra.get("arguments", {}),
                    "result": self._json_value(message.content),
                }
            )

        return message.content

    def _denormalize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []

        denormalized: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            denormalized.append(
                {
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": self._json_object(function.get("arguments", "")) or {},
                }
            )
        return denormalized

    def _split_for_compaction(
        self,
        messages: list[ContextMessage],
    ) -> tuple[list[ContextMessage], list[ContextMessage]]:
        turns = self._message_turns(messages)
        if len(turns) <= self.recent_turns:
            return [], messages
        older_turns = turns[: -self.recent_turns]
        recent_turns = turns[-self.recent_turns :]
        return self._flatten(older_turns), self._flatten(recent_turns)

    def _message_turns(
        self,
        messages: list[ContextMessage],
    ) -> list[list[ContextMessage]]:
        turns: list[list[ContextMessage]] = []
        current: list[ContextMessage] = []
        for message in messages:
            if message.role == "user" and current:
                turns.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            turns.append(current)
        return turns

    def _force_truncate_messages(
        self,
        conversation_id: str,
        messages: list[ContextMessage],
        *,
        extra_input_tokens: int = 0,
    ) -> list[ContextMessage]:
        if not messages:
            return []
        kept_groups: list[list[ContextMessage]] = []
        target = max(
            1,
            self._input_token_budget()
            - self._system_token_count(conversation_id)
            - extra_input_tokens,
        )
        groups = self._atomic_message_groups(messages)
        for group in reversed(groups):
            candidate_groups = [group, *kept_groups]
            candidate = self._flatten(candidate_groups)
            if kept_groups and self._message_token_count(candidate) > target:
                continue
            kept_groups = candidate_groups
        if not kept_groups:
            kept_groups = [groups[-1]]
        return self._flatten(kept_groups)

    def _repair_tool_call_sequence(
        self,
        conversation_id: str,
        messages: list[ContextMessage],
    ) -> list[ContextMessage]:
        repaired: list[ContextMessage] = []
        changed = False
        index = 0
        while index < len(messages):
            message = messages[index]
            required_ids = self._assistant_tool_call_ids(message)
            if not required_ids:
                repaired.append(message)
                index += 1
                continue

            group = [message]
            observed_ids: set[str] = set()
            next_index = index + 1
            while next_index < len(messages) and messages[next_index].role == "tool":
                tool_message = messages[next_index]
                group.append(tool_message)
                tool_call_id = str(tool_message.extra.get("tool_call_id", ""))
                if tool_call_id in required_ids:
                    observed_ids.add(tool_call_id)
                next_index += 1

            if len(group) == 1 and next_index == len(messages):
                repaired.extend(group)
                index = next_index
                continue
            if required_ids.issubset(observed_ids):
                repaired.extend(group)
            else:
                changed = True
                if message.content.strip():
                    repaired.append(ContextMessage(role="assistant", content=message.content))
            index = next_index

        if changed:
            self.store.replace_messages(
                conversation_id,
                self._storage_messages(repaired),
            )
            self._message_cache.pop(conversation_id, None)
        return repaired

    def _atomic_message_groups(
        self,
        messages: list[ContextMessage],
    ) -> list[list[ContextMessage]]:
        groups: list[list[ContextMessage]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            required_ids = self._assistant_tool_call_ids(message)
            if not required_ids:
                groups.append([message])
                index += 1
                continue

            group = [message]
            observed_ids: set[str] = set()
            next_index = index + 1
            while next_index < len(messages) and messages[next_index].role == "tool":
                tool_message = messages[next_index]
                group.append(tool_message)
                tool_call_id = str(tool_message.extra.get("tool_call_id", ""))
                if tool_call_id in required_ids:
                    observed_ids.add(tool_call_id)
                next_index += 1
                if required_ids.issubset(observed_ids):
                    break
            groups.append(group)
            index = next_index
        return groups

    def _assistant_tool_call_ids(self, message: ContextMessage) -> set[str]:
        if message.role != "assistant":
            return set()
        tool_calls = message.extra.get("tool_calls")
        if not isinstance(tool_calls, list):
            return set()
        return {
            str(tool_call.get("id", ""))
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and str(tool_call.get("id", ""))
        }

    def _flatten(
        self,
        groups: list[list[ContextMessage]],
    ) -> list[ContextMessage]:
        return [message for group in groups for message in group]

    def _arguments_json(self, arguments: Any) -> str:
        return arguments if isinstance(arguments, str) else self._json_dumps(arguments)

    def _json_dumps(self, value: Any, *, indent: int | None = None) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )

    def _json_object(self, value: Any) -> dict[str, Any] | None:
        parsed = self._json_value(value)
        return parsed if isinstance(parsed, dict) else None

    def _json_value(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _input_token_budget(self) -> int:
        """Calculate input token budget: context_window - reserved_output - safety."""
        return self._base_budget().input_budget_tokens

    def _compact_threshold_tokens(self) -> int:
        return self._base_budget().compact_threshold_tokens

    def _request_token_count(
        self,
        conversation_id: str,
        messages: list[ContextMessage],
        extra_input_tokens: int,
    ) -> int:
        return self._input_token_count(conversation_id, messages) + extra_input_tokens

    def _input_token_count(
        self,
        conversation_id: str,
        messages: list[ContextMessage],
    ) -> int:
        """Calculate total token count including system prompt."""
        return self._system_token_count(conversation_id) + self._message_token_count(
            messages
        )

    def _system_token_count(self, conversation_id: str) -> int:
        return self.token_counter.count_message(
            ContextMessage(
                role="system",
                content=self._system_message(conversation_id)["content"],
            )
        )

    def _message_token_count(self, messages: list[ContextMessage]) -> int:
        return sum(self.token_counter.count_message(m) for m in messages)

    def _base_budget(self) -> ContextBudget:
        input_budget = max(
            1,
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )
        ratio = min(1.0, max(0.1, self.compact_threshold_ratio))
        return ContextBudget(
            context_window_tokens=self.context_window_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
            input_budget_tokens=input_budget,
            compact_threshold_tokens=max(1, int(input_budget * ratio)),
        )
