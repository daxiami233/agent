# Agent Runtime

Local Python web agent runtime.

V1 scope:

- Browser-based interaction.
- Agent loop.
- Context engineering.
- SQLite-backed memory.
- Local skills.
- Tool registry.
- MCP tool adapter.
- Provider abstraction with an OpenAI SDK based provider.

This repository is currently scaffolded module by module. Runtime behavior will be implemented incrementally.

Provider environment variables:

```env
API_KEY=...
BASE_URL=...
MODEL=...
PROVIDER_API=auto
MAX_RETRIES=2
CONTEXT_WINDOW=128000
```

Provider capabilities:

- OpenAI SDK based generation: `OpenAIProvider.generate(...)`
- OpenAI SDK based streaming: `OpenAIProvider.stream(...)`
- `PROVIDER_API=auto`, `responses`, or `chat_completions`

Tools:

- `ToolRegistry` registers, lists, exports, and dispatches runtime tools.
- Built-in `weather` tool queries current weather for a requested location.

Web UI:

```bash
agent
```

The `agent` command starts a local FastAPI server and opens a React chat UI:

```bash
agent --host 127.0.0.1 --port 8765
```

The web UI uses an OpenAI-style chat layout with a left conversation list,
scrollable transcript, streaming assistant responses, reasoning/thinking
details, slash-command completion, and a bottom composer. It supports `/help`,
`/model`, and `/status`. If `CONTEXT_WINDOW` is configured, the status area
shows context-window usage and the configured size. Conversations are
persisted in `~/.agent-runtime/memory.sqlite3`, and each conversation sends its
own recent user/assistant message history back to the model.

While a response is streaming, the send button turns into a pause button. Click
it to cancel the current generation. Slash-command completion appears in a
floating command box, can be navigated with the up/down arrow keys, and Enter
executes the selected command.

Slash commands live in `web/commands/`; each command has its
own module.
