# Agent Runtime

Agent Runtime is a local Python agent runtime with a browser UI. It is designed
for experimenting with Agent + Tool + Skill workflows, context compression,
long-term memory, and OpenAI-compatible model providers.

The project is intentionally lightweight: the core runtime lives in Python, the
web UI is a React single-page app, and runtime state is stored locally.

## Features

- Browser-based chat UI with streaming responses.
- Multi-step agent loop with tool execution and observation feedback.
- OpenAI-compatible provider abstraction with Responses API and Chat
  Completions support.
- Tool registry with built-in shell, memory, and skill tools.
- Skill loading from `SKILL.md` files.
- SQLite-backed conversation memory and long-term memory support.
- Context compression with a single leading summary message.
- Large shell output compaction to avoid flooding the model context.
- Runtime logs for model requests, tool calls, context compression, and errors.
- Local project management from the web UI.

## Safety Notes

This runtime can expose local tools such as shell execution. Run it only in a
trusted local environment and review tool permissions before using it with
untrusted prompts or repositories.

Never commit a real `.env` file. Use `.env.example` as the template.

## Requirements

- Python 3.11+
- Node.js 18+ if you want to rebuild the web UI
- An OpenAI-compatible model endpoint

## Quick Start

```bash
git clone <repo-url>
cd agent
./scripts/setup.sh
```

Edit `.env` and set at least:

```env
API_KEY=your_api_key
BASE_URL=https://your-provider.example/v1
MODEL=your-model-name
```

Start the local web app:

```bash
./scripts/run.sh
```

Then open:

```text
http://127.0.0.1:8765
```

After `.env` is configured, you can also install and run in one command:

```bash
./scripts/dev.sh
```

Manual setup is also available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tokenizers]"
cp .env.example .env
python -m web.main --host 127.0.0.1 --port 8765
```

## Scripts

| Command | Purpose |
|---|---|
| `./scripts/setup.sh` | Create `.venv`, install Python dependencies, install/build the web UI when `npm` is available, and create `.env` if missing. |
| `./scripts/run.sh` | Start the local web server from `.venv`. |
| `./scripts/dev.sh` | Run setup and then start the server. |

`HOST`, `PORT`, `PYTHON`, and `VENV_DIR` can be overridden through environment
variables.

## Rebuilding the Web UI

The repository includes built static assets under `web/static` so the Python
entrypoint can serve the UI directly. Rebuild the frontend after editing files
under `web/src`:

```bash
cd web
npm install
npm run build
cd ..
```

## Configuration

Common environment variables:

| Variable | Description | Default |
|---|---|---|
| `API_KEY` | API key for the model provider | empty |
| `BASE_URL` | OpenAI-compatible API base URL | empty |
| `MODEL` | Model name | empty |
| `PROVIDER_API` | `auto`, `responses`, or `chat_completions` | `auto` |
| `MAX_RETRIES` | Provider retry count | `2` |
| `CONTEXT_WINDOW` | Total model context window in tokens | `32000` |
| `RESERVED_OUTPUT_TOKENS` | Tokens reserved for model output | `4000` |
| `CONTEXT_SAFETY_MARGIN` | Safety buffer for provider/tokenizer overhead | `1000` |
| `COMPACT_THRESHOLD_RATIO` | Ratio of input budget that triggers compression | `0.8` |
| `RECENT_TURNS` | Recent user turns kept before turn-level compression | `6` |
| `RAW_KEEP_RATIO` | Raw recent context kept during ratio compression | `0.6` |
| `MEMORY_BACKEND` | `sqlite` or `memory` | `sqlite` |
| `AGENT_RUNTIME_DATA_DIR` | Runtime data directory | `~/.agent-runtime` |
| `ENABLE_MEMORY_TOOLS` | Register memory tools | `true` |
| `ENABLE_SKILL_TOOLS` | Register skill tools | `true` |
| `ENABLE_SHELL_TOOL` | Register shell tool | `true` |
| `SHELL_TIMEOUT_SECONDS` | Shell command timeout | `30` |
| `SHELL_MAX_OUTPUT_CHARS` | Max shell output returned directly to model | `20000` |
| `SKILL_PATHS` | Extra skill directories, separated by OS path separator | empty |

## Runtime Data

By default, runtime state is stored under:

```text
~/.agent-runtime
```

This includes conversation storage, long-term memory, logs, and large tool
output artifacts. These files are local runtime data and should not be committed.

## Skills

Skills are loaded from directories containing a `SKILL.md` file. A minimal skill
looks like this:

```md
---
name: demo_echo_skill
description: A short description of what this skill helps the agent do.
triggers: [echo, demo]
---

# Demo Echo Skill

Describe when the agent should use this skill, what constraints apply, and what
tools or outputs are expected.
```

The built-in demo skill is available at:

```text
src/agent_runtime/skills/builtin/demo_echo_skill/SKILL.md
```

## Development

Run the Python test suite:

```bash
python -m pytest -q
```

Run focused tests while working on the agent loop or context engine:

```bash
python -m pytest tests/test_runtime.py tests/test_context.py -q
```

Rebuild the web UI:

```bash
cd web
npm run build
```

## Project Layout

```text
src/agent_runtime/
  agent.py                  Agent construction and high-level SDK API
  runtime/loop.py            Model/tool loop orchestration
  context/                   Context building, compression, token budgeting
  memory/                    Conversation and long-term memory stores
  providers/                 Provider abstraction and OpenAI-compatible clients
  skills/                    Skill manifests, loading, and registry
  tools/                     Tool registry and built-in tools
  logging.py                 Runtime log formatting

web/
  server.py                  FastAPI server
  session.py                 Browser session orchestration
  src/                       React frontend source
  static/                    Built frontend assets served by Python

tests/                       Unit and integration tests
```

## Current Status

This is an early local agent runtime for development and experimentation. The
core loop, tools, context compression, memory, skills, web UI, and tests are in
active development.
