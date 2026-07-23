# Cogtrix Agent

A modular AI assistant with 53 built-in tools, multi-provider LLM support, and intelligent memory management.

---

## What Is Cogtrix?

Cogtrix is an **interactive command-line AI assistant** that connects to large language models (LLMs) and extends them with tools — web search, file operations, code execution, deep reasoning, and more. You type a question or task; the agent reasons about it, calls tools as needed, and delivers the result.

**Works with:** [Ollama](https://ollama.com/) (local, free), OpenAI, Anthropic Claude, Google Gemini, and any OpenAI-compatible API (Groq, Together, vLLM, xAI, etc.)

**Highlights:**

- 53 built-in tools across 6 search providers, file I/O, shell, Python, HTTP, NLP, WhatsApp and Telegram messaging, and more
- Three memory modes optimized for conversation, coding, or strategic reasoning — with hybrid memory (rolling summary + semantic recall)
- Deep reasoning engine (Tree-of-Thought with iterative reflection) via `/think`
- Task delegation across multiple LLM models via `/delegate`
- Safety layer with human confirmation for sensitive operations
- Headless assistant mode — run as a WhatsApp/Telegram daemon with per-chat context isolation and shared knowledge
- Works out of the box with zero configuration if Ollama is running

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/NorthlandPositronics/Cogtrix.git
cd Cogtrix
uv sync            # or: pip install -r requirements.txt
```

Install optional provider and feature extras:

```bash
uv pip install "cogtrix[anthropic]"    # Anthropic Claude
uv pip install "cogtrix[google]"       # Google Gemini
uv pip install "cogtrix[api]"          # REST API server
uv pip install "cogtrix[mcp]"          # MCP server support
uv pip install "cogtrix[search]"       # Tavily, Exa, Brave, SerpAPI search
```

> **Prerequisite:** Python 3.13.x and [uv](https://docs.astral.sh/uv/) (recommended) or pip.

### 2. Start an LLM

The default backend is **Ollama** — local, free, no API key.

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen3:8b
```

That's it. No config file needed — Cogtrix connects to `localhost:11434` automatically.

> **Using OpenAI instead?** `export OPENAI_API_KEY="sk-..." && python cogtrix.py -p openai`
> **Using another provider?** See [Providers Guide](docs/PROVIDERS.md).

### 3. Run

```bash
uv run python cogtrix.py        # if you used uv sync
python cogtrix.py                # if you used pip install
```

### 4. Try it out

```
You: What is the capital of New Zealand?
You: /think search for top 10 news affecting the stock market
You: /tools
You: /help
You: /quit
```

---

## What Can I Do with Cogtrix?

Here are some things you can try right away:

**Research and questions:**
```
You: What were the biggest AI breakthroughs in 2025?
You: Compare PostgreSQL and MongoDB for a real-time analytics workload
```

**File operations and coding:**
```
You: Read main.py and suggest improvements
You: Write a Python function that validates email addresses and save it to utils.py
```

**Deep reasoning (uses the `/think` command):**
```
You: /think Design a microservices architecture for an e-commerce platform
You: /think Should we use Kubernetes or Docker Swarm? Budget is $500/month, team of 3
```

**Task delegation (splits work across multiple LLM models):**
```
You: /delegate Compare Python, Rust, and Go for web backend development
You: /delegate Research top 10 AI companies and their market cap
```

**Multi-step workflows:**
```
You: Search the web for the latest Python 3.13 features, summarize them, and write the summary to python313.md
```

The agent decides which tools to call, chains them together, and delivers a complete answer.

---

## Common Launch Examples

```bash
python cogtrix.py                            # Ollama default
python cogtrix.py -p openai -m gpt-4.1       # OpenAI
python cogtrix.py -M code                    # Code development memory mode
python cogtrix.py -M reasoning               # Strategic planning mode
python cogtrix.py --prompt "Summarize X"     # Single prompt, then exit
python cogtrix.py --prompt "Query" -o out.md # Save response to file
python cogtrix.py -m fast                    # Use a model alias from config
python cogtrix.py -y                         # Auto-approve all tool confirmations
python cogtrix.py -c ~/my-config.yaml        # Use a specific config file
python cogtrix.py --debug                    # Full debug logging
python cogtrix.py --assistant                  # Headless messaging daemon (WhatsApp/Telegram)
python cogtrix.py --assistant --debug          # Assistant mode with debug logging
```

---

## Configuration

Cogtrix works with zero configuration when Ollama is running on localhost. For anything more, create a config file in your project directory or home directory. Both JSON (`.cogtrix.json`) and YAML (`.cogtrix.yaml`) formats are supported:

**YAML** (`.cogtrix.yaml` — recommended, easier to read):

```yaml
provider: my-server

providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
    model: qwen3:8b
  openai:
    type: openai
    model: gpt-4.1-mini

services:
  tavily:
    api_key: "tvly-..."

models:
  fast: my-server/qwen3:8b
  smart: openai/gpt-4.1
```

**JSON** (`.cogtrix.json`):

```json
{
  "provider": "my-server",
  "providers": {
    "my-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "qwen3:8b"
    },
    "openai": {
      "type": "openai",
      "model": "gpt-4.1-mini"
    }
  },
  "services": {
    "tavily": { "api_key": "tvly-..." }
  },
  "models": {
    "fast": "my-server/qwen3:8b",
    "smart": "openai/gpt-4.1"
  }
}
```

**Configuration is loaded from** (highest priority first):

1. Command-line flags (`-p`, `-m`, `-M`, `-c`, etc.)
2. Environment variables (`COGTRIX_PROVIDER`, `COGTRIX_MODEL`, `COGTRIX_OLLAMA`, `OPENAI_API_KEY`, etc.)
3. Config file — pass a specific path with `-c ~/my-config.yaml`, or Cogtrix searches for `.cogtrix.json` / `.cogtrix.yaml` / `.cogtrix.yml` in the current directory, home directory, and `~/.config/cogtrix/`
4. Built-in defaults — Ollama on localhost, conversation mode, 25-message history

Full reference: **[Configuration Guide](docs/CONFIGURATION.md)**

---

## Interactive Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help [cmd]` | `/h`, `/?` | List commands or show detailed help |
| `/info` | `/i` | Show session info (provider, model, mode) |
| `/tools [search\|load\|enable\|disable]` | `/t`, `/tool` | List, search, load, or manage tools |
| `/think <task>` | `/T` | Run deep Tree-of-Thought reasoning |
| `/delegate <task>` | `/d` | Force task delegation across models |
| `/mode [name]` | `/M` | Show / switch memory mode |
| `/model [name]` | `/m` | Show / switch LLM model |
| `/provider [name]` | `/p` | Show / switch LLM provider |
| `/session [id]` | `/s` | Show / switch session |
| `/setup` | — | Launch interactive setup wizard |
| `/approve` | `/a` | Toggle tool auto-approval (also: `-y` at startup) |
| `/paste` | `/P` | Enter multi-line paste mode |
| `/clear` | `/c` | Clear conversation history |
| `/optimizer [prompt]` | `/o` | Toggle prompt optimizer / force-optimize a prompt |
| `/debug` | `/D` | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/mcp [restart]` | — | List or restart MCP server connections |
| `/quit` | `/exit`, `/q` | Exit |

Arrow keys, Home/End, and input history work out of the box (via `readline`).

---

## Built-in Tools (53)

| Category | Tools |
|----------|-------|
| **Search** (10) | DuckDuckGo (free), Tavily, Exa, Brave, Google, SerpAPI |
| **Files** (5) | `read_file`, `write_file`, `append_file`, `list_directory`, `file_info` |
| **System** (2) | `execute_shell_command`, `execute_python` |
| **Text & NLP** (10) | word count, find/replace, URLs, emails, compare, sentiment, summarize, keywords |
| **JSON & Math** (6) | parse, format, query, extract, convert, calculate |
| **Web** (2) | `http_get`, `http_post` |
| **Date & Weather** (4) | datetime, timezone, parse date, weather |
| **WhatsApp** (4) | `whatsapp_send`, `whatsapp_send_image`, `whatsapp_check`, `whatsapp_contacts` |
| **Telegram** (4) | `telegram_send`, `telegram_send_photo`, `telegram_check`, `telegram_contacts` |
| **Reasoning** (3) | `deep_think`, `delegate_task`, `delegate_parallel` |
| **Knowledge** (1) | `query_knowledge_base` (RAG) |

DuckDuckGo search works immediately with no setup. Premium search providers (Tavily, Exa, etc.) activate automatically when their API key is configured. WhatsApp messaging requires a [Waha](https://waha.devlike.pro/) container -- see the **[WhatsApp Guide](docs/WHATSAPP_GUIDE.md)**. Telegram requires a bot token from [@BotFather](https://t.me/BotFather) -- see the **[Telegram Guide](docs/TELEGRAM_GUIDE.md)**. See also [Search Providers](docs/CONFIGURATION.md#services-section) for details.

**On-demand tool loading:** At startup you'll see something like `Tools: [██████████░░] 41 on demand (3 unavailable)`. All tools start in an **on-demand pool** — the agent requests only the tools it needs for the current task via an internal `request_tools` meta-tool. This keeps the initial prompt lean and context usage efficient. Tools whose API keys are missing are marked as unavailable. You don't need to manage any of this — the agent handles it automatically. See [Tool Loading](docs/CONFIGURATION.md#tool-loading) for details.

Full parameter reference: **[Tools Reference](docs/TOOLS_REFERENCE.md)**

---

## Memory Modes

| Mode | Best for | Working memory |
|------|----------|----------------|
| `conversation` (default) | General chat, Q&A, research | 25 messages |
| `code` | Programming, debugging | 30 messages + file/error tracking |
| `reasoning` | Planning, architecture decisions | 30 messages + goal/decision tracking |

All modes include **hybrid memory**: older messages are automatically compressed into a rolling summary, and (when an embedding provider is available) stored for semantic recall — so the agent retains awareness of the full conversation even after messages leave the sliding window. Token-aware trimming ensures the context always fits the model's context window.

Switch at startup (`-M code`) or at runtime (`/mode code`). See **[Memory Modes](docs/MEMORY_MODES.md)**.

---

## Docker

```bash
docker pull ghcr.io/northlandpositronics/cogtrix:latest
docker run -it --network host ghcr.io/northlandpositronics/cogtrix:latest
```

The container includes all optional packages — search providers (Tavily, Exa, SerpAPI), Anthropic Claude, Google Gemini, MCP server support, and scientific computing (NumPy, SciPy). Use `--network host` so it can reach a local Ollama server.

**Passing configuration via environment variables:**

```bash
docker run -it --network host \
  -e COGTRIX_OLLAMA="192.168.1.100" \
  -e TAVILY_API_KEY="tvly-..." \
  -e OPENWEATHER_API_KEY="abc123" \
  ghcr.io/northlandpositronics/cogtrix:latest
```

**Mounting a config file:**

```bash
docker run -it --network host \
  -v "$HOME/.cogtrix.yaml:/app/.cogtrix.yaml:ro" \
  ghcr.io/northlandpositronics/cogtrix:latest
```

**Persisting session history across container restarts:**

```bash
docker run -it --network host \
  -v cogtrix-data:/app/data \
  ghcr.io/northlandpositronics/cogtrix:latest
```

**Running the API server in Docker:**

```bash
docker run -p 8000:8000 \
  -e COGTRIX_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  ghcr.io/northlandpositronics/cogtrix:latest api
```

Pass `api` as the final argument to start the FastAPI server instead of the interactive CLI.

---

## API Server

Cogtrix includes a REST + WebSocket API server built with FastAPI. It exposes 65 REST endpoints and 2 WebSocket streams, powering the React web frontend and enabling programmatic access from any HTTP client.

### Starting the API server

**Directly with uvicorn:**

```bash
# Generate a strong secret (required)
export COGTRIX_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"

uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**With dev auto-reload:**

```bash
uvicorn src.api.app:app --reload --port 8000
```

Once running, interactive API docs are available at `http://localhost:8000/api/v1/docs` (Swagger UI) and `http://localhost:8000/api/v1/redoc` (ReDoc).

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COGTRIX_JWT_SECRET` | Yes | — | JWT signing secret, minimum 32 characters. Generate with: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `COGTRIX_API_HOST` | No | `0.0.0.0` | Bind host for uvicorn (Docker entrypoint) |
| `COGTRIX_API_PORT` | No | `8000` | Bind port for uvicorn (Docker entrypoint) |
| `COGTRIX_API_WORKERS` | No | `1` | Number of uvicorn worker processes (Docker entrypoint) |
| `COGTRIX_DB_URL` | No | `sqlite+aiosqlite:///./data/api/cogtrix.db` | SQLAlchemy async database URL. Use `postgresql+asyncpg://` in production |
| `COGTRIX_CORS_ORIGINS` | No | Localhost + `app.cogtrix.ai` | Comma-separated list of allowed CORS origins |

### Authentication

The API uses JWT bearer tokens (`Authorization: Bearer <token>`). The first registered user is automatically granted the `admin` role. API keys (prefix `cgx_live_`) are available for programmatic/CI access and are accepted on the same `Authorization` header.

WebSocket connections that cannot set custom headers may pass the token as a `?token=<jwt>` query parameter.

### What the API covers

| Route group | Endpoints | Description |
|-------------|-----------|-------------|
| `POST /api/v1/auth/*` | 8 | Registration, login, token refresh, logout, profile, API key management |
| `/api/v1/sessions/*` | 5 | Create, list, get, update, delete sessions |
| `/api/v1/sessions/{id}/messages/*` | 3 | Send messages, list history, clear history |
| `/api/v1/sessions/{id}/memory/*` | 3 | Get memory state, switch mode, clear memory |
| `/api/v1/sessions/{id}/tools/*` | 4 | List, load, enable, disable tools |
| `/api/v1/config/*` | 12 | Read/write config, provider management, model aliases |
| `/api/v1/assistant/*` | 16 | Start/stop assistant mode, channel management, phonebook |
| `/api/v1/rag/*` | 5 | Upload documents, list, delete, query knowledge base |
| `/api/v1/mcp/*` | 5 | List servers, connect, disconnect, restart, list tools |
| `/api/v1/system/*` | 2 | Server info, shutdown |
| `/api/v1/health` | 2 | Liveness and readiness probes |
| `ws://host/ws/v1/sessions/{id}` | WS | Streaming agent turns, tool confirmation, token events |
| `ws://host/ws/v1/logs` | WS | Live log stream (admin only) |

Full reference: **[API Reference](docs/api/openapi.yaml)** | **[Client Contract](docs/api/client-contract.md)** | **[WebSocket Protocol](docs/api/websocket-protocol.md)**

---

## Quick Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "Connection refused" on startup | Ollama isn't running | `ollama serve` in another terminal |
| "Model not found" | Model not pulled yet | `ollama pull qwen3:8b` |
| No search results | DuckDuckGo rate-limited | Wait a moment and retry, or add a Tavily/Brave API key |
| Empty or garbled response | Model too small or OOM | Try a smaller model: `-m qwen3:8b` |
| Tool not appearing in `/tools` | Missing API key for that tool | Set the key in env or config (tools auto-hide when unconfigured) |
| "41 on demand (3 unavailable)" — what does that mean? | Normal — on-demand tool loading | 41 tools are ready for the agent to request; 3 are hidden because their API keys aren't configured. See [Tool Loading](docs/CONFIGURATION.md#tool-loading) |
| "Invalid API key" (OpenAI) | Key missing or expired | `export OPENAI_API_KEY="sk-..."` |
| Not sure if config is valid | Typo or wrong structure | `python cogtrix.py --check-config` |

For detailed debugging, run with `--debug` (logs every LLM call, tool input/output, and context info to `cogtrix.log`).

---

## Documentation

| Guide | What you'll learn |
|-------|-------------------|
| **[Configuration](docs/CONFIGURATION.md)** | Every config option, environment variables, search providers |
| **[Providers](docs/PROVIDERS.md)** | Step-by-step: Ollama, OpenAI, Groq, Together, vLLM |
| **[Memory Modes](docs/MEMORY_MODES.md)** | Conversation, code, and reasoning modes + hybrid memory (summary + recall) |
| **[Tools Reference](docs/TOOLS_REFERENCE.md)** | All 53 tools with parameters and examples |
| **[WhatsApp Guide](docs/WHATSAPP_GUIDE.md)** | Use Cogtrix as a WhatsApp assistant (with Docker Compose) |
| **[Telegram Guide](docs/TELEGRAM_GUIDE.md)** | Use Cogtrix as a Telegram assistant via a bot |
| **[Assistant Mode](docs/CONFIGURATION.md#assistant-mode)** | Run Cogtrix as a headless WhatsApp/Telegram messaging daemon |
| **[Deep Think](docs/DEEPTHINK.md)** | Tree-of-Thought reasoning engine internals |
| **[RAG Guide](docs/RAG_GUIDE.md)** | Build a knowledge base from your documents |
| **[API Reference](docs/api/openapi.yaml)** | OpenAPI 3.1 schema for the REST + WebSocket API |
| **[Client Contract](docs/api/client-contract.md)** | TypeScript API client contract with full type definitions |
| **[WebSocket Protocol](docs/api/websocket-protocol.md)** | Streaming session protocol and message types |
| **[Architecture](docs/ARCHITECTURE.md)** | System design, data flow, components |
| **[Development](docs/DEVELOPMENT.md)** | Add tools, memory modes, slash commands; testing |

**New here?** You're in the right place. Follow the [Quick Start](#quick-start) above to get running in under 5 minutes. Then:

- Want to connect OpenAI, Groq, or another LLM? See [Providers](docs/PROVIDERS.md).
- Want to customize settings, add search API keys, or set up messaging? See [Configuration](docs/CONFIGURATION.md).
- Want to know what all 53 tools do? See [Tools Reference](docs/TOOLS_REFERENCE.md).

---

## Testing

```bash
uv run pytest tests/ -v
```

---

## License

Copyright 2025-2026 Northland Positronics (FZE). All rights reserved.

This software is released under the **Cogtrix Source-Available License 1.0**. See [LICENSE](LICENSE) for full terms.
