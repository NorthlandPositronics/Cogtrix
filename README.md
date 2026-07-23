# Cogtrix

**An AI agent that lives on your laptop.** Talk to a local LLM, give it tools, let it do real work — read your files, search the web, write code, run shell commands, ping Slack. No API key required to start. Bring your own when you want to plug in GPT‑4, Claude, Gemini, or DeepSeek.

```
You> Find the five most-cited deep-learning papers from arXiv in 2025,
     summarize each in two sentences, and save the list to papers.md.

Cogtrix> http_get("api.semanticscholar.org/graph/v1/paper/search?...")
         http_get("https://arxiv.org/abs/2501.…")
         http_get("https://arxiv.org/abs/2502.…")
         http_get("https://arxiv.org/abs/2503.…")
         write_file("papers.md", "# Top arXiv DL papers 2025\n…")
         Done. 5 papers summarized — see papers.md.
```

That's one prompt, five tool calls, one file on disk. Cogtrix knows to query Semantic Scholar for citation counts (arXiv itself doesn't publish them), pulls each paper's abstract from arXiv, and chains the steps on its own.

---

## 🚀 Try it in 60 seconds

```bash
git clone https://github.com/NorthlandPositronics/Cogtrix.git
cd Cogtrix
uv sync
ollama pull qwen3:8b            # any GGUF model works
uv run python cogtrix.py
```

That's the whole install. No accounts, no keys, no SaaS. Cogtrix finds Ollama on `localhost:11434` by itself and loads 67 tools into the agent's toolbox.

Prefer cloud LLMs? `export OPENAI_API_KEY="sk-..." && uv run python cogtrix.py -m gpt-4.1`. Or any of Anthropic, Google, DeepSeek, Groq, Together, vLLM, xAI — anything that speaks the OpenAI API.

> **Need:** Python 3.13.x and [uv](https://docs.astral.sh/uv/). (No `uv`? Export a pip file with `uv export --no-dev --no-hashes -o requirements.txt`, then `pip install -r requirements.txt`.)

---

## 🎯 Three things that surprise people

### 1. Multi-step jobs from a single sentence

```
You> Read training_log.csv, plot the validation-loss curve,
     find the epoch where overfitting starts, and patch
     train.py to enable early stopping at that point.
```

Cogtrix reads the log, runs Python to compute the per-epoch loss delta, picks the inflection point, then applies the change to your training script with `patch_file`. Every shell or write action asks for confirmation first — you stay in control.

### 2. Deep reasoning when the answer matters

```
You> /think Design a real-time fraud-detection ML pipeline for 10M
     card transactions/day at sub-100ms p99 latency and 99.95% recall.
```

`/think` engages the **Tree‑of‑Thought** engine: Cogtrix proposes several candidate pipelines — feature store choice, model family, serving topology — scores each against your latency and recall targets, prunes, and explains the winner with the trade-offs against the runners-up. You see the *reasoning trail*, not just a verdict.

### 3. Parallel delegation across models

```
You> /delegate Compare LightGBM, XGBoost, and CatBoost for credit-default
     prediction on a heavily imbalanced dataset (positive rate ≈ 2%).
```

Cogtrix spawns three sub-agents in parallel — optionally on three different models — to dig into each library's class-imbalance handling, training cost, and inference latency, then synthesises a single comparison with a recommendation. Roughly the latency of one deep query, the breadth of three.

---

## 🧠 What's actually under the hood

| Capability | How Cogtrix does it |
|---|---|
| **Local-first** | Default backend is Ollama. Works offline, no telemetry, no rate limits. |
| **Multi-provider** | Ollama, OpenAI, Anthropic, Gemini, DeepSeek, plus any OpenAI-compatible endpoint. Switch with `/model`. |
| **67 built-in tools** | Files, Git, GitHub, shell, Python, HTTP, search (7 providers), text/NLP, math, scheduling, RAG, messaging — full list in [Tools Reference](docs/TOOLS_REFERENCE.md). |
| **Three memory modes** | `conversation` for chat, `code` for programming (tracks files + errors), `reasoning` for planning (tracks goals + decisions). All modes do hybrid memory — rolling summary plus semantic recall. |
| **Tool safety** | Sensitive tools (shell, write, patch) ask for confirmation. `-y` to auto-approve in trusted contexts. |
| **MCP support** | Connect to any Model Context Protocol server — Anthropic's MCP ecosystem works out of the box. |
| **Workflows** | Bundle a system prompt, knowledge base, and tool policy into a reusable named workflow with auto-detection. |
| **Headless mode** | Run as a WhatsApp or Telegram daemon (see below). |
| **REST + WebSocket API** | 159 endpoints, 2 WebSocket streams — drives the React web UI and any custom integration. |

---

## ⚙️ Configuration in one screenful

Cogtrix runs with zero config when Ollama is on localhost. For anything more, drop a YAML file in `.cogtrix.yaml` (project) or `~/.cogtrix.yaml` (global):

```yaml
providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
  openai:
    type: openai

models:
  default: local
  local:                          # everyday work — local qwen3 on a home GPU
    provider: my-server
    model: qwen3:8b
  fast: my-server/qwen3:8b        # same model, shorthand alias form
  smart: openai/gpt-4.1           # heavy reasoning, e.g. /think and /delegate

services:
  tavily:
    api_key: "tvly-..."           # cleaner results than DuckDuckGo at low volume
```

JSON works too (`.cogtrix.json`). Settings are resolved highest priority first: **CLI flags** → **environment variables** → **config file** → **built-in defaults**.

Full reference: **[Configuration Guide](docs/CONFIGURATION.md)**.

---

## 💬 Interactive commands

| Command | Aliases | What it does |
|---|---|---|
| `/help [cmd]` | `/h`, `/?` | List commands or detailed help |
| `/think <task>` | `/T` | Tree‑of‑Thought deep reasoning |
| `/delegate <task>` | `/d` | Parallel multi-model investigation |
| `/tools [search\|load\|enable\|disable]` | `/t`, `/tool` | Inspect and manage the toolbox |
| `/model [name]` | `/m` | Show or switch LLM |
| `/mode [name]` | `/M` | Show or switch memory mode (`conversation`, `code`, `reasoning`) |
| `/session [id]` | `/s` | Show or switch session |
| `/setup` | — | Interactive setup wizard |
| `/approve` | `/a` | Toggle tool auto-approval (also `-y` at startup) |
| `/paste` | `/P` | Multi-line paste mode |
| `/clear` | `/c` | Clear conversation history |
| `/optimizer [prompt]` | `/o` | Toggle prompt optimizer / force-optimize a prompt |
| `/mcp [restart [name]]` | — | Manage MCP server connections |
| `/info` | `/i` | Session info (provider, model, mode) |
| `/quit` | `/exit`, `/q` | Exit |
| `!<command>` | — | Inline shell, e.g. `!ls -la` |

Arrow keys, Home/End, and history all work via `readline`.

---

## 📂 What the tool library covers

| Category | Examples |
|---|---|
| **Search** | `web_search` (multi-provider fan-out + extract + structured output, citations included). Single canonical research tool; legacy `search_web` / `tavily_search` / `brave_search` / `google_search` / `exa_search` / `serpapi_search` / `searxng_search` are no longer in the agent catalogue, but the underlying functions remain importable for power users. |
| **Files** | `read_file`, `write_file`, `patch_file`, `append_file`, `list_directory`, `file_info` |
| **Git** | `git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`, `git_create_branch`, `git_checkout` |
| **GitHub** | `gh_create_issue`, `gh_comment_issue`, `gh_list_prs`, `gh_get_file` |
| **System** | `execute_shell_command`, `execute_python` |
| **Text & NLP** | word count, find/replace, URL/email extraction, sentiment, summarize, keywords, split, trim, compare |
| **Data** | `parse_json`, `format_json`, `query_json`, `extract_json`, `calculate` |
| **Web & Time** | `http_get`, `http_post`, `get_current_datetime`, `convert_timezone`, `parse_date`, `get_weather` |
| **Goal tracking** | `set_goal`, `add_subgoal`, `complete_goal`, `abandon_goal`, `list_goals` |
| **Scheduling** | `cron_add`, `cron_list`, `cron_remove` |
| **Agent & tasks** | `spawn_agent`, `send_to_agent`, `read_agent_inbox`, plus task-queue tools |
| **Reasoning** | `deep_think`, `delegate_task`, `delegate_parallel` |
| **Knowledge (RAG)** | `query_knowledge_base`, `save_to_knowledge_base` |
| **Messaging** | WhatsApp via [Waha](https://waha.devlike.pro/), Telegram via bot token |

**Tools auto-hide when their API keys are missing** — no errors, no clutter. The startup banner reports `Tools: [██████████░░] 41 on demand (3 unavailable)` and the agent loads what it needs through an internal `request_tools` meta-tool. You don't manage any of this. Full parameter reference: **[Tools Reference](docs/TOOLS_REFERENCE.md)**.

---

## 🧩 Memory modes

| Mode | Best for | Window |
|---|---|---|
| `conversation` (default) | General chat, Q&A, research | 25 messages |
| `code` | Programming, debugging | 30 messages + file & error tracking |
| `reasoning` | Planning, architecture decisions | 30 messages + goal & decision tracking |

All three include **hybrid memory**: older messages compress into a rolling summary, then (when an embedding provider is available) move to a semantic store. The agent stays aware of the full thread even after messages leave the window. Switch at startup (`-M code`) or runtime (`/mode code`). Details: **[Memory Modes](docs/MEMORY_MODES.md)**.

---

## 📱 Run Cogtrix as a WhatsApp / Telegram assistant

A genuinely uncommon feature: Cogtrix can run **headlessly as a messaging daemon**.
Wire it to a WhatsApp number through [Waha](https://waha.devlike.pro/) or to a Telegram bot via [@BotFather](https://t.me/BotFather), and it becomes an AI assistant your team or family talks to in their normal chat app.
Per-chat context isolation, shared knowledge base, scheduled campaigns, and workflow auto-detection — all the CLI's smarts, delivered through the channel people already use.

```bash
python cogtrix.py --assistant
```

Setup walk-throughs: **[WhatsApp Guide](docs/WHATSAPP_GUIDE.md)** · **[Telegram Guide](docs/TELEGRAM_GUIDE.md)**.

---

## 🐳 Docker

```bash
docker pull ghcr.io/northlandpositronics/cogtrix:latest
docker run -it --network host ghcr.io/northlandpositronics/cogtrix:latest
```

The image bundles every optional package (Anthropic, Google, MCP, all search providers, NumPy/SciPy). `--network host` lets it reach a local Ollama. Mount your config (`-v "$HOME/.cogtrix.yaml:/app/.cogtrix.yaml:ro"`) and persist sessions (`-v cogtrix-data:/data`). Append `api` to the docker command to launch the REST/WS server instead of the interactive CLI.

---

## 🔌 REST + WebSocket API

Cogtrix ships a FastAPI server that exposes **159 REST endpoints across 27 route groups** plus **2 WebSocket streams**. It's the same API the React web frontend uses.

```bash
export COGTRIX_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python -m src.api
# or: python -m src.api --debug --reload
```

Interactive docs at `http://localhost:8000/api/v1/docs` (Swagger) and `/api/v1/redoc`.

**Auth**: JWT bearer tokens (`Authorization: Bearer <token>`). First registered user gets the `admin` role automatically. API keys (prefix `cgx_live_`) can be created and managed via `/api/v1/auth/api-keys` and are accepted in the same `Authorization: Bearer` header — the request-auth dependency dispatches on prefix.

**WebSockets**: The session stream (`/ws/v1/sessions/{id}`) requires the JWT in the `Authorization` header; the `?token=<jwt>` query-parameter fallback was removed for security (#1128). The admin log stream (`/ws/v1/logs`) still accepts `?token=<jwt>` for clients that can't set custom WS headers.

Route map by group:

| Group | Count | Notes |
|---|---:|---|
| `auth/*` | 9 | Register, login, refresh, logout, logout-all, profile, API key CRUD |
| `agents/*` | 2 | List & get named agents |
| `sessions/*` | 6 | Create/list/get/update/delete sessions |
| `sessions/{id}/messages/*` | 3 | Send, list history, clear history |
| `sessions/{id}/memory/*` | 3 | Get state, switch mode, clear |
| `sessions/{id}/tools/*` | 4 | List, load, enable, disable |
| `config/*` | 15 | Read/write config, providers, models, setup wizard |
| `assistant/*` | 24 | Start/stop, channels, phonebook, outbound, campaigns |
| `assistant/workflows/*` | 11 | Workflow CRUD, documents, chat bindings |
| `tasks/*` | 5 | Background-task queue with log stream |
| `users/*` | 5 | User management (admin) |
| `rag/*` | 5 | RAG document & query CRUD |
| `mcp/*` | 5 | MCP server connections |
| `admin/*` | 7 | Org list, global stats, usage metrics, impersonation, audit log |
| `system/*` | 2 | Server info, shutdown |
| `health` | 3 | Liveness, readiness, full-readiness |
| `metrics` | 1 | Prometheus scrape endpoint |
| `organizations/*` | 1 | Update org-member role (other org CRUD lives in `admin/*`) |
| `teams/*` | 8 | Team management, membership |
| `workspaces/*` | 10 | Workspace CRUD, membership, scoped config |
| `plans/*` | 6 | Plan CRUD + `/org-plans/{id}` assignment |
| `usage/*` | 3 | Usage summary, per-event records, manual record |
| `enforcement/*` | 1 | Plan limit snapshot and headroom |
| `saml/*` | 3 | SAML 2.0 SSO: metadata, SSO, ACS |
| `scim/v2/*` | 7 | SCIM 2.0 provisioning (Okta, Azure AD) |
| `ldap/*` | 2 | LDAP/AD status, sync trigger |
| `jit/*` | 2 | JIT provisioning status, test |
| `cross-workspace/*` | 3 | Cross-workspace message bus |
| `billing/*` | 4 | Stripe Checkout, Customer Portal, subscription, webhook |
| `ws://host/ws/v1/sessions/{id}` | WS | Streaming agent turns, tool confirmation, token events |
| `ws://host/ws/v1/logs` | WS | Live log stream (admin only) |

Full reference: **[API Reference](docs/API/OPENAPI.yaml)** · **[Client Contract](docs/API/CLIENT_CONTRACT.md)** · **[WebSocket Protocol](docs/API/WEBSOCKET_PROTOCOL.md)**.

---

## 🛠️ Optional extras

```bash
uv pip install "cogtrix[anthropic]"   # Anthropic Claude
uv pip install "cogtrix[google]"      # Google Gemini
uv pip install "cogtrix[api]"         # REST API server + Stripe billing
uv pip install "cogtrix[mcp]"         # MCP server support
uv pip install "cogtrix[search]"      # Tavily, Exa, Brave, SerpAPI
uv pip install "cogtrix[rag]"         # RAG (needs C++ build tools)
uv pip install "cogtrix[saml]"        # SAML 2.0 SSO (needs libxmlsec1-dev on Linux)
uv pip install "cogtrix[ldap]"        # LDAP / Active Directory sync
```

---

## 🚦 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on startup | Ollama isn't running | `ollama serve` in another terminal |
| `Model not found` | Model not pulled yet | `ollama pull qwen3:8b` |
| No search results | DuckDuckGo rate-limited | Wait, retry, or add a Tavily/Brave key |
| Empty or garbled response | Model too small or OOM | Try a smaller model: `-m qwen3:8b` |
| Tool missing from `/tools` | API key for that tool isn't set | Set the key — tools auto-hide when unconfigured |
| `41 on demand (3 unavailable)` — meaning? | Normal on-demand loading | 41 tools ready to request, 3 hidden for missing keys ([details](docs/CONFIGURATION.md#tool-loading)) |
| `Invalid API key` (OpenAI) | Key missing or expired | `export OPENAI_API_KEY="sk-..."` |
| Not sure if config is valid | Typo or wrong structure | `python cogtrix.py --check-config` |

Detailed debugging: run with `--debug` (logs every LLM call, tool input/output, and context info to `cogtrix.log`).

---

## 📚 Documentation

| Guide | What's inside |
|---|---|
| [Configuration](docs/CONFIGURATION.md) | Every option, environment variable, search-provider key |
| [Providers](docs/PROVIDERS.md) | Step-by-step for Ollama, OpenAI, Anthropic, Google, DeepSeek, xAI, Groq, Together, vLLM |
| [Memory Modes](docs/MEMORY_MODES.md) | Conversation, code, reasoning + hybrid memory internals |
| [Tools Reference](docs/TOOLS_REFERENCE.md) | All 67 tools, parameters, examples |
| [WhatsApp Guide](docs/WHATSAPP_GUIDE.md) | Run Cogtrix as a WhatsApp assistant |
| [Telegram Guide](docs/TELEGRAM_GUIDE.md) | Run Cogtrix as a Telegram bot |
| [Deep Think](docs/DEEPTHINK.md) | Tree-of-Thought engine internals |
| [RAG Guide](docs/RAG_GUIDE.md) | Build a knowledge base from your documents |
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, components |
| [Development](docs/DEVELOPMENT.md) | Add tools, memory modes, slash commands; testing |
| [API Reference](docs/API/OPENAPI.yaml) | OpenAPI 3.1 schema (also available as [JSON](docs/API/OPENAPI.json)) |
| [Client Contract](docs/API/CLIENT_CONTRACT.md) | TypeScript API types |
| [WebSocket Protocol](docs/API/WEBSOCKET_PROTOCOL.md) | Streaming session protocol |

---

## 🧪 Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ -q -m "not agent_workflow and not live_llm and not docker"  # fast unit suite
uv run pytest tests/ -m live_llm -v --timeout=300                                # live LLM tests (needs Gemma container at :18080)
```

---

## 📜 License

Copyright 2025‑2026 Northland Positronics (FZE). Released under the **Cogtrix Source-Available License 1.0** — see [LICENSE](LICENSE) for full terms.
