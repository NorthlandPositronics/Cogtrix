# Cogtrix Configuration Reference

This page covers every way to configure Cogtrix — from the simplest environment variable to a full multi-provider config file. If you just want to get running, the [Quick Start in the README](../README.md#quick-start) is all you need; come back here when you want to customize.

## Table of Contents

- [Configuration Priority](#configuration-priority)
- [Configuration File](#configuration-file)
  - [Research Delegate Section](#research-delegate-section)
  - [Prompt Optimizer](#prompt-optimizer)
  - [Context Compression](#context-compression)
  - [MCP Servers](#mcp-servers)
  - [Tool Loading](#tool-loading)
  - [Assistant Mode](#assistant-mode)
  - [Assistant Guardrails](#assistant-guardrails)
- [Environment Variables](#environment-variables)
- [Command Line Arguments](#command-line-arguments)
  - [Setup Wizard](#setup-wizard)
- [Complete Configuration Example](#complete-configuration-example)
- [Debugging & Logging](#debugging--logging)

---

## Configuration Priority

Configuration is loaded from multiple sources with the following priority (highest to lowest):

1. **Command line arguments** — Override everything
2. **Environment variables** — Override config file
3. **Configuration file** (`.cogtrix.json` / `.cogtrix.yml` / `.cogtrix.yaml`) — Base settings
4. **Built-in defaults** — Fallback values

---

## Configuration File

Both JSON and YAML formats are supported. Create a config file in one of these locations (first found wins):

1. `./.cogtrix.json`
2. `./.cogtrix.yml` or `./.cogtrix.yaml`
3. `~/.cogtrix.json`
4. `~/.cogtrix.yml` or `~/.cogtrix.yaml`
5. `~/.config/cogtrix/cogtrix.json`
6. `~/.config/cogtrix/cogtrix.yml` or `~/.config/cogtrix/cogtrix.yaml`

Within each directory, JSON is checked first, then `.yml`, then `.yaml`.

### General Settings

```yaml
provider: ollama
model: qwen3:8b
session: default
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `"ollama"` | Active provider name |
| `model` | string | Provider-specific | Model to use (overrides provider default) |
| `session` | string | `"default"` | Session ID for memory persistence |

### Providers Section

Define named LLM providers with custom configurations:

```yaml
providers:
  my-ollama:
    type: ollama
    base_url: "http://192.168.1.100:11434"
    model: qwen3:8b
  openai:
    type: openai
    model: gpt-4.1-mini
    api_key: "sk-..."
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."
    model: llama-3.3-70b-versatile
```

> **Note:** The key `"providers"` is preferred. The legacy key `"inference"` still works as an alias for backward compatibility.

#### Provider Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `type` | string | Yes | Provider type: `"openai"`, `"ollama"`, `"anthropic"`, or `"google"` (case-insensitive) |
| `base_url` | string | No | API endpoint URL |
| `model` | string | No | Default model for this provider |
| `api_key` | string | No | API key (all providers except Ollama) |
| `tool_instructions` | string | No | Custom tool-call formatting instructions appended to the system prompt. Not injected by default — `bind_tools()` handles formatting at the API level. Set a non-empty string only for providers that need explicit guidance. |

#### Provider Types

| Type | Use For | Default Model | Default Base URL |
|------|---------|---------------|------------------|
| `openai` | OpenAI, Groq, Together, vLLM, LocalAI | `gpt-4.1-mini` | `https://api.openai.com/v1` |
| `ollama` | Ollama servers | `qwen3:8b` | `http://localhost:11434` |
| `anthropic` | Anthropic Claude | `claude-sonnet-4-5` | SDK default |
| `google` | Google Gemini | `gemini-2.5-flash` | SDK default |

**xAI (Grok)** uses `type: openai` with `base_url: "https://api.x.ai/v1"`. The setup wizard offers it as a named choice.

Optional dependencies: `langchain-anthropic` (`uv pip install "cogtrix[anthropic]"`), `langchain-google-genai` (`uv pip install "cogtrix[google]"`).

### Memory Section

Configure memory management:

```yaml
memory:
  mode: conversation
  modes:
    conversation:
      working_memory_size: 25
      summarization: true
      vector_recall_k: 3
    code:
      working_memory_size: 30
      max_files: 20
      summarization: true
      vector_recall_k: 3
    reasoning:
      working_memory_size: 30
      max_decisions: 20
      summarization: true
      vector_recall_k: 3
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `"conversation"` | Active memory mode |
| `modes` | object | `{}` | Mode-specific configurations |

#### Hybrid Memory Options (per mode)

All modes support hybrid memory — a combination of a sliding window, incremental summarization, and optional vector recall that keeps the agent aware of older conversation context.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `summarization` | bool | `true` | Enable LLM-based rolling summary of older messages. Set to `false` to save LLM calls on metered APIs. |
| `vector_recall_k` | int | `3` | Number of semantically similar past exchanges to retrieve per turn. Set to `0` to disable vector recall. |

Hybrid memory is automatically enabled when an LLM is available. The vector recall component additionally requires an embedding provider — Cogtrix attempts to auto-detect one at startup (tries Ollama's `nomic-embed-text` first, then falls back to OpenAI if `OPENAI_API_KEY` is set). If no embedding provider is available, vector recall is silently skipped while summarization still functions normally.

See [MEMORY_MODES.md](MEMORY_MODES.md) for detailed mode options and a full explanation of the hybrid memory system.

### RAG Section

Configure document ingestion for knowledge base:

```yaml
rag:
  docs_dir: docs
  vectordb_dir: data/vectordb
  chunk_size: 2000
  chunk_overlap: 200
  model: embed-local
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `docs_dir` | string | `"docs"` | Source documents directory |
| `vectordb_dir` | string | `"data/vectordb"` | Vector database output directory |
| `chunk_size` | int | `2000` | Text chunk size in characters |
| `chunk_overlap` | int | `200` | Overlap between chunks |
| `model` | string | `null` | Model name from the `models` registry to use for embeddings. Falls back to the active provider when not set. |

**Note:** The `model` field references a named entry in the top-level `models` registry. Define an embedding model there and point `rag.model` at it. The provider connection details (type, base_url, api_key) are resolved automatically from the matching provider config.

See [RAG_GUIDE.md](RAG_GUIDE.md) for detailed setup instructions.

### Models

The `models` registry gives short names to `provider/model` combinations. They are used by:

- The **`-m` CLI flag** — start Cogtrix with any model name: `python cogtrix.py -m fast`
- The **`/model` command** — switch at runtime: `/model coder`
- The **delegation tools** — the agent uses model names to pick the best model for a subtask
- The **`rag.model` field** — reference an embedding model by name

Define models at the **top level** of your config (preferred) or inside `delegate` for backward compatibility:

```yaml
models:
  fast: my-server/qwen3:8b
  smart: openai/gpt-4.1
  coder:
    provider: my-server
    model: qwen3-coder
    temperature: 0.3
  reasoning:
    provider: local-gpu
    model: qwen3:32b
    num_ctx: 32768
    temperature: 0.3
  embed-local:
    provider: local-gpu
    model: nomic-embed-text
```

> **Backward compatibility:** The key `model_aliases` still works in config files as an alias for `models`. New configs should use `models`.

#### Model Entry Formats

**String format** — `"provider/model"` or just `"model"`:

```yaml
fast: my-server/qwen3:8b
```

**Object format** — with additional overrides (`num_ctx`, `temperature`, `timeout`):

```yaml
coder:
  provider: my-server
  model: qwen3-coder
  temperature: 0.3
  timeout: 300
```

The object format fields `num_ctx` and `temperature` are model-level settings. They are applied at resolution time and override any defaults from the provider config. The `provider` field references a key in the `providers` section.

> **Note:** `num_ctx` is only effective for Ollama-type providers. It is silently ignored for OpenAI, Anthropic, and Google providers (which manage context windows via their own API parameters).

#### Using Models

```bash
python cogtrix.py -m fast       # Resolves to my-server/qwen3:8b
python cogtrix.py -m coder      # Resolves to my-server/qwen3-coder with temperature=0.3
```

At runtime:
```
You: /model fast
Switched to model qwen3:8b (my-server)
```

### Delegate Section

Configure task delegation to other models:

```yaml
delegate:
  enabled: true
  default_timeout: 60
  allowed_models:
    - coder
    - smart
    - fast
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable delegation |
| `default_timeout` | int | `60` | Default timeout in seconds |
| `allowed_models` | array | All models | Model names from the `models` registry the agent may delegate to |
| `allowed_providers` | array | All providers | Provider names allowed for delegation |

**`allowed_models`** restricts which model names the agent may use when delegating. If omitted, all entries in the `models` registry are available. This is the recommended way to control delegation scope — configure a broad set of models in `models`, then whitelist a subset in `allowed_models`:

```yaml
models:
  fast: my-server/qwen3:8b
  smart: openai/gpt-4.1
  coder: my-server/qwen3-coder

delegate:
  enabled: true
  allowed_models: [coder, smart]  # agent can only delegate to these two
```

**`allowed_providers`** restricts by provider name and is an additional guard. Both checks must pass for delegation to proceed.

> **Backward compatibility:** `delegate.models` still works for defining models scoped to the delegate section. If both top-level `models` and `delegate.models` are present, the top-level definition takes priority. The older `delegate.model_aliases` key is also still recognized.

### Research Delegate Section

When the user requests deep reasoning (via `/think` or "think deeply" in a prompt) and the agent has used web tools during its initial research, Cogtrix can spawn a **research delegate** — a sub-agent that re-fetches the same URLs with a much larger context budget and extracts structured, verbatim specifications instead of lossy summaries. The extracted content is then fed into the `deep_think` engine as high-fidelity context.

```yaml
research_delegate:
  enabled: true
  cap_ratio: 0.85
  timeout: 300
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable the research delegate pipeline |
| `cap_ratio` | float | `0.85` | Proportion of `max_context_tokens` allocated to the delegate's tool output cap. Higher values let the delegate load more page content. Clamped to 0.50–0.95. |
| `timeout` | int | `300` | Maximum seconds for the delegate agent to run. Clamped to 60–600. |

**How it works:**

1. The main agent runs its initial research (web searches, content fetching) with the normal output cap.
2. Cogtrix extracts the URLs the agent visited from its tool call history.
3. A research delegate agent is spawned with the same provider/model configuration. Its web tools are temporarily patched to allow output up to `cap_ratio × max_context_tokens × 4` characters.
4. The delegate is instructed to fetch each URL and extract **exact specifications** — schemas, field names, code examples, file paths — without summarizing or paraphrasing.
5. The delegate's structured output replaces the raw tool dumps as primary context for `deep_think`.
6. After the delegate finishes (or times out), the original tool output caps are restored.

**When to tune:**

- Set `enabled: false` if you don't use web research with deep thinking, or if you want to save LLM calls on a metered API.
- Increase `cap_ratio` toward `0.95` if the delegate's output is being truncated and you have a large context window.
- Increase `timeout` if the delegate is timing out on slow models or large pages.

### Prompt Optimizer

The prompt optimizer preprocesses complex user prompts before the agent executes them. It uses a one-shot LLM call to evaluate whether the prompt needs restructuring and rewrites it with a high-level approach and practical guardrails if needed.

```yaml
prompt_optimizer: true
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `prompt_optimizer` | bool | `true` | Enable/disable prompt optimization before agent execution |

**How it works:**

1. Prompts shorter than 400 characters skip optimization entirely (no LLM call).
2. The LLM evaluates the prompt — if already clear and actionable, it returns it unchanged.
3. If the prompt is complex or vague, it rewrites it to preserve the goal, add a high-level approach (phases/steps), and include practical guardrails.
4. The optimizer's system instructions are ephemeral — they do not persist in conversation history or affect subsequent prompts.

**Important:** The original prompt is always used for deep-think detection (`_user_wants_deep_think`) and memory context preparation. Only `run_agent()` receives the optimized version.

Set `prompt_optimizer: false` to disable this feature (e.g., when running automated pipelines where prompts are already structured).

### Context Compression

During long agent runs, tool outputs (file contents, shell output, search results) accumulate in the message history and are re-sent to the LLM on every cycle. Context compression summarizes old, large ToolMessages before each LLM call to reduce per-cycle token usage while preserving important context.

```yaml
# Simple toggle
context_compression: true

# Or with custom thresholds
context_compression:
  enabled: true
  model: fast
  min_age: 8       # call_model cycles before eligible (default: 6)
  min_chars: 6000  # minimum content length to qualify (default: 2000)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context_compression` | bool or object | `true` | Enable/disable context compression, or configure thresholds |
| `model` | string | `null` | Model alias or `provider/model` string for a dedicated compression LLM. Uses the main agent LLM when not set. |
| `min_age` | int | `6` | Number of `call_model` cycles a ToolMessage must survive before it becomes eligible for compression |
| `min_chars` | int | `2000` | Minimum character length of a ToolMessage's content to qualify for compression |

**How it works:**

1. On each `call_model` cycle, the compression pass checks whether total message size exceeds 72% of the context window.
2. ToolMessages that are both old enough (age >= `min_age`) and large enough (length >= `min_chars`) are compressed. Multiple eligible messages are compressed in parallel (up to 4 concurrent LLM calls).
3. The LLM preserves file paths, error messages, stack traces, line numbers, schemas, exact values, and code snippets while removing verbose prose and boilerplate.
4. When `model` is set, a dedicated LLM is used for compression instead of the main agent model — a smaller/faster model reduces latency.
5. Compressed messages are cached by `tool_call_id` to avoid re-summarizing.
6. Compression operates on a copy of the message list — graph state is never mutated.
7. On LLM failure, the compressor falls back to middle-truncation (`_truncate_tool_output`).

**When to tune:**

- Set `context_compression: false` if you have a very large context window and want to avoid the extra LLM calls.
- Set `model` to a fast/cheap model alias to avoid using the main agent model for compression. Without this, each compression call uses the same (potentially slow) model.
- Increase `min_age` if you find recent tool outputs are being compressed too early.
- Increase `min_chars` to only compress very large outputs (e.g., full file contents).

### Parallel Tool Execution

When the LLM emits multiple tool calls in a single response, Cogtrix can execute them concurrently using a thread pool instead of processing them sequentially.

```yaml
parallel_tool_execution: true
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `parallel_tool_execution` | bool | `true` | Enable/disable concurrent execution of independent tool calls |

**How it works:**

1. When the LLM returns multiple tool calls, a classification pass splits them into two groups:
   - **Serial-first** — `request_tools` calls and calls to tools not yet loaded (require auto-expansion). These run sequentially first.
   - **Parallel** — all other calls to already-active tools. These run concurrently via a `ThreadPoolExecutor` (up to 8 workers).
2. A single tool call in a batch skips pool overhead and runs inline.
3. `UserCancelledRun` from any tool stops all remaining execution immediately.
4. The system prompt instructs models to batch independent operations when possible.

**When to tune:**

- Set `parallel_tool_execution: false` if you experience issues with tools that have hidden shared state or if you need deterministic tool execution order.
- Models that support parallel tool calls (GPT-4o, Claude, Gemini) benefit most from this feature. Models that emit one call per response (some open-source/vLLM models) are unaffected.

### Allowed Write Paths

By default, file write operations (`write_file`, `append_file`) are restricted to the current working directory. You can extend this with additional directories:

```yaml
allowed_write_paths:
  - /data/output
  - /shared/workspace
```

This is especially useful in Docker deployments where the working directory differs from the application install path:

```bash
# Via environment variable (colon-separated)
docker run -it -e COGTRIX_ALLOWED_WRITE_PATHS="/app:/data" -w /tmp ghcr.io/northlandpositronics/cogtrix:latest

# Via CLI flag (repeatable)
cogtrix.py --allow-write-path /data/output --allow-write-path /shared/workspace
```

Read operations are not affected — they already allow access to both the working directory and the application install directory.

**Priority:** CLI (`--allow-write-path`) > env var (`COGTRIX_ALLOWED_WRITE_PATHS`) > config file.

### MCP Servers

Cogtrix can connect to external tool servers via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Configure servers in the `mcp_servers` section — each key is a server name:

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env:
      HOME: /home/user
    requires_confirmation: true
    timeout: 30

  remote-api:
    url: http://localhost:8000/sse
    headers:
      Authorization: "Bearer your-token"
    requires_confirmation: false
```

Transport is auto-detected from the config keys:
- **Stdio** (local process): set `command` and optionally `args`, `env`
- **SSE** (remote HTTP): set `url` and optionally `headers`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `command` | string | — | Executable to launch for stdio transport |
| `args` | array | `[]` | Command-line arguments for the executable |
| `env` | object | `null` | Environment variables (all values must be strings) |
| `url` | string | — | Full URL for SSE transport |
| `headers` | object | `null` | HTTP headers for SSE transport (e.g., auth tokens) |
| `requires_confirmation` | bool | `true` | Whether tools from this server need user confirmation |
| `timeout` | int | `30` | Per-call timeout in seconds |

MCP tools are registered into the on-demand pool and loaded by the agent via `request_tools`, just like built-in tools. They appear in `/tools` with an `[mcp]` tag.

**Prerequisite:** Install the MCP SDK: `uv pip install "cogtrix[mcp]"` (or `pip install mcp`). If the package is not installed and `mcp_servers` is configured, a warning is logged and servers are skipped.

Use `/mcp` to list connected servers and their tools. Use `/mcp restart [name]` to reconnect.

#### Docker Compose (SSE via supergateway)

When running Cogtrix in Docker, stdio MCP servers can't be spawned directly because each service runs in its own container. Use [supergateway](https://github.com/supercorp-ai/supergateway) to bridge stdio servers to SSE:

```yaml
# docker-compose.yml (excerpt)
mcp-filesystem:
  image: supercorp/supergateway
  command: >
    --stdio "npx -y @modelcontextprotocol/server-filesystem /data"
    --port 8000
  volumes:
    - mcp-workspace:/data
  expose:
    - "8000"
```

Then configure Cogtrix to connect via SSE:

```yaml
# .cogtrix.yaml
mcp_servers:
  filesystem:
    url: http://mcp-filesystem:8000/sse
    requires_confirmation: false
```

The shared `mcp-workspace` volume is mounted at `/data` in the MCP server and at `/app/mcp-data` in the Cogtrix container, giving both services read/write access to the same files. See the included `docker-compose.yml` for the full working setup.

### Tool Loading

When Cogtrix starts, you see a line like:

```
Tools : [██████████░░] 41 on demand (3 unavailable)
```

This means 41 tools are configured and ready to use, while 3 are hidden because their API keys aren't set. The progress bar shows the ratio of configured to total registered tools.

#### How it works

The agent starts with a single meta-tool called `request_tools`. Its description contains a catalog of every available tool. When the agent needs a tool, it calls `request_tools(add=["tool_a", "tool_b"])` and the system activates the requested tools before the agent's next turn. This keeps the initial prompt lean — only the tools relevant to the current task are loaded.

The agent can also release tools it no longer needs to keep its toolkit small:

```
request_tools(remove=["tool_a"])
```

Released tools return to the catalog and can be re-requested later.

#### Startup banner

| Element | Meaning |
|---------|---------|
| `[██████████░░]` | Ratio of configured tools to total registered (e.g. 41/44) |
| `41 on demand` | Tools the agent can request |
| `(3 unavailable)` | Tools hidden due to missing API keys |

#### What the agent sees

The `request_tools` tool description includes a one-line summary of every available tool, so the agent can choose intelligently. For example, if you ask a date question it will request `get_current_datetime`; if you ask to search the web it will request `search_web`.

#### Fuzzy name matching

If the agent tries to call a tool by an approximate name (e.g. `list_dir` instead of `list_directory`), Cogtrix resolves it automatically, activates the correct tool, and retries the request.

#### Overriding with `--tools`

Use the `--tools` CLI flag to bypass the on-demand system and load specific tools directly:

```bash
python cogtrix.py --tools none                    # No tools (pure LLM chat)
python cogtrix.py --tools minimal                 # Basic set only
python cogtrix.py --tools "search_web,calculate"  # Specific tools
```

When `--tools` is used, all specified tools are active immediately (no on-demand pool).

### Services Section

Configure API keys for external services (search providers, weather, etc.) in a single place:

```yaml
services:
  tavily:
    api_key: "tvly-..."
  exa:
    api_key: "exa-..."
  brave:
    api_key: "BSA..."
  serpapi:
    api_key: "..."
  google:
    api_key: "AIza..."
    cse_id: "abc123..."
  openweather:
    api_key: "..."
```

Tools that require an API key are **automatically hidden** from the agent when the key is not configured — no errors, they simply don't appear in the tool list.

#### Search Providers

Cogtrix includes six search providers. DuckDuckGo is always available with no setup. The other five require an API key and some require an additional Python package.

| Provider | Tools | Package | API Key | Free Tier |
|----------|-------|---------|---------|-----------|
| DuckDuckGo | `search_web`, `search_news` | Included (`ddgs`) | None | Unlimited |
| Tavily | `tavily_search`, `tavily_extract` | `tavily-python` | `TAVILY_API_KEY` | 1 000/month |
| Exa | `exa_search`, `exa_find_similar`, `exa_get_contents` | `exa-py` | `EXA_API_KEY` | 1 000/month |
| Brave | `brave_search` | Included (`requests`) | `BRAVE_API_KEY` | 2 000/month |
| Google | `google_search` | Included (`requests`) | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | 100/day |
| SerpAPI | `serpapi_search` | `google-search-results` | `SERPAPI_API_KEY` | 100/month |

**Installing optional search packages:**

Tavily, Exa, and SerpAPI need extra Python packages not included by default:

```bash
# All at once (recommended)
uv sync --extra search

# Or individually with pip
pip install tavily-python exa-py google-search-results
```

Brave and Google use only `requests`, which is already a core dependency.

#### Legacy service format

For backward compatibility, top-level service keys still work:

```json
{
  "openweather": { "api_key": "..." },
  "tavily":      { "api_key": "..." }
}
```

The `"services"` section takes priority when both are present.

### WhatsApp Messaging

Cogtrix can send and receive WhatsApp messages via a self-hosted [Waha](https://waha.devlike.pro/) Docker container. Run it alongside Cogtrix:

```bash
docker run -p 3000:3000 devlikeapro/waha
```

Then open `http://localhost:3000` in your browser, scan the QR code with your phone, and configure Cogtrix:

```yaml
services:
  whatsapp:
    waha_url: "http://localhost:3000"
    api_key: "yoursecretkey"
    session: default
    allow_send: true
    allow_receive: true
    require_confirmation: true
    filter_mode: whitelist
    contacts: ["+14155551234", "+442071234567"]
    phonebook:
      alice: "+14155551234"
      bob: "+442071234567"
    rate_limit: 30
    max_message_length: 4096
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `waha_url` | string | `"http://localhost:3000"` | Waha server URL |
| `api_key` | string | — | Waha `X-Api-Key` header value |
| `session` | string | `"default"` | Waha session name |
| `allow_send` | bool | `true` | Enable send tools (`whatsapp_send`, `whatsapp_send_image`) |
| `allow_receive` | bool | `true` | Enable receive tool (`whatsapp_check`) |
| `require_confirmation` | bool | `true` | Prompt user before sending messages |
| `filter_mode` | string | `"none"` | `"none"`, `"whitelist"`, or `"blacklist"` |
| `contacts` | array | `[]` | E.164 phone numbers for the filter list |
| `phonebook` | object | `{}` | Nickname → phone number map |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages to this length |

When both `allow_send` and `allow_receive` are `false`, no WhatsApp tools are loaded.

See [Tools Reference — WhatsApp](TOOLS_REFERENCE.md#whatsapp-messaging) for tool parameters and usage. For a complete step-by-step walkthrough, see the **[WhatsApp Guide](WHATSAPP_GUIDE.md)**.

### Telegram Messaging

Cogtrix can send and receive Telegram messages via a bot. Create a bot with [@BotFather](https://t.me/BotFather) and configure the token:

```yaml
services:
  telegram:
    bot_token: "123456:ABC-DEF..."
    allow_send: true
    allow_receive: true
    require_confirmation: true
    filter_mode: whitelist
    contacts: ["123456789", "@alice_username"]
    phonebook:
      alice: "123456789"
      team: "-1001234567890"
    rate_limit: 30
    max_message_length: 4096
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bot_token` | string | — | Bot token from @BotFather (required) |
| `allow_send` | bool | `true` | Enable send tools (`telegram_send`, `telegram_send_photo`) |
| `allow_receive` | bool | `true` | Enable receive tool (`telegram_check`) |
| `require_confirmation` | bool | `true` | Prompt user before sending messages |
| `filter_mode` | string | `"none"` | `"none"`, `"whitelist"`, or `"blacklist"` |
| `contacts` | array | `[]` | Chat IDs or @usernames for the filter list |
| `phonebook` | object | `{}` | Nickname → chat ID map |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages to this length |

**Quick setup:**

1. Message [@BotFather](https://t.me/BotFather) on Telegram and create a bot (`/newbot`)
2. Copy the bot token
3. Set `COGTRIX_TELEGRAM_TOKEN="123456:ABC-DEF..."` or add it to the config file
4. Start a chat with your bot on Telegram (send it `/start`)
5. Run Cogtrix — the Telegram tools appear automatically

**Note:** Telegram bots can only receive messages from users who have started a conversation with the bot first. The bot cannot initiate contact with unknown users.

When both `allow_send` and `allow_receive` are `false`, no Telegram tools are loaded.

See [Tools Reference — Telegram](TOOLS_REFERENCE.md#telegram-messaging) for tool parameters and usage. For a complete walkthrough, see the **[Telegram Guide](TELEGRAM_GUIDE.md)**.

### Assistant Mode

Run Cogtrix as a headless messaging daemon that maintains ongoing conversations over WhatsApp and Telegram. Launch with `--assistant`:

```bash
python cogtrix.py --assistant --log --debug
```

Configure under `services.assistant`:

```yaml
services:
  assistant:
    max_concurrent: 4          # concurrent LLM calls across all chats
    max_sessions: 50           # active chat sessions in memory
    idle_timeout: 3600         # seconds before idle session is evicted
    max_response_length: 4000  # truncate replies for messaging
    system_prompt: null        # null = built-in assistant persona
    excluded_tools: []         # additional tools to exclude (beyond defaults)
    channels:
      whatsapp:
        enabled: true
        poll_interval: 5       # seconds between polls
      telegram:
        enabled: true
        poll_interval: 1
        long_poll_timeout: 30  # Telegram long-polling timeout
    knowledge:
      enabled: true
      extraction_model: null   # model alias for fact extraction LLM
      recall_k: 5              # facts retrieved per query
      max_facts: 10000
    guardrails:
      enabled: true                    # master kill switch
      max_input_length: 4000           # chars
      unicode_checks: true             # invisible/RTL character detection
      input_patterns: []               # additional regex patterns to block
      rate_limit:
        per_minute: 10                 # per chat
        per_hour: 60                   # per chat
      encoding_detection:
        enabled: true                  # detect Morse/Base64/hex/leetspeak bypasses
        min_score: 0.6                 # 0.0-1.0; lower = more sensitive
      tool_call_guard:
        enabled: true                  # inspect tool arguments before execution
        injection_scan: true           # check all string args for injection patterns
        path_blocking: true            # block sensitive paths in file tool args
        exfiltration_detection: true   # detect secrets/PII in web tool URL args
        sensitive_paths: []            # additional path prefixes to block
      auto_blacklist:
        enabled: true                  # auto-blacklist repeat offenders
        max_violations: 2              # violations before blacklist triggers
        window_minutes: 30             # sliding window for violation count
      banned_output_strings: []        # system prompt fragments to redact
      block_urls_in_output: true       # strip URLs from responses
      pii_detection: true              # regex PII scanning on output
      llm_judge:
        enabled: false                 # opt-in (adds ~500ms-2s latency)
        model: null                    # model alias or provider/model
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_concurrent` | int | `4` | Maximum simultaneous agent runs across all chats |
| `max_sessions` | int | `50` | Maximum active chat sessions in memory |
| `idle_timeout` | float | `3600` | Seconds of inactivity before a session is evicted to disk |
| `max_response_length` | int | `4000` | Truncate agent responses to this length |
| `system_prompt` | string | `null` | Custom system prompt (null = built-in messaging persona) |
| `excluded_tools` | array | `[]` | Additional tools to exclude (messaging tools, shell, and write tools are always excluded) |
| `channels.{name}.enabled` | bool | `true` | Enable/disable a specific channel |
| `channels.{name}.poll_interval` | float | varies | Seconds between poll cycles |
| `channels.telegram.long_poll_timeout` | int | `30` | Telegram getUpdates timeout |
| `knowledge.enabled` | bool | `true` | Enable cross-chat fact extraction and recall |
| `knowledge.extraction_model` | string | `null` | Model alias for fact extraction (null = main LLM) |
| `knowledge.recall_k` | int | `5` | Number of facts recalled per query |
| `knowledge.max_facts` | int | `10000` | Maximum stored facts |
| `knowledge.data_dir` | string | `"data"` | Base directory for knowledge persistence (facts.json, FAISS index) |
| `guardrails.enabled` | bool | `true` | Master kill switch for all guardrails |
| `guardrails.max_input_length` | int | `4000` | Maximum input length in characters |
| `guardrails.unicode_checks` | bool | `true` | Detect invisible/RTL Unicode steganography |
| `guardrails.input_patterns` | array | `[]` | Additional regex patterns to block on input |
| `guardrails.rate_limit.per_minute` | int | `10` | Maximum messages per minute per chat |
| `guardrails.rate_limit.per_hour` | int | `60` | Maximum messages per hour per chat |
| `guardrails.encoding_detection.enabled` | bool | `true` | Detect encoding-based bypass attempts (Morse, Base64, hex, leetspeak) |
| `guardrails.encoding_detection.min_score` | float | `0.6` | Minimum detection score (0.0–1.0) to block a message. Lower values are more sensitive. |
| `guardrails.tool_call_guard.enabled` | bool | `true` | Inspect tool arguments before execution |
| `guardrails.tool_call_guard.injection_scan` | bool | `true` | Scan all string tool arguments for injection patterns |
| `guardrails.tool_call_guard.path_blocking` | bool | `true` | Block sensitive filesystem paths in file tool arguments |
| `guardrails.tool_call_guard.exfiltration_detection` | bool | `true` | Detect API keys, SSH keys, and SSNs in web tool URL/query arguments |
| `guardrails.tool_call_guard.sensitive_paths` | array | `[]` | Additional path prefixes to block in file tool arguments |
| `guardrails.auto_blacklist.enabled` | bool | `true` | Auto-blacklist chats that exceed the violation threshold |
| `guardrails.auto_blacklist.max_violations` | int | `2` | Number of security violations before a chat is blacklisted |
| `guardrails.auto_blacklist.window_minutes` | int | `30` | Sliding window (in minutes) for counting violations |
| `guardrails.banned_output_strings` | array | `[]` | Strings to redact from agent responses (e.g. system prompt fragments) |
| `guardrails.block_urls_in_output` | bool | `true` | Strip URLs from agent responses |
| `guardrails.pii_detection` | bool | `true` | Redact email, credit card, SSN, and private IP addresses from responses |
| `guardrails.llm_judge.enabled` | bool | `false` | Enable LLM-as-judge classifier (opt-in; adds ~500ms–2s latency) |
| `guardrails.llm_judge.model` | string | `null` | Model alias or `provider/model` for the judge LLM (null = main LLM) |

**How it works:**

1. One polling thread per channel checks for new messages at the configured interval.
2. New messages are dispatched to a thread pool for concurrent processing.
3. Each incoming message is checked by the `GuardrailPipeline` (rate limit, input validation, injection detection). Blocked messages receive a canned reply without reaching the agent.
4. Each `(channel, chat_id)` pair gets an independent `ConversationMemoryManager` — no context blending between chats.
5. The agent runs with the same tool pipeline as interactive mode (minus excluded tools).
6. After each turn, durable facts are extracted and stored in a shared knowledge store (`data/knowledge/facts.json`).
7. On each new message, relevant facts are recalled and injected into the agent's context — enabling cross-chat knowledge without exposing raw conversation history.
8. The agent response passes through output sanitization (PII redaction, URL stripping, banned string removal) before being sent.
9. SIGINT/SIGTERM triggers graceful shutdown: all sessions saved, knowledge store persisted.

**Prerequisites:** WhatsApp requires a running Waha container. Telegram requires a bot token. Both must be configured in their respective `services.whatsapp` / `services.telegram` sections.

### Assistant Guardrails

Every message handled by assistant mode passes through a `GuardrailPipeline` in `src/assistant/guardrails.py`. Guardrails run before the agent processes input, before each tool call executes, and again before the reply is sent to the channel. Configure under `services.assistant.guardrails` (shown in the config block above).

**Input pipeline order:** `blacklist → rate_limiter → input_guard → encoding_guard → llm_judge`

Rate limit violations are recorded but do not increment the security violation counter (and therefore cannot trigger auto-blacklisting on their own).

**Input guard details:**

- Length check: messages exceeding `max_input_length` characters are rejected.
- Unicode check: invisible characters and RTL override codepoints (used in steganographic injection) are detected and rejected. A UTF-8 BOM at position 0 is allowed.
- Injection patterns: 15 pre-compiled regexes cover common prompt injection and jailbreak patterns (DAN mode, persona override, system tag injection, etc.). Add site-specific patterns via `input_patterns`.

**Encoding detection:**

`EncodingDetectionGuard` scores each message with four independent sub-detectors (Morse code, Base64, hex encoding, leetspeak/ROT13), each returning 0–1. The maximum of the four scores is compared against `min_score` (default 0.6). Messages that exceed the threshold are rejected. Violations are counted toward auto-blacklisting. Tune `min_score` downward to catch more attempts (with higher false-positive risk) or upward to reduce false positives on legitimate content.

**Tool call guard:**

`ToolCallGuard` inspects tool arguments before each tool executes:

- **Injection scan** — checks all string arguments of any tool for prompt injection patterns.
- **Path blocking** — for file tools (`read_file`, `write_file`, etc.), rejects arguments that reference sensitive paths such as `/etc/`, `/proc/`, `.env` files, and private key files. Add custom prefixes via `sensitive_paths`.
- **Exfiltration detection** — for web tools (`search_web`, `http_request`, etc.), detects API keys, SSH keys, and SSNs embedded in URL or query arguments.

**Auto-blacklist:**

`ViolationTracker` maintains a per-chat sliding window of security violation timestamps. When a chat's violation count within the last `window_minutes` minutes reaches `max_violations`, all subsequent messages from that chat are rejected immediately (before any other check) with a blacklist reason. The blacklist state is persisted to `data/assistant/violations.json` and survives assistant restarts. Expired violations (older than the sliding window) are pruned on load.

**Output sanitization:**

- Markdown images are stripped (alt text preserved).
- HTML tags are removed.
- Strings listed in `banned_output_strings` are replaced with `[REDACTED]` (case-insensitive).
- PII is replaced with typed placeholders: `[EMAIL_REDACTED]`, `[CREDIT_CARD_REDACTED]`, `[SSN_REDACTED]`, `[IP_ADDRESS_REDACTED]`.
- URLs are replaced with `[link removed]` when `block_urls_in_output` is true.

**LLM judge:** When `llm_judge.enabled: true`, an additional LLM call classifies the input as SAFE or UNSAFE. The judge is fail-open — if the LLM call fails, the message is allowed through. Use `llm_judge.model` to point the judge at a fast/cheap model alias to avoid adding 500ms–2s to every request.

**Disabling:** Set `guardrails.enabled: false` to bypass the entire pipeline. The `GuardrailPipeline` still exists in the handler but all checks return safe immediately.

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `COGTRIX_PROVIDER` | LLM provider | `ollama` |
| `COGTRIX_MODEL` | Model name | `qwen3:8b` |
| `COGTRIX_SESSION` | Session ID | `my-project` |
| `COGTRIX_MEMORY_MODE` | Memory mode | `code` |
| `COGTRIX_OLLAMA` | Ollama server address (`host` or `host:port`) | `192.168.1.100` or `192.168.1.100:8080` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `XAI_API_KEY` | xAI (Grok) API key | `xai-...` |
| `OLLAMA_BASE_URL` | Ollama server URL (legacy, full URL) | `http://192.168.1.100:11434` |
| `OPENWEATHER_API_KEY` | OpenWeather API key | `abc123` |
| `COGTRIX_EMBEDDING_PROVIDER` | RAG embedding provider | `openai` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `TAVILY_API_KEY` | Tavily search API key | `tvly-...` |
| `EXA_API_KEY` | Exa search API key | `exa-...` |
| `BRAVE_API_KEY` | Brave search API key | `BSA...` |
| `GOOGLE_API_KEY` | Google Custom Search API key | `AIza...` |
| `GOOGLE_CSE_ID` | Google Programmable Search Engine ID | `abc123...` |
| `SERPAPI_API_KEY` | SerpAPI search API key | `...` |
| `COGTRIX_WHATSAPP_URL` | Waha server URL | `http://localhost:3000` |
| `COGTRIX_WHATSAPP_API_KEY` | Waha API key | `yoursecretkey` |
| `COGTRIX_WHATSAPP_SESSION` | Waha session name | `default` |
| `COGTRIX_TELEGRAM_TOKEN` | Telegram bot token | `123456:ABC-DEF...` |

---

## Command Line Arguments

### General Options

```bash
python cogtrix.py [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider name |
| `--model NAME` | `-m` | Model name or model alias from config |
| `--session ID` | `-s` | Session ID for memory persistence |
| `--memory-mode MODE` | `-M` | Memory mode: `conversation`, `code`, `reasoning` |
| `--config-file FILE` | `-c` | Path to a specific config file (JSON or YAML). Bypasses the automatic config file search. |
| `--no-confirm` | `-y` | Skip all tool safety confirmations (auto-approve file writes, shell commands, etc.) |
| `--output FILE` | `-o` | Save responses to file. Non-interactive: single write. Interactive: append each exchange as Markdown. |
| `--debug` | | Enable debug mode (auto-enables `--log` and `--verbose`) |
| `--verbose` | `-v` | Log full LLM interactions: tokens, thinking, tool calls |
| `--log [FILE]` | | Enable logging to file (default: `cogtrix.log`) |
| `--tools LIST` | | Comma-separated tools to load (default: all) |
| `--check-config` | | Validate configuration and exit |

### Non-interactive Mode

Process a single prompt and exit (useful for scripting and automation):

```bash
python cogtrix.py --prompt "What is 2+2?"
python cogtrix.py --prompt-file task.txt
python cogtrix.py --prompt "Summarize this" -o summary.md
python cogtrix.py --prompt "Generate JSON" --no-stream -o data.json
```

| Option | Short | Description |
|--------|-------|-------------|
| `--prompt TEXT` | | Send a single prompt and exit |
| `--prompt-file FILE` | | Read prompt from file and exit |
| `--output FILE` | `-o` | Write response to file |
| `--no-stream` | | Disable streaming output |

### Tool Filtering

Control which tools are loaded at startup:

```bash
python cogtrix.py --tools none                    # No tools (pure LLM chat)
python cogtrix.py --tools minimal                 # Basic set (file ops + calculate)
python cogtrix.py --tools "search_web,calculate"  # Specific tools only
```

### RAG Ingestion Options

```bash
python cogtrix.py --ingest [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--ingest` | Build vector database and exit |
| `--docs-dir PATH` | Documents directory |
| `--vectordb-dir PATH` | Vector database output directory |
| `--embedding-provider NAME` | Embedding provider: `openai`, `ollama`, or `google` |
| `--embedding-model NAME` | Embedding model name |

### Setup Wizard

The setup wizard generates a valid Cogtrix config file through an interactive three-phase process: scripted LLM bootstrap, conversational Q&A, and YAML validation and write. It works for both first-time setup and editing an existing config.

```bash
python cogtrix.py --setup
python cogtrix.py --setup --setup-output ~/myproject/.cogtrix.yaml
python cogtrix.py --setup --setup-docs https://example.com/cogtrix-config-docs
```

| Option | Description |
|--------|-------------|
| `--setup` | Launch the interactive setup wizard and exit |
| `--setup-docs URL` | Fetch configuration documentation from URL instead of the bundled `docs/CONFIGURATION.md`. Useful when running the wizard against a different documentation version. |
| `--setup-output FILE` | Write the generated config to this path (default: `~/.cogtrix.yaml`) |

**How the wizard works:**

1. **Scripted bootstrap** — detects `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, and Ollama at `localhost:11434`. Prompts for provider type (`ollama`, `openai`, `anthropic`, `google`, or `xai`), model name, and API key if needed. Tests LLM connectivity before proceeding.
2. **LLM conversation** — loads the configuration reference (bundled or fetched), loads any existing config from the standard search paths, and runs an interactive Q&A loop. The wizard LLM asks targeted questions and produces a complete YAML config in a code fence when it has enough information. Type `quit` at any prompt to cancel.
3. **Validation and write** — extracts the YAML from the LLM response, injects the real API key collected during bootstrap, validates the result via an internal config round-trip, shows a masked preview for confirmation, and writes the file.

**Notes:**

- The wizard detects an existing config automatically and asks whether to edit it or start fresh.
- API keys entered during bootstrap are injected into the final YAML, so the LLM never sees the actual key value.
- The output file is shown after writing: `Config written to: ~/.cogtrix.yaml`.

---

## Complete Configuration Example

Below is a full configuration in both YAML and JSON. Both formats are functionally identical — pick whichever you prefer.

### YAML (`.cogtrix.yaml`)

```yaml
provider: my-server
session: default

# ─── LLM Providers ──────────────────────────────────────────────
providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
    model: qwen3:8b
  openai:
    type: openai
    model: gpt-4.1
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."
    model: llama-3.3-70b-versatile
  local-gpu:
    type: ollama
    base_url: "http://192.168.1.101:11434"
    model: qwen3-coder:30b-a3b

# ─── External Services ──────────────────────────────────────────
services:
  tavily:
    api_key: "tvly-..."
  exa:
    api_key: "exa-..."
  brave:
    api_key: "BSA..."
  openweather:
    api_key: "..."
  whatsapp:
    waha_url: "http://localhost:3000"
    allow_send: true
    allow_receive: true
    filter_mode: whitelist
    contacts: ["+14155551234"]
    phonebook:
      alice: "+14155551234"
  telegram:
    bot_token: "123456:ABC-DEF..."
    phonebook:
      alice: "123456789"

# ─── Models (chat + embedding) ───────────────────────────────────
models:
  fast: my-server/qwen3:8b
  smart: openai/gpt-4.1
  coder:
    provider: local-gpu
    model: qwen3-coder:30b-a3b
    temperature: 0.3
  embed-local:
    provider: local-gpu
    model: nomic-embed-text

# ─── Memory ─────────────────────────────────────────────────────
memory:
  mode: conversation
  modes:
    conversation:
      working_memory_size: 25
      summarization: true
      vector_recall_k: 3
    code:
      working_memory_size: 30
      max_files: 20
      summarization: true
      vector_recall_k: 3
    reasoning:
      working_memory_size: 30
      max_decisions: 20
      summarization: true
      vector_recall_k: 3

# ─── RAG ────────────────────────────────────────────────────────
rag:
  docs_dir: docs
  vectordb_dir: data/vectordb
  model: embed-local

# ─── Delegation ─────────────────────────────────────────────────
delegate:
  enabled: true
  default_timeout: 60
  allowed_models: [fast, smart, coder]

# ─── Research Delegate ───────────────────────────────────────────
research_delegate:
  enabled: true
  cap_ratio: 0.85
  timeout: 300

# ─── Prompt Optimizer ────────────────────────────────────────────
prompt_optimizer: true

# ─── Context Compression ────────────────────────────────────────
context_compression:
  enabled: true
  model: fast
  min_age: 6
  min_chars: 2000

# ─── MCP Servers (requires: uv pip install "cogtrix[mcp]") ──────
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    requires_confirmation: true
    timeout: 30
  # remote-api:
  #   url: http://localhost:8000/sse
  #   headers:
  #     Authorization: "Bearer token"
  #   requires_confirmation: false

# ─── Assistant Guardrails (under services.assistant) ─────────────
# services:
#   assistant:
#     guardrails:
#       enabled: true
#       max_input_length: 4000
#       unicode_checks: true
#       input_patterns: []
#       rate_limit:
#         per_minute: 10
#         per_hour: 60
#       encoding_detection:
#         enabled: true
#         min_score: 0.6
#       tool_call_guard:
#         enabled: true
#         injection_scan: true
#         path_blocking: true
#         exfiltration_detection: true
#         sensitive_paths: []
#       auto_blacklist:
#         enabled: true
#         max_violations: 2
#         window_minutes: 30
#       banned_output_strings: []
#       block_urls_in_output: true
#       pii_detection: true
#       llm_judge:
#         enabled: false
#         model: null
```

### JSON (`.cogtrix.json`)

```json
{
  "provider": "my-server",
  "session": "default",

  "providers": {
    "my-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "qwen3:8b"
    },
    "openai": {
      "type": "openai",
      "model": "gpt-4.1"
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-...",
      "model": "llama-3.3-70b-versatile"
    },
    "local-gpu": {
      "type": "ollama",
      "base_url": "http://192.168.1.101:11434",
      "model": "qwen3-coder:30b-a3b"
    }
  },

  "services": {
    "tavily": { "api_key": "tvly-..." },
    "exa": { "api_key": "exa-..." },
    "brave": { "api_key": "BSA..." },
    "openweather": { "api_key": "..." },
    "whatsapp": {
      "waha_url": "http://localhost:3000",
      "allow_send": true,
      "allow_receive": true,
      "filter_mode": "whitelist",
      "contacts": ["+14155551234"],
      "phonebook": { "alice": "+14155551234" }
    },
    "telegram": {
      "bot_token": "123456:ABC-DEF...",
      "phonebook": { "alice": "123456789" }
    },
    "assistant": {
      "guardrails": {
        "enabled": true,
        "max_input_length": 4000,
        "unicode_checks": true,
        "input_patterns": [],
        "rate_limit": {
          "per_minute": 10,
          "per_hour": 60
        },
        "encoding_detection": {
          "enabled": true,
          "min_score": 0.6
        },
        "tool_call_guard": {
          "enabled": true,
          "injection_scan": true,
          "path_blocking": true,
          "exfiltration_detection": true,
          "sensitive_paths": []
        },
        "auto_blacklist": {
          "enabled": true,
          "max_violations": 2,
          "window_minutes": 30
        },
        "banned_output_strings": [],
        "block_urls_in_output": true,
        "pii_detection": true,
        "llm_judge": {
          "enabled": false,
          "model": null
        }
      }
    }
  },

  "models": {
    "fast": "my-server/qwen3:8b",
    "smart": "openai/gpt-4.1",
    "coder": {
      "provider": "local-gpu",
      "model": "qwen3-coder:30b-a3b",
      "temperature": 0.3
    },
    "embed-local": {
      "provider": "local-gpu",
      "model": "nomic-embed-text"
    }
  },

  "memory": {
    "mode": "conversation",
    "modes": {
      "conversation": { "working_memory_size": 25, "summarization": true, "vector_recall_k": 3 },
      "code": { "working_memory_size": 30, "max_files": 20, "summarization": true, "vector_recall_k": 3 },
      "reasoning": { "working_memory_size": 30, "max_decisions": 20, "summarization": true, "vector_recall_k": 3 }
    }
  },

  "rag": {
    "docs_dir": "docs",
    "vectordb_dir": "data/vectordb",
    "model": "embed-local"
  },

  "delegate": {
    "enabled": true,
    "default_timeout": 60,
    "allowed_models": ["fast", "smart", "coder"]
  },

  "research_delegate": {
    "enabled": true,
    "cap_ratio": 0.85,
    "timeout": 300
  },

  "prompt_optimizer": true,

  "context_compression": {
    "enabled": true,
    "model": "fast",
    "min_age": 6,
    "min_chars": 2000
  },

  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "requires_confirmation": true,
      "timeout": 30
    }
  }

}
```

> **Note:** Both examples use `"providers"` (preferred). The legacy key `"inference"` still works as an alias.

---

## Interactive Commands

See the [Interactive Commands table in the README](../README.md#interactive-commands) for the full list of slash commands, or type `/help` inside a running session.

**Tip:** Commands like `/mode`, `/model`, `/provider`, and `/session` work in two ways: run them without arguments to display the current value, or pass a name to switch at runtime (e.g. `/mode code`).

### Line Editing

The interactive prompt supports full line editing via Python's `readline` module:

- **Left/Right arrows** — Move cursor within the line
- **Home/End** — Jump to beginning/end of line
- **Up/Down arrows** — Navigate input history
- **Ctrl+A / Ctrl+E** — Beginning/end of line (Emacs-style)
- **Ctrl+W** — Delete previous word

This works out of the box on Linux and macOS. On Windows, install `pyreadline3` for equivalent functionality.

---

## Debugging & Logging

Enable logging to troubleshoot issues:

```bash
# Enable logging to default file (cogtrix.log)
python cogtrix.py --log

# Enable logging to specific file
python cogtrix.py --log ~/my-logs/session.log

# Log full LLM interactions (tokens, thinking, tool calls)
python cogtrix.py --log -v

# Enable debug mode (auto-enables --log and --verbose)
python cogtrix.py --debug
python cogtrix.py --debug --log ~/debug.log
```

### Log Levels

| Mode | Level | What's Logged |
|------|-------|---------------|
| `--log` | INFO | User messages, agent responses, tool calls, errors |
| `--log -v` | INFO | Above plus: full LLM interactions, tokens, thinking content |
| `--debug` | DEBUG | All of the above plus: message details, context info, tool inputs/outputs |

### What Gets Logged

| Event | Level | Example |
|-------|-------|---------|
| User message | INFO | `User: What's the weather?` |
| Agent response | INFO | `Agent response` |
| Tool execution | INFO | `Tool: get_weather` |
| Tool input | DEBUG | `Tool input: {'location': 'Auckland'}` |
| Tool output | DEBUG | `Tool output: Current weather in...` |
| Memory context | DEBUG | `Context: mode=conversation, 10 messages` |
| Errors | ERROR | `Tool failed: get_weather - Connection error` |

### Example Log Output

```
2025-01-15 10:30:15.123 [INFO] [a1b2c3d4] User: What's the weather in Auckland?
2025-01-15 10:30:15.124 [DEBUG] [a1b2c3d4] Context: mode=conversation, 5 messages, ~1200 tokens
2025-01-15 10:30:16.500 [INFO] [a1b2c3d4] Tool: get_weather
2025-01-15 10:30:16.500 [DEBUG] [a1b2c3d4] Tool input: {'location': 'Auckland, New Zealand', 'units': 'metric'}
2025-01-15 10:30:17.200 [DEBUG] [a1b2c3d4] Tool output: Current weather in Auckland: 18°C, partly cloudy...
2025-01-15 10:30:18.500 [INFO] [a1b2c3d4] Agent response
```

The `[a1b2c3d4]` is a request ID that groups all log entries for a single user query.

### Debugging Tips

1. **Tool not being called?** Check if the agent outputs JSON text instead of calling the tool. This may indicate conversation history issues — try a fresh session with `-s new_session`.

2. **Timeout errors?** The model may be slow. Check the provider's status and consider using a faster model.

3. **Connection errors?** Verify the provider URL and that the service is running.

---

## See Also

- [PROVIDERS.md](PROVIDERS.md) — Provider setup guides
- [MEMORY_MODES.md](MEMORY_MODES.md) — Memory mode details
- [RAG_GUIDE.md](RAG_GUIDE.md) — Knowledge base setup
