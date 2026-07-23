# Cogtrix Configuration Reference

This page covers every way to configure Cogtrix — from the simplest environment variable to a full multi-provider config file. If you just want to get running, the [Quick Start in the README](../README.md#quick-start) is all you need; come back here when you want to customize.

## Table of Contents

- [Configuration Priority](#configuration-priority)
- [Configuration File](#configuration-file)
  - [Providers Section](#providers-section)
  - [Models Section](#models-section)
  - [Research Delegate Section](#research-delegate-section)
  - [Decision Accountability](#decision-accountability)
  - [Prompt Optimizer](#prompt-optimizer)
  - [Context Compression](#context-compression)
  - [MCP Servers](#mcp-servers)
  - [Tool Loading](#tool-loading)
  - [Assistant Mode](#assistant-mode)
  - [Contact Prompts](#contact-prompts)
  - [Workflows](#workflows)
  - [Scheduled Reply Delivery](#scheduled-reply-delivery)
  - [Deferred Message Processing](#deferred-message-processing)
  - [Response Timing / Quiet Hours](#response-timing--quiet-hours)
  - [Assistant Guardrails](#assistant-guardrails)
- [Environment Variables](#environment-variables)
- [Command Line Arguments](#command-line-arguments)
  - [Setup Wizard](#setup-wizard)
- [Complete Configuration Example](#complete-configuration-example)
- [Migration Guide](#migration-guide)
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
session: default
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session` | string | `"default"` | Session ID for memory persistence |

The active model is selected via `models.default` (see [Models Section](#models-section)). The legacy top-level `provider` and `model` keys still work but are deprecated — they are auto-migrated at load time.

### Providers Section

Providers hold connection info only — the type, endpoint, and credentials needed to reach an LLM API. Model settings (model name, temperature, context window, max tokens) live in the [Models Section](#models-section) instead.

```yaml
providers:
  spark-cluster:
    type: openai
    base_url: "http://192.168.70.254:8080/v1"
    api_key: "sk-..."
  openai:
    type: openai
    api_key: "sk-..."
  local:
    type: ollama
    base_url: "http://localhost:11434"
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."
```

> **Note:** The key `"providers"` is preferred. The legacy key `"inference"` still works as an alias for backward compatibility.

#### Provider Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `type` | string | Yes | Provider type: `"openai"`, `"ollama"`, `"anthropic"`, or `"google"` (case-insensitive) |
| `base_url` | string | No | API endpoint URL |
| `api_key` | string | No | API key. Omit or leave empty for unauthenticated OpenAI-compatible endpoints (vLLM, LM Studio). Required for OpenAI, Groq, Together, Anthropic, Google, xAI, and DeepSeek. Not used by Ollama. |
| `tool_instructions` | string | No | Custom tool-call formatting instructions appended to the system prompt. Not injected by default — `bind_tools()` handles formatting at the API level. Set a non-empty string only for providers that need explicit guidance. |

#### Provider Types

| Type | Use For | Default Model | Default Base URL |
|------|---------|---------------|------------------|
| `openai` | OpenAI, Groq, Together, vLLM, LocalAI | `gpt-4.1-mini` | `https://api.openai.com/v1` |
| `ollama` | Ollama servers | `qwen3:8b` | `http://localhost:11434` |
| `anthropic` | Anthropic Claude | `claude-sonnet-4-5` | SDK default |
| `google` | Google Gemini | `gemini-2.5-flash` | SDK default |

**xAI (Grok)** and **DeepSeek** use `type: openai` with a custom `base_url` (`https://api.x.ai/v1` and `https://api.deepseek.com/v1` respectively). The setup wizard offers both as named choices and auto-detects `XAI_API_KEY` / `DEEPSEEK_API_KEY` from the environment.

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
  vectordb_dir: vectordb
  chunk_size: 2000
  chunk_overlap: 200
  model: embed-local
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `docs_dir` | string | `"docs"` | Source documents directory |
| `vectordb_dir` | string | `"vectordb"` | Vector database output directory |
| `chunk_size` | int | `2000` | Text chunk size in characters |
| `chunk_overlap` | int | `200` | Overlap between chunks |
| `model` | string | `null` | Model name from the `models` registry to use for embeddings. Falls back to the active provider when not set. |

**Note:** The `model` field references a named entry in the top-level `models` registry. Define an embedding model there and point `rag.model` at it. The provider connection details (type, base_url, api_key) are resolved automatically from the matching provider config.

See [RAG_GUIDE.md](RAG_GUIDE.md) for detailed setup instructions.

### Models Section

The `models` registry assigns short names to specific `provider/model` combinations. All model settings (model name, temperature, context window, max output tokens) live here. Providers hold only connection info.

The `models.default` key selects which model alias is active when Cogtrix starts. It is the primary way to choose which model to use.

```yaml
models:
  default: oss              # active model alias at startup

  oss:
    provider: spark-cluster
    model: gpt-oss
    temperature: 0.5

  gpt-4o:
    provider: openai
    model: gpt-4o
    temperature: 0.7

  local-qwen:
    provider: local
    model: qwen3:8b
    context_window: 131072

  embed:
    provider: spark-cluster
    model: qwen3-embedding
    temperature: 0.0

  regular: spark-cluster/gpt-oss   # string shorthand
```

The `models` registry is used by:

- **`models.default`** — selects the active model alias at startup
- The **`-m` CLI flag** — start Cogtrix with any model alias: `python cogtrix.py -m gpt-4o`
- The **`/model` command** — switch at runtime: `/model local-qwen`
- The **delegation tools** — the agent uses model aliases to pick the best model for a subtask
- The **`rag.model` field** — reference an embedding model by alias
- The **`context_compression.model` field** — reference a compression model by alias

> **Backward compatibility:** The key `model_aliases` still works in config files as an alias for `models`. New configs should use `models`.

#### Model Entry Formats

**String shorthand** — `"provider/model"` creates a minimal model entry with no overrides:

```yaml
models:
  regular: spark-cluster/gpt-oss
  fast: local/qwen3:8b
```

**Object format** — full control over all model-level settings:

```yaml
models:
  coder:
    provider: local
    model: qwen3-coder
    temperature: 0.3
    context_window: 32768
    max_tokens: 8192
```

#### Model Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | Yes | References a key in the `providers` section |
| `model` | string | Yes | Model name as the provider expects it |
| `temperature` | float | No | Sampling temperature, 0.0–2.0 |
| `context_window` | int | No | Context window size in tokens (>= 256). Forwarded to Ollama as `num_ctx`; silently ignored for OpenAI, Anthropic, and Google. Accepted aliases: `context_length`, `num_ctx`. |
| `max_tokens` | int | No | Maximum output tokens per LLM call (>= 1) |

#### Using Models

```bash
python cogtrix.py -m oss          # Use the "oss" model alias
python cogtrix.py -m local-qwen   # Use local-qwen with its configured context_window
```

At runtime:
```
You: /model gpt-4o
Switched to model gpt-4o (openai)
```

The `/model` command lists all aliases with an active marker (`*`) next to the current selection. The `/provider` command is read-only — use `/model` to switch models.

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
| `auto` | bool | `false` | When `true`, automatically trigger the research delegate whenever the agent's tool output exceeds `auto_threshold` of the context window. |
| `auto_threshold` | float | `0.50` | Fraction of context used by tool output that triggers automatic delegation when `auto: true`. |

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

### Decision Accountability

Decision accountability adds an explicit self-debate layer to the agent's reasoning. When enabled, the agent is instructed to produce a structured plan with assumptions and evidence, then generate a counter-plan ("why this might be wrong") before acting. Responses where the adjusted confidence falls below the threshold receive an uncertainty note so you can review before proceeding.

**Off by default** — this feature is opt-in. Enable it for high-stakes autonomous work (code changes, shell commands, API calls) where traceable reasoning matters.

```yaml
decision_accountability:
  enabled: true
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the self-debate prompt and response parsing. Off by default — opt in when you need traceable reasoning. |
| `min_confidence_threshold` | float | `7.0` | Adjusted confidence (0–10) below which the agent appends an uncertainty note. Adjusted confidence = base confidence − 1.0 per identified critical flaw. |
| `require_counter_plan` | bool | `true` | When true, the accountability prompt instructs the agent to always produce a counter-plan before acting. |
| `report_uncertainty` | bool | `true` | When true, responses where adjusted confidence falls below the threshold receive a visible `⚠️ Decision accountability:` note. |

**How it works:**

1. When `enabled: true`, Cogtrix appends the accountability block to the system prompt at session start. The block instructs the agent to structure each action plan using named delimiters (`---PLAN---`, `---ASSUMPTIONS---`, `---EVIDENCE---`, `---CONFIDENCE---`, `---COUNTER-PLAN---`, `---FLAWS---`).
2. After every model response, Cogtrix parses this structure from the response text.
3. The confidence score is adjusted: each identified critical flaw reduces it by 1.0.
4. When the adjusted confidence falls below `min_confidence_threshold` and `report_uncertainty: true`, a note is appended to the response:

```
⚠️ Decision accountability: confidence 5.0/10 with 2 critical flaw(s): Missing validation;
No rollback path. Adjusted confidence 3.0/10 is below threshold 7.0. Proceeding with caution.
```

5. The full structured output (plan, assumptions, evidence, counter-plan, flaws, confidence) is logged at INFO level under `decision_accountability:` for auditing.

**Interaction with `/think`:**

Decision accountability and Deep Think (`/think`) are independent features. Deep Think explores multiple solution branches in parallel. Decision accountability adds a plan/counter-plan layer to every agent action turn. Both can be active at the same time.

**When to use:**

- Enable for agents running autonomously on sensitive tasks (file edits, git operations, deployments).
- Keep disabled for conversational sessions, simple lookups, or when using fast/small models where the extra prompt tokens would hurt performance.
- The additional ~600 tokens in the system prompt add roughly 0.2–0.5s to TTFT depending on provider; no extra LLM calls are made (the agent reasons within its own response).

### Task Ownership Classifier

The task ownership classifier analyses each user prompt before the agent starts to determine whether the request asks the agent to *execute* an action or *explain* how to do it. This prevents the agent from acting when the user only wants information (e.g. "check how to install gh" should explain, not install).

**On by default.** Disable only if you want the agent to always proceed without ownership analysis.

```yaml
task_ownership_classifier:
  enabled: true            # set to false to disable entirely
  llm_fallback: false      # when true, calls the LLM for ambiguous prompts (adds latency)
  ambiguous_action: ask    # what to do when ownership is ambiguous: ask | inform | execute
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable pre-prompt ownership classification. |
| `llm_fallback` | bool | `false` | Use an LLM micro-call for borderline cases. Improves accuracy at the cost of added latency. Off by default. |
| `ambiguous_action` | string | `"ask"` | How to handle a prompt whose ownership cannot be determined. `ask` — inject a clarification constraint into the system prompt and run the agent normally; the agent asks one focused question and does not execute until intent is confirmed; `inform` — treat as informational; `execute` — treat as execution request. |

### Pre-Action Confirmation

When enabled, the agent is instructed to state what it is about to do and ask "Shall I proceed?" before any irreversible operation (delete, install, deploy to production, drop table, overwrite data).

**Off by default** — opt in for autonomous workflows where you want explicit consent gates before destructive actions.

```yaml
pre_action_confirmation:
  enabled: true
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Inject the pre-action confirmation prompt. The agent will pause and ask before irreversible operations. |

**Exception:** If the user's message already contains explicit execution consent ("go ahead", "yes do it", "proceed", "confirmed"), the agent skips the confirmation step.

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
  model: fast
  min_age: 8       # call_model cycles before eligible (default: 6)
  min_chars: 6000  # minimum content length to qualify (default: 2000)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context_compression` | bool or object | `true` | Enable/disable context compression, or configure thresholds |
| `enabled` | bool | `true` | Enable/disable compression when using the object form |
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

By default, file write operations (`write_file`, `append_file`, `patch_file`) are restricted to the current working directory. You can extend this with additional directories:

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

Read operations default to the working directory and application install directory. To allow reads from additional directories, use `allowed_read_paths`.

**Priority:** CLI (`--allow-write-path`) > env var (`COGTRIX_ALLOWED_WRITE_PATHS`) > config file.

### Allowed Read Paths

By default, file read operations (`read_file`, `list_directory`) are restricted to the current working directory and the application install directory. You can extend this with additional directories for read access:

```yaml
allowed_read_paths:
  - /workspace
  - /data/external
```

This is especially useful in Docker deployments where the project is mounted at a different location than the working directory. For example, if you mount your project at `/workspace` but the container's working directory is `/app`:

```bash
# Via environment variable (colon-separated)
docker run -v /home/user/project:/workspace:ro \
    -e COGTRIX_ALLOWED_READ_PATHS=/workspace \
    cogtrix --prompt "Analyze /workspace/docs"

# Via CLI flag (repeatable)
cogtrix.py --allow-read-path /workspace
```

**Priority:** CLI (`--allow-read-path`) > env var (`COGTRIX_ALLOWED_READ_PATHS`) > config file.

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

#### Docker (SSE via supergateway)

When running Cogtrix in Docker, stdio MCP servers can't be spawned directly because each service runs in its own container. Use [supergateway](https://github.com/supercorp-ai/supergateway) to bridge stdio servers to SSE:

```yaml
# Example: run supergateway as a separate container
# docker run -d --name mcp-filesystem -v mcp-data:/data supercorp/supergateway \
#   --stdio "npx -y @modelcontextprotocol/server-filesystem /data" --port 8000
```

Then configure Cogtrix to connect via SSE:

```yaml
# .cogtrix.yaml
mcp_servers:
  filesystem:
    url: http://mcp-filesystem:8000/sse
    requires_confirmation: false
```

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

#### Pinning tools with `--activate-tools`

Use `--activate-tools` to pin specific tools as active on startup while keeping the on-demand system for everything else:

```bash
python cogtrix.py --activate-tools web_search,shell
python cogtrix.py --activate-tools query_knowledge_base -M code
```

Pinned tools stay active across prompt cycles — they are not auto-unloaded between turns. You can unpin them interactively with `/tools unload <name>`.

#### Two-tier tool loading

Tools loaded by the agent via `request_tools` during a turn are **agent-loaded** — they are automatically unloaded at the start of the next prompt cycle so the LLM doesn't carry stale tools between turns. Tools loaded manually (via `/tools load`, `--activate-tools`, or the API `PATCH /sessions/{id}/tools` endpoint) are **pinned** — they persist across prompt cycles until explicitly unloaded.

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
| SearXNG | `searxng_search` | Included (`requests`) | `SEARXNG_BASE_URL` | Self-hosted |

**Configuring SearXNG:**

[SearXNG](https://docs.searxng.org/) is a self-hosted meta-search engine. To enable it, set `SEARXNG_BASE_URL` to your instance URL:

```bash
export SEARXNG_BASE_URL=http://localhost:8888
```

Or in config YAML:

```yaml
services:
  searxng:
    base_url: http://localhost:8888
```

The `searxng_search` tool becomes available automatically once the URL is configured.

**Installing optional search packages:**

Tavily, Exa, and SerpAPI need extra Python packages not included by default:

```bash
# All at once (recommended)
uv sync --extra search

# Or individually with pip
pip install tavily-python exa-py google-search-results
```

Brave, Google, and SearXNG use only `requests`, which is already a core dependency.

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
    filter_mode: allow
    contacts: ["+14155551234", "+442071234567"]
    phonebook:
      alice: "+14155551234"
      bob: "+442071234567"
    contact_prompts:
      alice: |
        You are replying to Alice on behalf of the user.
        Be friendly and casual.
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
| `filter_mode` | string | `"none"` | `"none"`, `"allow"`, `"ignore"`, or `"blacklist"`. Legacy `"whitelist"` maps to `"allow"`. |
| `contacts` | array | `[]` | E.164 phone numbers for the filter list |
| `phonebook` | object | `{}` | Nickname → phone number map |
| `contact_prompts` | object | `{}` | Per-contact system prompts (see [Contact Prompts](#contact-prompts)) |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages to this length |
| `overview_limit` | int | `50` | Maximum number of chats returned per overview poll cycle. A warning is logged when the response reaches this limit, indicating that some chats may have been missed. |
| `message_fetch_limit` | int | `50` | Maximum number of messages fetched per chat per poll cycle. Prevents silent message loss when a chat receives more messages than the limit between poll intervals. |
| `ignore_archived` | bool | `true` | Skip archived chats during polling. When enabled, chats marked as archived in WhatsApp are not fetched or processed. |
| `ignore_older_than` | string | — | Skip messages older than this duration. Accepts human-readable strings like `"24h"`, `"30m"`, `"7d"`, `"1d12h"`. Disabled by default (all messages processed). |
| `lid_negative_ttl` | float | `300.0` | Cache duration (seconds) for failed LID-to-phone resolutions |

Filter mode behaviour:
- `none` — respond to all contacts
- `allow` — only respond to contacts in the `contacts` list
- `ignore` — skip listed contacts (no response, message kept)
- `blacklist` — delete the message and archive the chat for listed contacts

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
    filter_mode: allow
    contacts: ["123456789", "@alice_username"]
    phonebook:
      alice: "123456789"
      team: "-1001234567890"
    contact_prompts:
      alice: |
        You are replying to Alice on behalf of the user.
        Be friendly and casual.
    rate_limit: 30
    max_message_length: 4096
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bot_token` | string | — | Bot token from @BotFather (required) |
| `allow_send` | bool | `true` | Enable send tools (`telegram_send`, `telegram_send_photo`) |
| `allow_receive` | bool | `true` | Enable receive tool (`telegram_check`) |
| `require_confirmation` | bool | `true` | Prompt user before sending messages |
| `filter_mode` | string | `"none"` | `"none"`, `"allow"`, `"ignore"`, or `"blacklist"`. Legacy `"whitelist"` maps to `"allow"`. |
| `contacts` | array | `[]` | Chat IDs or @usernames for the filter list |
| `phonebook` | object | `{}` | Nickname → chat ID map |
| `contact_prompts` | object | `{}` | Per-contact system prompts (see [Contact Prompts](#contact-prompts)) |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages to this length |
| `ignore_older_than` | string | — | Skip messages older than this duration. Accepts human-readable strings like `"24h"`, `"30m"`, `"7d"`, `"1d12h"`. Disabled by default (all messages processed). |

Filter mode behaviour:
- `none` — respond to all contacts
- `allow` — only respond to contacts in the `contacts` list
- `ignore` — skip listed contacts (no response, message kept)
- `blacklist` — delete the message and archive the chat for listed contacts

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
    debounce_seconds: 3.0      # quiet window before rapid messages are batched
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
      datamarking: true                # Microsoft Spotlighting prompt injection defense
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
| `excluded_tools` | array | `[]` | Additional tools to exclude. Messaging tools, shell, write, and read tools are always excluded. Queue management tools (`schedule_reply`, `queue_reply`, `edit_last_reply`, `list_scheduled_messages`, `edit_scheduled_message`, `cancel_scheduled_message`) and deferral tools (`defer_processing`, `suppress_reply`) can also be added here. |
| `debounce_seconds` | float | `3.0` | Quiet window in seconds before rapid messages from the same chat are batched into a single agent turn. Increase to tolerate longer bursts; decrease for faster single-message response. |
| `dispatch_interval` | float | `30.0` | Seconds between scheduler checks for due messages |
| `channels.{name}.enabled` | bool | `true` | Enable/disable a specific channel |
| `channels.{name}.poll_interval` | float | varies | Seconds between poll cycles |
| `channels.<name>.poll_interval_min` | float | base interval | Minimum poll interval (seconds); polling backs off on idle, recovers on activity |
| `channels.<name>.poll_interval_max` | float | `60.0` | Maximum poll interval during idle backoff |
| `channels.<name>.poll_backoff_factor` | float | `1.5` | Multiplier when no messages received (clamped `>= 1.0`) |
| `channels.<name>.poll_recovery_factor` | float | `2.0` | Divisor when messages received (clamped `>= 1.0`) |
| `channels.telegram.long_poll_timeout` | int | `30` | Telegram getUpdates timeout |
| `knowledge.enabled` | bool | `true` | Enable cross-chat fact extraction and recall |
| `knowledge.extraction_model` | string | `null` | Model alias for fact extraction (null = main LLM) |
| `knowledge.recall_k` | int | `5` | Number of facts recalled per query |
| `knowledge.max_facts` | int | `10000` | Maximum stored facts |
| `knowledge.data_dir` | string | `"data"` | Base directory for knowledge persistence (facts.json, FAISS index) |
| `guardrails.datamarking` | bool | `true` | Enable Microsoft Spotlighting (datamarking) — interleaves a random token at word boundaries in user messages so the LLM treats them as data, not instructions |
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
2. New messages are passed to `MessageBuffer`, which resets a per-chat debounce timer. When the timer expires (after `debounce_seconds` of silence from that chat), all buffered messages are concatenated and dispatched as a single agent turn via `handle_batch()`. A single message with no follow-ups dispatches immediately after the quiet window.
3. Each incoming message (or batch) is checked by the `GuardrailPipeline` (rate limit, input validation, injection detection). Blocked messages receive a canned reply without reaching the agent.
4. Each `(channel, chat_id)` pair gets an independent `ConversationMemoryManager` — no context blending between chats.
5. The agent runs with the same tool pipeline as interactive mode (minus excluded tools). Up to eight message management tools are injected per turn: `schedule_reply`, `queue_reply`, `edit_last_reply` (only when a prior message ID is available), `list_scheduled_messages`, `edit_scheduled_message`, `cancel_scheduled_message`, `defer_processing` (only when deferral is enabled and below max depth), and `suppress_reply` (only during re-processing passes).
6. After each turn, durable facts are extracted and stored in a shared knowledge store (`data/knowledge/facts.json`).
7. On each new message, relevant facts are recalled and injected into the agent's context — enabling cross-chat knowledge without exposing raw conversation history.
8. The agent response is routed by `_route_response`: edit and schedule paths run independently — both can fire in the same turn. Output is sanitized (PII redaction, URL stripping, banned string removal) in each delivery branch before being sent and before being written to memory.
9. SIGINT/SIGTERM triggers graceful shutdown: all sessions saved, knowledge store persisted.

**Prerequisites:** WhatsApp requires a running Waha container. Telegram requires a bot token. Both must be configured in their respective `services.whatsapp` / `services.telegram` sections.

### Contact Prompts

`contact_prompts` lets operators assign a per-contact system prompt that **replaces** the default assistant system prompt entirely for that contact. Configure it inside the channel config (`services.whatsapp` or `services.telegram`), keyed by the same names used in `phonebook`.

```yaml
services:
  whatsapp:
    phonebook:
      alice: "+1234567890"
    contact_prompts:
      alice: |
        You are replying to Alice on behalf of the user.
        Be friendly and casual. Use the schedule_reply tool
        to delay responses by 1-3 hours.
      # Or reference a file:
      # alice: /path/to/alice_prompt.txt
```

**Matching:** the handler looks up the incoming message's phone number / chat ID in `phonebook` to find a contact name, then checks `contact_prompts` for that name. If no match is found, or the resolved prompt is empty, the default assistant system prompt is used unchanged.

**File paths:** a value that starts with `/`, `~`, `./`, or `../` is treated as a file path. Relative paths are resolved against the `data_dir` and must remain inside it (path containment enforced). All other values are used as inline prompt text.

### Workflows

Workflows bundle a system prompt, a per-workflow FAISS knowledge base, and a tool policy into a named, reusable unit. A chat can be bound to a workflow manually, via auto-detection, or inherited from a `contact_prompts` entry. Workflows are stored as YAML files in `data/workflows/<id>/workflow.yaml`.

**Workflow definition (`data/workflows/bike-sales/workflow.yaml`):**

```yaml
id: "bike-sales"                     # Required. URL-safe slug (must match directory name).
name: "Bike Sales Assistant"         # Required. Human-readable label.
description: "Specialist assistant for bike sales inquiries"

system_prompt: |                     # Optional. Overrides global system_prompt.
  You are a specialist bike sales advisor...
# system_prompt_file: prompts/bike.txt  # Alternative: path to a file (resolved against data_dir).

knowledge_base: true                 # If true, per-workflow FAISS index at
                                     # data/workflows/bike-sales/vectordb/faiss_index/
                                     # is searched alongside the global index.

tool_policy:
  excluded_tools: []                 # Tools to block for this workflow.
  additional_approved_tools: []      # Tools auto-approved without confirmation.

auto_detect:
  enabled: false
  keywords: ["bike", "bicycle"]      # Case-insensitive substring matches.
  patterns: ["\\bbike\\b"]           # Python regex patterns.
  min_confidence: 1                  # Keyword matches needed to trigger.
```

**Chat-to-workflow bindings** are stored in `data/workflows/bindings.json` and managed via the API or auto-detection:

```json
{
  "whatsapp::14155551234@c.us": "bike-sales",
  "telegram::987654321": "support-desk"
}
```

**Resolution order** (first match wins):

1. **Explicit binding** — `bindings.json` entry for this `session_key`
2. **Contact prompt fallback** — if a `contact_prompts` entry exists for the sender, it is used as an ephemeral workflow (not persisted)
3. **Auto-detect** — if any workflow has `auto_detect.enabled: true`, incoming messages are scored against keywords and regex patterns; the highest-scoring workflow above `min_confidence` is assigned and persisted as a binding
4. **No match** — global `system_prompt` and default tool policy apply

**API management:** 11 CRUD endpoints at `/api/v1/assistant/workflows/` — create, list, get, update, delete workflows; upload and manage per-workflow documents; bind and unbind chats. See the [API Reference](api/openapi.yaml) for details.

**Per-workflow knowledge base:** when `knowledge_base: true`, upload documents to `data/workflows/<id>/docs/` via the API. A FAISS index is built at `data/workflows/<id>/vectordb/faiss_index/` and searched alongside the global index when the `query_knowledge_base` tool runs for a chat bound to that workflow.

### Scheduled Reply Delivery

Up to eight message management tools are injected automatically when assistant mode is active — no extra config is required to enable them. They can be blocked via `excluded_tools` if not needed.

| Tool | Purpose |
|------|---------|
| `schedule_reply` | Queue a reply for deferred delivery. Provide the full reply text and a delay in minutes (1–1440). |
| `queue_reply` | Append a message after the queue tail for this chat. Supports multiple calls per turn with optional `gap_minutes` spacing. |
| `edit_last_reply` | Edit/replace the most recently sent message in this chat. Only available after at least one reply has been sent in the session. |
| `list_scheduled_messages` | List pending queued messages. Filter by `recipient` (phone/name substring), `chat_id` (exact), or `contact_name` (phonebook key). Returns short IDs for use with the edit and cancel tools. |
| `edit_scheduled_message` | Update the text and/or reschedule the delivery time of a pending message (identified by short ID prefix). |
| `cancel_scheduled_message` | Cancel a specific pending message so it will not be delivered. |
| `defer_processing` | Postpone the reasoning pass without sending any reply. Only injected when deferral is enabled and below max depth. |
| `suppress_reply` | Send nothing and skip memory update. Only injected during re-processing passes. |

The agent decides when to use these tools based on instructions in its system prompt (or a `contact_prompts` entry).

**Scheduled message behavior:**

- Queued messages are persisted to `data/assistant/schedule.json` and survive restarts. Each record includes a `recipient` field (human-readable phone, username, or display name) for filtering via `list_scheduled_messages`.
- Delivery is retried up to 3 times on failure, with backoffs of 30 s, 2 min, and 10 min.
- Messages that are still pending more than 2 hours past their scheduled time are marked `expired`.
- Terminal-state messages (sent, cancelled, failed, expired) are cleaned up after 24 hours.
- When a new message arrives from the same chat, any pending scheduled reply for that chat is cancelled automatically.

**Message editing behavior:**

- `edit_last_reply` calls `Channel.edit_message()` on the channel that originally sent the message. WhatsApp and Telegram both support message editing; channels that do not implement it return a failure result (the tool reports the error to the agent but does not raise an exception).
- Only one edit per agent turn is allowed (idempotency guard). If the agent calls `edit_last_reply` multiple times in a single turn, only the first call takes effect.

### Deferred Message Processing

The deferral system lets the agent postpone its reasoning pass via the `defer_processing` tool. Messages arriving during a deferral are coalesced with the original batch and re-processed together when the timer fires.

```yaml
services:
  assistant:
    deferral:
      enabled: true
      max_depth: 3
      check_interval: 10
      stale_threshold: 7200
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable the deferral system |
| `max_depth` | int | `3` | Maximum re-processing depth (prevents infinite deferral loops) |
| `check_interval` | float | `10.0` | Seconds between checks for due deferrals |
| `stale_threshold` | float | `7200.0` | Seconds before a deferred record is considered stale and cancelled |

Deferred records are persisted to `data/assistant/deferrals.json` and survive restarts. Records in `"firing"` state are reset to `"pending"` on reload (at-least-once semantics).

### Outbound Campaigns

The campaign system enables multi-contact outbound messaging with automatic follow-ups, escalation, and goal classification. Campaigns are managed via the API (`/api/v1/assistant/campaigns/*`) and tracked in `data/assistant/campaigns.json`.

```yaml
services:
  assistant:
    campaigns:
      enabled: true
      check_interval: 60
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable the campaign system |
| `check_interval` | float | `60.0` | Seconds between follow-up check passes |

**Campaign lifecycle:**

1. **Create** — `POST /api/v1/assistant/campaigns` with name, goal, instructions, and target contacts
2. **Launch** — `POST /api/v1/assistant/campaigns/{id}/launch` sends initial outbound to all pending targets (or set `auto_launch: true` on create)
3. **Track** — incoming replies from campaign targets are tracked automatically; the `report_campaign_outcome` tool is injected so the agent can classify each target as `completed`, `failed`, or `in_progress`
4. **Follow-up** — the background thread sends follow-ups to non-responsive targets after `follow_up_interval_hours` (default 24h); escalates after `max_follow_ups` (default 3)
5. **Complete** — campaign auto-completes when all targets reach a terminal state (completed, failed, or escalated)

**Per-campaign settings** (set at creation time):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_follow_ups` | int | `3` | Maximum follow-ups per target before escalation (0–20) |
| `follow_up_interval_hours` | float | `24.0` | Hours between follow-up attempts (0.5–720) |

### Response Timing / Quiet Hours

`response_timing` under `services.assistant` defers scheduled replies that would be delivered during a contact's quiet hours. Entries are keyed by contact name; `_default` applies to any contact without a specific entry.

```yaml
services:
  assistant:
    response_timing:
      _default:
        timezone: "UTC"
        quiet_hours: [23, 8]   # 11 pm to 8 am
      alice:
        timezone: "America/New_York"
        quiet_hours: [22, 7]   # 10 pm to 7 am EST
```

| Field | Type | Description |
|-------|------|-------------|
| `timezone` | string | IANA timezone name (e.g. `"Asia/Dubai"`, `"America/New_York"`). Defaults to `"UTC"` if omitted or invalid. |
| `quiet_hours` | `[start, end]` | Two-element list of hours (0–23). The quiet window runs from `start` up to (but not including) `end`. Wraps midnight when `start > end` (e.g. `[23, 8]` covers 11 pm–8 am). `start` and `end` must differ. |

When a scheduled reply's delivery time falls inside the quiet window, the scheduler defers it to the moment the window ends (`end_hour:00` in the contact's timezone).

Quiet hours only affect the `MessageScheduler` — they do not block immediate (non-scheduled) replies.

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
| `COGTRIX_CONFIG_FILE` | Path to a specific config file (bypasses automatic search) | `/etc/cogtrix/config.yaml` |
| `COGTRIX_MODEL` | Active model alias (sets `models.default` at runtime) | `oss` |
| `COGTRIX_SESSION` | Session ID | `my-project` |
| `COGTRIX_MEMORY_MODE` | Memory mode | `code` |
| `COGTRIX_DATA_DIR` | Root directory for data storage | `./data` |
| `COGTRIX_ALLOWED_READ_PATHS` | Colon-separated list of absolute directory paths the agent is allowed to read. When set, restricts file read operations to these directories. | `/workspace:/data/external` |
| `COGTRIX_ALLOWED_WRITE_PATHS` | Colon-separated extra write-allowed paths | `/app:/data` |
| `COGTRIX_OLLAMA` | Ollama server address (`host` or `host:port`) | `192.168.1.100` or `192.168.1.100:8080` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `XAI_API_KEY` | xAI (Grok) API key | `xai-...` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | `sk-...` |
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
| `SEARXNG_BASE_URL` | SearXNG instance URL. When set, enables the `searxng_search` tool. | `http://localhost:8888` |
| `COGTRIX_WHATSAPP_URL` | Waha server URL | `http://localhost:3000` |
| `COGTRIX_WHATSAPP_API_KEY` | Waha API key | `yoursecretkey` |
| `COGTRIX_WHATSAPP_SESSION` | Waha session name | `default` |
| `COGTRIX_TELEGRAM_TOKEN` | Telegram bot token | `123456:ABC-DEF...` |
| `COGTRIX_JWT_SECRET` | JWT signing secret for API mode (min 32 chars, required) | `your-secret-key-at-least-32-chars` |
| `COGTRIX_DB_URL` | Database URL for API mode (default: SQLite `aiosqlite`) | `postgresql+asyncpg://user:pass@host/db` |
| `COGTRIX_CORS_ORIGINS` | Comma-separated CORS allowed origins for API mode | `http://localhost:5173,https://app.example.com` |
| `COGTRIX_API_HOST` | API server bind host (default `0.0.0.0`) | `127.0.0.1` |
| `COGTRIX_API_PORT` | API server bind port (default `8000`) | `3001` |
| `COGTRIX_API_WORKERS` | Number of uvicorn workers (default `1`) | `4` |

#### Docker Healthcheck

The container image includes a built-in healthcheck that probes `GET /api/v1/health` using Python's stdlib `urllib` (no `curl` or `wget` required). This enables `depends_on: condition: service_healthy` in docker-compose:

```yaml
services:
  cogtrix:
    image: ghcr.io/northlandpositronics/cogtrix:latest
    command: ["api"]
    environment:
      COGTRIX_JWT_SECRET: "your-secret-key-at-least-32-chars"
    ports:
      - "8000:8000"

  webui:
    image: ghcr.io/northlandpositronics/cogtrix-webui:latest
    depends_on:
      cogtrix:
        condition: service_healthy
    ports:
      - "5173:80"
```

The healthcheck runs every 30 seconds with a 5-second deadline (4-second socket timeout + 1 second margin), starting 15 seconds after container launch. It only passes in API mode — CLI and assistant modes do not expose the health endpoint.

---

## Command Line Arguments

### General Options

```bash
python cogtrix.py [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--model NAME` | `-m` | Active model alias from the `models` registry |
| `--session ID` | `-s` | Session ID for memory persistence |
| `--memory-mode MODE` | `-M` | Memory mode: `conversation`, `code`, `reasoning` |
| `--config-file FILE` | `-c` | Path to a specific config file (JSON or YAML). Bypasses the automatic config file search. |
| `--data-dir PATH` | | Root directory for data storage (history, vectordb, assistant state) |
| `--no-confirm` | `-y` | Skip all tool safety confirmations (auto-approve file writes, shell commands, etc.) |
| `--output FILE` | `-o` | Save responses to file. Non-interactive: single write. Interactive: append each exchange as Markdown. |
| `--debug` | | Enable debug mode (auto-enables `--log` and `--verbose`) |
| `--verbose` | `-v` | Log full LLM interactions: tokens, thinking, tool calls |
| `--log [FILE]` | | Enable logging to file (default: `cogtrix.log`) |
| `--tools LIST` | | Comma-separated tools to load (default: all) |
| `--activate-tools LIST` | | Comma-separated tools to pin as active on startup |
| `--assistant` | | Run as a headless WhatsApp/Telegram messaging daemon |
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

### Pinning Tools at Startup

Pin specific on-demand tools as active without changing the overall tool filter:

```bash
python cogtrix.py --activate-tools web_search,shell,write_file
```

Pinned tools persist across prompt cycles (unlike agent-loaded tools which are cleared between turns). Unpin interactively with `/tools unload <name>`.

### RAG Ingestion Options

```bash
python cogtrix.py --ingest [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--ingest` | Build vector database and exit |
| `--docs-dir PATH` | Documents directory |
| `--vectordb-dir PATH` | Vector database output directory |
| `--embedding-provider NAME` | Embedding provider: `openai` or `ollama` |
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

1. **Scripted bootstrap** — detects `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, and Ollama at `localhost:11434`. Prompts for provider type (`ollama`, `openai`, `anthropic`, `google`, `xai`, or `deepseek`), model name, and API key if needed. Tests LLM connectivity before proceeding.
2. **LLM conversation** — loads the configuration reference (bundled or fetched), loads any existing config from the standard search paths, and runs an interactive Q&A loop. The wizard LLM asks targeted questions and produces a complete YAML config in a code fence when it has enough information. Type `quit` at any prompt to cancel.
3. **Validation and write** — extracts the YAML from the LLM response, injects the real API key collected during bootstrap, validates the result via an internal config round-trip, shows a masked preview for confirmation, and writes the file.

**Notes:**

- The wizard detects an existing config automatically and asks whether to edit it or start fresh.
- The API key field echoes `*` for each character typed. The masked preview shows the first 3 and last 4 characters (e.g. `sk-***4bcd`) for keys ≥ 10 characters, or `***` for shorter keys.
- Leave the API key blank for endpoints that do not require authentication (vLLM, LM Studio, and other self-hosted OpenAI-compatible servers).
- All values entered during bootstrap (provider type, base URL, model, API key) are preserved as defaults if the connection test fails — retry without re-entering unchanged fields.
- API keys entered during bootstrap are injected into the final YAML, so the LLM never sees the actual key value.
- The output file is shown after writing: `Config written to: ~/.cogtrix.yaml`.

**Docker auto-start:** When running the official container image, the container automatically launches the setup wizard if all of the following are true: (1) no command-line arguments were passed to the container, (2) no config file exists at `/app/.cogtrix.yaml` or `/app/.cogtrix.json`, (3) none of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `COGTRIX_OLLAMA`, or `OLLAMA_BASE_URL` is set, and (4) stdin is a TTY. This simplifies first-run setup:

```bash
docker run -it -v ~/.cogtrix.yaml:/app/.cogtrix.yaml ghcr.io/northlandpositronics/cogtrix:latest
# → wizard starts automatically, writes config to the mounted path
```

---

## Complete Configuration Example

Below is a full configuration in both YAML and JSON. Both formats are functionally identical — pick whichever you prefer.

### YAML (`.cogtrix.yaml`)

```yaml
session: default

# ─── LLM Providers (connection info only) ───────────────────────
providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
  openai:
    type: openai
    api_key: "sk-..."
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."
  local-gpu:
    type: ollama
    base_url: "http://192.168.1.101:11434"

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
    filter_mode: allow
    contacts: ["+14155551234"]
    phonebook:
      alice: "+14155551234"
  telegram:
    bot_token: "123456:ABC-DEF..."
    phonebook:
      alice: "123456789"

# ─── Models (chat + embedding) ───────────────────────────────────
models:
  default: fast             # active model alias at startup
  fast: my-server/qwen3:8b
  smart:
    provider: openai
    model: gpt-4.1
    temperature: 0.7
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
  vectordb_dir: vectordb
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

# ─── Decision Accountability ────────────────────────────────────
# Off by default. Enable for high-stakes autonomous work.
decision_accountability:
  enabled: false
  min_confidence_threshold: 7.0
  require_counter_plan: true
  report_uncertainty: true

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
  "session": "default",

  "providers": {
    "my-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434"
    },
    "openai": {
      "type": "openai",
      "api_key": "sk-..."
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-..."
    },
    "local-gpu": {
      "type": "ollama",
      "base_url": "http://192.168.1.101:11434"
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
      "filter_mode": "allow",
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
    "default": "fast",
    "fast": "my-server/qwen3:8b",
    "smart": {
      "provider": "openai",
      "model": "gpt-4.1",
      "temperature": 0.7
    },
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
    "vectordb_dir": "vectordb",
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

  "decision_accountability": {
    "enabled": false,
    "min_confidence_threshold": 7.0,
    "require_counter_plan": true,
    "report_uncertainty": true
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

> **Note:** Both examples use `"providers"` (preferred). The legacy key `"inference"` still works as an alias. `models.default` sets the active model alias; the deprecated top-level `provider` and `model` keys are auto-migrated on load.

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

## Migration Guide

### Migrating from the old provider/model format

In earlier versions, model settings (model name, temperature, `context_window`) were placed inside the provider entry, and the active model was selected via top-level `provider` and `model` keys:

```yaml
# Old format — still accepted but deprecated
provider: my-server
model: qwen3:8b

providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
    model: qwen3:8b
    temperature: 0.5
```

**What changed:** Providers now hold only connection info. All model settings live in the `models` registry. The active model is selected via `models.default`.

**New format:**

```yaml
# New format
providers:
  my-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"

models:
  default: main
  main:
    provider: my-server
    model: qwen3:8b
    temperature: 0.5
```

**Auto-migration:** Old configs continue to work without changes. When Cogtrix loads a config that has model fields (`model`, `temperature`, `context_window`) inside a provider entry, they are automatically migrated to the `models` registry as a new entry named after that provider. Similarly, top-level `provider` and `model` keys are mapped to `models.default` by matching against existing registry entries. A log warning is emitted for each migrated field.

**Recommended steps to update manually:**

1. Move `model`, `temperature`, `context_window`, and `max_tokens` out of each provider entry and into a named entry in the `models` section.
2. Set `models.default` to the alias you want active at startup.
3. Remove the top-level `provider` and `model` keys.
4. Update `--provider` CLI usage to `--model <alias>`.
5. Replace the `COGTRIX_PROVIDER` environment variable with `COGTRIX_MODEL=<alias>`.

### Environment variable changes

| Old | New | Notes |
|-----|-----|-------|
| `COGTRIX_PROVIDER=ollama` | `COGTRIX_MODEL=my-alias` | Set any model alias defined in `models` |
| `COGTRIX_MODEL=qwen3:8b` | `COGTRIX_MODEL=my-alias` | Now expects a registry alias, not a bare model name |

### CLI flag changes

| Old | New |
|-----|-----|
| `--provider ollama` | `--model <alias>` |
| `-p ollama` | `-m <alias>` |

### New feature: Decision Accountability (ADR-0052)

**No migration required.** The feature is opt-in (`enabled: false` by default) and adds no breaking changes to existing configuration.

To enable, add the following to your `.cogtrix.yaml`:

```yaml
decision_accountability:
  enabled: true
```

If you have an existing config that previously contained a `decision_accountability` key with a dict value (from a pre-release build that used a different schema), replace it with the scalar fields shown above. The old dict form is no longer read by the parser.

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

### API Server Logging

The API server (`python -m src.api`) supports the same logging flags with one addition: **debug log streaming**.

```bash
python -m src.api --debug                      # DEBUG/INFO → stdout, WARNING+ → stderr
python -m src.api --debug --log-file /tmp/api.log  # all levels → file (overrides streaming)
python -m src.api --log                        # INFO → cogtrix-api.log
python -m src.api --log-file /var/log/api.log  # INFO → specified file
```

When `--debug` is used without `--log-file`, log output is split across standard streams:
- **DEBUG and INFO** messages go to **stdout**
- **WARNING, ERROR, and CRITICAL** messages go to **stderr**

This is useful for `docker logs`, live terminals, and log aggregators that distinguish stdout from stderr. When `--log-file` is provided, it takes priority and all levels are written to the file.

The `COGTRIX_LOG_STREAM=1` environment variable is set internally to propagate the stream mode to the application lifespan.

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
