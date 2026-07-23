# Cogtrix Architecture

This document describes how Cogtrix is built: the layers, components, and data flows that make everything work. It's aimed at developers who want to understand the internals, extend the system, or contribute code. For user-facing guides, see the [README](../README.md) and [Configuration](CONFIGURATION.md). For a full documentation index including ADRs and internal docs, see [docs/INDEX.md](INDEX.md).

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Security Model](#security-model)

---

## System Overview

Cogtrix is a modular LangChain-based AI agent built with a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Interface Layer                         │
│                      (cogtrix.py ~4300 LOC)                        │
│  • Interactive & non-interactive modes  • Slash command system   │
│  • Session management  • Tool management (load/enable/disable)  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                     Configuration Layer                          │
│                       (src/config.py)                            │
│  • Multi-provider config  • Priority resolution  • Validation    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      Agent Core Layer                            │
│                    (src/agent/core.py)                           │
│  • CogtrixState schema  • LLM factory  • System prompt builder     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                       Safety Layer                               │
│                    (src/agent/safety.py)                         │
│  • Tool confirmation (y/n/a/d/D/c)  • Approval & denial tracking │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      Tool Registry                               │
│                     (src/registry.py)                            │
│  • Dynamic discovery  • Tool wrapping  • Metadata management     │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
┌───────────────┴───────────────┐ ┌───────────────┴───────────────┐
│        Memory System          │ │        Tool Modules           │
│       (src/memory/)           │ │       (src/tools/)            │
│  • Mode managers              │ │  • 51 built-in tools          │
│  • Context preparation        │ │  • Auto-discovery             │
│  • JSON persistence           │ │  • Pydantic schemas           │
└───────────────────────────────┘ └───────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Assistant Mode (src/assistant/)              │
│  • Headless WhatsApp/Telegram daemon  • Per-chat isolation      │
│  • Channel abstraction  • Concurrent message handling           │
│  • Shared knowledge store  • Session lifecycle management       │
│  • Security guardrails  (input/output/rate-limit/encoding/      │
│    tool-call/auto-blacklist/LLM judge)                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Setup Wizard (src/setup_wizard.py)              │
│  • Interactive --setup  • Provider bootstrap with retry         │
│  • Ollama model listing  • Rich markdown rendering + spinner   │
│  • LLM-guided Q&A  • YAML validation and write                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Directory Structure

```
src/
├── config.py              # Configuration management
├── registry.py            # Tool discovery and registration
├── logging_config.py      # Logging infrastructure
├── setup_wizard.py        # Interactive --setup configuration wizard
├── mcp_client.py          # MCP server lifecycle, tool discovery, LangChain integration
├── providers/
│   ├── __init__.py        # Registry: create_chat_model(), create_embeddings(), PROVIDER_TYPES
│   ├── defaults.py        # Default models, embedding models, base URLs, env var names, presets
│   ├── openai.py          # OpenAI and compatible APIs (xAI, vLLM, Groq, Together)
│   ├── ollama.py          # Ollama local inference
│   ├── anthropic.py       # Anthropic Claude
│   └── google.py          # Google Gemini
├── agent/
│   ├── core.py            # LangGraph agent setup
│   └── safety.py          # Tool confirmation wrapper
├── memory/
│   ├── base.py            # Abstract base classes
│   ├── factory.py         # Memory mode factory
│   ├── manager.py         # Base memory manager + hybrid memory logic
│   ├── context.py         # Context data structures
│   ├── json_store.py      # JSON file persistence
│   ├── summarizer.py      # LLM-based incremental summarization
│   ├── recall.py          # Per-session FAISS vector store for semantic recall
│   └── modes/
│       ├── conversation.py  # General chat mode
│       ├── code.py          # Code development mode
│       └── reasoning.py     # Planning/reasoning mode
├── assistant/
│   ├── __init__.py        # Package exports
│   ├── channel.py         # Channel ABC + IncomingMessage dataclass
│   ├── channels/
│   │   ├── whatsapp.py    # WhatsApp via Waha
│   │   └── telegram.py    # Telegram Bot API
│   ├── session.py         # ChatSession + ChatSessionManager
│   ├── handler.py         # Message → agent → reply pipeline
│   ├── poller.py          # Per-channel polling threads
│   ├── knowledge.py       # Cross-chat fact extraction + recall
│   ├── guardrails.py      # Security guardrails (input/output/rate-limit/LLM judge)
│   └── service.py         # Main orchestrator (AssistantService)
├── rag/
│   ├── __init__.py        # RAG module
│   └── ingest.py          # Document ingestion
└── tools/
    ├── brave_search.py    # Brave Search API
    ├── calculator.py      # Math expressions
    ├── datetime_tool.py   # Date/time utilities
    ├── deep_think.py      # Tree-of-Thought reasoning
    ├── delegate.py        # Task delegation
    ├── exa_search.py      # Exa semantic search (3 tools)
    ├── file_ops.py        # File operations
    ├── google_search.py   # Google Custom Search API
    ├── http_request.py    # HTTP requests
    ├── json_tool.py       # JSON processing
    ├── nlp_tools.py       # NLP (sentiment, summarization)
    ├── python_exec.py     # Python execution
    ├── rag.py             # Knowledge base queries
    ├── serpapi_search.py  # SerpAPI (Google/Bing structured)
    ├── shell.py           # Shell commands
    ├── tavily_search.py   # Tavily AI search (2 tools)
    ├── text_tools.py      # Text processing
    ├── weather.py         # Weather information
    ├── web_search.py      # DuckDuckGo search
    ├── whatsapp.py        # WhatsApp messaging (4 tools)
    ├── _whatsapp_client.py # Waha HTTP client
    ├── telegram.py        # Telegram messaging (4 tools)
    └── _telegram_client.py # Telegram Bot API client
```

---

## Core Components

### 1. Configuration (`src/config.py`)

Manages multi-source configuration with priority resolution.

**Key Classes:**

```python
@dataclass
class ProviderConfig:
    name: str
    type: str              # "openai", "ollama", "anthropic", or "google"
    base_url: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    # Runtime-only fields — NOT parsed from the providers section.
    # Populated by _resolve_model() when a ModelConfig is applied.
    temperature: Optional[float]
    num_ctx: Optional[int]
    tool_instructions: Optional[str]

@dataclass
class ModelConfig:
    provider: str              # references a key in Config.providers
    model: str                 # actual model name at the provider
    num_ctx: Optional[int]
    temperature: Optional[float]

@dataclass
class Config:
    provider: str
    model: Optional[str]
    session: str
    providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]
    memory_mode: str
    rag: RAGConfig
    debug: bool
    verbose: bool
    log_file: Optional[str]
    # ... more fields
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_config(cli_args)` | Load and merge configuration from all sources |
| `find_config_file()` | Locate `.cogtrix.{json,yml,yaml}` in cwd, home, or `~/.config/cogtrix/` |
| `Config.get_provider_config(name)` | Get provider configuration by name |
| `Config.list_providers()` | List all available provider names |
| `Config.resolve_embedding_config()` | Resolve `rag.model` via the `models` registry to `(provider_type, model, base_url, api_key)` |

### 2. CLI Interface (`cogtrix.py`)

The entry point handles both interactive and non-interactive modes.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `SlashCommandRegistry` | Registers, resolves, and dispatches `/commands` |
| `SlashCommand` | Dataclass: name, handler, help text, aliases |
| `ToolCallLogger` | `BaseCallbackHandler` for LLM/tool observability |
| `readline` import | Enables arrow keys, Home/End, input history |

**Slash Command Dispatch:**

```
User Input
    │
    ├── Starts with "/"  → SlashCommandRegistry.dispatch()
    │                        ├── /help, /info, /tools, /mode, /model, /provider
    │                        ├── /session, /setup, /debug, /verbose, /approve, /optimizer, /clear
    │                        ├── /think <task> → deep_think() directly
    │                        ├── /delegate <task> → forced delegation pipeline
    │                        ├── /paste → multi-line input mode
    │                        └── /quit → exit
    │
    └── Regular text     → Agent processing pipeline
```

**Built-in Commands:**

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | `/h`, `/?` | List commands or detailed help |
| `/quit` | `/exit`, `/q` | End session |
| `/info` | `/i` | Session information (provider, model, system prompt size, mode) |
| `/tools` | `/t`, `/tool` | List / manage tools (load, enable, disable) |
| `/mcp` | | List or restart MCP server connections |
| `/think` | `/T` | Deep Tree-of-Thought reasoning |
| `/delegate` | `/d` | Force task delegation across models |
| `/mode` | `/M` | Show / switch memory mode |
| `/model` | `/m` | Show / switch LLM model |
| `/provider` | `/p` | Show / switch LLM provider |
| `/session` | `/s` | Show / switch session |
| `/setup` | | Launch the interactive setup wizard |
| `/approve` | `/a` | Auto-approve all tool confirmations |
| `/optimizer` | `/o` | Toggle prompt optimizer |
| `/debug` | `/D` | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/paste` | `/P` | Enter multi-line paste mode |
| `/clear` | `/c` | Clear conversation history |

Hidden commands (not in `/help` listing):

| Command | Aliases | Description |
|---------|---------|-------------|
| `/system_prompt` | `/sp` | Display the full system prompt |

### 3. Agent Core (`src/agent/core.py`)

Defines the state schema, system prompt, and LLM factory. The actual agent graph is built in `cogtrix.py`.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `CogtrixState` | TypedDict with `messages: Annotated[Sequence[BaseMessage], add_messages]` |
| `build_system_prompt(mode_additions, tool_instructions)` | Generate system prompt with mode context and optional tool-call formatting |
| `create_llm_from_provider_config(config)` | LLM factory from ProviderConfig |
| `build_agent_executor(tools, ...)` | Legacy ReAct agent builder (used by delegates) |
| `DEFAULT_TOOL_INSTRUCTIONS` | Raw-JSON tool-call formatting instructions (not injected by default; available for explicit opt-in via `tool_instructions` config) |

**Agent Architecture:**

The main agent uses a custom LangGraph `StateGraph` built by `_build_agent_graph()` in `cogtrix.py`. It has three nodes with conditional routing:

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│      Custom LangGraph StateGraph    │
│                                     │
│  ┌───────────┐    ┌──────────────┐  │
│  │call_model │───▶│process_tools │  │
│  │ (LLM call)│◀───│(execute+     │  │
│  └───────────┘    │ expand tools)│  │
│       │           └──────────────┘  │
│       │ (no tool calls)             │
│       ▼                             │
│  ┌──────────────┐                   │
│  │handle_phantom│ (fuzzy-match      │
│  │              │  unknown tools)    │
│  └──────────────┘                   │
└─────────────────────────────────────┘
    │
    ▼
Agent Response
```

### 4. Safety Layer (`src/agent/safety.py`)

Wraps sensitive tools with confirmation prompts.

**Mechanism:**

1. Tools marked with `requires_confirmation: True` are wrapped
2. Wrapper intercepts execution and prompts user
3. User can: Yes — allow once (`y`), No — deny once (`n`), Allow all — approve
   for session (`a`), Disable tool (`d`), Forbid all tools (`f`), or Cancel
   workflow (`c`)
4. Session-scoped sets track approvals (`approvals`), denials (`_denials`),
   and dynamically loaded tools (`_loaded_tools`)
5. Disabled tools are also blocked at the expansion point in `process_tools`

**Sensitive Tools:**
- `execute_shell_command`
- `execute_python`
- `write_file`
- `append_file`
- `http_post`
- `whatsapp_send`, `whatsapp_send_image` (configurable)
- `telegram_send`, `telegram_send_photo` (configurable)

### 5. Tool Registry (`src/registry.py`)

Discovers and loads tools dynamically.

**Discovery Process:**

```
1. Scan src/tools/ directory
       │
       ▼
2. Import each .py module
       │
       ▼
3. Look for TOOL_CONFIG or TOOL_CONFIGS
       │
       ▼
4. Extract function, schema, metadata
       │
       ▼
5. Create LangChain StructuredTool
       │
       ▼
6. Register in tools dictionary
```

**Tool Configuration Format:**

```python
TOOL_CONFIG = {
    "name": "function_name",
    "description": "What this tool does",
    "input_schema": PydanticModel,
    "requires_confirmation": False,
}
```

**API Key Gating:** Modules that export an `is_configured()` function are checked before registration. If it returns `False`, the module is skipped — its tools never appear in the agent's toolbox. This is used by search providers, weather, and WhatsApp to hide themselves when their API keys or services are unavailable.

### 6. MCP Client (`src/mcp_client.py`)

Connects to external MCP (Model Context Protocol) servers, discovers their tools, and exposes them as LangChain StructuredTool objects.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `MCPServerConfig` | Dataclass: server name, command/args (stdio) or url/headers (SSE), confirmation, timeout |
| `MCPConnection` | Manages one server: async connect, list_tools, call_tool, close via `AsyncExitStack` |
| `MCPManager` | Manages all connections; runs a background asyncio event loop for sync↔async bridging |

**Async Bridging:** The MCP SDK is fully async. `MCPManager` creates a dedicated `asyncio` event loop on a daemon thread. Sync callers use `asyncio.run_coroutine_threadsafe()` + `.result(timeout)` to block until the async operation completes.

**Tool Discovery Flow:**

```
Config: mcp_servers section
    │
    ▼
MCPManager.connect_all()
    │
    ├── For each server:
    │   ├── Start process (stdio) or connect (SSE)
    │   ├── ClientSession.initialize()
    │   └── session.list_tools() → MCP Tool objects
    │
    ▼
Convert each MCP Tool → LangChain StructuredTool
    │
    ├── JSON Schema → Pydantic model (via create_model)
    ├── Sync wrapper function → MCPManager.call_tool()
    └── Metadata: source="mcp", server name, requires_confirmation
    │
    ▼
Register in ToolRegistry → available_tools pool (on-demand)
```

**Name collision handling:** When an MCP tool has the same name as a built-in tool (e.g., `read_file`), it is automatically prefixed with the server name to prevent shadowing.

**Configuration:** `mcp_servers` section in config file. Transport is auto-detected: `command` → stdio, `url` → SSE. Optional dependency: `mcp` package (`uv pip install "cogtrix[mcp]"`).

### 7. Memory System (`src/memory/`)

Pluggable memory management with multiple modes and a shared hybrid layer.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      MemoryFactory                               │
│                   create(mode, store, ...)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Conversation  │     │     Code      │     │   Reasoning   │
│    Memory     │     │   Memory      │     │    Memory     │
│   Manager     │     │   Manager     │     │   Manager     │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │ BaseMemoryManager │
                    │  (hybrid layer)   │
                    │  ┌─────────────┐  │
                    │  │ Summarizer  │  │  ← src/memory/summarizer.py
                    │  │ (LLM-based) │  │
                    │  └─────────────┘  │
                    │  ┌─────────────┐  │
                    │  │VectorStore  │  │  ← src/memory/recall.py
                    │  │(FAISS, opt.)│  │
                    │  └─────────────┘  │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │ JsonFileMemory    │
                    │     Store         │
                    └───────────────────┘
```

**Base Interface:**

```python
class BaseMemoryManager(ABC):
    def prepare_context(self, user_input: str) -> MemoryContext
    def update(self, user_input: str, ai_response: str,
               agent_messages: list[Any] | None = None) -> None
    def save(self) -> None
    def load(self) -> None
    def get_system_prompt_additions(self) -> str

    # Hybrid memory (called by subclasses)
    def set_llm(self, llm: Any) -> None
    def set_embeddings(self, embedding_fn: Any, embedding_model: str) -> None
    def _maybe_summarize(self, messages, window_size) -> None
    def _build_hybrid_prefix(self, user_input: str) -> str | None
```

**Hybrid Memory:** All modes inherit a hybrid memory layer from `BaseMemoryManager`. When an LLM is injected via `set_llm()`, messages that fall outside the sliding window are incrementally summarized every 6 messages. When an embedding function is injected via `set_embeddings()`, evicted messages are also embedded into a per-session FAISS index for semantic recall. Both are injected into the context prefix by `_build_hybrid_prefix()`. See [Memory Modes — Hybrid Memory System](MEMORY_MODES.md#hybrid-memory-system) for details.

**Message Timestamps:** Every message is automatically stamped with a UTC timestamp. The user message is stamped in `prepare_context()` (when the user sends input) and the AI message is stamped in `update()` (when the response arrives). This lets the LLM see elapsed time between turns. Timestamps are stored as ISO 8601 strings (`2026-02-14T15:23:05Z`) and injected into message content as `[2026-02-14 15:23:05 UTC]` at context-preparation time. Old sessions without timestamps load normally (backward-compatible).

**Token-Aware Trimming:** Before messages are sent to the LLM, `prepare_messages_with_context()` in `src/agent/core.py` estimates the total token count and trims the oldest history messages (or truncates oversized individual messages) to fit the model's context window. The `max_tokens` parameter is also dynamically capped to prevent negative values.

### 8. RAG System (`src/rag/`)

Document ingestion and knowledge base queries.

**Ingestion Pipeline:**

```
Documents (docs/)
       │
       ▼
┌─────────────────┐
│  Document       │  PDF, Markdown, CSV, TXT
│  Loaders        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text           │  chunk_size: 2000
│  Splitter       │  chunk_overlap: 200
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │  Ollama (default), OpenAI, or Google
│  Model          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS          │  data/vectordb/faiss_index
│  Vector Store   │
└─────────────────┘
```

### 9. Assistant Mode (`src/assistant/`)

Headless daemon that maintains ongoing conversations over WhatsApp and Telegram with per-chat context isolation and shared cross-chat knowledge.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     AssistantService                             │
│                   (src/assistant/service.py)                     │
│  • Channel discovery  • Signal handling  • Graceful shutdown    │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐
│ChannelPoller  │   │ChatSession    │   │SharedKnowledge        │
│(poller.py)    │   │Manager        │   │Store                  │
│               │   │(session.py)   │   │(knowledge.py)         │
│• 1 thread per │   │               │   │                       │
│  channel      │   │• get_or_create│   │• extract_and_store()  │
│• Eviction     │   │• evict_idle() │   │• recall()             │
│  thread (60s) │   │• save_all()   │   │• FAISS + keyword      │
└───────┬───────┘   └───────┬───────┘   │  fallback             │
        │                   │           └───────────────────────┘
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│Channel (ABC)  │   │ChatSession    │
│(channel.py)   │   │               │
│               │   │• session_key  │
│• poll()       │   │• memory_mgr   │
│• send()       │   │• lock (per-   │
│• is_ready()   │   │  session)     │
└──┬─────────┬──┘   └───────────────┘
   │         │
   ▼         ▼
WhatsApp  Telegram
Channel   Channel
```

**Message Processing Flow:**

```
Incoming Message (from channel.poll())
    │
    ▼
┌─────────────────────────────────────┐
│ 1. MessageHandler.handle()          │
│    - Acquire session.lock           │
│    - Update last_activity timestamp │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Guardrails: check_input()        │
│    - Rate limit check (per-chat)    │
│    - Input length + Unicode check   │
│    - Injection pattern matching     │
│    - LLM judge (if enabled)         │
│    → blocked: send canned reply     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Memory: prepare_context()        │
│    - Load per-chat history          │
│    - Build context prefix           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Knowledge: recall()              │
│    - Semantic search (FAISS) or     │
│      keyword fallback               │
│    - Inject "Known facts" into      │
│      context prefix                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Agent: run_agent()               │
│    - Same pipeline as interactive   │
│    - Excluded tools filtered out    │
│    - All tools auto-approved        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. Guardrails: record + sanitize    │
│    - Record message for rate limit  │
│    - Strip PII, images, HTML, URLs  │
│    - Redact banned strings          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. Memory: update() + save()        │
│    - Persist sanitized response     │
│    - Per-chat history stored after  │
│      PII/URL stripping              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 8. Knowledge: extract_and_store()   │
│    - LLM extracts entity-centric    │
│      durable facts from the turn    │
│    - Deduplication by hash          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 9. Truncate + channel.send()        │
│    - Cap at max_response_length     │
│    - Send reply via channel         │
└─────────────────────────────────────┘
```

**Concurrency Model:**

- `ThreadPoolExecutor(max_workers=max_concurrent)` processes different chats in parallel
- Per-session `threading.Lock` serializes messages within the same chat
- One polling thread per channel (WhatsApp short-poll 5s, Telegram long-poll 30s)
- Session eviction thread runs every 60s

**Two-Layer Memory:**

| Layer | Scope | Storage | Purpose |
|-------|-------|---------|---------|
| Per-chat context | Private to each `(channel, chat_id)` | `data/history/{session_key}.json` | Independent conversation history, summarization, vector recall |
| Shared knowledge | Cross-chat | `data/knowledge/facts.json` + `data/vectordb/knowledge/` | Entity-centric facts recalled when relevant to any chat |

**Default Excluded Tools:** `whatsapp_send`, `whatsapp_check`, `whatsapp_send_image`, `whatsapp_contacts`, `telegram_send`, `telegram_check`, `telegram_send_photo`, `telegram_contacts`, `shell`, `write_file`, `append_file`

**Security Guardrails (`src/assistant/guardrails.py`):**

Every message in assistant mode passes through a `GuardrailPipeline` that wraps seven
independent components:

| Component | What it does |
|-----------|-------------|
| `ViolationTracker` | Tracks security violations per chat. Auto-blacklists a chat after N violations within a sliding time window. Rate limit violations are excluded from the violation count. Blacklist state is persisted to `data/assistant/violations.json` and survives restarts. |
| `ChatRateLimiter` | Per-chat sliding window (per-minute + per-hour). Thread-safe deque scan. |
| `InputGuard` | Length limit, Unicode steganography detection, 15 pre-compiled injection regex patterns, optional custom patterns. |
| `EncodingDetectionGuard` | Detects encoding-based bypass attempts (Morse code, Base64, hex, leetspeak/ROT13). Scores each message with four sub-detectors (0–1 each); blocks when max score exceeds configurable threshold (default 0.6). |
| `ToolCallGuard` | Inspects tool arguments before execution. Injection scan across all arguments; path blocking for file tools (blocks /etc/, /proc/, .env, id_rsa, and configurable custom paths); exfiltration detection for web tools (flags API keys, SSH keys, and SSNs in URL/query arguments). |
| `LLMJudge` | Opt-in LLM-as-judge classifier. Fail-open on error. Disabled by default (adds 500ms–2s). |
| `OutputGuard` | Strips markdown images, HTML tags, banned strings, PII (email, credit card, SSN, private IP), and URLs (matched case-insensitively). Runs before memory update so conversation history only stores sanitized content. |

The pipeline order for input checks is: `blacklist → rate_limiter → input_guard → encoding_guard → llm_judge`.
Rate limit violations are recorded but do not increment the violation counter that drives auto-blacklisting.
Output sanitization runs after the agent responds, before memory is updated and before `channel.send()`,
so only sanitized content is stored in conversation history. Tool call inspection runs inside the
`process_tools` node before each tool is executed. The entire pipeline is bypassed when
`guardrails.enabled: false`.

Performance without the LLM judge is under 0.5ms total — negligible compared to LLM
inference latency (1–30s). The `GuardrailPipeline` is constructed in `service.py` with an
optional judge LLM and injected into `MessageHandler`.

---

## Data Flow

### Request Processing Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Memory: prepare_context()        │
│    - Capture user timestamp (UTC)   │
│    - Get working memory             │
│    - Inject timestamps into context │
│    - Build context prefix           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Agent: graph.stream()            │
│    - Compress old ToolMessages      │
│      (context_compression)          │
│    - Process with LLM (call_model)  │
│    - Execute tools (process_tools)  │
│    - Stream preserves state on      │
│      RecursionError                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Safety: check confirmation       │
│    - Prompt if required             │
│    - Execute or deny                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Memory: update()                 │
│    - Stamp user msg with saved ts   │
│    - Stamp AI msg with current ts   │
│    - Add to history                 │
│    - Update mode-specific tracking  │
│    - Trigger hybrid memory:         │
│      • Summarize if batch ready     │
│      • Feed evicted msgs to vector  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Memory: save()                   │
│    - Persist history to JSON file   │
│    - Persist hybrid meta (summary)  │
│    - Persist vector index (FAISS)   │
└─────────────────────────────────────┘
    │
    ▼
Response to User
```

### Tool Execution Flow

```
Agent decides to use tool
    │
    ▼
┌─────────────────────────────────────┐
│ Tool Registry: get tool             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Safety Wrapper: check               │
│                                     │
│ Disabled or forbid-all?              │
│   └── Yes → Return denial silently  │
│ Requires confirmation?              │
│   ├── No  → Execute directly        │
│   └── Yes → Prompt user             │
│             ├── y → Yes (once)      │
│             ├── a → Allow all       │
│             ├── d → Disable tool    │
│             ├── f → Forbid all      │
│             ├── c → Cancel workflow │
│             └── n → No (deny once)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Tool Function: execute              │
└─────────────────────────────────────┘
    │
    ▼
Result to Agent
```

### Research Delegate and Deep Think Pipeline

When the user requests deep reasoning and the agent has used web tools, an enhanced pipeline runs between steps 2 and 4 above:

```
Agent finished initial research (step 2)
    │
    ▼
┌─────────────────────────────────────┐
│ 2a. Detect web tool usage           │
│     _agent_used_web_tools()         │
│     _extract_fetched_urls()         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2b. Research Delegate               │
│     _run_research_delegate()        │
│     - Spawn sub-agent               │
│     - Temporarily patch output caps │
│       to cap_ratio × max_context    │
│     - Re-fetch URLs with high cap   │
│     - Extract structured specs      │
│     - Restore original caps         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2c. Deep Think                      │
│     _force_deep_think()             │
│     - Prefer research_context over  │
│       raw tool_outputs              │
│     - Run Tree-of-Thought engine    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2d. Execution Phase (if needed)     │
│     _run_execution_phase()          │
│     - Check: prompt requests action │
│       but agent made no write calls │
│     - Re-prompt agent to create     │
│       files based on deep_think     │
│       output                        │
└─────────────────────────────────────┘
    │
    ▼
Continue to step 4 (Memory: update)
```

**Execution phase trigger:** When a prompt requests file actions (`_prompt_requests_action()`) but the agent produced only text with no `write_file` / `append_file` calls (`_agent_performed_writes()` returns False), the orchestrator feeds the analysis back to the agent via `_run_execution_phase()` so it can act on it. Write tools are available throughout both phases.

**Configurable via:** The `research_delegate` section in the config file (note: config-file overrides are not yet active; runtime defaults are hardcoded in `cogtrix.py`). See [CONFIGURATION.md — Research Delegate Section](CONFIGURATION.md#research-delegate-section).

---

## Extension Points

### Adding a New Tool

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed instructions.

### Adding a New Memory Mode

1. Create `src/memory/modes/my_mode.py`
2. Extend `BaseMemoryManager`
3. Implement required methods
4. Register in `src/memory/factory.py`

### Adding a New Provider Type

Supported types: `openai`, `ollama`, `anthropic`, `google`. OpenAI-compatible services (xAI, Groq, vLLM) use `type: openai` with a custom `base_url`.

To add a new native provider type:

1. Create `src/providers/<name>.py` with `create_chat_model()` and (optionally) `create_embeddings()` functions, and `CHAT_AVAILABLE` / `EMBEDDINGS_AVAILABLE` booleans.
2. Register the module in `src/providers/__init__.py` by adding it to the `_MODULES` dict.
3. Add default model, embedding model, and base URL entries to `src/providers/defaults.py`.
4. Update `src/config.py` if additional validation is needed in `_parse_providers_section()`.

### Adding a New Interface

The agent core is interface-agnostic:

```python
from src.config import load_config
from src.registry import ToolRegistry
from src.agent.core import build_agent_executor, create_llm_from_provider_config
from src.memory import MemoryFactory, JsonFileMemoryStore

# Load config
config = load_config()

# Setup tools
registry = ToolRegistry()
registry.load_all_tools()
tools = list(registry.tools.values())

# Create LLM and agent
provider_config = config.get_provider_config()
llm = create_llm_from_provider_config(provider_config)
agent = build_agent_executor(tools, llm=llm)

# Setup memory
memory_store = JsonFileMemoryStore()
memory_manager = MemoryFactory.create(mode="conversation", store=memory_store)
memory_manager.load()

# Use agent
context = memory_manager.prepare_context(user_input)
for chunk in graph.stream({"messages": context.messages}, stream_mode="values"):
    result = chunk  # last chunk holds final state
memory_manager.update(user_input, str(result))
memory_manager.save()
```

---

## Security Model

### Tool Safety

| Category | Tools | Confirmation |
|----------|-------|--------------|
| Read-only | `read_file`, `list_directory`, `search_web` | No |
| Sensitive | `execute_shell_command`, `write_file`, `execute_python` | Yes |
| External | `http_post` | Yes |

### Assistant Mode Guardrails

Assistant mode adds a dedicated security layer in `src/assistant/guardrails.py`. See [Section 8 — Security Guardrails](#security-guardrails-srcassistantguardrailspy) for the full description. The pipeline now includes `EncodingDetectionGuard`, `ToolCallGuard`, and `ViolationTracker` (auto-blacklist) in addition to the original four components. Configuration is in the `services.assistant.guardrails` config block.

### API Key Management

- Store in environment variables (preferred)
- Or in config file (`.cogtrix.yaml` / `.cogtrix.json`)
- Never commit to version control (`.gitignore` excludes config files and `.env*`)
- Keys hidden in logs (`api_key: "***"`)

### Session Isolation

- Each session has separate memory file
- Approvals are session-scoped (not persisted)
- History stored in `data/history/{session_id}.json`

---

## Dependencies

Dependencies are managed via `pyproject.toml` (with `uv`) and exported to `requirements.txt` for pip compatibility.

### Core

| Package | Purpose |
|---------|---------|
| `langchain-core` | Base LangChain functionality |
| `langchain-openai` | OpenAI LLM and embeddings integration |
| `langchain-ollama` | Ollama LLM and embeddings integration |
| `langchain-community` | Community tool integrations |
| `langgraph` | StateGraph agent implementation |
| `pydantic` | Schema validation |
| `python-dotenv` | Environment variable loading |

### Optional Provider Packages

| Package | Install | Purpose |
|---------|---------|---------|
| `langchain-anthropic` | `uv pip install "cogtrix[anthropic]"` | Anthropic Claude support |
| `langchain-google-genai` | `uv pip install "cogtrix[google]"` | Google Gemini support |

### Tools

| Package | Purpose |
|---------|---------|
| `ddgs` | DuckDuckGo search |
| `textblob` | NLP tools |
| `faiss-cpu` | Vector store |
| `pypdf` | PDF loading |
| `python-docx` | DOCX file support |
| `tiktoken` | Token counting |

### CLI

| Package | Purpose |
|---------|---------|
| `rich` | Terminal formatting |

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) — Configuration reference
- [DEVELOPMENT.md](DEVELOPMENT.md) — Extension guide
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — Tool documentation
