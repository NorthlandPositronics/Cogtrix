# Cogtrix Architecture

This document describes how Cogtrix is built: the layers, components, and data flows that make everything work. It's aimed at developers who want to understand the internals, extend the system, or contribute code. For user-facing guides, see the [README](../README.md) and [Configuration](CONFIGURATION.md). For a full documentation index including ADRs and internal docs, see [docs/INDEX.md](INDEX.md).

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Security Model](#security-model)
- [Dependencies](#dependencies)

---

## System Overview

Cogtrix is a modular LangChain-based AI agent built with a layered architecture:

```
┌──────────────────────────────────┐ ┌──────────────────────────────────┐
│      CLI Interface Layer         │ │      API Layer (src/api/)        │
│  (cogtrix.py + src/cli/ +        │ │  FastAPI REST + WebSocket        │
│   src/ui/)                       │ │  JWT auth  •  Session registry   │
│  • Interactive & batch modes     │ │  Token streaming  •  DB layer    │
│  • Slash commands                │ │  (aiosqlite / asyncpg)           │
│  • Tool management               │ │                                  │
└────────────────┬─────────────────┘ └─────────────────┬────────────────┘
                 └──────────────────┬──────────────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────┐
│                     Configuration Layer                          │
│                       (src/config.py)                            │
│  • Multi-provider config  • Priority resolution  • Validation    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                    Orchestration Layer                           │
│                   (src/orchestration/)                           │
│  • Agent execution pipeline  • Session state  • Intent detection │
│  • Context compression  • Research/execution phases             │
│  • Provider/model switching with snapshot/rollback              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      Agent Core Layer                            │
│                    (src/agent/core.py)                           │
│  • CogtrixState schema  • LLM factory  • System prompt builder  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                       Safety Layer                               │
│                    (src/agent/safety.py)                         │
│  • Tool confirmation (y/n/a/d/f/c)  • Approval & denial tracking │
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
│  • Mode managers              │ │  • 68 built-in tools          │
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
│  • Workflow system (per-chat prompt, knowledge base, tool policy)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Setup Wizard (src/setup_wizard.py)              │
│  • Interactive --setup  • Provider bootstrap with retry         │
│  • Ollama model listing  • Rich markdown rendering + spinner    │
│  • LLM-guided Q&A  • YAML validation and write                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Directory Structure

```
src/
├── _version.py            # Single source of truth: __version__, __copyright__
├── config.py              # Configuration management
├── registry.py            # Tool discovery and registration
├── logging_config.py      # Logging infrastructure with secret scrubbing
├── setup_wizard.py        # Interactive --setup configuration wizard
├── mcp_client.py          # MCP server lifecycle, tool discovery, LangChain integration
├── providers/
│   ├── __init__.py        # Registry: create_chat_model(), create_embeddings(), PROVIDER_TYPES
│   ├── defaults.py        # Default models, embedding models, base URLs, env var names, presets
│   ├── openai.py          # OpenAI and compatible APIs (xAI, vLLM, Groq, Together)
│   ├── ollama.py          # Ollama local inference
    │   ├── anthropic.py       # Anthropic Claude
    │   └── google.py          # Google Gemini
├── orchestration/
│   ├── run_config.py      # AgentRunConfig dataclass (29 fields: LLM, tools, compression, ownership classifier, decision accountability)
│   ├── runner.py          # run_agent() entry point, response extraction, ToolCallLogger
│   ├── graph.py           # LangGraph StateGraph: call_model, process_tools, handle_phantom
│   ├── nodes/             # Graph node implementations
│   │   ├── call_model.py  # LLM invocation node (binds tools, prepends system message)
│   │   ├── process_tools.py  # Tool execution + on-demand tool expansion
│   │   └── recovery.py    # Recovery strategies (step limit, phantom, recursion)
│   ├── phases.py          # Deep think, execution phase, step-limit recovery
│   ├── research_delegate.py  # Source diversity tracking and contradiction detection for research
│   ├── reflection_delegate.py  # Decision accountability and counter-argumentation (ADR-0052)
│   ├── intent.py          # Intent detection, task complexity, ownership classification pipeline
│   ├── compression.py     # Context compression: apply_message_compression()
│   ├── session_state.py   # SessionState dataclass (replaces 7 former module-level globals)
│   └── session_orchestrator.py  # SessionOrchestrator snapshot()/rollback()
├── agent/
│   ├── core.py            # CogtrixState schema, system prompt builder, LLM factory
│   ├── safety.py          # SafeTool wrapper, create_safe_tool_wrapper(), ConfirmationUI protocol
│   └── agents_md.py   # AGENTS.md parser — load named AgentDefinition objects from Markdown
├── api/
│   ├── app.py             # FastAPI application factory (create_app()), CORS, lifespan, routers
│   ├── auth.py            # JWT validation (HS256), get_current_user, require_admin, validate_api_key
│   ├── session_bridge.py  # ApiSession dataclass, ApiSessionRegistry (warm/evict/save_all)
│   ├── confirmation.py    # ApiConfirmationUI — WebSocket tool confirmation replacing CLI Rich panel
│   ├── turn_runner.py     # run_message_turn(): normal/think/delegate modes via asyncio.to_thread
│   ├── callbacks.py       # WebSocketCallbackHandler: token/tool_start/tool_end streaming
│   ├── ws.py              # ConnectionManager, ServerMessage/ClientMessage envelopes, replay buffer
│   ├── pagination.py      # encode_cursor/decode_cursor/paginate_list, compound keyset cursors
│   ├── validation.py      # Request validation helpers
│   ├── assistant_lifecycle.py  # Assistant start/stop for API mode
│   ├── stripe_client.py   # Lazy Stripe SDK wrapper: api_key init, client factory
│   ├── __main__.py        # uvicorn entry point
│   ├── db/
│   │   ├── engine.py      # Async SQLAlchemy engine (aiosqlite/asyncpg), get_db() dependency
│   │   ├── models.py      # ORM models: User, Organization, Team, Workspace, WorkspaceMembership,
│   │   │                  #   ApiSessionRecord, Message, RefreshToken, ApiKey, Plan, UsageRecord
│   │   └── repositories/
│   │       ├── users.py         # UserRepository (atomic first-user admin election)
│   │       ├── sessions.py      # SessionRepository
│   │       ├── messages.py      # MessageRepository
│   │       ├── tokens.py        # RefreshTokenRepository
│   │       ├── api_keys.py      # ApiKeyRepository
│   │       ├── organizations.py  # OrganizationRepository (multi-tenancy)
│   │       ├── teams.py         # TeamRepository, TeamMembershipRepository
│   │       ├── workspaces.py    # WorkspaceRepository
│   │       ├── plans.py         # PlanRepository
│   │       └── usage.py         # UsageRepository (metering)
│   ├── routes/
│   │   ├── auth.py           # Registration, login, refresh, logout, logout-all, profile, API key CRUD (9 endpoints)
│   │   ├── sessions.py       # Session lifecycle: create, list, get, update, delete (6 endpoints)
│   │   ├── messages.py       # Send message, list history, clear history; session WebSocket lives here (3 REST + 1 WS)
│   │   ├── memory.py         # Get memory state, switch mode, clear memory (3 endpoints)
│   │   ├── tools.py          # List tools, load, enable, disable (4 endpoints)
│   │   ├── config.py         # Read/write config sections, provider management, model aliases (15 endpoints)
│   │   ├── assistant.py      # Start/stop assistant, channel management, phonebook, outbound, campaigns (24 endpoints)
│   │   ├── workflows.py      # Workflow CRUD, per-workflow docs, chat bindings (11 endpoints)
│   │   ├── users.py          # User management: list, create, update role, delete (5 admin endpoints)
│   │   ├── admin.py          # Org list, global stats, usage metrics, impersonation, audit log, superadmin stats (7 endpoints)
│   │   ├── rag.py            # Upload documents, list, delete, query knowledge base (5 endpoints)
│   │   ├── mcp.py            # List servers, connect, disconnect, restart, list tools (5 endpoints)
│   │   ├── system.py         # Server info, shutdown; live log WebSocket (2 REST + 1 WS endpoint)
│   │   ├── health.py         # Liveness, readiness, full readiness probes (3 endpoints)
│   │   ├── metrics.py        # Prometheus metrics export (1 endpoint)
│   │   ├── plans.py          # Plan CRUD + org-plan assignment (5 + 1 = 6 endpoints)
│   │   ├── usage.py          # Usage summary, per-event records, manual record (3 endpoints)
│   │   ├── enforcement.py    # Plan limit snapshot and headroom (1 endpoint)
│   │   ├── organizations.py  # Update org-member role; most org CRUD lives in admin.py (1 endpoint)
│   │   ├── teams.py          # Team management, membership (8 endpoints)
│   │   ├── workspaces.py     # Workspace CRUD, membership, scoped config (10 endpoints)
│   │   ├── cross_workspace.py # Cross-workspace message bus: post, read, delete (3 endpoints)
│   │   ├── saml.py           # SAML 2.0 SSO: metadata, SSO, ACS (3 endpoints)
│   │   ├── scim.py           # SCIM 2.0 user provisioning — RFC 7643/7644 (7 endpoints)
│   │   ├── ldap.py           # LDAP/AD sync: status, trigger (2 endpoints)
│   │   ├── jit.py            # Just-in-time provisioning: status, test (2 endpoints)
│   │   ├── billing.py        # Stripe billing: Checkout, Customer Portal, subscription, webhook (4 endpoints)
│   │   ├── agents.py         # Named agent configuration: list, get (2 endpoints)
│   │   └── tasks.py          # Background task queue: submit, list, get, cancel, log (5 endpoints)
│   ├── schemas/
│   │   ├── auth.py           # Auth request/response models
│   │   ├── session.py        # Session models
│   │   ├── message.py        # Message models
│   │   ├── memory.py         # Memory models
│   │   ├── tool.py           # Tool models
│   │   ├── config.py         # Config models
│   │   ├── assistant.py      # Assistant models
│   │   ├── rag.py            # RAG models
│   │   ├── mcp.py            # MCP models
│   │   ├── system.py         # System models
│   │   ├── user.py           # User management models
│   │   ├── workflow.py       # Workflow models
│   │   ├── organization.py   # Organization, multi-tenancy models
│   │   ├── plans.py          # Plan, PlanLimits models
│   │   ├── teams.py          # Team, TeamMembership models
│   │   ├── workspaces.py     # Workspace, WorkspaceMembership models
│   │   └── common.py         # APIResponse[T] envelope, CursorPage[T], APIError
│   └── tasks/
│       └── rag.py         # Background task helper for async document ingestion
├── cli/
│   ├── args.py            # parse_arguments(), ColorHelpFormatter, ANSI color helpers
│   ├── banner.py          # print_startup(), LOGO_LINES
│   ├── input.py           # read_multiline(), run_inline_shell(), readline history
│   └── escape_monitor.py  # EscapeMonitor: daemon thread translating Escape to SIGINT
├── ui/
│   ├── spinner.py         # ActivityIndicator animated terminal spinner
│   └── input_session.py   # PromptSession wrapper with CPR disabled (prevents PTY tight-loop)
├── prompt/
│   └── optimizer.py       # optimize_prompt(): one-shot LLM prompt rewriter
├── memory/
│   ├── base.py            # Abstract base classes
│   ├── factory.py         # Memory mode factory
│   ├── mode_selector.py   # Heuristic memory mode classifier (conversation / code / reasoning)
│   ├── manager.py         # BaseMemoryManager + hybrid memory logic + _sanitize_session_id()
│   ├── context.py         # MemoryContext data structure
│   ├── json_store.py      # JSON file persistence
│   ├── summarizer.py      # LLM-based incremental summarization
│   ├── distillation.py    # Extract durable facts from rolling memory summary
│   ├── facts.py           # Persistent distilled facts for long-lived sessions
│   ├── recall.py          # Per-session FAISS vector store for semantic recall
│   ├── tier_cache.py      # Tiered Context Cache (TCC) data structures and assembly
│   └── modes/
│       ├── conversation.py  # General chat mode
│       ├── code.py          # Code development mode
│       └── reasoning.py     # Planning/reasoning mode with section-freshness gating
├── assistant/
│   ├── __init__.py        # Package exports
│   ├── channel.py         # Channel ABC + IncomingMessage dataclass
│   ├── channels/
│   │   ├── whatsapp.py    # WhatsApp via Waha
│   │   └── telegram.py    # Telegram Bot API
│   ├── session.py         # ChatSession + ChatSessionManager
│   ├── handler.py         # Message → agent → reply pipeline
│   ├── datamarking.py     # Microsoft Spotlighting: per-call random token injection
│   ├── poller.py          # Per-channel polling threads
│   ├── knowledge.py       # Cross-chat fact extraction + recall
│   ├── guardrails.py      # Security guardrails (input/output/rate-limit/LLM judge)
│   ├── deferral.py      # DeferralManager — deferred re-processing passes
│   ├── scheduler.py     # MessageScheduler — deferred reply delivery
│   ├── campaign.py      # CampaignManager — multi-contact outbound campaigns
│   ├── workflows.py     # WorkflowRegistry: YAML definitions, bindings, auto-detect
│   └── service.py       # Main orchestrator (AssistantService)
├── rag/
│   ├── __init__.py        # RAG module
│   └── ingest.py          # Document ingestion
├── tools/
│   ├── configure.py       # Tool config factories: load_tools(), build_tool_catalog(),
│   │                      #   apply_output_cap(), TOOL_PRESETS, configure_* functions
│   ├── resolver.py        # resolve_tool_name(): canonical fuzzy tool-name resolver
│   ├── agent_messaging.py # Agent inbox messaging (2 tools: send_to_agent, read_agent_inbox)
│   ├── agent_tools.py     # Sub-agent lifecycle (5 tools: spawn_agent, get_task_status, …)
│   ├── brave_search.py    # Brave Search API
│   ├── calculator.py      # Math expressions
│   ├── calendar_tools.py  # Google Calendar integration (gated)
│   ├── checkpoint.py      # Checkpoint / finding persistence
│   ├── cron_tools.py      # Scheduled task management (3 tools: cron_add, cron_list, …)
│   ├── datetime_tool.py   # Date/time utilities
│   ├── deep_think.py      # Tree-of-Thought reasoning
│   ├── delegate.py        # Task delegation (2 tools)
│   ├── email_tools.py     # Email via SMTP/IMAP (gated)
│   ├── exa_search.py      # Exa semantic search (3 tools)
│   ├── extend_run.py      # Extend agent recursion limit mid-run
│   ├── file_ops.py        # File operations (6 tools, incl. patch_file)
│   ├── generate_tests.py  # Auto test generation (gated)
│   ├── git_tools.py       # Git operations (7 tools)
│   ├── github_tools.py    # GitHub API tools (gated)
│   ├── goal_tools.py      # Goal/subgoal tracking (5 tools: set_goal, add_subgoal, …)
│   ├── google_search.py   # Google Custom Search API
│   ├── http_request.py    # HTTP requests with SSRF protection
│   ├── json_tool.py       # JSON processing
│   ├── nlp_tools.py       # NLP (sentiment, summarization)
│   ├── python_exec.py     # Python execution
│   ├── rag.py             # Knowledge base queries
│   ├── searxng_search.py  # SearXNG self-hosted search (gated via SEARXNG_BASE_URL)
│   ├── self_improve.py    # Self-improvement suggestions (gated)
│   ├── semantic_tool_index.py # Semantic tool description index
│   ├── serpapi_search.py  # SerpAPI (Google/Bing structured)
│   ├── shell.py           # Shell commands
│   ├── slack_tools.py     # Slack messaging (1 tool: cogtrix_slack_post_message)
│   ├── tavily_search.py   # Tavily AI search (2 tools)
│   ├── text_tools.py      # Text processing
│   ├── weather.py         # Weather information
│   ├── web_search.py      # DuckDuckGo search (2 tools: search_web, search_news)
│   ├── whatsapp.py        # WhatsApp messaging (4 tools)
│   ├── _whatsapp_client.py # Waha HTTP client
│   ├── telegram.py        # Telegram messaging (4 tools)
│   ├── _telegram_client.py # Telegram Bot API client
│   └── report_progress.py # Report progress updates during long tasks
└── utils/
    └── atomic_write.py    # atomic_write_json() context manager for safe crash-proof JSON writes
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
    api_key: Optional[str]
    tool_instructions: Optional[str]

@dataclass
class ModelConfig:
    provider: str              # references a key in Config.providers
    model: str                 # actual model name at the provider
    context_window: Optional[int]
    temperature: Optional[float]
    max_tokens: Optional[int]
    timeout: int               # per-request LLM timeout in seconds (default 180)

@dataclass
class Config:
    active_model_alias: Optional[str]
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
| `Config.resolve_llm_config()` | Returns `(ProviderConfig_copy, ModelConfig)` for the active model |
| `Config.resolve_llm_config_for(alias)` | Same but for a named alias or `provider/model` shorthand |
| `Config.resolve_embedding_config()` | Resolve `rag.model` via the `models` registry to `(provider_type, model, base_url, api_key)` |

`ProviderConfig` validates `type` at construction time via `__post_init__` — unrecognized types
raise `ConfigError`. `ModelConfig` validates inference parameters: `temperature` must be in `[0.0,
2.0]`, `context_window` must be `>= 256`, `max_tokens` must be `>= 1` — all violations raise
`ConfigError`. The `context_window` field is forwarded to Ollama as `num_ctx` by the provider
registry; it is silently dropped for OpenAI, Anthropic, and Google.

### 2. CLI Interface (`cogtrix.py` + `src/cli/`)

The entry point handles both interactive and non-interactive modes. CLI utility code is extracted into `src/cli/`.

**Key Components:**

| Component | Location | Purpose |
|-----------|----------|---------|
| `SlashCommandRegistry` | `cogtrix.py` | Registers, resolves, and dispatches `/commands` |
| `SlashCommand` | `cogtrix.py` | Dataclass: name, handler, help text, aliases |
| `parse_arguments()` | `src/cli/args.py` | CLI argument parsing |
| `ColorHelpFormatter` | `src/cli/args.py` | Argparse formatter with bold headers |
| `color_enabled()`, `bold()`, `dim()` | `src/cli/args.py` | ANSI color helpers |
| `print_startup()`, `LOGO_LINES` | `src/cli/banner.py` | Startup banner |
| `read_multiline()` | `src/cli/input.py` | Multi-line paste input |
| `run_inline_shell()` | `src/cli/input.py` | `!<cmd>` inline shell execution |
| `EscapeMonitor` | `src/cli/escape_monitor.py` | Daemon thread: Escape key → SIGINT |
| `ActivityIndicator` | `src/ui/spinner.py` | Animated terminal spinner |

`EscapeMonitor` runs as a background daemon thread. On Unix terminals with a real tty, it enters
cbreak mode during LLM inference, detects standalone Escape bytes (as opposed to escape sequences
such as arrow keys), restores the terminal, and sends SIGINT to the main process. The existing
`KeyboardInterrupt` handler in `cogtrix.py` catches this signal. It is a no-op on non-tty stdin,
Windows, or assistant mode.

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
    │                        ├── /mcp [restart [name]] → list or restart MCP servers
    │                        ├── /memory, /agents, /tasks, /spawn, /goal, /undo, /compact, /retry, /export
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
| `/mcp [restart [name]]` | | List or restart MCP server connections |
| `/think` | `/T` | Deep Tree-of-Thought reasoning |
| `/delegate` | `/d` | Force task delegation across models |
| `/mode` | `/M` | Show / switch memory mode |
| `/model` | `/m` | Show / switch LLM model |
| `/provider` | `/p` | List configured providers (read-only) |
| `/session` | `/s` | Show / switch session |
| `/setup` | | Launch the interactive setup wizard |
| `/approve` | `/a` | Auto-approve all tool confirmations |
| `/optimizer` | `/o` | Toggle prompt optimizer |
| `/debug` | `/D` | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/paste` | `/P` | Enter multi-line paste mode |
| `/clear` | `/c` | Clear conversation history |
| `/memory` | `/mem` | Show / inspect current memory state and rolling summary |
| `/agents` | | List or inspect named agents from AGENTS.md; supports `reload` |
| `/tasks` | `/task` | List background tasks; view details or filter by status |
| `/spawn` | | Submit a background task for a named agent |
| `/goal` | `/goals` | Manage session goals: set, complete, abandon, list |
| `/undo` | | Remove the last exchange from conversation memory |
| `/compact` | | Compress context in place — summarise old messages without deleting history |
| `/retry` | | Re-run the last prompt through the agent |
| `/export` | `/save` | Export conversation to markdown or HTML |

Note: the `/help` listing groups commands into four categories (Session & Config, Tools & Reasoning, Logging, Input & Other). The following command is available but not shown in `/help` — it is excluded from the four main categories by design. The `SlashCommand` dataclass has no `hidden` attribute; commands are excluded from `/help` categorisation through help grouping logic:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/system_prompt` | | Display the full system prompt |

### 3. Orchestration Layer (`src/orchestration/`)

The agent execution pipeline was extracted from `cogtrix.py` into a dedicated package. This is the central coordination layer between the CLI and the agent core.

#### 3a. AgentRunConfig (`src/orchestration/run_config.py`)

`AgentRunConfig` is a dataclass that bundles the 29 fields required by `run_agent()`, `build_agent_graph()`, and `run_execution_phase()`. Using a single config object instead of positional kwargs makes omissions visible as `AttributeError` at call time and keeps function signatures readable.

```python
@dataclass
class AgentRunConfig:
    # Core LLM and tools
    llm: Any = None
    system_prompt: str | None = None
    available_tools: dict[str, Any] | None = None
    active_tools_list: list[Any] | None = None
    preset_tools: set[str] | None = None
    max_context_tokens: int | None = None

    # Context compression
    context_compression: bool = True
    compression_min_age: int | None = None
    compression_min_chars: int | None = None
    compression_llm: Any = None

    # Session and UI
    tool_call_guard: Any | None = None
    session_state: Any = None
    confirmation_ui: Any | None = None
    on_tool_expansion: Any | None = None

    # Execution options
    parallel_tool_execution: bool = True
    git_native: bool = False
    tool_context_limit_pct: float = 0.80
    tier_cache_enabled: bool = True

    # Performance caches (excluded from equality/repr)
    bound_cache: OrderedDict | None = None
    compression_cache: OrderedDict | None = None

    # Decision accountability (ADR-0052)
    decision_accountability_enabled: bool = False
    decision_accountability_report_uncertainty: bool = True
    decision_accountability_min_confidence: float = 7.0

    # Task ownership classifier
    task_ownership_classifier_enabled: bool = True
    task_ownership_classifier_llm_fallback: bool = False
    task_ownership_ambiguous_action: str = "ask"   # "ask" | "inform" | "execute"

    # Pre-action confirmation gate
    pre_action_confirmation_enabled: bool = False
```

Individual kwargs are preserved in `run_agent()` for backward compatibility; when `config` is `None`, a temporary `AgentRunConfig` is assembled from the flat kwargs.

#### 3b. Agent Runner (`src/orchestration/runner.py`)

`run_agent()` is the main entry point for a prompt-to-response cycle. It builds the agent graph, streams its execution, and handles errors and edge cases.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `run_agent()` | Main entry point: build graph, stream, return response string |
| `extract_response(result)` | Walk backward through messages to find the last AIMessage with content |
| `extract_ai_content(msg)` | Handle string, list, multimodal, and reasoning-token content |
| `has_phantom_tool_call(result)` | Detect empty AIMessage with `finish_reason=tool_calls` |
| `format_agent_error(e)` | Categorize exceptions into user-friendly Markdown error messages |
| `ToolCallLogger` | Tracks tool invocations by `call_id` for accurate duration measurement |
| `classify_task_ownership(prompt, ...)` | 3-layer pipeline defined in `intent.py`; called by `run_agent()` to determine execution ownership |
| `_build_ownership_constraint(mode)` | Generates system-prompt constraint text for INFORM/ADVISE/AMBIGUOUS modes; called inside `run_agent()` to inject the constraint before the graph starts |

Before `run_agent()` starts the LangGraph graph, it calls `classify_task_ownership()` to determine
whether the prompt is an execution request (EXECUTE), an information request (INFORM/ADVISE), or
ambiguous. For INFORM and ADVISE modes, `_build_ownership_constraint()` generates a system-prompt
constraint that instructs the agent not to execute operations — only explain. For AMBIGUOUS mode
with `ambiguous_action="ask"`, the agent is constrained to ask one focused clarifying question
before proceeding.

`ToolCallLogger` uses `call_id` (LangChain's unique per-call ID) as the timing key so concurrent calls to the same tool each get accurate duration measurements. Stale entries are evicted after 10 minutes to prevent memory leaks.

`format_agent_error()` categorizes `NotFoundError`, `AuthenticationError`, `RateLimitError`, `APIConnectionError`, `Timeout`, `BadRequestError`, `InternalServerError`, and `ServiceUnavailableError`, plus Ollama-specific connection and model-not-found errors. SDK internals (request/response bodies, headers) are stripped from messages before display.

#### 3c. Agent Graph (`src/orchestration/graph.py`)

`build_agent_graph()` constructs a custom LangGraph `StateGraph` with three nodes and conditional routing.

**Graph Structure:**

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
│  │handle_phantom│ (empty AIMessage  │
│  │              │  with tool_calls  │
│  │              │  finish_reason)   │
│  └──────────────┘                   │
└─────────────────────────────────────┘
    │
    ▼
Agent Response
```

**Node responsibilities:**

- `call_model` — binds active tools to the LLM via `bind_tools()`, prepends the system message, optionally runs context compression, and invokes the model. Uses `_bound_cache` (an `OrderedDict` with LRU eviction at capacity 8) keyed by a tuple of active tool names. The cache is only rebuilt when the active tool set changes, avoiding redundant `bind_tools()` calls on every LLM invocation.
- `process_tools` — iterates the last AIMessage's `tool_calls`, executes known tools, and handles
unknown tool names via `_resolve_tool_name()`. When the agent calls a tool not yet in the active
set, `process_tools` auto-loads it from the on-demand pool (up to `_MAX_TOOL_EXPANSIONS = 3`
auto-expansions per graph run). The `on_tool_expansion` callback decouples spinner updates from the
orchestration layer — `graph.py` has no UI imports. `request_tools` calls are processed via
`_detect_tool_request()` which scans only the current iteration's messages, not the full history.
Mid-turn guidance messages injected by this node are sent as `HumanMessage` (not `SystemMessage`) to
remain compatible with providers that reject `SystemMessage` outside position 0.
- `handle_phantom` — injects a correction hint when the model returns an empty AIMessage with
`finish_reason=tool_calls` (a malformed-JSON failure mode seen in vLLM and some inference servers).
The correction hint is sent as a `HumanMessage`, not a `SystemMessage`, so it is accepted by
providers that reject `SystemMessage` outside position 0 (Qwen3, strict vLLM deployments). Retries
up to `_MAX_PHANTOM_RETRIES = 3` times before injecting a fallback response.

**Key exports from `graph.py`:**

| Symbol | Purpose |
|--------|---------|
| `build_agent_graph(config, registry, approvals, ...)` | Build and compile the StateGraph |
| `ToolManagementRequest` | Dataclass: `add: list[str]`, `remove: list[str]`, `has_changes: bool` |
| `DEFAULT_RECURSION_LIMIT` | 90 (approximately 45 tool calls) |

#### 3d. Pipeline Phases (`src/orchestration/phases.py`)

Houses the post-agent pipeline stages that run between the initial agent response and the memory update.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `extract_fetched_urls(messages)` | Harvest URLs from the agent's tool call history |
| `run_research_delegate(urls, config, ...)` | Spawn sub-agent to re-fetch URLs with high context budget |
| `force_deep_think(task, context, config, ...)` | Invoke Tree-of-Thought engine with gathered research |
| `run_execution_phase(analysis, config, ...)` | Re-run agent to act on analysis output |
| `recover_from_step_limit(graph, result, ...)` | Recover from RecursionError or step-limit apology |
| `is_step_limit_apology(response)` | Detect LLM responses claiming insufficient steps |
| `build_llm_for_decomposition(config)` | Create an LLM instance via provider registry for delegate tasks |
| `WEB_TOOL_NAMES` | frozenset of 9 web tool names used for research-delegate detection |

`build_llm_for_decomposition` delegates to the provider registry in `src/providers/`, keeping `phases.py` provider-agnostic. Research delegate sub-agents use shallow-copied tool objects to avoid shared-state mutation between the delegate and the main agent.

#### 3e. Intent Detection (`src/orchestration/intent.py`)

Classifies user prompts to determine which pipeline branches to activate.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `user_wants_deep_think(prompt)` | Regex detection of deep-thinking requests |
| `user_wants_delegation(prompt)` | Detect explicit delegation requests |
| `prompt_requests_action(prompt)` | Verb+target matching for action-oriented prompts |
| `classify_think_task(prompt, llm)` | LLM-based classification into one of 23 `ThinkCategory` definitions |
| `THINK_CATEGORIES` | Tuple of 23 `ThinkCategory` objects with gather/analysis templates |
| `TaskComplexity` | Enum: `SIMPLE`, `MODERATE`, `COMPLEX_ACTION`, `COMPLEX_RESEARCH` |
| `classify_task_complexity(prompt)` | Regex-based complexity classification; determines step-limit and auto-loads search tools |
| `OwnershipMode` | Enum: `EXECUTE`, `INFORM`, `ADVISE`, `AMBIGUOUS` |
| `OwnershipResult` | Dataclass: `mode`, `confidence`, `is_reversible`, `raw_signal`, `inferred_action` |
| `classify_task_ownership(prompt, llm, ...)` | 3-layer ownership pipeline: structural regex → optional LLM micro-call (10s timeout) → reversibility override |

Each `ThinkCategory` defines keywords for fast pattern matching, a `gather_template` for Stage 1 data collection, an `analysis_preamble` and `stage2_task_framing` for the deep_think call, and a `tool_intensive` flag. When `tool_intensive=True`, automatic `_force_deep_think` is suppressed in normal prompts — the `/think` command still works.

#### 3f. Context Compression (`src/orchestration/compression.py`)

Summarizes old, large `ToolMessage` objects before each LLM call to reduce token usage without losing task-critical context.

**Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `COMPRESSION_MIN_AGE_CYCLES` | 6 | Minimum `call_model` cycles before a message is eligible |
| `COMPRESSION_MIN_CHARS` | 2000 | Minimum message length to qualify for compression |
| `_COMPRESSION_THRESHOLD_RATIO` | 0.72 | Total message size / context window before compression runs |

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `apply_message_compression(messages, ...)` | Entry point; returns a (possibly modified) message list |
| `compress_tool_message(content, tool_name, llm)` | One-shot LLM summarization; falls back to `truncate_tool_output()` |
| `truncate_tool_output(text, max_chars)` | Middle-truncation with informative ellipsis |

Compression is skipped entirely for providers with fewer than 16,384 context tokens (where simple
truncation is cheaper). Eligible messages are compressed in parallel using
`concurrent.futures.ThreadPoolExecutor` (up to 4 workers) rather than sequentially, so large batches
of stale tool outputs are summarized in a single wall-clock pass. Once compressed, results are
cached by `tool_call_id` and reused. The compression pass operates on a copy of the message list —
graph state is never mutated.

#### 3g. Session State (`src/orchestration/session_state.py`)

`SessionState` replaces 7 former module-level globals in `cogtrix.py` with a proper dataclass. One instance is created per interactive session and passed into `build_agent_graph()` and `create_safe_tool_wrapper()`.

```python
@dataclass
class SessionState:
    denials: set[str]           # tools blocked via 'd' or /tools disable
    deny_all: bool              # blanket forbid (reset each prompt cycle)
    no_confirm: bool            # True in assistant mode (auto-approve)
    approvals: set[str]         # tools auto-approved via 'a' or /approve
    loaded_tools: set[str]      # agent-loaded + pinned tools currently active
    pinned_tools: set[str]      # manually pinned tools (persist across prompt cycles)
    all_tool_descriptions: dict[str, str]   # process-scoped, populated at startup
    all_tool_originals: dict[str, Any]      # process-scoped, populated at startup
    checkpoint_store: Any | None = None     # CheckpointStore for checkpoint tool
    _lock: threading.Lock       # guards denials/deny_all against concurrent access
```

**Two-tier tool loading:**
- **Agent-loaded** tools are added to `loaded_tools` when the LLM calls `request_tools`. They are cleared at prompt boundaries by `reset_for_new_prompt()` (`loaded_tools &= pinned_tools`).
- **Pinned** tools are in both `loaded_tools` and `pinned_tools`. They are loaded manually via `/tools load`, `PATCH /sessions/{id}/tools`, or `--activate-tools` and persist across prompt cycles until explicitly unloaded.

**Lifetime mapping:**
- `denials`, `loaded_tools`, `pinned_tools`, `approvals` — session-scoped; cleared on session switch via `reset_for_new_session()`
- `deny_all` — per-prompt; reset at the start of each new prompt cycle via `reset_for_new_prompt()` (`loaded_tools &= pinned_tools` also runs here)
- `no_confirm` — process-scoped; set once at startup
- `all_tool_descriptions`, `all_tool_originals` — process-scoped; populated once at startup

In assistant mode, each call receives its own `SessionState(no_confirm=True)` to prevent any state bleeding between concurrent conversations.

**Thread-safe denial access:** `SessionState` carries a `threading.Lock` (`_lock`) that guards `denials` and `deny_all` against TOCTOU races between the tool-executor `ThreadPoolExecutor` (8 workers) and API/CLI mutation paths. All callers must use the helper methods rather than accessing fields directly:

| Method | Purpose |
|--------|---------|
| `is_denied(tool_name)` | Atomic check: `deny_all or tool_name in denials` |
| `deny_tool(tool_name)` | Atomic add to `denials` |
| `allow_tool(tool_name)` | Atomic remove from `denials` |
| `set_deny_all()` | Atomic set `deny_all = True` |
| `get_denials_snapshot()` | Immutable `frozenset` snapshot for safe off-lock inspection |

`reset_for_new_session()` and `reset_for_new_prompt()` also acquire `_lock` before mutating `denials`/`deny_all`.

#### 3h. Session Orchestrator (`src/orchestration/session_orchestrator.py`)

`SessionOrchestrator` provides snapshot/rollback semantics for model, mode, and session switches. It captures mutable state before a switch attempt and can restore it if the switch fails (for example, if the new model fails to initialize).

```python
class SessionOrchestrator:
    def snapshot(self, *, memory_manager, system_prompt,
                 registry_tools, available_tools, tools) -> SessionSnapshot
    def rollback(self, snap: SessionSnapshot, *, tools_list) -> dict[str, Any]
```

`SessionSnapshot` captures: `active_model_alias`, `memory_mode`, `memory_config`, `session`, `system_prompt`, `memory_manager`, `registry_tools`, `available_tools`, and `tools`. The orchestrator does not perform switches itself — all switch logic remains in `cogtrix.py`; the orchestrator only captures and restores state.

### 4. Agent Core (`src/agent/core.py`)

Defines the state schema, system prompt builder, and LLM factory. The actual graph is built in `src/orchestration/graph.py`.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `CogtrixState` | TypedDict with `messages: Annotated[Sequence[BaseMessage], add_messages]` |
| `build_system_prompt(mode_additions, tool_instructions)` | Generate system prompt with mode context |
| `create_llm_from_provider_config(config)` | Legacy LLM factory (delegates use only) |
| `build_agent_executor(tools, ...)` | Legacy ReAct agent builder (used by delegates) |
| `prepare_messages_with_context(...)` | Token-aware trimming before messages are sent to the LLM |
| `DEFAULT_TOOL_INSTRUCTIONS` | Raw-JSON tool-call formatting (opt-in via `tool_instructions` config) |

`build_system_prompt()` generates generic task instructions only. Deep-think and delegation guidance lives in the respective tool descriptions, not in the system prompt.

`prepare_messages_with_context()` estimates total token count and trims the oldest history messages (or truncates oversized individual messages) to fit the model's context window. The `max_tokens` parameter is dynamically capped to prevent negative values.

### 5. Safety Layer (`src/agent/safety.py`)

Wraps sensitive tools with confirmation prompts.

**Mechanism:**

1. Tools marked with `requires_confirmation: True` are wrapped via `create_safe_tool_wrapper()`
2. Wrapper intercepts execution and calls `session_state.is_denied(tool_name)` — an atomic check under `SessionState._lock` — to eliminate TOCTOU races between tool-executor threads and API/CLI mutations to `deny_all`/`denials`
3. When `session_state.no_confirm` is `True` (assistant mode), the tool is auto-approved silently
4. Otherwise, the confirmation prompt is shown; the user can choose:
   - `y` — Yes, allow once
   - `n` — No, deny once (agent may retry)
   - `a` — Allow all (approve this tool for the session)
   - `d` — Disable tool (block for the session)
   - `f` — Forbid all further tool requests (reset on next prompt)
   - `c` — Cancel the current agent workflow
5. The prompt is wrapped in `try/finally` so `ui.resume_spinner()` is guaranteed even if `KeyboardInterrupt` or `UserCancelledRun` is raised
6. Disabled tools are also blocked at the expansion point in `process_tools` — the agent cannot load them even via `request_tools`

**Sensitive Tools (default):**
- `execute_shell_command`
- `execute_python`
- `write_file`
- `patch_file`
- `append_file`
- `http_post`
- `git_add`, `git_commit`, `git_create_branch`, `git_checkout` (confirmation required)
- `whatsapp_send`, `whatsapp_send_image` (configurable)
- `telegram_send`, `telegram_send_photo` (configurable)

**Key Protocol:**

```python
@runtime_checkable
class ConfirmationUI(Protocol):
    def render_prompt(self, tool_name, tool_args) -> str
    def read_choice(self) -> str
    def pause_spinner(self) -> None
    def resume_spinner(self) -> None
```

CLI passes a Rich-based implementation; tests use stubs; assistant mode passes `None` (which triggers `no_confirm` auto-approval).

### 6. Tool Configuration (`src/tools/configure.py` and `src/tools/resolver.py`)

#### Tool Configuration Factories (`src/tools/configure.py`)

Centralizes all `configure_*` functions that wire up API keys, provider settings, and runtime parameters for each tool module. Called by `cogtrix.py` at startup and on provider/model switches.

**Key Exports:**

| Symbol | Purpose |
|--------|---------|
| `load_tools(tool_filter)` | Load tools from `src/tools/` directory |
| `build_tool_catalog(tools)` | Build `{name: one-line-description}` dict |
| `apply_output_cap(tool, max_chars)` | Patch a tool's output cap in-place |
| `compute_tool_output_cap(max_context_tokens)` | Derive per-tool output limit (10% of context, minimum 8192 chars) |
| `filter_unconfigured_tools(registry)` | Remove tools whose `is_configured()` returns False |
| `TOOL_PRESETS` | Dict of preset names to sets of tool names |
| `apply_tool_preset(preset, tools)` | Filter a tool list to a named preset |
| `create_request_tools_tool(...)` | Build the `request_tools` meta-tool with current catalog |
| `configure_delegate_tools(...)` | Update delegate tool with current provider config |
| `configure_*` functions | Per-module wiring (e.g. `configure_brave_search`, `configure_exa_search`) |

All `configure_*` functions use atomic reference swap (`_config = {**_config, **new}`) so concurrent reads always see a consistent snapshot.

#### Fuzzy Tool Name Resolver (`src/tools/resolver.py`)

`resolve_tool_name()` is the canonical resolver used by both the `process_tools` node and the `request_tools` factory.

```python
def resolve_tool_name(
    requested: str,
    available_tools: dict[str, Any],
    active_tool_names: set[str] | None = None,
) -> tuple[str | None, str]:
```

Resolution order:
1. Exact match in the on-demand pool → `("name", "available")`
2. Exact match among active tools → `("name", "active")`
3. Fuzzy match (Jaccard token overlap + substring/prefix bonuses) → `("name", "available"|"active")`
4. No match → `(None, "none")`

The fuzzy threshold is `FUZZY_MATCH_THRESHOLD = 0.65` (raised from 0.40 to prevent false positives such as `read_file`↔`write_file`). Token overlap uses underscore-split normalization. Substring containment and token-prefix hits add score bonuses (prefix bonus: +0.35) so abbreviated names (e.g., `search` → `search_web`, `shell_exec` → `shell_execute`) resolve correctly.

### 7. Tool Registry (`src/registry.py`)

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

### 8. MCP Client (`src/mcp_client.py`)

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
    ├── asyncio.gather(*[connect(s) for s in servers])  ← parallel connections
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

**Parallel startup:** `connect_all()` launches all server connections concurrently via `asyncio.gather()`, reducing total startup time from N×timeout to max(individual timeouts) when multiple servers are configured.

**Name collision handling:** When an MCP tool has the same name as a built-in tool (e.g., `read_file`), it is automatically prefixed with the server name to prevent shadowing.

**Configuration:** `mcp_servers` section in config file. Transport is auto-detected: `command` → stdio, `url` → SSE. Optional dependency: `mcp` package (`uv pip install "cogtrix[mcp]"`).

### 9. Memory System (`src/memory/`)

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
    def save(self) -> None        # non-blocking: uses alive check, not join()
    def load(self) -> None
    def get_system_prompt_additions(self) -> str

    # Hybrid memory (called by subclasses)
    def set_llm(self, llm: Any) -> None
    def set_embeddings(self, embedding_fn: Any, embedding_model: str) -> None
    def _maybe_summarize(self, messages, window_size) -> None
    def _build_hybrid_prefix(self, user_input: str) -> str | None
```

**Hybrid Memory:** All modes inherit a hybrid memory layer from `BaseMemoryManager`. When an LLM is
injected via `set_llm()`, messages that fall outside the sliding window are incrementally summarized
every 6 messages. Summarization and embedding both run on a background daemon thread named
`memory-slow-path` — they never block the user response. `save()` checks whether the background
thread is alive rather than blocking on `join()`. When an embedding function is injected via
`set_embeddings()`, evicted messages are also embedded into a per-session FAISS index for semantic
recall. Both are injected into the context prefix by `_build_hybrid_prefix()`.

**Meaningful-Content Gate:** Summarization will not fire on short or tool-heavy exchanges. It only runs when at least `_MIN_MEANINGFUL_MSGS_FOR_SUMMARY = 4` messages (2 full H+A pairs) and `_MIN_MEANINGFUL_CHARS_FOR_SUMMARY = 5000` characters of real conversational content exist outside the sliding window.

**Reasoning Mode Section-Freshness Gating (F5):** The reasoning memory mode tracks the turn each prefix section was last modified. Sections older than `prefix_max_stale_turns` (default 3) are omitted from the context prefix to reduce per-call token overhead. This prevents stale planning context from consuming tokens on every LLM call.

**Message Timestamps:** Every message is automatically stamped with a UTC timestamp. The user message is stamped in `prepare_context()` and the AI message in `update()`. Timestamps are stored as ISO 8601 strings and injected as `[2026-02-14 15:23:05 UTC]` at context-preparation time. Old sessions without timestamps load normally (backward-compatible).

**Token-Aware Trimming:** `prepare_messages_with_context()` in `src/agent/core.py` estimates total token count before sending to the LLM and trims the oldest history messages (or truncates oversized individual messages) to fit the context window.

**Session ID Sanitization:** `_sanitize_session_id()` in `src/memory/manager.py` (shared helper) replaces `..`, `/`, `\`, and null bytes in session IDs before use as filesystem paths, then enforces `resolve()` + prefix containment.

### 10. Prompt Optimizer (`src/prompt/optimizer.py`)

`optimize_prompt()` rewrites complex user prompts for clarity using a one-shot LLM call. The optimizer's system instructions are ephemeral — they do not persist in conversation history.

**Gates (checked in order, skip if any matches):**

1. Prompt shorter than `PROMPT_OPTIMIZER_MIN_LENGTH` (400 chars) — always skip
2. Prompt between 400–600 chars and starts with a clear action verb (`create`, `write`, `analyze`, etc.) — skip
3. LLM self-gates: decides whether restructuring is needed; returns original if not
4. Any exception — fail-safe: return original unchanged

A random nonce delimiter is injected around the optimized prompt in the LLM response to prevent prompt injection attacks from embedding instructions in the optimized text.

**Config:** `prompt_optimizer: true` (default) in config file or `Config.prompt_optimizer`.

**Call sites:** non-interactive (before `run_single_prompt`) and interactive (after `user_wants_deep_think()` but before `run_agent()`). Memory and deep-think detection always see the original prompt, not the optimized one.

### 11. RAG System (`src/rag/`)

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

The embedding model is resolved from the `models` registry via `Config.resolve_embedding_config()`. The default embedding provider is `"ollama"`.

### 12. Assistant Mode (`src/assistant/`)

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
│    - Blacklist check                │
│    - Rate limit check (per-chat)    │
│    - Input length + Unicode check   │
│    - Injection pattern matching     │
│    - LLM judge (if enabled)         │
│    → blocked: send canned reply     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Workflow: resolve()              │
│    - Explicit binding → contact     │
│      prompt → auto-detect → default │
│    - Sets system prompt, knowledge  │
│      base, and tool policy          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Memory: prepare_context()        │
│    - Load per-chat history          │
│    - Build context prefix           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Knowledge: recall()              │
│    - Semantic search (FAISS) or     │
│      keyword fallback               │
│    - Inject "Known facts" into      │
│      context prefix                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. Agent: run_agent()               │
│    - Same pipeline as interactive   │
│    - Excluded tools filtered out    │
│    - SessionState(no_confirm=True)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. Guardrails: record + sanitize    │
│    - Record message for rate limit  │
│    - Strip PII, images, HTML, URLs  │
│    - Redact banned strings          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 8. Memory: update() + save()        │
│    - Persist sanitized response     │
│    - Per-chat history stored after  │
│      PII/URL stripping              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 9. Knowledge: extract_and_store()   │
│    - Dispatched to background pool  │
│      (max 2 workers) — non-blocking │
│    - LLM extracts entity-centric    │
│      durable facts from the turn    │
│    - Deduplication by hash          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│10. Truncate + channel.send()        │
│    - Cap at max_response_length     │
│    - Send reply via channel         │
└─────────────────────────────────────┘
```

**Concurrency Model:**

- `ThreadPoolExecutor(max_workers=max_concurrent)` processes different chats in parallel
- Per-session `threading.Lock` serializes messages within the same chat
- One polling thread per channel with adaptive interval — backs off exponentially on idle (factor 1.5, max 60 s), recovers on activity (factor 2.0); configurable per channel
- Channels are initialized concurrently via `ThreadPoolExecutor(max_workers=2)` in `_discover_channels`
- Session eviction thread runs every 60s; `evict_idle()` saves sessions outside the registry lock
- `available_tools` and `active_tools` are shallow-copied per call in `MessageHandler` so concurrent sessions cannot mutate each other's tool sets
- Deep-think calls are capped at 4 concurrent via `_deep_think_sem` (module-level semaphore)
- Knowledge extraction runs on a background `ThreadPoolExecutor(max_workers=2)` so `extract_and_store()` never blocks message delivery

**WhatsApp Polling Optimizations:**

- Two-phase polling: overview snapshot comparison → per-chat message fetch
- `_can_skip_chat()` pre-filter skips HTTP fetch for chats that won't pass the contact filter (group chats always pass)
- `_prefetch_lids()` resolves uncached `@lid` identifiers in parallel via `ThreadPoolExecutor(max_workers=min(n, 8))`
- LRU/TTL `_lid_cache` (1024 entries, negative TTL 300 s configurable, protected by `_lid_cache_lock`)
- Per-chat exponential error backoff (30 s base, 2x escalation, 300 s cap)

**Two-Layer Memory:**

| Layer | Scope | Storage | Purpose |
|-------|-------|---------|---------|
| Per-chat context | Private to each `(channel, chat_id)` | `data/history/{session_key}.json` | Independent conversation history, summarization, vector recall |
| Shared knowledge | Cross-chat | `data/knowledge/facts.json` + `data/vectordb/knowledge/` | Entity-centric facts recalled when relevant to any chat |

**Default Excluded Tools:** `whatsapp_send`, `whatsapp_check`, `whatsapp_send_image`, `whatsapp_contacts`, `telegram_send`, `telegram_check`, `telegram_send_photo`, `telegram_contacts`, `execute_shell_command`, `execute_python`, `write_file`, `append_file`, `read_file`, `read_pdf`, `list_directory`, `file_info`

**Security Guardrails (`src/assistant/guardrails.py`):**

Every message in assistant mode passes through a `GuardrailPipeline` that wraps seven independent components:

| Component | What it does |
|-----------|-------------|
| `ViolationTracker` | Tracks security violations per chat. Auto-blacklists a chat after N violations within a sliding time window. Rate limit violations are excluded from the violation count. Blacklist state is persisted to `data/assistant/violations.json` and survives restarts. |
| `ChatRateLimiter` | Per-chat sliding window (per-minute + per-hour). Thread-safe deque scan. |
| `InputGuard` | Length limit, Unicode steganography detection, 15 pre-compiled injection regex patterns, optional custom patterns. |
| `EncodingDetectionGuard` | Detects encoding-based bypass attempts (Morse code, Base64, hex, leetspeak/ROT13). Scores each message with four sub-detectors (0–1 each); blocks when max score exceeds configurable threshold (default 0.6). |
| `ToolCallGuard` | Inspects tool arguments before execution. Injection scan across all arguments; path blocking for file tools (blocks /etc/, /proc/, .env, id_rsa, and configurable custom paths); exfiltration detection for web tools (flags API keys, SSH keys, and SSNs in URL/query arguments). |
| `LLMJudge` | Opt-in LLM-as-judge classifier. Fail-closed on error (blocks content if the judge crashes or returns empty — intentional secure-by-default). Disabled by default (adds 500ms–2s). |
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

**Workflow System (`src/assistant/workflows.py`):**

`WorkflowRegistry` loads YAML workflow definitions from `data/workflows/<id>/workflow.yaml`. Each workflow bundles:

- **System prompt** — replaces the global system prompt for bound chats
- **Per-workflow knowledge base** — FAISS index built from documents uploaded to `data/workflows/<id>/documents/`
- **Tool policy** — lists of excluded and additionally approved tools

Chat-to-workflow resolution order:

1. **Explicit binding** — `data/workflows/bindings.json` maps `session_key` → `workflow_id`
2. **Contact prompts fallback** — ephemeral, not persisted (backward compat)
3. **Auto-detect** — keyword/regex scoring against incoming message; binding is persisted on match
4. **No match** — global defaults (base system prompt, no workflow knowledge base)

Thread-safe via `threading.RLock`. `MessageHandler` accepts `workflow_registry` and delegates resolution via `WorkflowRegistry.resolve()`. API CRUD at `/api/v1/assistant/workflows/` (11 endpoints).

### 13. API Layer (`src/api/`)

A FastAPI application that exposes Cogtrix capabilities over HTTP and WebSocket. It is the backend for the React web frontend and supports programmatic access via API keys. For the full endpoint reference and WebSocket protocol, see [docs/api/](api/).

**Application factory:** `app.py` exports `create_app()`, which mounts all routers under `/api/v1` (REST) and `/ws/v1` (WebSocket), configures CORS, and registers a lifespan context manager that initialises the database, tool registry, and session registry on startup and flushes them on shutdown.

**Authentication:** `auth.py` issues HS256 JWTs (1-hour access tokens, 30-day refresh tokens) via `COGTRIX_JWT_SECRET` (minimum 32 chars). FastAPI dependencies `get_current_user` and `require_admin` enforce auth on protected routes. `validate_api_key` handles static API keys stored in the database.

**Session lifecycle:**

```
POST /api/v1/sessions           → creates ApiSessionRecord in DB (idle)
First request to session        → get_or_warm() loads memory + builds LLM + AgentRunConfig
POST …/messages  or  WS message → run_message_turn() executes one agent turn
Idle 30+ minutes                → ApiSessionRegistry.evict_idle() saves + removes
Process shutdown                → save_all() flushes all live sessions
```

**Turn execution (`turn_runner.py`):** `run_message_turn()` runs `run_agent()` via
`asyncio.to_thread` (never blocks the event loop). Three execution modes are supported: `normal`
(standard agent run), `think` (deep reasoning via `_run_think_pipeline`), and `delegate` (parallel
sub-agent delegation via `_run_delegate_pipeline`). Intermediate agent states (`analyzing`,
`deep_thinking`, `researching`, `delegating`) are streamed to the WebSocket for frontend progress
visibility. Both pipeline helpers check `session.cancel_event.is_set()` between phases; if
cancellation arrives during post-processing, an `except asyncio.CancelledError` block resets
`session.agent_state = "idle"` before re-raising so the session is never left in a stale
intermediate state. The `done` sentinel message uses `asyncio.wait_for(put(), timeout=5.0)` and
catches only `TimeoutError` — `asyncio.Queue.put()` never raises `QueueFull` (BUG-AUDIT-001). The
`callbacks.py` `WebSocketCallbackHandler` measures tool duration with `time.monotonic()` (consistent
with `ToolCallLogger` in `runner.py`; immune to NTP clock-adjustment drift). Prompt character
computation in `on_llm_start` is gated behind `log.isEnabledFor(logging.DEBUG)` to avoid an O(n)
string scan on every LLM call in production.

**WebSocket streaming (`ws.py`):** `ConnectionManager` maintains one WebSocket per session; a second
connection gracefully displaces the first. Messages are typed via `ServerMessage` / `ClientMessage`
Pydantic envelopes. A 30-second reconnect buffer with sequence-based replay handles brief
disconnections. Server message types include `token`, `tool_start`, `tool_end`,
`tool_confirm_request`, `agent_state`, `memory_update`, `error`, `done`, and `pong`.

**Tool confirmation (`confirmation.py`):** `ApiConfirmationUI` implements the `ConfirmationUI` Protocol from `safety.py` over WebSocket, replacing the CLI Rich panel. `read_choice()` polls at 0.5 s intervals with a 5-minute timeout (defaults to deny). `_POLL_INTERVAL` and `_TIMEOUT_SECONDS` are module-level constants so they are assigned once at import time rather than on every poll iteration (BUG-AUDIT-002).

**Database layer (`db/`):** Async SQLAlchemy with `aiosqlite` by default; switch to `asyncpg` via
`COGTRIX_DB_URL`. ORM models: `User`, `Organization`, `Team`, `Workspace`, `WorkspaceMembership`,
`Plan`, `UsageRecord`, `ApiSessionRecord`, `Message`, `RefreshToken`, `ApiKey`. All access goes
through repository classes in `db/repositories/`. Schema migrations use Alembic (`alembic upgrade
head`); 11 migrations ship with the project (0001–0011) covering the full schema including the
enterprise multi-tenancy layer and Stripe billing fields.

**Key patterns:**

| Pattern | Detail |
|---------|--------|
| Response envelope | `APIResponse[T]`: `{"data": T \| null, "error": APIError \| null}` |
| Paginated lists | `CursorPage[T]`: opaque base64url cursors; compound `(created_at, id)` keyset for stable ordering |
| Blocking I/O | All memory, LLM init, file stat, config I/O, network calls, and assistant JSON persistence (`violation_tracker.save`, `knowledge_store.save`, `scheduler.edit_message`, `scheduler.cancel_message`) run via `asyncio.to_thread` |
| Duplicate warm | Per-session `asyncio.Event` in `ApiSessionRegistry` prevents concurrent `warm_session()` races |
| Thread safety | `WebSocketCallbackHandler` uses `call_soon_threadsafe` + `_try_put_nowait` for per-token enqueue |

**Interactive docs:** Swagger UI at `/api/v1/docs`; ReDoc at `/api/v1/redoc`; OpenAPI schema at `/api/v1/openapi.json`.

### 14. Utilities (`src/utils/`)

Shared helpers used by multiple packages across the codebase.

**`atomic_write.py`** — `atomic_write_json(path)` is a context manager that yields a `(tmp_path,
fd)` pair. On clean exit it renames the temporary file to the target path, making the write atomic
from the OS perspective. On any `BaseException` the rename is skipped, leaving the target file
intact. Used by `src/assistant/scheduler.py`, `src/assistant/knowledge.py`,
`src/assistant/deferral.py`, `src/tools/file_ops.py`, and `src/memory/manager.py` to prevent corrupt
JSON on process crash or signal.

---

## Data Flow

### Request Processing Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Prompt Optimizer (optional)      │
│    - Skip if < 400 chars            │
│    - LLM decides if restructuring   │
│      is needed                      │
│    - Fail-safe: returns original    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Memory: prepare_context()        │
│    - Capture user timestamp (UTC)   │
│    - Get working memory             │
│    - Inject timestamps into context │
│    - Build context prefix           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Agent: graph.stream()            │
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
│ 4. Safety: check confirmation       │
│    - Prompt if required             │
│    - Execute or deny                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Memory: update()                 │
│    - Stamp user msg with saved ts   │
│    - Stamp AI msg with current ts   │
│    - Add to history                 │
│    - Update mode-specific tracking  │
│    - Trigger hybrid memory:         │
│      • Summarize if batch ready     │
│        (background thread)          │
│      • Feed evicted msgs to vector  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. Memory: save()                   │
│    - Non-blocking: alive check only │
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
│ deny_all or in denials?              │
│   └── Yes → Return denial silently  │
│ no_confirm (assistant mode)?        │
│   └── Yes → Auto-approve            │
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

When the user requests deep reasoning and the agent has used web tools, an enhanced pipeline runs between steps 3 and 5 above:

```
Agent finished initial research (step 3)
    │
    ▼
┌─────────────────────────────────────┐
│ 3a. Detect web tool usage           │
│     WEB_TOOL_NAMES check            │
│     extract_fetched_urls()          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3b. Research Delegate               │
│     run_research_delegate()         │
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
│ 3c. Deep Think                      │
│     force_deep_think()              │
│     - Prefer research_context over  │
│       raw tool_outputs              │
│     - Run Tree-of-Thought engine    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3d. Execution Phase (if needed)     │
│     run_execution_phase()           │
│     - Check: prompt requests action │
│       but agent made no write calls │
│     - Re-prompt agent to create     │
│       files based on deep_think     │
│       output                        │
└─────────────────────────────────────┘
    │
    ▼
Continue to step 5 (Memory: update)
```

**Execution phase trigger:** When a prompt requests file actions (`prompt_requests_action()`) but the agent produced only text with no `write_file` / `append_file` calls (`agent_performed_writes()` returns False), the orchestrator feeds the analysis back to the agent via `run_execution_phase()` so it can act on it. Write tools are available throughout both phases.

All pipeline functions live in `src/orchestration/phases.py`. The `research_delegate` section is fully configurable via `.cogtrix.yaml` — all fields (`enabled`, `timeout`, `cap_ratio`, `auto`, `auto_threshold`) are parsed by `src/config.py` and forwarded to the research delegate at runtime.

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
2. Register the module in `src/providers/__init__.py` by adding it to the `_MODULES` dict. Provider modules are cached via `@functools.cache` on `_load_provider()` to avoid redundant imports.
3. Add default model, embedding model, and base URL entries to `src/providers/defaults.py`.
4. Update `src/config.py` if additional validation is needed in `_parse_providers_section()`.

### Adding a New Interface

The agent core is interface-agnostic. The orchestration layer provides a clean `run_agent()` entry point:

```python
from src.config import load_config
from src.registry import ToolRegistry
from src.providers import create_chat_model_from_configs
from src.orchestration.runner import run_agent
from src.orchestration.run_config import AgentRunConfig
from src.orchestration.session_state import SessionState
from src.memory import MemoryFactory, JsonFileMemoryStore

# Load config
config = load_config()

# Setup tools
registry = ToolRegistry()
registry.load_all_tools()
tools = list(registry.tools.values())

# Create LLM
provider_config, model_config = config.resolve_llm_config()
llm = create_chat_model_from_configs(provider_config, model_config)

# Setup memory
memory_store = JsonFileMemoryStore()
memory_manager = MemoryFactory.create(mode="conversation", store=memory_store)
memory_manager.load()

# Build run config
agent_config = AgentRunConfig(
    llm=llm,
    active_tools_list=tools,
    session_state=SessionState(),
)

# Use agent
context = memory_manager.prepare_context(user_input)
response = run_agent(
    user_input=user_input,
    history_messages=context.messages,
    registry=registry,
    approvals=set(),
    config=agent_config,
)
memory_manager.update(user_input, response)
memory_manager.save()
```

---

## Security Model

### Tool Safety

| Category | Tools | Confirmation |
|----------|-------|--------------|
| Read-only | `read_file`, `list_directory`, `search_web`, `git_status`, `git_diff`, `git_log` | No |
| Sensitive | `execute_shell_command`, `write_file`, `patch_file`, `append_file`, `execute_python`, `git_add`, `git_commit`, `git_create_branch`, `git_checkout` | Yes |
| External | `http_post` | Yes |

Denial checks (`is_denied()`) are atomic under `SessionState._lock`, eliminating TOCTOU races between tool-executor threads and API/CLI mutations. `_confirmation_lock` in `create_safe_tool_wrapper()` is a separate, narrower lock that serializes the confirmation prompt UI so concurrent tool calls do not interleave their prompts.

### File Path Safety

All file operations (`read_file`, `write_file`, `patch_file`, `append_file`, `list_directory`, `file_info`) validate paths through `_validate_path()` in `src/tools/file_ops.py` before any filesystem access.

**Allowed roots:**

| Operation | Allowed directories |
|-----------|---------------------|
| **Read** | Current working directory (`cwd`) **and** the application install directory (`_APP_DIR`) |
| **Write** | Current working directory (`cwd`) only |

The dual-root model exists for Docker deployments where the working directory is set to a scratch path (e.g., `-w /tmp`) while the application source lives at `/app`. Without `_APP_DIR`, the agent cannot read project documentation or source files, forcing expensive web searches instead.

**Path traversal protection:** Paths containing `..` are resolved and must still fall within an allowed root after resolution. Symlinks are followed by `Path.resolve()` before the containment check.

**Concurrent append safety:** `append_file` uses `_RefLock` — a ref-counted `threading.Lock` wrapper — obtained from an LRU `OrderedDict` capped at 256 entries. Eviction skips entries with `ref_count > 0` to prevent use-after-eviction where two threads hold different lock objects for the same path.

### HTTP Request SSRF Protection

`src/tools/http_request.py` uses Python's `ipaddress` module to block requests to loopback, private, link-local, reserved, and unspecified IP ranges. DNS resolution is checked on every returned address (not just the first) to prevent DNS rebinding attacks.

### Atomic Tool Configuration Updates

All `configure_*` functions in `src/tools/configure.py` use atomic reference swap (`_config = {**_config, **new}`) so concurrent reads always see a consistent snapshot. This prevents partial configuration states during provider or model switches.

### Assistant Mode Guardrails

Assistant mode adds a dedicated security layer in `src/assistant/guardrails.py`. See [Section 12 —
Security Guardrails](#security-guardrails-srcassistantguardrailspy) for the full description. The
pipeline includes `EncodingDetectionGuard`, `ToolCallGuard`, and `ViolationTracker` (auto-blacklist)
in addition to the original four components. Configuration is in the `services.assistant.guardrails`
config block.

### API Key Management

- Store in environment variables (preferred)
- Or in config file (`.cogtrix.yaml` / `.cogtrix.json`)
- Never commit to version control (`.gitignore` excludes config files and `.env*`)
- Keys hidden in logs (`api_key: "***"`)

### Session Isolation

- Each session has separate memory file
- Approvals and denials are session-scoped (not persisted)
- History stored in `data/history/{session_id}.json`
- Session IDs are sanitized via `_sanitize_session_id()` before filesystem use

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
| `faiss-cpu` | Vector store (optional, `cogtrix[rag]`) |
| `pypdf` | PDF loading (optional, `cogtrix[rag]`) |
| `python-docx` | DOCX file support (optional, `cogtrix[rag]`) |
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
