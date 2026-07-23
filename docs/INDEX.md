# Cogtrix Documentation

## User Guides

| Guide | Description |
|-------|-------------|
| [Configuration](CONFIGURATION.md) | Config file format, all settings, environment variables |
| [Providers](PROVIDERS.md) | LLM provider setup (OpenAI, Ollama, Anthropic, Google, xAI) |
| [Tools Reference](TOOLS_REFERENCE.md) | All built-in tools with parameters and examples |
| [Memory Modes](MEMORY_MODES.md) | Conversation, code, and reasoning memory modes |
| [Deep Think](DEEPTHINK.md) | Tree-of-Thought reasoning, think categories, research pipeline |
| [RAG Guide](RAG_GUIDE.md) | Document ingestion, embeddings, retrieval-augmented generation |

## Integration Guides

| Guide | Description |
|-------|-------------|
| [WhatsApp](WHATSAPP_GUIDE.md) | WhatsApp assistant mode via Waha |
| [Telegram](TELEGRAM_GUIDE.md) | Telegram bot assistant mode |

## Developer Guides

| Guide | Description |
|-------|-------------|
| [Architecture](ARCHITECTURE.md) | System architecture, component design, data flow |
| [Development](DEVELOPMENT.md) | Adding tools, memory modes, slash commands, testing |
| [Bug Hunting](BUG_HUNTING.md) | Manual testing checklist for QA |

## Architecture Decision Records

| ADR | Status | Description |
|-----|--------|-------------|
| [ADR-007](adr/007-unify-tool-safety.md) | Proposed | Unify tool safety into `src/agent/safety.py` |
| [ADR-0008](adr/0008-async-memory-update.md) | Accepted | Background memory update after agent response |
| [ADR-0009](adr/0009-eliminate-live-list-reference-in-background-thread.md) | Proposed | Eliminate live list reference in background thread |
| [ADR-0010](adr/0010-remove-cogtrix-module-import-from-orchestration.md) | Proposed | Remove reverse import from orchestration into cogtrix.py |
| [ADR-0011](adr/0011-spinner-coupling-in-graph-process-tools.md) | Implemented | Decouple UI spinner from graph process_tools node |
| [ADR-0012](adr/0012-extract-session-switching-into-session-manager.md) | Proposed | Extract live-session switching into SessionOrchestrator |
| [ADR-0013](adr/0013-extract-version-constants-from-cogtrix-entry-point.md) | Proposed | Extract version constants from cogtrix.py into src/_version.py |
| [ADR-0014](adr/0014-extract-tool-preset-and-apply-preset-to-configure.md) | Proposed | Extract TOOL_PRESETS and apply_tool_preset into src/tools/configure.py |
| [ADR-0015](adr/0015-tool-activation-workflow-audit.md) | Proposed | Tool activation workflow audit and remediation plan |
| [ADR-0016](adr/0016-architectural-review-2026-02.md) | Accepted | Architectural review — February 2026 |
| [ADR-0017](adr/0017-cross-provider-model-resolution.md) | Proposed | Cross-provider model resolution for `/model` command |
| [ADR-0018](adr/0018-agent-persistence-for-search-tasks.md) | Proposed | Agent persistence for search-heavy tasks |
| [ADR-0019](adr/0019-architecture-review-2026-02.md) | Informational | Architecture review — February 2026 (round 2) |
| [ADR-0020](adr/0020-architecture-review-round-12.md) | Informational | Architecture review — round 12 |
| [ADR-0021](adr/0021-architecture-review-round-13.md) | Informational | Architecture review — round 13 |
| [ADR-0022](adr/0022-architecture-review-round-15.md) | Accepted | Architecture review — round 15 (final audit) |
| [ADR-0023](adr/0023-performance-optimization-plan.md) | Accepted | Performance optimization plan — TTFT and token budget |
| [ADR-0024](adr/0024-parallel-tool-execution.md) | Accepted | Parallel tool execution via ThreadPoolExecutor |
| [ADR-0025 Architecture](adr/0025-architecture-review-round-22.md) | Informational | Architecture review — round 22 |
| [ADR-0025 Latency](adr/0025-latency-audit-round-22.md) | Informational | Latency audit — round 22 |
| [ADR-0026](adr/0026-python-exec-session-lock-consistency.md) | Accepted | Python exec session lock consistency |
| [ADR-0027 Architecture](adr/0027-architecture-review-round-23.md) | Informational | Architecture review — round 23 |
| [ADR-0027 Performance](adr/0027-performance-audit-round-23.md) | Informational | Performance audit — round 23 |

## Internal Documentation

| Document | Description |
|----------|-------------|
| [AI Interaction Audit v3](prompts/ai-interaction-audit-v3.md) | Round 3 prompt and interaction audit (final) |
