# Cogtrix Documentation

## Repositories & Packages

| Project | GitHub Repository | Container Registry |
|---------|------------------|--------------------|
| Cogtrix (backend / CLI / API) | [NorthlandPositronics/Cogtrix](https://github.com/NorthlandPositronics/Cogtrix) | [`ghcr.io/northlandpositronics/cogtrix`](https://github.com/NorthlandPositronics/Cogtrix/pkgs/container/cogtrix) |
| Cogtrix WebUI (React frontend) | [NorthlandPositronics/Cogtrix-WebUI](https://github.com/NorthlandPositronics/Cogtrix-WebUI) | [`ghcr.io/northlandpositronics/cogtrix-webui`](https://github.com/NorthlandPositronics/Cogtrix-WebUI/pkgs/container/cogtrix-webui) |

**Pull examples:**
```bash
docker pull ghcr.io/northlandpositronics/cogtrix:latest
docker pull ghcr.io/northlandpositronics/cogtrix-webui:latest
```

---

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

## API Documentation

| Guide | Description |
|-------|-------------|
| [OpenAPI Schema](api/openapi.yaml) | OpenAPI 3.1 specification for all 87 REST endpoints ([JSON](api/openapi.json)) |
| [Client Contract](api/client-contract.md) | TypeScript types, API client patterns, and WebSocket example code |
| [WebSocket Protocol](api/websocket-protocol.md) | Streaming message types, authentication, connection lifecycle |
| [WebUI Development Guide](api/webui-development-guide.md) | Page map, component hierarchy, integration patterns, and state management for React developers |
| [Test Specification](api/test-specification.md) | Master test plan: schema validation, endpoint integration, DB repository, and WebSocket tests |

## Developer Guides

| Guide | Description |
|-------|-------------|
| [Architecture](ARCHITECTURE.md) | System architecture, component design, data flow |
| [Development](DEVELOPMENT.md) | Adding tools, memory modes, slash commands, testing |
| [Bug Hunting](BUG_HUNTING.md) | Manual testing checklist for QA |

## Architecture Decision Records

| ADR | Status | Description |
|-----|--------|-------------|
| [ADR-0007](adr/0007-unify-tool-safety.md) | Proposed | Unify tool safety into `src/agent/safety.py` |
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
| [ADR-0025b Latency](adr/0025b-latency-audit-round-22.md) | Informational | Latency audit — round 22 |
| [ADR-0026](adr/0026-python-exec-session-lock-consistency.md) | Accepted | Python exec session lock consistency |
| [ADR-0027 Architecture](adr/0027-architecture-review-round-23.md) | Informational | Architecture review — round 23 |
| [ADR-0027b Performance](adr/0027b-performance-audit-round-23.md) | Informational | Performance audit — round 23 |
| [ADR-0028](adr/0028-architectural-review-march-2026.md) | Informational | Architectural review — March 2026 |
| [ADR-0029](adr/0029-assistant-mode-architectural-review.md) | Informational | Assistant mode architectural review: datamarking and subsystem audit |
| [ADR-0030](adr/0030-comprehensive-architectural-review-march-2026.md) | Informational | Comprehensive architectural review — March 2026 |
| [ADR-0031](adr/0031-list-scheduled-messages-filter-parameters.md) | Accepted | Extended filter parameters for `list_scheduled_messages` |
| [ADR-0032](adr/0032-whatsapp-polling-performance-optimizations.md) | Accepted | WhatsApp polling performance optimizations |
| [ADR-0033](adr/0033-architecture-and-performance-review-march-2026-round2.md) | Informational | Architecture and performance review — March 2026 (round 2) |
| [ADR-0034](adr/0034-sprint-audit-fixes-march-2026.md) | Accepted | Sprint audit fixes — March 2026 (performance, concurrency, correctness) |
| [ADR-0035](adr/0035-architecture-performance-review-round3.md) | Proposed | Architecture and performance review — round 3 |
| [ADR-0036](adr/0036-deferred-audit-fixes-march-2026.md) | Proposed | Deferred audit fixes — March 2026 (concurrency, security, correctness) |
| [ADR-0037](adr/0037-deferred-audit-fixes-round4-march-2026.md) | Proposed | Deferred audit fixes — round 4, March 2026 (performance, architecture, tests) |
| [ADR-0038](adr/0038-deferred-round4-final-batch-march-2026.md) | Proposed | Deferred round 4 final batch — March 2026 (PERF-801/803/809, ARCH-037-06/08) |
| [ADR-0040](adr/0040-round6-audit-sprint.md) | Proposed | Round 6 audit sprint plan — BUG-083–090, ARCH-040, PERF-1001–1008 |
| [ADR-0041](adr/0041-queue-append-tool-for-sequential-scheduling.md) | Accepted | Queue-append tool for sequential scheduled message delivery |
| [ADR-0042 AI Designer](adr/0042-ai-designer-recommendations.md) | Accepted | AI designer recommendations for `defer_processing` tool |
| [ADR-0042b Defer Reply](adr/0042b-defer-reply-tool.md) | Proposed | Deferred message processing (`defer_processing` tool) |
| [ADR-0043](adr/0043-sprint3-architecture-audit.md) | Proposed | Sprint 3 architecture audit |
| [ADR-0044](adr/0044-sprint4-architecture-audit.md) | Proposed | Sprint 4 architecture audit |
| [ADR-0045](adr/0045-api-validation-error-translation-layer.md) | Accepted | API validation error translation layer |
| [ADR-0046](adr/0046-bug-fix-batch-march-2026.md) | Proposed | Bug fix batch — March 2026 (security, correctness, cleanup) |
| [ADR-0047 Architecture](adr/0047-architecture-review-round-5-march-2026.md) | Accepted | Architecture review — round 5, March 2026 |
| [ADR-0047b Performance](adr/0047b-performance-audit-round5-ttft-and-total-latency.md) | Proposed | Performance audit — TTFT and total request latency |
| [ADR-0048](adr/0048-workflow-system.md) | Accepted | Workflow system — per-chat knowledge bases, tool policies, auto-detection |
| [ADR-0049](adr/0049-architecture-review-holistic-audit-march-2026.md) | Informational | Architecture review — holistic audit, March 2026 |

## Bug Reports

The [`docs/bugs/`](bugs/) directory contains 62 bug report files produced during
automated and manual audit sweeps. Each file documents findings from a single
audit session with BUG-IDs, root-cause analysis, and fix status.

## Internal Documentation

| Document | Description |
|----------|-------------|
| [AI Interaction Audit v3](prompts/ai-interaction-audit-v3.md) | Round 3 prompt and interaction audit (final) |
