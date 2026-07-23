# Cogtrix Documentation

## User Guides

| Guide | Description |
|-------|-------------|
| [Configuration](CONFIGURATION.md) | Config file format, all settings, environment variables |
| [Providers](PROVIDERS.md) | LLM provider setup (OpenAI, Ollama, Anthropic, Google, xAI) |
| [Tools Reference](TOOLS_REFERENCE.md) | All 51 built-in tools with parameters and examples |
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
| [Input Prompt UX](ux/input-prompt.md) | UX spec for interactive input prompt |
| [AI Interaction Audit v3](prompts/ai-interaction-audit-v3.md) | Round 3 prompt and interaction audit (final) |

## Bug Reports

Chronological sweep reports produced by the `bug_hunter` agent. Each report covers a targeted scope; fix-verification reports confirm that findings from the previous round were resolved.

| Report | Date | Scope |
|--------|------|-------|
| [Full Sweep](bugs/2026-02-25-full-sweep.md) | 2026-02-25 | Round 1: cogtrix.py, all src/ modules |
| [Post-Refactor Sweep](bugs/2026-02-25-post-refactor-sweep.md) | 2026-02-25 | Round 1: post-refactor verification |
| [Fix Verification](bugs/2026-02-25-fix-verification.md) | 2026-02-25 | Round 1: fix verification + new findings |
| [Uncovered Modules](bugs/2026-02-25-uncovered-modules-sweep.md) | 2026-02-25 | Round 2: memory, MCP, intent, tools |
| [Round 3 Sweep](bugs/2026-02-26-round3-sweep.md) | 2026-02-26 | Round 3: providers, config, SSRF, threading |
| [Round 3 Verification](bugs/2026-02-26-fix-verification.md) | 2026-02-26 | Round 3: fix verification |
| [Round 4 Sweep](bugs/2026-02-26-round4-sweep.md) | 2026-02-26 | Round 4: P2 sprint changes |
| [Round 4 P2 Sprint](bugs/2026-02-26-round4-p2-sprint-sweep.md) | 2026-02-26 | Round 4: P2 fix sprint results |
| [Round 4 Re-Audit](bugs/2026-02-26-round4-phase6-reaudit.md) | 2026-02-26 | Round 4: phase 6 re-audit |
| [Round 5 Sweep](bugs/2026-02-26-round5-sweep.md) | 2026-02-26 | Round 5: P1 sprint — executor misuse, HTTP buffering, slash command fallthrough |
| [Exhaustive Sweep](bugs/2026-02-26-exhaustive-sweep.md) | 2026-02-26 | Round 6: full codebase audit of orchestration and runner |
| [P1 Sprint Re-Audit](bugs/2026-02-26-p1-sprint-reaudit.md) | 2026-02-26 | Round 6: targeted re-audit of P1 sprint changed files |
| [Arch Review: Escape Monitor](bugs/2026-02-27-arch-review-escape-monitor.md) | 2026-02-27 | Architectural review of escape_monitor, spinner, confirmation UI |
| [Escape Monitor Sweep](bugs/2026-02-27-escape-monitor-sweep.md) | 2026-02-27 | Round 7: escape monitor, spinner integration, Ctrl+C / prefill flow |
| [Exhaustive Sweep](bugs/2026-02-27-exhaustive-sweep.md) | 2026-02-27 | Round 8: full sweep — escape_monitor, spinner, config, providers |
| [Performance Audit](bugs/2026-02-27-performance-audit.md) | 2026-02-27 | Round 8: TTFT, token budget, hot path, per-cycle overhead |
| [Round 10 Deep Dive](bugs/2026-02-27-round10-deep-dive.md) | 2026-02-27 | Round 10: escape_monitor, input, spinner, REPL, orchestration |
| [Round 12](bugs/2026-02-27-round-12.md) | 2026-02-27 | Round 12: post-sprint sweep — REPL loop, orchestration layer |
| [Round 13](bugs/2026-02-27-round-13.md) | 2026-02-27 | Round 13: round 12 fix verification, slash commands, tool management |
| [Round 14](bugs/2026-02-27-round-14.md) | 2026-02-27 | Round 14: round 13 critical-fix verification |
| [Round 15](bugs/2026-02-27-round-15.md) | 2026-02-27 | Round 15: final verification and sweep |
| [Round 16](bugs/2026-02-27-round-16.md) | 2026-02-27 | Round 16: exhaustive post-sprint sweep |
| [Round 16 Architecture Review](bugs/2026-02-27-arch-round16-final.md) | 2026-02-27 | Round 16: final architecture review |
| [Round 16 Performance Audit](bugs/2026-02-27-round16-performance-audit.md) | 2026-02-27 | Round 16: TTFT, token budget, hot path, per-cycle overhead |
| [Round 17](bugs/2026-02-27-round-17.md) | 2026-02-27 | Round 17: post-sprint sweep |
| [Round 17 Architecture Review](bugs/2026-02-27-arch-round17-final.md) | 2026-02-27 | Round 17: final architecture review |
| [Round 17 Performance Audit](bugs/2026-02-27-round17-performance-audit.md) | 2026-02-27 | Round 17: PERF-203/204 verification + residual findings |
| [Round 18](bugs/2026-02-28-round-18.md) | 2026-02-28 | Round 18: brief findings |
| [Round 18 Full](bugs/2026-02-28-round18-sweep.md) | 2026-02-28 | Round 18: full audit |
| [Round 19](bugs/2026-02-28-round-19.md) | 2026-02-28 | Round 19: sweep |
| [Round 20](bugs/2026-02-28-round-20.md) | 2026-02-28 | Round 20: findings |
| [Round 21](bugs/2026-02-28-round-21.md) | 2026-02-28 | Round 21: comprehensive audit |
| [Round 22](bugs/2026-02-28-round-22.md) | 2026-02-28 | Round 22: holistic optimization audit |
| [Round 23](bugs/2026-02-28-round-23.md) | 2026-02-28 | Round 23: holistic optimization audit |
| [Round 27](bugs/2026-02-28-round-27.md) | 2026-02-28 | Round 27: holistic optimization audit |
| [Round 27 Architecture Review](bugs/2026-02-28-arch-round27-review.md) | 2026-02-28 | Round 27: architecture review |
| [Performance Audit: TTFT](bugs/2026-02-28-performance-audit-ttft.md) | 2026-02-28 | Round 27: TTFT performance audit |
| [Exhaustive Sweep](bugs/2026-03-01-exhaustive-sweep.md) | 2026-03-01 | Round 28: exhaustive sweep — thread safety, compression, arg correction |
