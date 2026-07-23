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

## Internal Documentation

| Document | Description |
|----------|-------------|
| [Refactoring Plan](architecture/refactoring-plan.md) | Planned structural refactoring of cogtrix.py |
| [Input Prompt UX](ux/input-prompt.md) | UX spec for interactive input prompt |
| [AI Interaction Audit v1](prompts/ai-interaction-audit-v1.md) | Round 1 prompt and interaction audit |
| [AI Interaction Audit v2](prompts/ai-interaction-audit-v2.md) | Round 2 prompt and interaction audit |

## Bug Reports

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
