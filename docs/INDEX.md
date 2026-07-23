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
| [API Overview](API/OVERVIEW.md) | Entry point — transport layers, authentication model, surface map |
| [OpenAPI Schema](API/OPENAPI.yaml) ([JSON](API/OPENAPI.json)) | OpenAPI 3.1 specification for the REST surface — YAML and JSON forms are equivalent |
| [Client Contract](API/CLIENT_CONTRACT.md) | TypeScript types, API client patterns, and WebSocket example code |
| [WebSocket Protocol](API/WEBSOCKET_PROTOCOL.md) | Streaming message types, authentication, connection lifecycle |
| [WebUI Development Guide](API/WEBUI_DEVELOPMENT_GUIDE.md) | Page map, component hierarchy, integration patterns, and state management for React developers |
| [Enterprise API](API/ENTERPRISE.md) | Enterprise-tier endpoints (SAML, LDAP, JIT, teams, workspaces) |

## Developer Guides

| Guide | Description |
|-------|-------------|
| [Architecture](ARCHITECTURE.md) | System architecture, component design, data flow |
| [Concurrency Policy](architecture/CONCURRENCY.md) | `ThreadPoolExecutor` usage rules, `invoke_with_timeout` helper, the four shared pools |
| [Development](DEVELOPMENT.md) | Adding tools, memory modes, slash commands, testing |
| [Agent Complexity Test Fleet](../tests/agent_complexity/README.md) | Multi-container Docker fleet that exercises the agent across complexity tiers |

## Reference & Planning

| Document | Status | Description |
|----------|--------|-------------|
| [Versioning](VERSIONING.md) | Active | Versioning and stability policy |
| [Migration](MIGRATION.md) | Planned | v1.0 migration guide draft |
| [Tools Authoring](TOOLS_AUTHORING.md) | Active | How to add built-in, file-drop, and entry-point tools |

---

## Internal Documentation (Private)

Material that is internal-only, in flux, or aimed at a private audience —
including the full **Architecture Decision Record (ADR) set**, the
**master API test specification**, **internal prompt design notes**,
**quality and audit-run metrics**, **UX research**, **internal bug
investigations**, the **core and enterprise roadmaps**, the **security
audit report**, and the **team bug-hunting process doc** —
lives in the private
[`NorthlandPositronics/cogtrix-docs`](https://github.com/NorthlandPositronics/cogtrix-docs)
repository, mounted into this tree as a git submodule at
[`docs/optional/`](optional/).

Authorised contributors can fetch it with:

```bash
git submodule update --init docs/optional
```

The catalogue of contents lives at `docs/optional/INDEX.md` once the
submodule is initialised. Public visitors without access to that
repository do not need it — Cogtrix builds, runs, and ships without
`docs/optional/` populated.
