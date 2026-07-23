# Changelog

## [0.1.23](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.22...v0.1.23) (2026-03-20)


### Bug Fixes

* **ci:** post status checks to correct SHA after Contents API commit ([d18249c](https://github.com/NorthlandPositronics/Cogtrix/commit/d18249c3f79ae0606daea1c3aefbda2fb604d7ed))
* **ci:** post status checks to the new commit SHA after Contents API push ([a63e146](https://github.com/NorthlandPositronics/Cogtrix/commit/a63e14675f95b8cede988917a6bd53ca813309db))

## [0.1.22](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.21...v0.1.22) (2026-03-20)


### Bug Fixes

* **ci:** sign uv.lock update commit via GitHub Contents API ([#27](https://github.com/NorthlandPositronics/Cogtrix/issues/27)) ([ba4ccc1](https://github.com/NorthlandPositronics/Cogtrix/commit/ba4ccc1ee1ae7e2a764aa16a21d57a2fec4d8a52))
* **ci:** use jq --rawfile to avoid ARG_MAX on base64-encoded uv.lock ([#34](https://github.com/NorthlandPositronics/Cogtrix/issues/34)) ([61b68c3](https://github.com/NorthlandPositronics/Cogtrix/commit/61b68c350db494d69076ec178971da55c99a6dc1))
* **ci:** use jq temp file to avoid arg-too-long on large uv.lock ([#30](https://github.com/NorthlandPositronics/Cogtrix/issues/30)) ([a7a6733](https://github.com/NorthlandPositronics/Cogtrix/commit/a7a67334aecb1a7eee367a78a88e2b8a8cb0d9e5))
* **deps:** resolve 7 security vulnerabilities in dependencies ([#32](https://github.com/NorthlandPositronics/Cogtrix/issues/32)) ([94cf6d5](https://github.com/NorthlandPositronics/Cogtrix/commit/94cf6d570403c707a0f61f12fb5d79aea80383c4))
* use jq --rawfile to bypass ARG_MAX on large uv.lock ([b05d0a1](https://github.com/NorthlandPositronics/Cogtrix/commit/b05d0a10fe69f44fd114b46eda321131917425c9))

## [0.1.21](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.20...v0.1.21) (2026-03-20)


### Bug Fixes

* **ci:** add --extra api and timeout to release workflow ([34087f6](https://github.com/NorthlandPositronics/Cogtrix/commit/34087f681b7189236bea7834284e26ed516e459c))
* **ci:** add --extra api and timeout to release workflow; make Docker publish advisory ([d8c76f7](https://github.com/NorthlandPositronics/Cogtrix/commit/d8c76f7bce971308d05568000bbda08e4cecbf8f))
* **ci:** add statuses: write permission to release-please workflow ([4115270](https://github.com/NorthlandPositronics/Cogtrix/commit/41152706924387bcc6a1e604f7f75e4610bb0c6a))
* **ci:** add statuses: write permission to release-please workflow ([ef30656](https://github.com/NorthlandPositronics/Cogtrix/commit/ef3065623cbd44f8c8f9ad11a9cd69503c98c86a))
* **ci:** add statuses: write permission to release-please workflow ([024f167](https://github.com/NorthlandPositronics/Cogtrix/commit/024f167c1607b33ee750c1575447b99a7396d5ba))

## [0.1.20](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.19...v0.1.20) (2026-03-20)


### Features

* **ci:** guard main source branch, fix CI dependencies and hanging tests ([2a96cb7](https://github.com/NorthlandPositronics/Cogtrix/commit/2a96cb7d8efa96cbcda15a755585914c25a1b68d))
* **ci:** guard main source branch, fix CI dependencies and hanging tests ([06dba98](https://github.com/NorthlandPositronics/Cogtrix/commit/06dba98bfc5a91393f8a88fd4cfd33286ad18634))
* **ci:** guard main source branch, fix CI dependencies and hanging tests ([#17](https://github.com/NorthlandPositronics/Cogtrix/issues/17)) ([2a96cb7](https://github.com/NorthlandPositronics/Cogtrix/commit/2a96cb7d8efa96cbcda15a755585914c25a1b68d))

## [Unreleased]

### Breaking Changes

* **config:** Provider/model separation refactor — `ProviderConfig` now holds connection info only (`type`, `base_url`, `api_key`, `tool_instructions`); all inference parameters (`model`, `temperature`, `num_ctx`, `max_tokens`) belong exclusively in `ModelConfig`. `models.default` selects the active model alias. Legacy top-level `provider`/`model` keys and model fields inside `providers:` entries are auto-migrated but deprecated.
* **cli:** `--provider` / `-p` CLI flag removed; use `--model` / `-m` with a model alias instead
* **cli:** `/provider` command is now read-only (lists providers); use `/model` to switch models
* **config:** `COGTRIX_PROVIDER` environment variable removed; use `COGTRIX_MODEL` instead
* **api:** `POST /config/provider` endpoint removed (returns 410 Gone)

### Features

* **assistant:** Level 1 outbound messaging — `POST /api/v1/assistant/outbound` admin endpoint sends operator-initiated messages to phonebook contacts via the agent pipeline (bypasses input guardrails, applies output guardrails, updates memory)
* **assistant:** Level 2 campaign system — multi-contact outbound campaigns with per-target progress tracking, automatic follow-ups when contacts don't reply, escalation after max attempts, and agent-classified goal completion via `report_campaign_outcome` tool; 6 API endpoints for CRUD + launch; persistence to `data/assistant/campaigns.json`; background follow-up thread with configurable check interval
* **tools:** two-tier tool loading — agent-loaded tools auto-unload after each prompt cycle; manually loaded tools (via `/tools load`, `--activate-tools`, or API `PATCH`) are pinned and persist until explicitly unloaded
* **cli:** `--activate-tools LIST` flag pins comma-separated tools as active at startup
* **cli:** `/tools unload <name>` command to unpin and return a tool to the on-demand pool
* **rag:** `query_knowledge_base` auto-activates (pinned) when a FAISS knowledge base exists; dynamic description shows index count and size
* **rag:** multi-index search — queries both global CLI index and per-document API indexes, merges and deduplicates results
* **api:** tool status now includes `"pinned"` for manually loaded tools in `ToolStatus` enum; API version bumped to 1.1.0
* **api:** REST + WebSocket API layer with JWT authentication, session management, streaming agent turns, tool management, memory control, RAG document endpoints, config management, MCP server management, and assistant mode control (65 REST endpoints + 2 WebSocket streams)
* **api:** API key authentication (`cgx_live_` prefix) for programmatic and CI access
* **api:** setup wizard API for interactive configuration via HTTP
* **api:** live log streaming via WebSocket at `ws://host/ws/v1/logs` (admin only)
* **config:** `Config.resolve_llm_config()` and `resolve_llm_config_for(alias)` — new primary LLM resolution methods returning `(ProviderConfig, ModelConfig)` tuples
* **providers:** `create_chat_model_from_configs(provider_config, model_config)` — new dual-config LLM factory replacing the old single-config path
* **config:** `_parse_providers_section()` auto-migrates model fields from provider entries to the models registry for backward compatibility
* **whatsapp:** track locally-archived chats in `_locally_archived` set to prevent re-processing after WhatsApp auto-unarchives (BUG-113)

### Bug Fixes

* **campaign:** `Campaign.from_dict` no longer mutates the caller's dict via `pop()` — uses `get()` instead (BUG-221)
* **campaign:** `on_reply` releases the lock before calling `save()` to avoid blocking other threads during disk I/O (BUG-222)
* **campaign:** `_process_follow_ups` re-checks `target.status` under lock at the escalation branch to prevent racing with concurrent `mark_target_outcome` (BUG-223)
* **campaign:** `launch()` sets target to `"active"` before `handle_outbound` call so `on_reply()` can match replies arriving during the send window (BUG-224)
* **campaign:** `start()` validates handler is wired via `set_handler()` and guards thread creation under the lock to prevent duplicate follow-up threads (BUG-225)
* **api:** campaign CRUD routes (`create`, `update`, `delete`) now wrapped with `asyncio.to_thread` to prevent blocking the event loop during file I/O (BUG-226)
* **api:** `_validate_campaign_id` enforces UUID regex on all campaign path parameters to prevent injection (BUG-227)
* **api:** `status_filter` query param on `GET /campaigns` typed as `CampaignStatus` for Pydantic validation (rejects invalid values with 422)
* **api:** `_resolve_contact` extracted as unified phonebook lookup — `send_outbound` now prefers active channels (matching campaign target resolution behavior)
* **api:** `stop_assistant` route uses `executor.shutdown(wait=True)` to drain in-flight agent turns before `session_mgr.save_all()`, eliminating a race between executor threads and memory persistence (data-loss fix; mirrors `service.py` `_handle_shutdown` behaviour)
* **api:** `WebSocketCallbackHandler` now tracks `tool_call_count` and `_extract_token_counts` in `turn_runner.py` returns it, so the `done` WebSocket message reports actual tool invocations instead of a hardcoded `0`
* **api:** atomic `INSERT…SELECT` for admin role election in `create_with_role_election` eliminates registration race condition
* **api:** per-session `asyncio.Event` in `ApiSessionRegistry._pending` prevents duplicate `warm_session` calls for concurrent requests targeting the same session (TOCTOU fix)
* **api:** DB session threaded through route `Depends(get_db)` into auth helpers, eliminating redundant database connections
* **api:** fix silent token degradation in `get_current_user_optional` — supplied tokens that are expired or invalid now re-raise instead of falling through to anonymous access (P0 security fix)
* **api:** catch `IntegrityError` in registration endpoint for concurrent duplicate username/email submissions
* **api:** bulk `DELETE` for `clear_history` with `keep_last` parameter (performance)
* **api:** protect `_cancel_requested` flag with lock in `ApiConfirmationUI.cancel()` (thread safety)
* **api:** fix `WSLogHandler` crash, `turn_runner` blocking save, and RAG flush/delete without commit
* **api:** path traversal guard in RAG upload, WebSocket close ordering, event loop blocking in turn runner
* **assistant:** resolve BUG-091 through BUG-112 across deferral system and assistant subsystem
* **api:** reset `_cancel_requested = False` at the start of `render_prompt()` so cancellation from a previous turn does not silently deny all future tool confirmations (P0)
* **api:** `_validate_doc_id()` UUID regex guard in RAG endpoints prevents path traversal via document ID parameters
* **api:** `_snapshot_sessions()`, `_snapshot_scheduler_queue()`, `_snapshot_deferral_records()` helpers copy dicts under lock before iteration, eliminating five race conditions in assistant route handlers
* **api:** `warm_session()` in `session_bridge.py` wraps `_build_memory_manager` and `_build_llm` with `asyncio.to_thread` to prevent blocking the event loop during session warm-up
* **api:** `clear_memory` and `switch_memory_mode` in `memory.py` route blocking `mm.clear()`, `old_mm.save()`, and `new_mm.load()` calls through `asyncio.to_thread`
* **api:** `ConnectionManager.connect()` in `ws.py` releases `_lock` before closing the displaced WebSocket connection to avoid holding the lock across I/O
* **api:** `stop_assistant()` wraps blocking service shutdown calls with `asyncio.to_thread`
* **api:** RAG document list endpoint uses compound `(created_at, id)` keyset cursor for stable pagination ordering; `_doc_to_out` disk I/O (file stat) runs via `asyncio.to_thread` when paginating
* **api:** deleting a session now calls `manager.disconnect(session_id)` to close any orphaned WebSocket connection before archiving the record
* **api:** `get_chat_messages` correctly derives message count by calling `get_messages()` when available (was always returning 0)
* **api:** fix agent amnesia in `turn_runner._build_history()` — `prepare_context()` was called with no arguments (silently raising `TypeError`) and its return value was accessed via `.get()` on a dataclass (silently raising `AttributeError`), causing every turn to start with empty history; fixed by forwarding `user_input` and accessing `.messages` attribute directly (P0)
* **api:** `asyncio.CancelledError` in `run_message_turn()` was swallowed without re-raising, breaking `asyncio.Task.cancel()` semantics; fixed by adding `raise` after cleanup (P0)
* **api:** `get_or_warm()` in `session_bridge.py` now saves the discarded `ApiSession` when a concurrent warmer wins the race, preventing a memory manager resource leak (P1)
* **api:** `memory_manager.update()` and `.save()` in `turn_runner` now run via `asyncio.to_thread` to prevent blocking the event loop on the threading lock and file I/O (P1)
* **api:** `_http_exception_handler` now maps non-dict exception detail to a status-appropriate error code via `_STATUS_CODE_MAP` instead of always returning `code="INTERNAL_ERROR"` for 4xx responses (P1)
* **api:** `check_provider_health` in `routes/config.py` now runs `create_chat_model_from_configs()` via `asyncio.to_thread` to prevent network I/O stalling the event loop (P1)
* **api:** `reload_config` in `routes/config.py` now runs `Config()` file I/O via `asyncio.to_thread` (P1)
* **api:** fix `MemoryUpdatePayload.tokens_used` schema example — was `"1200"` (string) for an `int` field, producing a malformed OpenAPI schema (P2)
* **api:** fix fake-lock data races in `assistant.py` routes — `remove_from_blacklist`, `list_knowledge`, `search_knowledge`, and `delete_fact` created anonymous `threading.Lock()` as fallback instead of using the actual object lock, providing zero mutual exclusion against concurrent readers or writers; all four routes now acquire the real `violation_tracker._lock` or `knowledge_store._lock` (P0)
* **api:** `delete_fact` in `assistant.py` contained dead unreachable code after `raise HTTPException` refactor — removed (P1)
* **api:** `start_assistant` in `assistant.py` now creates the LLM via `asyncio.to_thread(create_chat_model_from_configs, ...)` instead of calling it synchronously in the async handler, preventing event loop blocking on provider initialization (P1)
* **api:** `patch_session` in `sessions.py` now calls `_build_llm` via `asyncio.to_thread` when provider or model changes, preventing event loop blocking on LLM initialization (P1)
* **api:** `_config_to_out` in `config.py` refactored — new `_read_raw_yaml()` async helper offloads config file I/O to a thread pool via `asyncio.to_thread`; `_config_to_out` signature changed from `is_admin: bool` to `raw_yaml: str | None` so the async I/O happens in the caller (P1)
* **api:** `run_message_turn()` now updates `session.last_activity` at the end of each successful turn so the 30-minute idle eviction TTL resets correctly — previously a long-running agent turn would age out the session mid-execution, causing the next request to re-warm from DB (BUG-120)
* **api:** `run_message_turn()` now fully implements `mode='think'` and `mode='delegate'` — think mode wires `classify_think_task` → optional research delegate → `force_deep_think`; delegate mode wires `force_delegation` with parallel sub-agent execution; all blocking LLM calls run via `asyncio.to_thread`; new `agent_state` values (`analyzing`, `deep_thinking`, `researching`, `delegating`) stream progress to the frontend (BUG-122)
* **assistant:** removed dead conditional guard in `MessageHandler._run_agent()` — the `if defer_state/suppress_state` branch and its fallthrough were identical; the guard was a no-op that added confusion without preventing any unintended behaviour (BUG-121)
* **security:** remove `copy` from `SAFE_MODULES` in `python_exec.py` — `copy.deepcopy` invokes `__reduce_ex__` via C code, bypassing sandbox attribute guards (SEC-01)
* **security:** cap unbounded `.*` in 2 guardrails injection patterns to `.{0,200}` to prevent ReDoS on attacker-controlled assistant input (SEC-02)
* **security:** cap unbounded `.*?` in 3 `DEEP_THINK_TRIGGERS` patterns and 2 `DELEGATION_TRIGGERS` patterns to bounded `.{0,80}?` / `.{3,80}?` to prevent ReDoS (SEC-03/04)
* **security:** `resolve_data_path()` now returns the resolved absolute path, closing a TOCTOU window for symlink attacks between validation and file open (SEC-05)
* **shell:** switch from `subprocess.run` to `Popen` with `start_new_session=True` + `os.killpg()` on timeout — kills the entire process group instead of just the direct `/bin/sh` child, preventing orphaned grandchild processes
* **delegate:** fix `_validate_json_response` fence stripping — now finds the matching closing fence after the opening `` ```json `` instead of unconditionally stripping the last triple-backtick, which corrupted responses containing additional code blocks
* **delegate:** change `circuit_breaker.check_availability()` to `circuit_breaker._check_availability_locked()` inside `with _circuit_breaker_lock:` block to fix redundant reentrant lock acquisition
* **calculator:** tighten `_safe_pow` guard from `exp >= 10_000` to `abs(exp) >= 1_000` — prevents `9999**9999` (39K-digit computation) and catches negative exponents
* **graph:** parallel tool `future.result()` now has a 10-minute timeout — on timeout an error `ToolMessage` is produced instead of hanging indefinitely (BUG-202)
* **graph:** `_detect_tool_request` normalizes bare string args to single-element lists so `{"add": "web_search"}` works the same as `{"add": ["web_search"]}` (BUG-204)
* **graph:** auto-expansion dedup cache key now uses the resolved (post-fuzzy-match) tool name for correct cross-name deduplication
* **api:** `run_message_turn()` calls `reset_for_new_prompt()` at turn start to clear ephemeral tools and `deny_all`, matching CLI prompt boundaries (BUG-198)
* **api:** `warm_session()` populates `session_state.all_tool_originals` from tool registry so unload/disable can restore canonical tool objects (BUG-199)
* **api:** `patch_session_tools` acquires `turn_lock` around all `run_config` mutations to prevent races with in-flight agent turns (BUG-196)
* **api:** `_classify_tool_status` only reports `"auto_approved"` when the tool is also in `loaded_tools` — an approval on an on-demand tool does not imply it is active
* **cli:** `/tools disable` now removes from both `pinned_tools` and `loaded_tools` in addition to adding to `denials` (BUG-197)
* **rag:** `_has_faiss_index()` validates actual FAISS index files exist before adding directories to the search list (BUG-200)
* **rag:** `_collect_faiss_dirs()` applies containment check on resolved `idx` path to prevent symlink traversal via intermediate components (BUG-191)
* **rag:** multi-index search uses `similarity_search_with_score` with score-based sorting for cross-index relevance ranking (BUG-193)
* **delegate:** `future.cancel()` now called in the `remaining <= 0` timeout branch to prevent leaked LLM threads (BUG-195)
* **compression:** warning log emitted on compression LLM timeout before truncation fallback
* **compression:** `as_completed()` now receives a pool-level timeout so hung LLM calls trigger truncation fallback instead of blocking the agent turn indefinitely (BUG-207)
* **api:** error messages in `run_message_turn()` use `put_nowait` (not blocking `await put()`) and the `done` message uses `asyncio.wait_for(put(), timeout=5.0)` to prevent deadlock on bounded queue in REST-only sessions (BUG-209)
* **rag:** `knowledge_base_stats()` and `_build_description()` wrap `iterdir()`/`stat()` calls with `OSError` handling to survive permission errors and TOCTOU races on FAISS directories (BUG-211)
* **tools:** `configure_rag_tool()` catches `(ImportError, OSError)` instead of just `ImportError` so a broken FAISS directory does not crash startup (BUG-211)
* **cleanup:** remove dead `_has_phantom_tool_call` alias and unused import from `cogtrix.py` (BUG-206)
* **api:** all workflow API `{workflow_id}` path parameters validated against `^[a-zA-Z0-9][a-zA-Z0-9_-]*$` regex at the route boundary before reaching filesystem operations (BUG-212)
* **api:** `update_workflow` route uses `dataclasses.replace()` to build a copy instead of mutating the live registry object, preventing concurrent readers from seeing partially-updated state (BUG-213)
* **assistant:** `_load_prompt_from_value` now resolves relative paths (e.g. `./`, `../`) against `data_dir` with `is_relative_to` containment, closing a path traversal bypass (BUG-214)
* **api:** `upload_workflow_document` validates resolved file path stays inside `data_dir` via `is_relative_to` before writing (BUG-216)
* **api:** `on_llm_new_token` sets `final=True` only when `tool_call_count > 0 AND len(_tool_starts) == 0`, preventing premature final-response marking during intermediate tool reasoning (BUG-218)
* **compression:** inner `future.result()` now has `timeout=120` so the per-future `TimeoutError` handler is reachable (was dead code without a timeout) (BUG-219)
* **assistant:** `_auto_detect` returns the highest-scoring workflow that meets `min_confidence`, not the first alphabetical match (BUG-220)
* **api:** `update_workflow` route returns `_wf_to_out(updated)` instead of the stale pre-update object

### Features

* **assistant:** workflow system — `WorkflowRegistry` loads YAML workflow definitions from `data/workflows/<id>/workflow.yaml`; each workflow bundles a system prompt, per-workflow FAISS knowledge base, and tool policy; chat-to-workflow bindings persisted in `data/workflows/bindings.json`; resolution order: explicit binding → contact_prompts fallback → auto-detect (keyword/regex scoring) → global default; API CRUD at `/api/v1/assistant/workflows/` (11 endpoints)
* **api:** user management — 4 admin-only endpoints: list all users, create user, update role, delete user; `UserRepository` extended with `list_all()`, `update_role()`, `delete()` methods
* **api:** `TokenPayload.final` boolean field distinguishes preamble tokens (before tool calls) from the final response after all tools complete
* **api:** `SessionCreateRequest.name` auto-generates `"Session YYYY-MM-DD HH:MM"` via `default_factory`
* **api:** `ConfigOut` now includes `system_prompt` and `guardrails` fields for WebUI consumption
* **api:** `_run_think_pipeline` checks `session.cancel_event.is_set()` between pipeline phases (classify → research → deep_think) to avoid proceeding to expensive phases after cancel
* **config:** auto-migrated model aliases use `"{provider}/{model}"` format (e.g. `"openai/gpt-4.1-mini"`) instead of bare provider name

### Performance

* **compression:** convert eager `_COMPRESSION_POOL` (4 threads spawned at import) to lazy `_get_compression_pool()` with double-checked locking — threads only created when compression actually runs (PERF-01)
* **runner:** fix `ToolCallLogger._evict_stale` calling `time.monotonic()` twice — reuses `now` parameter for cutoff calculation, eliminating a redundant syscall (PERF-02)
* **intent:** hoist `_cat_by_name` dict comprehension to module-level `_THINK_CAT_BY_NAME` — avoids rebuilding a 23-entry dict on every `classify_think_task()` call (PERF-03)

### Build

* add `api` optional dependency group to `pyproject.toml` (FastAPI, uvicorn, SQLAlchemy async, aiosqlite, alembic, python-jose, passlib)
* update `Dockerfile` for API mode support (uvicorn, alembic migrations at startup)
* update `docker-entrypoint.sh` with `api` / `--api` mode that runs `alembic upgrade head` then starts uvicorn

## [0.1.19](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.18...v0.1.19) (2026-03-05)


### Features

* **api:** assistant auto-start on API server boot ([838dfd6](https://github.com/NorthlandPositronics/Cogtrix/commit/838dfd6b11ac4f3e523cb0f4e4cd9764c18f0671))
* **api:** assistant auto-start on API server boot ([5c361da](https://github.com/NorthlandPositronics/Cogtrix/commit/5c361dab74d297b8d029c55fed6443c5c8aaa73c))


### Bug Fixes

* BUG-118 protect compress_tool_message against prompt injection ([93a7c9c](https://github.com/NorthlandPositronics/Cogtrix/commit/93a7c9c4159f61eb2abbfdfbcd27d624ee3cf11a))
* move Pydantic serialisation outside asyncio.Lock in ConnectionManager.send ([df78f2e](https://github.com/NorthlandPositronics/Cogtrix/commit/df78f2eb04356fa34c3571053a2825e0fc0fe59a))
* ProjectForge audit sprint 1 — 9 API bugs + 1 security fix ([5f6c5f6](https://github.com/NorthlandPositronics/Cogtrix/commit/5f6c5f6e26ef5691a3c28303bc90ea7e04d56282))
* resolve BUG-115 and BUG-117 in API turn runner concurrency ([36f1d35](https://github.com/NorthlandPositronics/Cogtrix/commit/36f1d35a1e359c56b0a68bf65f2928997eb78f0b))
* resolve BUG-116 — ApiConfirmationUI.render_prompt unblocks displaced callers ([e0ef573](https://github.com/NorthlandPositronics/Cogtrix/commit/e0ef57319d69e92024dc2f661d6c7ce28961d50c))
* resolve BUG-120 and BUG-126 in api/routes/sessions.py ([547ba15](https://github.com/NorthlandPositronics/Cogtrix/commit/547ba1535f49d5dbccd2a1649b60e54219a0c085))
* resolve BUG-122, BUG-113, and PERF-03 in session_bridge ([8080f65](https://github.com/NorthlandPositronics/Cogtrix/commit/8080f65347d935e0563f781c4ea1c8edd84e987e))

## [0.1.18](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.17...v0.1.18) (2026-03-05)


### Documentation

* holistic documentation audit — fix tool count, accuracy, and completeness ([c106408](https://github.com/NorthlandPositronics/Cogtrix/commit/c106408c4df4bf62cdbb100403ac824eed2f8390))

## [0.1.17](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.16...v0.1.17) (2026-03-05)


### Bug Fixes

* add missing `api` optional-dependency group to pyproject.toml ([a1a3af1](https://github.com/NorthlandPositronics/Cogtrix/commit/a1a3af1f43591582bb73bba11d262b36539df047))

## [0.1.16](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.15...v0.1.16) (2026-03-05)


### Features

* **api:** REST + WebSocket API layer ([d711932](https://github.com/NorthlandPositronics/Cogtrix/commit/d711932280889996d198f3f26a40430cb06b7d39))
* **api:** REST + WebSocket API layer with JWT auth, session management, and streaming agent turns ([7e6e4c2](https://github.com/NorthlandPositronics/Cogtrix/commit/7e6e4c2b083879739d7341d392bef930d229f29d))


### Bug Fixes

* exclude src/api from pyright — optional deps not in CI base install ([1927383](https://github.com/NorthlandPositronics/Cogtrix/commit/19273830863ef469b57adaf38e395dbd3c8a20d6))
* resolve CI failures — ruff B008/UP046 ignores, bandit B108, test import guards ([f4d6a5c](https://github.com/NorthlandPositronics/Cogtrix/commit/f4d6a5cc6477dde99f02041da4a79ee793328b79))
* skip API test files gracefully when fastapi is not installed ([05e8949](https://github.com/NorthlandPositronics/Cogtrix/commit/05e894912fdf52d9fa04f55dc2176ac16f9427c1))

## [0.1.15](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.14...v0.1.15) (2026-03-04)


### Bug Fixes

* resolve BUG-091 through BUG-099 in deferral system and adjacent files ([2406222](https://github.com/NorthlandPositronics/Cogtrix/commit/2406222e769c54b86ac1ec125def172f6d1b7ef1))
* resolve BUG-100 through BUG-104 in assistant subsystem ([efb9fd6](https://github.com/NorthlandPositronics/Cogtrix/commit/efb9fd6dbdb8d1c5c1a154fbec46a1f3d9ed2560))
* resolve BUG-105 through BUG-108 and BUG-094 partial fix ([8b69c61](https://github.com/NorthlandPositronics/Cogtrix/commit/8b69c617dcfd880479156af17111c50b4b5c6aad))
* resolve BUG-109 through BUG-112 and skip recovery on defer/suppress ([aca5f07](https://github.com/NorthlandPositronics/Cogtrix/commit/aca5f07049b78d59493245a02d9ca5aa48cc87c7))

## [0.1.14](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.13...v0.1.14) (2026-03-03)


### Features

* add defer_processing and suppress_reply tools for deferred message reasoning ([d910c1b](https://github.com/NorthlandPositronics/Cogtrix/commit/d910c1b006a8e24d09edc770a936f284d0d15e53))
* **assistant:** add queue_reply tool for sequential message delivery ([72e7267](https://github.com/NorthlandPositronics/Cogtrix/commit/72e72671bc3d427621009573920d324b0391a083))


### Bug Fixes

* apply round-6 audit sprint B fixes (BUG-083, BUG-084, PERF-1004, ARCH-040-10, ARCH-040-05, ARCH-040-04, ARCH-040-12, ARCH-040-06, ARCH-040-09, PERF-1007) ([0b4e9c2](https://github.com/NorthlandPositronics/Cogtrix/commit/0b4e9c2552df69ab35dbb1e122debdd9561ef7ea))
* **assistant:** apply queue_reply audit fixes (BUG-079, BUG-080, BUG-081, H1, H2) ([09d7ecf](https://github.com/NorthlandPositronics/Cogtrix/commit/09d7ecf29e8dfc497d04db668ea8ccdf4eba3c2b))
* **assistant:** apply round-6 sprint-A audit fixes ([75cb8b1](https://github.com/NorthlandPositronics/Cogtrix/commit/75cb8b1370f324e724272aa9cb7f1a571d0a0c7e))
* extract atomic_write_json utility and fix fd leaks (BUG-030, BUG-062, BUG-075) ([98b4595](https://github.com/NorthlandPositronics/Cogtrix/commit/98b4595691f6327d2aa273d7c62cdf28db4b6f1f))
* **guardrails:** remove broken ViolationTracker save debounce ([e382969](https://github.com/NorthlandPositronics/Cogtrix/commit/e38296969bdf8fde3efb39d6997e0ebe3ce448da))
* round-6 holistic audit — 18 findings across 14 files ([095e86a](https://github.com/NorthlandPositronics/Cogtrix/commit/095e86aceb12e1f971b36b9e1f87280c92888396))
* Sprint 1 — critical safety and correctness fixes (round 7) ([81e77fe](https://github.com/NorthlandPositronics/Cogtrix/commit/81e77feb309709f715aa96fb2bc0545f50bf20dd))
* sprint 3 — wizard template, SSRF guard, circuit breaker lock, MCP TOCTOU, LRU merge ([f861757](https://github.com/NorthlandPositronics/Cogtrix/commit/f8617570ab6926a74205ea2399863d04ab677ff3))
* sprint 4 — flush_all timer race, module-level thread pools for compression and tool execution ([c092e27](https://github.com/NorthlandPositronics/Cogtrix/commit/c092e27d41df0154b29406f23e3d0a1112d862c7))


### Documentation

* update CLAUDE.md for round-7 bug fixes across all 4 sprints ([7551967](https://github.com/NorthlandPositronics/Cogtrix/commit/7551967b0fac881f47122431313a7ff525768a8b))

## [0.1.13](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.12...v0.1.13) (2026-03-03)


### Bug Fixes

* apply Round 5 audit fixes (BUG-078/079/080/082, ARCH-039, PERF-901/902/907) ([2adf625](https://github.com/NorthlandPositronics/Cogtrix/commit/2adf625b65dbfe62a7dc2ccfd5b19bc7893af625))
* merge channel-specific config into WhatsApp/Telegram channel constructors ([840791a](https://github.com/NorthlandPositronics/Cogtrix/commit/840791af00dcd0bcbba34a74c0b907b778e82684))
* resolve 9 bugs and performance issues (BUG-074/075/076/077, PERF-802/804/806, ARCH-037-07/11) ([6291a65](https://github.com/NorthlandPositronics/Cogtrix/commit/6291a65d1aa44a245da3d36836ffcadca4dc7043))
* WAHA client-side filtering, factorial cap, rag atomic swap, and CI workflow ([712007c](https://github.com/NorthlandPositronics/Cogtrix/commit/712007c7411b449244c617a9e50aae9ad9534582))

## [0.1.12](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.11...v0.1.12) (2026-03-03)


### Features

* **assistant:** add filter_mode renames and blacklist delete/archive ([2eeae99](https://github.com/NorthlandPositronics/Cogtrix/commit/2eeae993c3fbca4298f83699e56d216c54846fc8))
* **assistant:** add message debounce buffer, edit_last_reply tool, and batch handling ([4f77f30](https://github.com/NorthlandPositronics/Cogtrix/commit/4f77f30fffe2fbf9b19c7668964ce6770d4bee25))
* **assistant:** add message editing, queue management tools, and bug fixes ([d571508](https://github.com/NorthlandPositronics/Cogtrix/commit/d571508588fd0209f94c48a744d1a1a3a45f57a6))
* **assistant:** add scheduler queue management tools and recipient tracking ([ef4a740](https://github.com/NorthlandPositronics/Cogtrix/commit/ef4a74090e9f5a1643e0569c4550ab47c7446469))
* **scheduler:** add chat_id and contact_name filters to list_scheduled_messages ([d27b9df](https://github.com/NorthlandPositronics/Cogtrix/commit/d27b9df80a1d35d4395ec5ce0ffb20e00aebea05))
* **whatsapp:** add two-phase polling tests and fix snapshot eviction order ([df3e588](https://github.com/NorthlandPositronics/Cogtrix/commit/df3e588b190bab6cebac8c7a797ddbeea1403c79))
* **whatsapp:** implement two-phase polling architecture ([a34422a](https://github.com/NorthlandPositronics/Cogtrix/commit/a34422a5d59a8d3796ead418d0c140e728b715c0))


### Bug Fixes

* apply Round 3 audit fixes (BUG-068/069/071/072, ARCH-035-01/03/13) ([16dcba1](https://github.com/NorthlandPositronics/Cogtrix/commit/16dcba10a36b4aea39472aaeaaed501f21f56c70))
* **assistant:** fix 5 polling and duration bugs (BUG-055 through BUG-059) ([552e0a3](https://github.com/NorthlandPositronics/Cogtrix/commit/552e0a30f3a26001cb2057f5f34c54dfe36daadb))
* **assistant:** remove early return in _route_response so edit+schedule both fire ([0a7b1ea](https://github.com/NorthlandPositronics/Cogtrix/commit/0a7b1ead78460d3ee74d13e4dfcfd0d96991dacb))
* close leaked file descriptors and fix TOCTOU race in MCP loop creation ([cf775b3](https://github.com/NorthlandPositronics/Cogtrix/commit/cf775b3a9da21b111be55e67c2cfe202ba2368fc))
* implement ADR-0036 deferred audit fixes (6 items) ([3348baf](https://github.com/NorthlandPositronics/Cogtrix/commit/3348bafa40744b14ae41d5fad5f2886670e3da04))
* **scheduler:** add lock+idempotency to tool closures and coerce from_dict types ([254d411](https://github.com/NorthlandPositronics/Cogtrix/commit/254d4119725eb29f972ce182ff552d2d32ce1b28))
* **whatsapp:** fix 6 polling bugs (BUG-049 through BUG-054) ([893be0b](https://github.com/NorthlandPositronics/Cogtrix/commit/893be0be2ce560612405416c93ef05dadfaa04d3))
* **whatsapp:** rewrite polling to two-phase architecture, fix stale dedup cache ([df2acab](https://github.com/NorthlandPositronics/Cogtrix/commit/df2acab9d3f0915043e88bd30e34c608903bd1c4))


### Documentation

* holistic documentation revision syncing all docs with codebase ([e70576d](https://github.com/NorthlandPositronics/Cogtrix/commit/e70576da66ddb8e34abb55ce8108baf496d39460))

## [0.1.11](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.10...v0.1.11) (2026-03-02)


### Bug Fixes

* cache off-by-one and fd leaks in bound-cache, guardrails, knowledge, and json_store ([191c86f](https://github.com/NorthlandPositronics/Cogtrix/commit/191c86fd9853167a93e7c586c3ab943409ca8e90))
* deep-copy active_tools_list in run_execution_phase and cap compression fallback at _FALLBACK_MAX_CHARS ([e89e501](https://github.com/NorthlandPositronics/Cogtrix/commit/e89e5016b8a03980f0e2c9b3738b380c10f0ccb1))
* eliminate orphaned tool-call chains, thread-safe slow-path counter, and LRU cache writeback ordering ([f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c005c91259aff3a5e9a5a213b109d9a3ea8))
* Round 8 bug fixes, documentation revision, and assistant refactor ([38c7a2b](https://github.com/NorthlandPositronics/Cogtrix/commit/38c7a2b8137de6cd30d093b28c02cf6331f57c74))


### Documentation

* holistic documentation revision — fix tool count, memory window, changelog gaps ([dc83c57](https://github.com/NorthlandPositronics/Cogtrix/commit/dc83c57ec812d67be696aa9dc3e3fe81778affa4))

## [0.1.10](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.9...v0.1.10) (2026-03-02)


### Features

* **assistant:** add per-contact system prompts ([6c9d1c7](https://github.com/NorthlandPositronics/Cogtrix/commit/6c9d1c79bfcd0854fda434d5771b071a1db5989b))


### Bug Fixes

* **assistant:** contact prompt replaces system prompt, fix save_all/cleanup bugs, update docs ([6f68eb5](https://github.com/NorthlandPositronics/Cogtrix/commit/6f68eb59c84b58a914c8788d6bcd34f1a4928682))
* **assistant:** dynamic scheduler dispatch, crash-safe knowledge save, fix ViolationTracker._save lock ([992b6d8](https://github.com/NorthlandPositronics/Cogtrix/commit/992b6d8c365e837f266137c7074c01d9e432eb6a))
* **assistant:** fix [@lid](https://github.com/lid) contact prompt matching, persist stale expiry, respect excluded_tools ([791e703](https://github.com/NorthlandPositronics/Cogtrix/commit/791e7031f3c4e82116cb8a265aa1ed1421d7afae))
* **assistant:** fix four datamarking and PII bugs in handler.py ([8f1d259](https://github.com/NorthlandPositronics/Cogtrix/commit/8f1d25949a56ed7fd3e077182910bb03d27ec1e0))
* **assistant:** fix Telegram update replay, contact prompt spoofing, and rate limiter bypass ([b4bcb77](https://github.com/NorthlandPositronics/Cogtrix/commit/b4bcb77048be476535ece49002d582da99d9ada9))
* **assistant:** multi-round audit bug fixes and optimizations ([66bf868](https://github.com/NorthlandPositronics/Cogtrix/commit/66bf86816406ec6e5696792ee4ffd01136c09131))
* **assistant:** resolve remaining audit bugs and add scheduled reply prompt ([dab52e0](https://github.com/NorthlandPositronics/Cogtrix/commit/dab52e0c9da587fc2e193fb403fec6f34b0f7cae))
* correct documentation inaccuracies across config, README, and docs ([8bbb241](https://github.com/NorthlandPositronics/Cogtrix/commit/8bbb241a3af18af57cc2565723e766aedbf304f2))
* **scheduler:** recover in-flight messages on restart and add architectural review ([940cfb1](https://github.com/NorthlandPositronics/Cogtrix/commit/940cfb1d28fbec652ae3069c1177a7ca89504a74))
* **whatsapp:** add session start, LID resolution, and chats overview to client ([c54e8d6](https://github.com/NorthlandPositronics/Cogtrix/commit/c54e8d6d6f3b4480095657627fd419173d3da304))
* **whatsapp:** fix poll() chatId bug and add [@lid](https://github.com/lid) sender support ([a390e1c](https://github.com/NorthlandPositronics/Cogtrix/commit/a390e1cd5da01c54e8aecd8da61aa4b086b91b78))
* **orchestration:** fix active_tools_list mutation, compression fallback cap, LRU writeback ordering, and cache off-by-one (BUG-031..035) ([e89e501](https://github.com/NorthlandPositronics/Cogtrix/commit/e89e501), [f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c0))
* **memory:** fix orphaned tool-chain cleanup, thread-safe slow-path counter, and fd leaks in json_store, guardrails, and knowledge (BUG-033,036,038..040) ([191c86f](https://github.com/NorthlandPositronics/Cogtrix/commit/191c86f))


### Performance Improvements

* **compression:** raise context compression threshold from 0.50 to 0.72 (PERF-001) ([f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c0))


### Documentation

* document contact_prompts, schedule_reply, and response_timing in CONFIGURATION.md ([a24e762](https://github.com/NorthlandPositronics/Cogtrix/commit/a24e7622987d49c82b9d9b6725e22817f863cd79))
* document datamarking defense and scheduler recovery ([bca6437](https://github.com/NorthlandPositronics/Cogtrix/commit/bca6437594f3215f5ca5d86da83fe7d25344eff8))

## [0.1.9](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.8...v0.1.9) (2026-03-01)


### Features

* add user-constraint trust rule and milestone focus guidance to system prompt ([468e376](https://github.com/NorthlandPositronics/Cogtrix/commit/468e376b4c2110b7da93b066793237913aadaf2c))


### Bug Fixes

* add thread safety to ToolCallLogger, deduplicate tool-call key computation, and synchronize deep_think progress callback ([d54bfbd](https://github.com/NorthlandPositronics/Cogtrix/commit/d54bfbd1d6525d536c595b0b61c52151276ecbf7))
* address 5 low-severity bugs and perf issues (BUG-1829, BUG-1848, PERF-1101/1102/1103) ([3e4a583](https://github.com/NorthlandPositronics/Cogtrix/commit/3e4a583907121ddbfd756519fad31d67abf6a2a4))
* address three concurrent-safety and injection bugs ([24b0bc6](https://github.com/NorthlandPositronics/Cogtrix/commit/24b0bc643255c015b1b02d67c3e93d166f67dd6a))
* comprehensive security, thread-safety, and correctness audit (56 bugs fixed) ([7d6ad16](https://github.com/NorthlandPositronics/Cogtrix/commit/7d6ad167dac6d5f48113adac091b24b85e90ba7e))
* correct misleading log message, add CLI mutual-exclusion guard, and add hasattr guard in _TokenAccumulator ([1694041](https://github.com/NorthlandPositronics/Cogtrix/commit/16940416f7f7be764c33761169c11f420e1cbd30))
* factorial DoS cap, JSON dot-path guard, exception types, session_state wiring ([b4e3c80](https://github.com/NorthlandPositronics/Cogtrix/commit/b4e3c805ebe39664dc151b6a13d00e58121562b3))
* guard against four High-severity bugs (BUG-1837..1840) ([40c2319](https://github.com/NorthlandPositronics/Cogtrix/commit/40c23197387cead25ca028f8651517ea568f6fac))
* handle plain host:badport in _parse_ollama_address and early tmp_path assignment in setup wizard ([f3a65a2](https://github.com/NorthlandPositronics/Cogtrix/commit/f3a65a23b03e7e2d9cb250ba010c41ab699ec27c))
* MCP unsupported type warning, close_all iteration safety, handler approvals copy, guardrails test ([dbfa1e2](https://github.com/NorthlandPositronics/Cogtrix/commit/dbfa1e202c1837f4155c279356c441434fe045bc))
* **memory:** persist and restore mode-specific state across session restarts ([b5c07a3](https://github.com/NorthlandPositronics/Cogtrix/commit/b5c07a346390525a557ab45b813af20872cd2b0d))
* patch three confirmed bugs — SSRF in delegate URL re-fetch, intent false positives, and secret masking ([325cf99](https://github.com/NorthlandPositronics/Cogtrix/commit/325cf99397b9c2e618049e558398d44c6ae59541))
* path traversal guard in resolve_data_path and SSRF header blocking ([75999aa](https://github.com/NorthlandPositronics/Cogtrix/commit/75999aaf3c20d367c68e5d5518c5394fa7229e83))
* prevent ANSI corruption on non-TTY, fix spinner TOCTOU, and harden inline shell ([3cbd464](https://github.com/NorthlandPositronics/Cogtrix/commit/3cbd46423a1f49fae1667d59a5b76ca9ea26f6cf))
* resolve 4 medium-severity bugs (BUG-1852, BUG-1853, SEC-0802) ([9ac9b79](https://github.com/NorthlandPositronics/Cogtrix/commit/9ac9b79e7dc63a5d78adaa21fdee2ae4783147d9))
* resolve five medium-severity bugs (BUG-1828, BUG-1842, BUG-1844, BUG-1845, PERF-1100) ([1a453c9](https://github.com/NorthlandPositronics/Cogtrix/commit/1a453c9bb604e8598815763d20a8dc5f96867876))
* setup live-reload missing state rebuilds and ProviderConfig in-place mutation ([4cca9e4](https://github.com/NorthlandPositronics/Cogtrix/commit/4cca9e41c406cb2f9c98be86374218b24506b07a))
* warn on unsupported base_url in Google provider, atomic config swaps, broaden json_store exception handling ([34ffcb0](https://github.com/NorthlandPositronics/Cogtrix/commit/34ffcb051d6f3343119a8cbd311dce44a04d4956))


### Performance Improvements

* move hot-closure imports and regex literals to module scope ([d8a71f2](https://github.com/NorthlandPositronics/Cogtrix/commit/d8a71f211f1a6a432cdc4bb5dc59b2ba030385d0))


### Documentation

* update CLAUDE.md, AGENTS.md, README.md, and architecture docs for Rounds 6-10 ([86443a8](https://github.com/NorthlandPositronics/Cogtrix/commit/86443a80abc2f8d7fbe671073e6c1ffb948c625d))

## [0.1.8](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.7...v0.1.8) (2026-02-28)


### Features

* add top-level data_dir config option ([3cd3907](https://github.com/NorthlandPositronics/Cogtrix/commit/3cd3907d2427f13669d36e83d70d41b4b1be70c6))
* add top-level data_dir config option for all data storage paths ([ae13eb3](https://github.com/NorthlandPositronics/Cogtrix/commit/ae13eb308438f09274f4152d42bcd2bc85ba681e))
* **docker:** optimize Dockerfile with selective COPY and slim .dockerignore ([1c7311e](https://github.com/NorthlandPositronics/Cogtrix/commit/1c7311e81803409ef3586321d117f61bf2a6a420))
* **milestone:** add spinner context prefix and report_progress tool ([957066a](https://github.com/NorthlandPositronics/Cogtrix/commit/957066a60fa1e372cc315c491660042d066428e7))
* **milestone:** wire progress tracking into cogtrix.py (step 5) ([850bfb6](https://github.com/NorthlandPositronics/Cogtrix/commit/850bfb66dd8e3beaa50f3098db1d783c0c9ff6a5))
* **optimizer:** add Milestone/PromptPlan types and plan_milestones param ([79707f6](https://github.com/NorthlandPositronics/Cogtrix/commit/79707f698810cad0f35ee63a9771b7db44fde437))


### Bug Fixes

* resolve Round 26 audit bugs and performance issues ([2f8f6af](https://github.com/NorthlandPositronics/Cogtrix/commit/2f8f6af200b419a89bcf163d6d5a8a84500e61c7))


### Documentation

* add Round 26 audit reports (bugs, performance, architecture) ([c2c8a67](https://github.com/NorthlandPositronics/Cogtrix/commit/c2c8a67ae433b10ce6f118e93d63b7936818938c))

## [0.1.7](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.6...v0.1.7) (2026-02-28)


### Features

* add --allow-write-path flag to permit writes outside cwd ([dcd64ea](https://github.com/NorthlandPositronics/Cogtrix/commit/dcd64ea252e1ebbd83b31a7972dcd5af8f9a85e7))
* Rounds 19-25 — parallel execution, security hardening, and performance optimizations ([1510cb9](https://github.com/NorthlandPositronics/Cogtrix/commit/1510cb9fe488af7cbe3032d3aab027ae4b461b57))


### Bug Fixes

* **docker:** add missing extras for Anthropic, Google, MCP, and science ([4d6d33d](https://github.com/NorthlandPositronics/Cogtrix/commit/4d6d33df31a120733bed5f074d81c27c8635ba12))
* **file_ops:** allow read access to app install directory outside cwd ([9f362b4](https://github.com/NorthlandPositronics/Cogtrix/commit/9f362b4c558527bfdd0606e667a4c7b1b53cd57e))
* handle string allowed_write_paths, cap bound cache, copy config.available_tools ([aedab52](https://github.com/NorthlandPositronics/Cogtrix/commit/aedab5214730d24c6f5bddbc6ccc8b634f12a6aa))


### Performance Improvements

* connect MCP servers concurrently in connect_all() ([415ac13](https://github.com/NorthlandPositronics/Cogtrix/commit/415ac1362f16fe438a81bc5dfbc5de7437d9cd15))
* single-pass token estimation and parallel compression LLM calls ([1182068](https://github.com/NorthlandPositronics/Cogtrix/commit/1182068c77c1845007b35f3c020444676eb67a41))


### Documentation

* update documentation for Round 25 bug fixes and performance improvements ([66377ad](https://github.com/NorthlandPositronics/Cogtrix/commit/66377ade33372f78dbb696e9b2f22afb55b771f1))

## [0.1.6](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.5...v0.1.6) (2026-02-28)


### Features

* add optimizer feedback message and parallel tool execution ADR ([ff1986c](https://github.com/NorthlandPositronics/Cogtrix/commit/ff1986c6e349e10176d016e119533f71beb10061))
* implement parallel tool execution in process_tools node ([d016206](https://github.com/NorthlandPositronics/Cogtrix/commit/d016206610fe82e69110d4b48fed9c2a5c4b596e))


### Bug Fixes

* address three medium-severity bugs (BUG-1402, BUG-1403, BUG-1404) ([a9c97a9](https://github.com/NorthlandPositronics/Cogtrix/commit/a9c97a97efa58b0f755d5e183946983393e6fd6f))
* address three security/correctness bugs in runner and python_exec ([c0405fb](https://github.com/NorthlandPositronics/Cogtrix/commit/c0405fb6813e9454b1a191339fc6ca7d49bc55a0))
* break parallel futures loop on cancel and snapshot dict before iteration ([75e3016](https://github.com/NorthlandPositronics/Cogtrix/commit/75e3016d21459e90124b4f29b0ce6bf598b8a81c))
* bump version to 0.1.5 and fix release-please extra-files path ([8f74fa5](https://github.com/NorthlandPositronics/Cogtrix/commit/8f74fa57256999e3752541aa67e32330e9bd7a3f))
* close fd race, spinner dirty-terminal, and stderr fd leak ([8039cbd](https://github.com/NorthlandPositronics/Cogtrix/commit/8039cbd6f716c55736ba14aab104f15965a6cbad))
* derive no_confirm from self._session_state when set, falling back to True. ([1b1667e](https://github.com/NorthlandPositronics/Cogtrix/commit/1b1667e54a43293f86545e829382bb78e1c22c79))
* empty API key re-prompt, output cap module resolution, and wizard prompt injection ([9c89f0b](https://github.com/NorthlandPositronics/Cogtrix/commit/9c89f0b7854f474adf4354f54dc4068c449129b1))
* guard parallel block on cancel, fix UserCancelledRun in auto-expansion, pass parallel_tool_execution to MessageHandler, defer tool_list on cache-hit ([8ae422f](https://github.com/NorthlandPositronics/Cogtrix/commit/8ae422f72407ba57e02fc4a2c5c35eb2622f7a53))
* log tracebacks on agent/tool errors and defer old-LLM close until after swap ([574f95a](https://github.com/NorthlandPositronics/Cogtrix/commit/574f95ae3d52f05fd7f551f765c743d7bb57b298))
* **mcp:** upgrade connection cleanup log level and fix inter-server collision detection ([82f29b5](https://github.com/NorthlandPositronics/Cogtrix/commit/82f29b5d3bd3315fdcd131bdb0ea92cc1fb1619e))
* **orchestration:** stop serial-first loop on cancel and guard stale cache merge-back ([88dea96](https://github.com/NorthlandPositronics/Cogtrix/commit/88dea965eeaf35e30a1ad9579ab5668168ef022f))
* resolve 3 HIGH-severity bugs in approve toggle, event loop leak, and spinner race ([ca27ecb](https://github.com/NorthlandPositronics/Cogtrix/commit/ca27ecb4ab533e0a85b1322a0cd4f06e29c0f8a8))
* resolve Pyright type error in graph.py classification pass ([0efc0b5](https://github.com/NorthlandPositronics/Cogtrix/commit/0efc0b5cc0b365d9f1a681f3b06475b36a63f1ae))
* restore provider_config after rollback and respect no_confirm in MessageHandler ([1b1667e](https://github.com/NorthlandPositronics/Cogtrix/commit/1b1667e54a43293f86545e829382bb78e1c22c79))
* Round 18 audit fixes + holistic documentation revision ([f0fcf7a](https://github.com/NorthlandPositronics/Cogtrix/commit/f0fcf7a9704bd213073fd29145b40a8a0ca88951))
* Round 19-20 bug fixes, guardrails cleanup, and audit docs ([8097160](https://github.com/NorthlandPositronics/Cogtrix/commit/8097160e35cecc3e4b11d29e1d48a43c955b86b5))
* round 21 bug fixes — thread safety, cancel propagation, config wiring ([6aedc28](https://github.com/NorthlandPositronics/Cogtrix/commit/6aedc287609cf2eb47d7f4b49f871f048e4c76fd))
* round 22 bug fixes — cache isolation, cancel guards, session locks ([5459140](https://github.com/NorthlandPositronics/Cogtrix/commit/54591401ade342ab49c818db8ac957b20ff21e3f))
* **runner:** eliminate cache race condition in concurrent assistant mode ([0feb507](https://github.com/NorthlandPositronics/Cogtrix/commit/0feb507465c45d346eb30088dd88fad9e41659e4))
* serialize _turn_count/_section_ts in reasoning memory, fix falsy-string KeyError in compression, and bijective session ID sanitization ([80d379a](https://github.com/NorthlandPositronics/Cogtrix/commit/80d379a550aaa292482684b6d7f9bee7586edaf4))


### Performance Improvements

* move function-level imports to module level in runner.py ([dae4cd3](https://github.com/NorthlandPositronics/Cogtrix/commit/dae4cd39d27ce03b6625a0fb66a41b589bb53fb0))
* persist _bound_cache and compression_cache across graph rebuilds ([b799c2f](https://github.com/NorthlandPositronics/Cogtrix/commit/b799c2f06ea4b79fc5137d660d71e8e573582f62))
* run optimize_prompt() concurrently with prepare_context() to reduce TTFT ([817cc9b](https://github.com/NorthlandPositronics/Cogtrix/commit/817cc9b5c495c8e13018ae852c2204cf05ca1e27))


### Documentation

* add ADRs 0015-0022, bug reports rounds 10-18, and update lockfile ([141bb8c](https://github.com/NorthlandPositronics/Cogtrix/commit/141bb8c36941cda26b0c04a7fc70f46ffd6aaf2c))
* add Round 18 bug hunt report ([858edd3](https://github.com/NorthlandPositronics/Cogtrix/commit/858edd30299c4da77f35e8858ae0b124f1836671))
* add Round 23 bug report and audit findings ([7f732f3](https://github.com/NorthlandPositronics/Cogtrix/commit/7f732f3e58dfaa00e70651b4374e654c3a87242c))
* add Round 24 audit reports (bugs, performance, architecture) ([bbcdcca](https://github.com/NorthlandPositronics/Cogtrix/commit/bbcdcca3fec43e1a4375f7434fbb9f397adf59f4))
* holistic documentation revision — fix accuracy drift and fill gaps ([af82852](https://github.com/NorthlandPositronics/Cogtrix/commit/af82852bfd159a1cd93164489eaf96f11f5ce1d0))
* update CLAUDE.md with parallel tool execution architecture ([d7653ec](https://github.com/NorthlandPositronics/Cogtrix/commit/d7653ec211e5f1a754fd15d9bcd5d675695f2387))

## [0.1.5](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.4...v0.1.5) (2026-02-27)


### Features

* add /setup slash command and MCP filesystem server to docker-compose ([67acf2d](https://github.com/NorthlandPositronics/Cogtrix/commit/67acf2d936794bd01bb8dbec2495a566b8ba191c))
* add cross-provider model resolution for /model command ([c63ee26](https://github.com/NorthlandPositronics/Cogtrix/commit/c63ee2662908a0a868994684101390a9cfbac2f5))
* add Escape key cancellation and Ctrl+C prompt re-editing ([1641afb](https://github.com/NorthlandPositronics/Cogtrix/commit/1641afb7496f4cbbc17338b6f28f4f2a2d4a4165))
* add search persistence guidance to system prompt and tool descriptions ([b505d2f](https://github.com/NorthlandPositronics/Cogtrix/commit/b505d2ffe39ca2184405966fb13b787d4c5ff12d))
* encourage http_get follow-up on search result URLs in system prompt and tool descriptions ([8555988](https://github.com/NorthlandPositronics/Cogtrix/commit/8555988c3b000839cc201f03c3711180106fbf1e))
* increase default RAG chunk_size from 1200 to 2000 characters ([eb1cc0a](https://github.com/NorthlandPositronics/Cogtrix/commit/eb1cc0abb3eea4f3a1f69e47abf17d7cfc87239a))
* **memory:** run summarization and embedding on a background daemon thread ([615084a](https://github.com/NorthlandPositronics/Cogtrix/commit/615084a0ace41dc3fb148d3fd50f8ac7f25d36d4))


### Bug Fixes

* add __post_init__ validation to ProviderConfig (BUG-056) ([0fc34e5](https://github.com/NorthlandPositronics/Cogtrix/commit/0fc34e517210ac16a9c6c45076e6ec87ed2271a0))
* add 150ms warmup drain to prevent false Escape detection ([16b8384](https://github.com/NorthlandPositronics/Cogtrix/commit/16b838485d512d3a471dab7e6ae0dbca315b4c91))
* add bounded eviction to circuit breaker registry ([de42711](https://github.com/NorthlandPositronics/Cogtrix/commit/de42711241f3230d06480c73ce3ecb777d68dd03))
* add FAISS index lock and safe coercions for model numeric fields ([9a014ea](https://github.com/NorthlandPositronics/Cogtrix/commit/9a014ea4782704eff5b27399fe1f04771929a330))
* add injection-resistant delimiters to prompt optimizer ([13f1c7f](https://github.com/NorthlandPositronics/Cogtrix/commit/13f1c7fe6d8e256c5f1234775da0dc2afc3e4786))
* add warning for non-numeric IPv6 port, cap docs URL size, guard spinner stop race ([61b5783](https://github.com/NorthlandPositronics/Cogtrix/commit/61b5783ad2cbe8fc9790394f034001f022b176c3))
* address BUG-088/089/090/091 and PERF-009/011 bugs ([601621c](https://github.com/NorthlandPositronics/Cogtrix/commit/601621ccad6f28cbafa32549844718a462db78e0))
* address HIGH-severity bugs 083-087 ([dc0d65e](https://github.com/NorthlandPositronics/Cogtrix/commit/dc0d65ec53b524db8951c0bdcdf4826bb525f9ba))
* apply Round 17 audit fixes (ARCH-400..403) ([636d6b1](https://github.com/NorthlandPositronics/Cogtrix/commit/636d6b165d1aaad448fc4ca13ccd39032a0ab7eb))
* apply Round 17 audit fixes (PERF-300, PERF-301, ARCH-404, ARCH-405) ([1477d45](https://github.com/NorthlandPositronics/Cogtrix/commit/1477d45a1157ff2ffa91de319440d692f779dac7))
* apply Round 17 audit fixes BUG-702..705 ([3e6bb5b](https://github.com/NorthlandPositronics/Cogtrix/commit/3e6bb5b1aedeed08644d1bce9ddacbcb9edf55f2))
* **assistant:** create per-call SessionState to isolate concurrent chats ([80ed2fa](https://github.com/NorthlandPositronics/Cogtrix/commit/80ed2fa3c05ecb0c4c3af925d668c0c1ac9b94ae))
* **assistant:** hold lock during _index_facts to prevent FAISS race condition ([33fe641](https://github.com/NorthlandPositronics/Cogtrix/commit/33fe641a8c9547b7476b03108cf61bd7d6724b4e))
* **assistant:** prevent shared dict/list mutation across concurrent sessions ([06431d0](https://github.com/NorthlandPositronics/Cogtrix/commit/06431d077551cd2a24b80af0ccaaee4c1bcbce19))
* BUG-042/051/044/047 — prompt injection, exception logging, MCP restart tools, registry fallback ([c9d8a9d](https://github.com/NorthlandPositronics/Cogtrix/commit/c9d8a9dd0ef1cec19688a98293b864aadeae1d4b))
* BUG-200/201/202 -- memory error prefixes, CGNAT SSRF, auth error ([67f92d4](https://github.com/NorthlandPositronics/Cogtrix/commit/67f92d443d5dad28cb46dc2861a0c9b6d054db63))
* close streaming response in _follow_redirects before raising or redirecting ([0df0058](https://github.com/NorthlandPositronics/Cogtrix/commit/0df0058af973219a26b67cd30457dd40277e0603))
* **compression:** increase fallback truncation ratio from 50% to 75% ([0ad4f76](https://github.com/NorthlandPositronics/Cogtrix/commit/0ad4f76d0c34500478b4e3fb9b24f3a30b58fa71))
* **concurrency:** protect _status_callback with a lock and hoist import time to module scope ([c88c76e](https://github.com/NorthlandPositronics/Cogtrix/commit/c88c76e719fd7eabe3b3dc1c2c0c374daa5bc320))
* correct IPv6 URL formatting and RAG vectordb_dir alignment ([c223e70](https://github.com/NorthlandPositronics/Cogtrix/commit/c223e703057c85dd360a118c2b07caae96779ffa))
* **deep_think:** shallow-copy LLM per thread in _call_llm_parallel ([0b095e0](https://github.com/NorthlandPositronics/Cogtrix/commit/0b095e0157a80b570f1caa674ca6f94215e1109f))
* defer cbreak entry to monitor thread and remove prefill redisplay ([12339b0](https://github.com/NorthlandPositronics/Cogtrix/commit/12339b0707e6f704a6e256da9aba76cdb5270750))
* **delegate:** acquire _circuit_breaker_lock around all check_availability() call sites ([d30c7d0](https://github.com/NorthlandPositronics/Cogtrix/commit/d30c7d0701a8ba5167938e6bc108e1cf9e1d8cb5))
* **delegate:** eliminate shared-object mutation race in run_research_delegate ([4e8e3f6](https://github.com/NorthlandPositronics/Cogtrix/commit/4e8e3f6a960441c8733f9f93d686a23481698f63))
* eliminate _delegate_tools race condition, align ModelConfig errors, remove trivial executor ([e6b9d62](https://github.com/NorthlandPositronics/Cogtrix/commit/e6b9d625264a96ef266e9f2c40906e405b15f0ef))
* eliminate circuit breaker check_availability() race condition ([9233648](https://github.com/NorthlandPositronics/Cogtrix/commit/9233648dda13fffcc9817dc5f9ab457c20419e17))
* eliminate DNS rebinding TOCTOU, dead code, prompt lies, and circuit-breaker gaps ([74ef3ff](https://github.com/NorthlandPositronics/Cogtrix/commit/74ef3ff490ecb0a80622cd701857f95476599c32))
* eliminate race conditions in python_exec and http_request ([a71aa60](https://github.com/NorthlandPositronics/Cogtrix/commit/a71aa60821f1641562cb96d511d82b5230556c5c))
* eliminate thread-unsafe global fallback in python_exec session routing ([29149da](https://github.com/NorthlandPositronics/Cogtrix/commit/29149da46259f09f5b0ff522f0c5d0a0f57a41ec))
* ensure spinner is always resumed and make circuit breaker thread-safe ([d58605d](https://github.com/NorthlandPositronics/Cogtrix/commit/d58605da74019311d96db2d51676ee7adc6dde6e))
* **escape-monitor:** fix 5 bugs in warmup, stop, drain, and error handling ([0ce4ca8](https://github.com/NorthlandPositronics/Cogtrix/commit/0ce4ca8d484bf7b4eaf32927828d5cf764cc9116))
* exclude execute_python from assistant mode and scrub secrets in _log ([5877a23](https://github.com/NorthlandPositronics/Cogtrix/commit/5877a23054af411396ba5bdb15aeff380cf9bd52))
* **file_ops:** block absolute path writes outside cwd (BUG-006) ([844b605](https://github.com/NorthlandPositronics/Cogtrix/commit/844b605df16fe28a64d89120efa95ad2e35a0a8b))
* **file_ops:** block absolute paths outside cwd in _validate_path ([db11fb7](https://github.com/NorthlandPositronics/Cogtrix/commit/db11fb70703d2759a88b44d55371da2fcaa3e40b))
* **file_ops:** eliminate TOCTOU race in read_file by removing pre-checks ([6b96d3f](https://github.com/NorthlandPositronics/Cogtrix/commit/6b96d3f2a9a952226adbd5084082302bee06df4d))
* **file_ops:** remove unreachable is_write cwd check in _validate_path() ([4b691d2](https://github.com/NorthlandPositronics/Cogtrix/commit/4b691d2b4f2698313156d0803316bc47d997c051))
* fix nullable anyOf/oneOf in MCP schema and turn_count in wrong method ([445ab56](https://github.com/NorthlandPositronics/Cogtrix/commit/445ab56a05ee2d8494c92c6eee34e10aafa52167))
* flush stale stdin bytes to prevent false Escape detection on first prompt ([f4a2087](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a2087014f18086b4900ffa62d2a40a3d30cb7c))
* guard int/float coercions against non-numeric strings and fix phases.py provider bypass ([3720a45](https://github.com/NorthlandPositronics/Cogtrix/commit/3720a45226813f196eda0b99e82d702403534143))
* **guardrails:** eliminate TOCTOU in ChatRateLimiter and fix leet-score false positives on numeric tokens ([4da74b9](https://github.com/NorthlandPositronics/Cogtrix/commit/4da74b9b712c50083210e50cac5c8ccdb2368b84))
* **guardrails:** wire ViolationTracker persist_path and release lock before disk write ([f98f286](https://github.com/NorthlandPositronics/Cogtrix/commit/f98f28670caea2c827267abd703efb4cdcbaf6c9))
* handle nested triple-backticks in _extract_yaml greedy fallback ([0e636e1](https://github.com/NorthlandPositronics/Cogtrix/commit/0e636e166fc2166e597aeda468fefe20c4c1fd77))
* handle ToolMessage in on_tool_end and clean stale tool_lookup entry ([311f31c](https://github.com/NorthlandPositronics/Cogtrix/commit/311f31cdede474da62fde713d7be3484899f5660))
* **handler:** record rate-limit before agent run and extract knowledge before sanitize ([9ddbb74](https://github.com/NorthlandPositronics/Cogtrix/commit/9ddbb74918ad07d3043e09d8b5e3ff90448011e7))
* harden security in setup_wizard, whatsapp channel, and python_exec sandbox ([b6e8d97](https://github.com/NorthlandPositronics/Cogtrix/commit/b6e8d976b0181f944224147d0b14bf4f8969ae91))
* **http:** eliminate DNS rebinding TOCTOU in http_request tool (BUG-074) ([90003db](https://github.com/NorthlandPositronics/Cogtrix/commit/90003dbc5716aee3312f620b135757144ccd1a1d))
* **http:** evict stale entries from _recent_failures in _record_failure ([c22f4b3](https://github.com/NorthlandPositronics/Cogtrix/commit/c22f4b3292797616c67d197225fdfe56a6114365))
* **http:** handle iter_content exceptions gracefully in _read_bounded_response ([3527c1f](https://github.com/NorthlandPositronics/Cogtrix/commit/3527c1f12668912c2fc5cf81dc705913491c449c))
* **http:** stream responses to avoid buffering large bodies in memory ([65fbaca](https://github.com/NorthlandPositronics/Cogtrix/commit/65fbaca3da5da36e5f88ed0ea495d8a682c7ce41))
* implement P2 correctness fixes F2, F9 (partial) ([4549b9b](https://github.com/NorthlandPositronics/Cogtrix/commit/4549b9bb0491fdc27232f032069daa517fe26dbd))
* **intent:** widen explain-verb proximity guard threshold from 30 to 50 ([35a1827](https://github.com/NorthlandPositronics/Cogtrix/commit/35a1827a6bb86802964af9bb1db771eb79ed41a9))
* lazy history paths and monotonic clock in ViolationTracker ([e61ce79](https://github.com/NorthlandPositronics/Cogtrix/commit/e61ce79d39a464798f202984c6f491da7791f3d2))
* **logging:** scrub secrets from tool inputs in on_tool_start ([71b6672](https://github.com/NorthlandPositronics/Cogtrix/commit/71b667238671afa11b3bfb079f64fdc55cf265de))
* **logging:** scrub secrets from tool_args before logging LLM_TOOL_CALL ([20f5f23](https://github.com/NorthlandPositronics/Cogtrix/commit/20f5f2386e7f8eae48d2eec006a9adcbdc3bd21f))
* **mcp:** handle complex JSON Schema types and builtin tool name collisions ([d8b446e](https://github.com/NorthlandPositronics/Cogtrix/commit/d8b446ed265b3a57e523c2736ab94d25e8f43903))
* **memory:** atomic write in save_history() to prevent corrupt JSON on crash ([d14844f](https://github.com/NorthlandPositronics/Cogtrix/commit/d14844fcbcb56a36fc36cb6d51e18afab6230991))
* **memory:** guard _save_hybrid_meta reads under lock, atomic write, shallow-copy batch ([db09700](https://github.com/NorthlandPositronics/Cogtrix/commit/db09700b669022b147ea557dbf29c50555783adf))
* move deny_all/denials check inside confirmation_lock to eliminate TOCTOU race ([c81850f](https://github.com/NorthlandPositronics/Cogtrix/commit/c81850f54019eb45f244ded75150b8db80740947))
* move pause_spinner inside try block and lock circuit breaker record calls ([0823571](https://github.com/NorthlandPositronics/Cogtrix/commit/08235710fe6b3e35f78ad9700b0fc3cca856ff8e))
* parse num_ctx/temperature from providers section and add missing env var handlers ([887971d](https://github.com/NorthlandPositronics/Cogtrix/commit/887971d04acdb6993934e27f7fe2bbab37979b8c))
* pass AIMessage to _detect_tool_request so explicit tool loading works ([e900b50](https://github.com/NorthlandPositronics/Cogtrix/commit/e900b502921b94143ff369d1bf1a642069dd109c))
* pass session_state to run_agent/run_execution_phase and split _enter_cbreak try blocks ([8a2b0fb](https://github.com/NorthlandPositronics/Cogtrix/commit/8a2b0fb03a825411bcc16f8d528af80ad0b801e0))
* persist memory on exit and update hybrid test for background threading ([afaf00f](https://github.com/NorthlandPositronics/Cogtrix/commit/afaf00f4918ed05e9dc98abd1e908d981261c384))
* persist ViolationTracker blacklist state across restarts ([051f262](https://github.com/NorthlandPositronics/Cogtrix/commit/051f26278681c111f1890698bd19ad8313f384a8))
* **phases:** make extract_turn_messages robust with isinstance and boundary anchor ([13ecdc9](https://github.com/NorthlandPositronics/Cogtrix/commit/13ecdc943873152060b5e95abd71401f4a8da83e))
* preserve wizard model key and rebuild compression LLM on switch ([e0a0577](https://github.com/NorthlandPositronics/Cogtrix/commit/e0a05775c9d0f366563b548a8888546d62588102))
* prevent double optimizer invocation on /o force-optimize command ([6f18b0c](https://github.com/NorthlandPositronics/Cogtrix/commit/6f18b0cbf43f6179f66219eeaf0ca2a250b99b4a))
* **prompt:** sanitize delimiter strings in optimizer to prevent injection ([cb16850](https://github.com/NorthlandPositronics/Cogtrix/commit/cb168506b7cd883eacde6900db8a5c4f95a85215))
* propagate UserCancelledRun and show spinner during optimize_prompt ([b97e7fb](https://github.com/NorthlandPositronics/Cogtrix/commit/b97e7fb0c7e22b8bfbe550954609f233ee244680))
* Pydantic v1/v2 copy compat and RAG subdir recursion (BUG-049, BUG-050) ([2f63861](https://github.com/NorthlandPositronics/Cogtrix/commit/2f638612fa71d40eafeadc3fb94122cd5848727e))
* **python_exec:** close sandbox escape via type.__dict__ descriptor chain ([0aa183e](https://github.com/NorthlandPositronics/Cogtrix/commit/0aa183e3acd6c7ad8048ef06ffabd1ce87ff2811))
* **python_exec:** replace substring module check with AST imports and add LRU eviction ([01f87fc](https://github.com/NorthlandPositronics/Cogtrix/commit/01f87fc31f8496a7d86dfff447713cff0d242cab))
* **registry:** remove single-schema fallback in fallback tool discovery ([f6fee62](https://github.com/NorthlandPositronics/Cogtrix/commit/f6fee62a7a128043de16159043aa3771e3f7ae52))
* remove reverse import in handler.py and sync AgentRunner Protocol ([708f440](https://github.com/NorthlandPositronics/Cogtrix/commit/708f440baba2b68391588a6c1e1744f6a0770101))
* reset _deny_all in run_single_prompt and guard _compress_one exceptions ([9c8738d](https://github.com/NorthlandPositronics/Cogtrix/commit/9c8738d301366a0ba34b216f821296d9090de04e))
* resolve 5 P2/P3 bugs across orchestration, config, and CLI ([98ca0e5](https://github.com/NorthlandPositronics/Cogtrix/commit/98ca0e5b75ce9f5ecc252c4853cac5e1e7ea7b51))
* resolve circular imports in src/orchestration/phases.py ([5845590](https://github.com/NorthlandPositronics/Cogtrix/commit/5845590a23dbaaf01f8f0f61c83f2569185d8c16))
* resolve four bugs across mcp_client, deep_think, memory, and poller ([158124b](https://github.com/NorthlandPositronics/Cogtrix/commit/158124bfbc865cf5672050c5e1f950437e0d5e48))
* resolve graph.py dependency issues after module extraction ([ec229a4](https://github.com/NorthlandPositronics/Cogtrix/commit/ec229a4665340938934bf123e86bcb2afa9e1b28))
* resolve spinner deadlock and WEB_TOOL_NAMES divergence ([e950613](https://github.com/NorthlandPositronics/Cogtrix/commit/e950613d428e9a33ed9c7b80cba3ae89c0f97abf))
* scan only current iteration result_msgs in _detect_tool_request ([e57c196](https://github.com/NorthlandPositronics/Cogtrix/commit/e57c196ba789bfda2a6defd466316a26e0d414d7))
* **security:** add Unicode bidi isolate codepoints and fix circuit breaker lock race ([1443b1d](https://github.com/NorthlandPositronics/Cogtrix/commit/1443b1d2297df5092a8b40512b7a50175e35e16a))
* **security:** atomic writes for knowledge/violation stores; sanitize SDK errors in agent error formatter ([7aaab51](https://github.com/NorthlandPositronics/Cogtrix/commit/7aaab511f7588577e080acd0c5efb67767429f90))
* **security:** block SSRF via redirect in http_get and http_post ([cf7499c](https://github.com/NorthlandPositronics/Cogtrix/commit/cf7499c09e4e88469e2334bec3fda1766be6d665))
* **security:** close sandbox escape via runtime getattr/setattr (BUG-016) ([601e7f9](https://github.com/NorthlandPositronics/Cogtrix/commit/601e7f91dfae15b3c588b71565227f27eb4d5888))
* **security:** correct shell tool exclusion name and add homoglyph normalization to guardrails ([943c894](https://github.com/NorthlandPositronics/Cogtrix/commit/943c8949d11bfe2a50a8f7db55f8617016b3e02f))
* **security:** normalize paths before prefix-checking in ToolCallGuard and recall.py ([98a3af0](https://github.com/NorthlandPositronics/Cogtrix/commit/98a3af0a7907cd20fe94541d4a9cc7cefdfdf8fc))
* **security:** replace string-based SSRF checks with ipaddress+socket validation ([dd7813a](https://github.com/NorthlandPositronics/Cogtrix/commit/dd7813a81999c90d6471715718772b349d50de19))
* **security:** scrub LLM output, add xai- key prefix, and fix sandbox hasattr ([c0bba06](https://github.com/NorthlandPositronics/Cogtrix/commit/c0bba0622da1178de91932e5df62cac0a60b5471))
* send response before memory update and add RLock to SessionVectorStore ([1df44ac](https://github.com/NorthlandPositronics/Cogtrix/commit/1df44ac946554d8f125cc3f78df92b7b732d72bd))
* serialize ViolationTracker disk writes under lock and replace O(n²) list pops with O(n) slice ([1e025cd](https://github.com/NorthlandPositronics/Cogtrix/commit/1e025cdaa9a2beb4232c6822b3b383cb0672e995))
* **shell:** return accurate message when command fails with no output ([e187979](https://github.com/NorthlandPositronics/Cogtrix/commit/e187979eab7c03f12e5429e29aa15e54748adb0e))
* split auto_expansion_count, distinguish active/unknown tool errors, add fuzzy rename guidance ([4da7445](https://github.com/NorthlandPositronics/Cogtrix/commit/4da7445a30d7ed3bbf2e80bb8927aec8157cef6d))
* **ssrf:** block link-local, RFC 6598, IPv6 ULA, and IPv4-mapped loopback in _validate_url ([ed7778b](https://github.com/NorthlandPositronics/Cogtrix/commit/ed7778b417bfba78e8474af856c7c3156575856c))
* stop LLM echoing timestamps, fix pipe table rendering, dim shell output ([686a230](https://github.com/NorthlandPositronics/Cogtrix/commit/686a2304e0e5858f3142dddddfe4c53d6073cea4))
* stop mutating shared ProviderConfig during model resolution ([c180c1f](https://github.com/NorthlandPositronics/Cogtrix/commit/c180c1fd55c10cbf49f9e9e0cde677460fab9974))
* stop spinner race, re-raise UserCancelledRun, add exit hint ([e8abd26](https://github.com/NorthlandPositronics/Cogtrix/commit/e8abd26bb7e977fcce6463cb755ba1ebc6abdd89))
* **thread-safety:** atomic config swap and stderr lock for concurrent tool calls ([ae68ac8](https://github.com/NorthlandPositronics/Cogtrix/commit/ae68ac8928d9a746433ad594fb4cb493711458ab))
* **threadpool:** replace context-manager executors to prevent timeout blocking ([7fff03c](https://github.com/NorthlandPositronics/Cogtrix/commit/7fff03cf2bdf50592b7de4d65a6d651a48e4162f))
* three bug fixes — no_confirm bypass, list-form ToolMessage content, lock scope ([497fdbe](https://github.com/NorthlandPositronics/Cogtrix/commit/497fdbefdd0c197dec452803c66724f6e501b5dc))
* three correctness bugs in check_config, execution phase, and output cap ([b6b7413](https://github.com/NorthlandPositronics/Cogtrix/commit/b6b74132e26918bb39024548a26468747fe42f95))
* validate negative integer config fields and normalize provider type case ([3227cdd](https://github.com/NorthlandPositronics/Cogtrix/commit/3227cddfb64ed49cf8ed0e9d23efdb7394f24a8b))


### Performance Improvements

* cache bind_tools() result in call_model to avoid redundant schema rebuilds ([53cfe9b](https://github.com/NorthlandPositronics/Cogtrix/commit/53cfe9bd5b6271825ff427a1b8a94628eeee43a6))
* **deep_think:** remove duplicate ISOLATION WARNING from tool description ([0faa38b](https://github.com/NorthlandPositronics/Cogtrix/commit/0faa38b9792046d2f0a005288b8d1061135404f6))
* implement P2 TTFT optimizations (F1, F3, F8) ([a4ac8d9](https://github.com/NorthlandPositronics/Cogtrix/commit/a4ac8d969c45a5535ac8481d791fa5fe1d27b7c4))
* lift tool_lookup rebuild and skip blocking join in save() ([9f9c24b](https://github.com/NorthlandPositronics/Cogtrix/commit/9f9c24ba3d09cbb4978dc0194e566217c8f3723e))
* **memory:** gate stale reasoning prefix sections to reduce TTFT (F5) ([9f78f11](https://github.com/NorthlandPositronics/Cogtrix/commit/9f78f11eecc6cc5373081df6b08fd893f75cf693))
* **optimizer:** raise length gate from 150 to 400 chars, action-verb skip to 600 ([447f746](https://github.com/NorthlandPositronics/Cogtrix/commit/447f74690e01ee328eba571c4b3196485e61d27a))
* remove blocking _wait_for_background from prepare_context, cap escape drain loop ([d9de96f](https://github.com/NorthlandPositronics/Cogtrix/commit/d9de96ffdf06d592764306eb641a3ed30e2b0655))
* remove duplicate search guidance, hoist time import, cache SystemMessage ([1b8db52](https://github.com/NorthlandPositronics/Cogtrix/commit/1b8db52f9467473759f5700d2ce34b836a6a39fd))
* **request_tools:** remove tool name list from description to reduce token usage ([665b36b](https://github.com/NorthlandPositronics/Cogtrix/commit/665b36b6805591164ae4a2755c1707db53c3d449))
* tune memory and compression thresholds (P2 batch) ([3487a46](https://github.com/NorthlandPositronics/Cogtrix/commit/3487a4613868a8921dbf00c3c4e852edddf42b2a))


### Documentation

* add ADR 007 — unify tool safety architecture decision record ([91ddcdd](https://github.com/NorthlandPositronics/Cogtrix/commit/91ddcdd10dfe4e63fd9bef55dd50d4cae2cd4b4a))
* add architecture refactoring plan ([820169c](https://github.com/NorthlandPositronics/Cogtrix/commit/820169c9ecf94149f21fda894fd0b79d2901da16))
* add post-refactor bug sweep and AI interaction audit reports ([645be13](https://github.com/NorthlandPositronics/Cogtrix/commit/645be138960d89bbe8d0af246eb2dc2237860182))
* add Round 3 audit ADRs and findings reports ([a44f0c9](https://github.com/NorthlandPositronics/Cogtrix/commit/a44f0c97d6c6e2a741e54680187b1d2a81dccfa4))
* fix tool count inconsistency — standardise to 51 across all docs ([85fd20a](https://github.com/NorthlandPositronics/Cogtrix/commit/85fd20a23e4d1288f1c18653ee9fdcb90873cda3))
* holistic documentation revision + P4 bug fixes + Round 6 audit prep ([28ef1ee](https://github.com/NorthlandPositronics/Cogtrix/commit/28ef1ee9b0541a1b83cd50b552c016c56aee958d))
* update CLAUDE.md and bug reports to reflect all ProjectForge audit fixes ([8380e08](https://github.com/NorthlandPositronics/Cogtrix/commit/8380e08ae606b5b5b9b69504cf273eaaaead35a1))
* update CLAUDE.md for Round 3 fixes and add verification report ([1f7409d](https://github.com/NorthlandPositronics/Cogtrix/commit/1f7409d6492260970c8791bd1c5d365ecffb2b9e))
* update CLAUDE.md to reflect new module structure ([aad17a6](https://github.com/NorthlandPositronics/Cogtrix/commit/aad17a6308a4523731bd3f732ab83c087f964ba9))
* update documentation to reflect bug fix changes ([b7135bc](https://github.com/NorthlandPositronics/Cogtrix/commit/b7135bc0c0d1e2996fdb7fdbf6a3e54ae527a9de))

## [0.1.4](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.3...v0.1.4) (2026-02-22)


### Bug Fixes

* **ci:** align release workflow with CI pipeline ([4a8e783](https://github.com/NorthlandPositronics/Cogtrix/commit/4a8e783861b1f602f2b578657e6bde17cf6dafa1))
* **ci:** exclude integration tests and set bandit threshold in release workflow ([a5f7af3](https://github.com/NorthlandPositronics/Cogtrix/commit/a5f7af30a01c279c9d9d1f8644483e1258f6b434))

## [0.1.3](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.2...v0.1.3) (2026-02-22)


### Features

* activity indicator (spinner) during LLM processing ([178a1cc](https://github.com/NorthlandPositronics/Cogtrix/commit/178a1ccfb51a900a6d056f3429550c01a813e794))
* add --setup, --setup-docs, --setup-output CLI flags and dispatch handler ([07da36f](https://github.com/NorthlandPositronics/Cogtrix/commit/07da36f551c84fc48abe8fb5d455bd4f8c6d7756))
* add /optimizer &lt;prompt&gt; to force-optimize and run a prompt ([2210421](https://github.com/NorthlandPositronics/Cogtrix/commit/22104217cb9f2599cb5efb5b4e4707e7301d4cfc))
* add /optimizer command, rename /noconfirm to /approve, add aliases ([b4d221d](https://github.com/NorthlandPositronics/Cogtrix/commit/b4d221d9fd2d66f169fcc16b726e484631c05a19))
* add /tools enable/disable subcommands and on-demand tool status ([256e486](https://github.com/NorthlandPositronics/Cogtrix/commit/256e486e2d7fffdabe93c33903f22003201951de))
* add /tools load subcommand and [loaded] status tag ([ac9058e](https://github.com/NorthlandPositronics/Cogtrix/commit/ac9058edc472fc20befa9f26909ca48c603201af))
* add agent message-handling workflow integration tests ([b757ff6](https://github.com/NorthlandPositronics/Cogtrix/commit/b757ff69eb9dde2ec0fdc6dac3ec39ad71214f45))
* add Anthropic/Google provider extras, expand Docker multi-provider setup ([51e517b](https://github.com/NorthlandPositronics/Cogtrix/commit/51e517b8a1fe50343d8e4e802403efe5070cc88c))
* add CLI flags and file-ref config for assistant system prompt ([1cb1487](https://github.com/NorthlandPositronics/Cogtrix/commit/1cb1487d11fa3041e8d1b5489499ead85f0c1f89))
* add colored response stats, fix TokenAccumulator, show system prompt in /info ([ead03ca](https://github.com/NorthlandPositronics/Cogtrix/commit/ead03caa6b77712314970b22b3c656a2a5e5fc6a))
* add colorized --help page with structured argument groups ([953240f](https://github.com/NorthlandPositronics/Cogtrix/commit/953240f624af486454adb3a5b8da23012918c52c))
* add dedicated compression model support via context_compression.model config ([29f3670](https://github.com/NorthlandPositronics/Cogtrix/commit/29f3670b0d09d6ecc33339fb64161606a4ca93d2))
* add delegation visibility, /delegate command, and auto-delegation ([a85ef95](https://github.com/NorthlandPositronics/Cogtrix/commit/a85ef95b418815464f382006795a9ae3dbf4b568))
* add EncodingDetectionGuard and ToolCallGuard to guardrail pipeline ([4445a6d](https://github.com/NorthlandPositronics/Cogtrix/commit/4445a6d80f21fa1ba664c30d24ecb03cdbe46ad8))
* add hidden /system_prompt command, show only prompt size in /info ([253a902](https://github.com/NorthlandPositronics/Cogtrix/commit/253a902f62bcbdb33d806c2883a8c87a90c359d7))
* add in-loop context compression for old ToolMessages ([dc3ba95](https://github.com/NorthlandPositronics/Cogtrix/commit/dc3ba956d1787ad4160a31590483ec9979be8c49))
* add inline shell commands, bright prompt, history resilience ([cb49340](https://github.com/NorthlandPositronics/Cogtrix/commit/cb493400c8615acbeba4b0292aed74b3d304d498))
* add MCP client manager module ([fc70705](https://github.com/NorthlandPositronics/Cogtrix/commit/fc7070511b9e9102865141734fb32c4d429ea5b7))
* add MCP config, registry helpers, optional dep, and documentation ([4e7d815](https://github.com/NorthlandPositronics/Cogtrix/commit/4e7d815faa30f2b2a09427871b1335cae5e35ae8))
* add prompt optimizer, StateGraph stream recovery, and tool instructions ([7aaed09](https://github.com/NorthlandPositronics/Cogtrix/commit/7aaed099dd318c4779c707d8a84d9dc78a9c2bba))
* add provider registry, auto-launch wizard, and fix assistant guardrails ([49db710](https://github.com/NorthlandPositronics/Cogtrix/commit/49db710d9bf9e48686f2b5253882d75e1d5431a2))
* add SharedKnowledgeStore for cross-chat fact extraction and recall ([2117a7b](https://github.com/NorthlandPositronics/Cogtrix/commit/2117a7beed83f8acd0c94a4f49171ee1b2d2ec77))
* add Telegram messaging tool, WhatsApp guide, and Docker Compose ([1a3be08](https://github.com/NorthlandPositronics/Cogtrix/commit/1a3be0859ff66c4f1c1fa07c1d45c1f8c7fd2959))
* add tool_call_guard callback to _build_agent_graph and run_agent ([b6ec413](https://github.com/NorthlandPositronics/Cogtrix/commit/b6ec41318dfd02682bf19f84894b67b087934220))
* add UTC timestamps to message history and harden type safety ([6425276](https://github.com/NorthlandPositronics/Cogtrix/commit/642527640093a62fb80576eabb001f8543025077))
* add ViolationTracker and auto-blacklist to GuardrailPipeline ([16bbb3a](https://github.com/NorthlandPositronics/Cogtrix/commit/16bbb3a538c941457003f8158bd61148d274ca0f))
* add WhatsApp messaging tool and fix bugs in contact filtering ([a1d8e61](https://github.com/NorthlandPositronics/Cogtrix/commit/a1d8e61ae24d2866e282318c0b161da417b20bd1))
* allow model to release tools via request_tools meta-tool ([164623a](https://github.com/NorthlandPositronics/Cogtrix/commit/164623adab4a23434a3be8bd54ff53644d894bf2))
* auto version bump and release on merge to main ([fb2b89f](https://github.com/NorthlandPositronics/Cogtrix/commit/fb2b89f0e9bac86ccb1b198ab99293af978877bc))
* default to Ollama for zero-config out-of-box experience ([4c942af](https://github.com/NorthlandPositronics/Cogtrix/commit/4c942af28c391c57555881357a32839854e30780))
* default to Ollama for zero-config out-of-box experience ([98c2c7b](https://github.com/NorthlandPositronics/Cogtrix/commit/98c2c7b03575c3f031cfec6f8d5e1c6a67b73706))
* display response stats (elapsed time and token usage) after agent replies ([f4a25d6](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a25d629fc4353136db8ed6de01986dd70a72fa))
* double spinner phrases, gradient color, trailing space ([f8aa7b5](https://github.com/NorthlandPositronics/Cogtrix/commit/f8aa7b5693bcc86bbe6bd2de26b0d72fa548575d))
* enhance setup wizard with Rich rendering, spinner, API key reuse, and tests ([c14ae85](https://github.com/NorthlandPositronics/Cogtrix/commit/c14ae8550d6bd497bc2e0b61bdccc339cedc8d72))
* expand spinner to 80 phrases with humorous tech messages ([c49cb79](https://github.com/NorthlandPositronics/Cogtrix/commit/c49cb794e47ce18a48080b514b63cb02081cf7eb))
* expand task classification to 23 categories and fix force deep_think override ([e71204b](https://github.com/NorthlandPositronics/Cogtrix/commit/e71204b21621280e807b74494b3018527e2ee2c2))
* expand tool confirmation prompt with disable/deny-all/cancel options ([65f787a](https://github.com/NorthlandPositronics/Cogtrix/commit/65f787a3cbcddc13f47a73a8c461ba332ac369b5))
* hybrid /think pipeline with category-aware prompts ([eede19b](https://github.com/NorthlandPositronics/Cogtrix/commit/eede19b32062c1e887e1ec27bd95a8a1085f4f0e))
* implement hybrid memory (sliding window + summary + vector recall) ([12835af](https://github.com/NorthlandPositronics/Cogtrix/commit/12835af3aec15960b46636a4b3138b26aa4e5b71))
* implement Sprint 1 of assistant mode (WhatsApp/Telegram daemon) ([a971ac5](https://github.com/NorthlandPositronics/Cogtrix/commit/a971ac5075cd9d81d12386805cc5445b970195be))
* increase working memory for code and reasoning modes ([dc44c66](https://github.com/NorthlandPositronics/Cogtrix/commit/dc44c66bf0b4a67747150f4026055bdf76c779f0))
* integrate MCP server support into CLI and slash commands ([1ab64f6](https://github.com/NorthlandPositronics/Cogtrix/commit/1ab64f6fcf49d132ed90821ce5e4063222002a92))
* move --setup to early-exit position and add auto-launch wizard ([ac348f0](https://github.com/NorthlandPositronics/Cogtrix/commit/ac348f01d1eb4d92b2ce31f5134913180c0261dd))
* provider registry, auto-launch wizard, assistant guardrail fixes ([5624729](https://github.com/NorthlandPositronics/Cogtrix/commit/56247295d7c37ba9d5fc4e22aede0797db50f729))
* publish multi-arch Docker image to GHCR ([d5cbcaa](https://github.com/NorthlandPositronics/Cogtrix/commit/d5cbcaa5763b79a63279f2f047823423daf34bc3))
* publish multi-arch Docker image to GHCR on main push ([a93dce2](https://github.com/NorthlandPositronics/Cogtrix/commit/a93dce2d438e43f707f3d48a7b5b26e5c8f612f8))
* randomize spinner phrase order on each start ([5c479dc](https://github.com/NorthlandPositronics/Cogtrix/commit/5c479dc12eec625ecb72a0428f5a5c7293b931fd))
* replace custom version-bump with release-please ([fecd792](https://github.com/NorthlandPositronics/Cogtrix/commit/fecd792c06565fd95c3d9b790d5b685f69d94c44))
* replace Panel with Rule+Padding for LLM response display ([c051c80](https://github.com/NorthlandPositronics/Cogtrix/commit/c051c80230195a1c308167ad76dc221f06ae6a97))
* restyle setup wizard with ANSI color and box-drawing UI ([541a06b](https://github.com/NorthlandPositronics/Cogtrix/commit/541a06be989da71294c12ad32436975133ef9826))
* restyle tool confirmation prompt labels and hotkeys ([9d0ca5c](https://github.com/NorthlandPositronics/Cogtrix/commit/9d0ca5c566d0d9ea9bbd6fea76f900f8f29c0165))
* sequential intro phrases then random fun phrases ([74500bf](https://github.com/NorthlandPositronics/Cogtrix/commit/74500bf50fa85f633a5b4c91f54a1c205067b265))
* show spinner during forced deep_think invocations ([236852a](https://github.com/NorthlandPositronics/Cogtrix/commit/236852a26fd66cf0037ae25ac6eeaa456c18cb75))
* start agent with only request_tools, all other tools on demand ([8099962](https://github.com/NorthlandPositronics/Cogtrix/commit/80999626b6d38b9ef8511b8d231cc8115688a8bf))
* tool-capable delegates and empty context validation ([ecfc45c](https://github.com/NorthlandPositronics/Cogtrix/commit/ecfc45c658261639326833067c49bcce98b630c0))


### Bug Fixes

* add execution phase so agent acts on analysis instead of just describing ([b16e3b9](https://github.com/NorthlandPositronics/Cogtrix/commit/b16e3b9116e2d314df16309d1bd28089d5a5375a))
* add nosec markers for Bandit false positives (B311, B110) ([a9062b8](https://github.com/NorthlandPositronics/Cogtrix/commit/a9062b87d62f6af4a0bef3cfce9d02a0ff4c81bc))
* add rollback to /mode switch to prevent state corruption on failure ([bea84ac](https://github.com/NorthlandPositronics/Cogtrix/commit/bea84ac1eaa30b30cac9d0e63a4f73063d6261db))
* add update_id tracking for Telegram deduplication and update docs ([e0223a4](https://github.com/NorthlandPositronics/Cogtrix/commit/e0223a4fb4ad57ec9c557426cdbd269c895f7ade))
* auto-activate on-demand tools and eliminate retry loops ([d691fd4](https://github.com/NorthlandPositronics/Cogtrix/commit/d691fd430b41ae25b0e8c7d3550d2bafec6cc925))
* break inner agent loop immediately after request_tools runs ([99de5ad](https://github.com/NorthlandPositronics/Cogtrix/commit/99de5ad4a51fd724be2e1c5e77d57b3126242797))
* category-aware Stage 2 framing for /think pipeline ([ca6d246](https://github.com/NorthlandPositronics/Cogtrix/commit/ca6d24656ce70309a3b26f2a054ffa6347561617))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([8954943](https://github.com/NorthlandPositronics/Cogtrix/commit/895494376ec73b74f1f0d54a515d578a97276619))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([65ef262](https://github.com/NorthlandPositronics/Cogtrix/commit/65ef2629ab989dda2ea315dd7797b97d01b29a5a))
* **ci:** move pytestmark below imports to fix ruff E402 ([82b8fec](https://github.com/NorthlandPositronics/Cogtrix/commit/82b8feca5193aac6b5db142facb631239d14341b))
* **ci:** resolve pyright type-ignore placement and bandit B310 warnings ([3a28a68](https://github.com/NorthlandPositronics/Cogtrix/commit/3a28a682eb69a29e8c08da3cf2df4201338ba967))
* **ci:** set bandit severity threshold to medium (-ll) ([1890edd](https://github.com/NorthlandPositronics/Cogtrix/commit/1890edd72768a8c38066370a3278c66b5fb4724f))
* clamp summarization index and harden JSON brace parsing ([659e4da](https://github.com/NorthlandPositronics/Cogtrix/commit/659e4daff0461f320b482556618fd242083a430a))
* classifier label normalization and tool output error filter ([89a951b](https://github.com/NorthlandPositronics/Cogtrix/commit/89a951b2e5c8ba1c9e23af0d0f8acf1893fca35b))
* clean up delegate tool alias resolution and remove dead max_depth ([69993de](https://github.com/NorthlandPositronics/Cogtrix/commit/69993de033e1fcf945f7728bf767eecebc75d964))
* correct /mode working memory display and add delegation tests ([70c6852](https://github.com/NorthlandPositronics/Cogtrix/commit/70c6852d362d2eadae4f21e109fb053dcae34e00))
* correct YAML output, RLock evict_idle, and guardrail false positives ([74c9fab](https://github.com/NorthlandPositronics/Cogtrix/commit/74c9fab1ca7af75443166a7c47dd196adffeac2e))
* detect unconfigured provider before LLM init ([bc3d5d8](https://github.com/NorthlandPositronics/Cogtrix/commit/bc3d5d83efd2f596778ec12c61fd273e4215c9b5))
* escape braces in deep_think prompts and guard escape flag to string context ([d999b7a](https://github.com/NorthlandPositronics/Cogtrix/commit/d999b7af0023e2736475ddbd7d6f74d06aebd1b1))
* give delegates all tools (active + on-demand) from the start ([acb316e](https://github.com/NorthlandPositronics/Cogtrix/commit/acb316ecfc9933205b7c422127a4f47695770754))
* graceful error messages for common configuration issues ([a4c9518](https://github.com/NorthlandPositronics/Cogtrix/commit/a4c9518d0da9597cdda65de989b1e92dfab94e5d))
* graceful error messages for common configuration issues ([62b14a4](https://github.com/NorthlandPositronics/Cogtrix/commit/62b14a417c2fb744750cba962b9cf2da5e6a6380))
* harden delegate tool review — docstring, tests, and exports ([2562d32](https://github.com/NorthlandPositronics/Cogtrix/commit/2562d327ba228d340cc167763b761762f614ab9c))
* harden input validation and add defensive checks (QA report) ([da3af41](https://github.com/NorthlandPositronics/Cogtrix/commit/da3af41deedc0fbec102d3aefdac6907cfeec2b8))
* harden input validation and correct misleading docstrings ([55c3d51](https://github.com/NorthlandPositronics/Cogtrix/commit/55c3d5139c02ab2f9bc968e70af1d0420b4963ba))
* harden memory system against multimodal content and ToolMessage corruption ([722ddda](https://github.com/NorthlandPositronics/Cogtrix/commit/722ddda26a5af00ea83d665ec1dd2b36721d1ddd))
* include search packages in Docker image ([93febfc](https://github.com/NorthlandPositronics/Cogtrix/commit/93febfc088bbd62f9aa60d5c8d3e56f92f5b5541))
* isolate config tests from env vars and use logging in config parser ([4c6df7d](https://github.com/NorthlandPositronics/Cogtrix/commit/4c6df7d2a87e7104abed51767b1a543c3bcea50c))
* keep request_tools meta-tool when all on-demand tools are activated ([e6e57b6](https://github.com/NorthlandPositronics/Cogtrix/commit/e6e57b6057ef5b491172e6abb279d548c73fcaff))
* lowercase Docker image name in release workflow ([2d30e82](https://github.com/NorthlandPositronics/Cogtrix/commit/2d30e820625d7a29700d0db25e631a876cc4527c))
* mode rollback, credit card regex, json wildcard, and docs ([ec35865](https://github.com/NorthlandPositronics/Cogtrix/commit/ec358656bb4b4bf722b579c2f4af46619472b6e8))
* optimize Dockerfile and harden .dockerignore ([049e8a3](https://github.com/NorthlandPositronics/Cogtrix/commit/049e8a39ff59497b9a6c8d68a0757553789a0784))
* pass force as boolean in version-bump ref update ([655059e](https://github.com/NorthlandPositronics/Cogtrix/commit/655059e4d986819b06ff89523979652ba4c9413a))
* pause spinner during deep_think progress output ([c58fc30](https://github.com/NorthlandPositronics/Cogtrix/commit/c58fc30cb92e7ff70f1d338ac438587ef9a8c0d3))
* persist full agent tool chain in history for iterative continuation ([52cfa75](https://github.com/NorthlandPositronics/Cogtrix/commit/52cfa75fffcc6b5bed977385e3958307fdf0ccfa))
* pin dependency versions and update requirements.txt ([4e34d90](https://github.com/NorthlandPositronics/Cogtrix/commit/4e34d90bbbed2096811a3418711c3975339207a8))
* preserve partial results in history for iterative refinement ([1e659e8](https://github.com/NorthlandPositronics/Cogtrix/commit/1e659e8fd2f30a65dea519cb3c4fadaeffd6e67b))
* prevent context window overflow during long agent runs ([2fe81c5](https://github.com/NorthlandPositronics/Cogtrix/commit/2fe81c5d568fb7471e5d879c7148dbf429cb2b94))
* prevent deep_think from producing meta-descriptions instead of answers ([72e7c88](https://github.com/NorthlandPositronics/Cogtrix/commit/72e7c88533e4902e1301d26452c9f9f126ccdda4))
* prevent delegation of tool-intensive tasks to LLM-only delegates ([e7678c1](https://github.com/NorthlandPositronics/Cogtrix/commit/e7678c1d15ef0df7d153aaf77a51de92b9e1df0a))
* prevent tool loss on release and mode switch ([ae15979](https://github.com/NorthlandPositronics/Cogtrix/commit/ae1597906dfe12c5e31733b01a63e58309b927e1))
* remove contradictory 'do not invent facts' from synthesis categories ([52c4650](https://github.com/NorthlandPositronics/Cogtrix/commit/52c4650efa4d7ea31ed38b9478ad49e2b84cf504))
* remove cycle-interval trigger from context compression ([5979284](https://github.com/NorthlandPositronics/Cogtrix/commit/5979284219ca4acc7ea2d2097e2c2453ef2d91f1))
* remove redundant guard and update query_json docstring ([4615f3c](https://github.com/NorthlandPositronics/Cogtrix/commit/4615f3c834f08c5443ede202de144c365be40d06))
* remove skip condition and add actions permission ([afdcf33](https://github.com/NorthlandPositronics/Cogtrix/commit/afdcf33b092fe5daf50c20c02d008aa69be0e1d7))
* rename release-please config to expected filename ([922ff3c](https://github.com/NorthlandPositronics/Cogtrix/commit/922ff3cc2abfa245701391f92c985d016ca6f1e7))
* resolve 12 bugs across orchestration, deep_think, and file_ops ([c5a78c5](https://github.com/NorthlandPositronics/Cogtrix/commit/c5a78c507c6032402d2c789df4e55e405c637f10))
* resolve delegation bugs with allowed_models and multimodal responses ([aed0051](https://github.com/NorthlandPositronics/Cogtrix/commit/aed00519236beeb6d16b02c180f618ded565c60d))
* resolve error filtering, display, and resource cleanup bugs ([5007dd1](https://github.com/NorthlandPositronics/Cogtrix/commit/5007dd1454a22ae020ecdad19952941b08f5850b))
* resolve MCP tool name closure, collision, unpacking, restart registry, and timeout bugs ([0f39a2d](https://github.com/NorthlandPositronics/Cogtrix/commit/0f39a2d41481f7f80f2b3886ea2ee0d85e01c71e))
* resolve multiple bugs across core agent and tool modules ([64958ed](https://github.com/NorthlandPositronics/Cogtrix/commit/64958ed84af1e393a1256773cb6171a535d7e172))
* resolve slash_cmds state desync, Rich markup injection, MCP restart, and token trim bugs ([50f65e4](https://github.com/NorthlandPositronics/Cogtrix/commit/50f65e42b80734c041b466be5f8aa6fbc7e89387))
* resolve stale agent_executor and false positive in execution phase ([d86e598](https://github.com/NorthlandPositronics/Cogtrix/commit/d86e5987826912332690a27bd015b2a8ade57d3f))
* resolve token estimation, message mutation, and timestamp bugs ([c092dc6](https://github.com/NorthlandPositronics/Cogtrix/commit/c092dc62aaf794f9d899c20266607ebd3b86deb3))
* resolve type-safety issues and apply formatting ([8e9e9cc](https://github.com/NorthlandPositronics/Cogtrix/commit/8e9e9ccc0e8e0567f623c787404b13f55c99f69c))
* show spinner during /think slash command ([ff2fdef](https://github.com/NorthlandPositronics/Cogtrix/commit/ff2fdefe354dc3e86ea8fe26bc39621f62098107))
* spinner line not clearing between frames ([f4a63e5](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a63e5b57a922484412a5466601a64446058c33))
* stop injecting raw-JSON tool instructions into system prompt ([465a399](https://github.com/NorthlandPositronics/Cogtrix/commit/465a399581e77289788a12af33c01de5a2f5d71f))
* suppress primp native stderr leaking into spinner output ([55261b4](https://github.com/NorthlandPositronics/Cogtrix/commit/55261b414243575e52f58aeefa8df534f10abaf5))
* suppress pyright reportInvalidTypeForm on CogtrixState.messages ([d4b1909](https://github.com/NorthlandPositronics/Cogtrix/commit/d4b190943db8397652cfc393490cbb7fdd7a28b4))
* suppress ruff F401 for TYPE_CHECKING-only import in safety.py ([7c5af4c](https://github.com/NorthlandPositronics/Cogtrix/commit/7c5af4c51977bbd4dfe8e1835add8e428e376c9c))
* switch rollback, json multi-bracket paths, and doc corrections ([a00f5f3](https://github.com/NorthlandPositronics/Cogtrix/commit/a00f5f3d59250606cdeeea9f0c561880540f5c89))
* sync tools list after outer agent rebuild ([3876751](https://github.com/NorthlandPositronics/Cogtrix/commit/387675185a1e63a8ce709585cc852bd0968a0308))
* tool confirmation panel rendering and bare slash crash ([437c47b](https://github.com/NorthlandPositronics/Cogtrix/commit/437c47b4c61a3050f3c36493a8669215151a4194))
* use native ARM64 Linux runner for Docker build ([925d95e](https://github.com/NorthlandPositronics/Cogtrix/commit/925d95e7bd128886002cb8ed10d6d752e45840ac))
* use native runners for Docker builds, remove QEMU emulation ([b611365](https://github.com/NorthlandPositronics/Cogtrix/commit/b611365dca6fa67bb634643ea55fa134bb4abf2d))
* use temp files for GitHub API blob creation in version-bump ([082dfa0](https://github.com/NorthlandPositronics/Cogtrix/commit/082dfa089bd9e39230947374bdebe625a3cce2e0))
* validate relative paths against cwd to catch symlink traversal ([b459abc](https://github.com/NorthlandPositronics/Cogtrix/commit/b459abc04cdd75913261cf39875b38a70a52d64b))
* wire LLM into new memory managers on mode/session switch ([745dd9b](https://github.com/NorthlandPositronics/Cogtrix/commit/745dd9bd6b068c806b7cb4039b4e4676cc8defb7))


### Performance Improvements

* parallelize context compression LLM calls ([d565a47](https://github.com/NorthlandPositronics/Cogtrix/commit/d565a4784f99194db215d0a19c5d8fdfef350b22))


### Documentation

* add detailed Telegram assistant guide ([ed4fad0](https://github.com/NorthlandPositronics/Cogtrix/commit/ed4fad0a11c57d6a1b8a2d19173f6ae4ffbb5066))
* add YAML examples, document allowed_models, and add /delegate command ([07895e4](https://github.com/NorthlandPositronics/Cogtrix/commit/07895e4328f5fb6f943213b696ed8f59c6a4bf82))
* document assistant mode in README, ARCHITECTURE, and DEVELOPMENT ([a6ab6b5](https://github.com/NorthlandPositronics/Cogtrix/commit/a6ab6b5d3ff78c9a4c908b2f2d162f8df25e1601))
* document hybrid memory system across all guides ([0353a6a](https://github.com/NorthlandPositronics/Cogtrix/commit/0353a6a3808e882ba44bf5a265983e462d1e8119))
* document tool presets and on-demand auto-activation ([50b5372](https://github.com/NorthlandPositronics/Cogtrix/commit/50b537235a162e15be218f1f531c3f7dce519111))
* fix clone URL, tool counts, and missing telegram references ([4b0c6ec](https://github.com/NorthlandPositronics/Cogtrix/commit/4b0c6ec4d3c5e81e8d75c9e74099472f2ef3409a))
* fix stray backtick, outdated description, and wrong linter commands ([47be75a](https://github.com/NorthlandPositronics/Cogtrix/commit/47be75a3916ac964eea4419dfdad357898608cf4))
* improve documentation for newcomers and consistency ([99d2916](https://github.com/NorthlandPositronics/Cogtrix/commit/99d29161b757d83b2b509c829a84058e56c084b2))
* improve getting-started guide and document search provider setup ([2c15462](https://github.com/NorthlandPositronics/Cogtrix/commit/2c15462374d14174aa5cdb73d5d009d359cec510))
* improve getting-started guide and document search provider setup ([7d1c036](https://github.com/NorthlandPositronics/Cogtrix/commit/7d1c036f74212f6bde9cdb34038b55b5f8789ea0)), closes [#22](https://github.com/NorthlandPositronics/Cogtrix/issues/22)
* overhaul documentation for clarity and consistency ([10ad02c](https://github.com/NorthlandPositronics/Cogtrix/commit/10ad02cda8608a513eea7ae7e3d0ed1bad1917cf))
* polish README and fix misleading RAG directory example ([ae93500](https://github.com/NorthlandPositronics/Cogtrix/commit/ae93500649a79cb3f0ff5325fd099e19e9916425))
* standardize all config examples to YAML across guides ([d033bd7](https://github.com/NorthlandPositronics/Cogtrix/commit/d033bd7e28487fbe0f66fa8a217fbfe6b12d4638))
* update context compression docs for parallelization and model override ([b4132e2](https://github.com/NorthlandPositronics/Cogtrix/commit/b4132e2be6b8a3349d7dd044b80b66e083f5ffc8))
* update DEEPTHINK.md quotes to match tightened system prompt ([c02d3a0](https://github.com/NorthlandPositronics/Cogtrix/commit/c02d3a0a224f616f6a9b42f50c987ac5aff5682e))
* update documentation to reflect on-demand tool loading ([ea36469](https://github.com/NorthlandPositronics/Cogtrix/commit/ea3646931f394f11cf6b508ea501d01c9d8cab5f))
* update tool_instructions description in CONFIGURATION.md ([835af41](https://github.com/NorthlandPositronics/Cogtrix/commit/835af41d858471503b3fc723af315e9e69e68d02))

## [0.1.2](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.1...v0.1.2) (2026-02-16)


### Bug Fixes

* include search packages in Docker image ([6b6eb02](https://github.com/NorthlandPositronics/Cogtrix/commit/6b6eb02aa0bbbe74adf4e5bbb940ece4859e6501))


### Documentation

* improve getting-started guide and document search provider setup ([bbf113d](https://github.com/NorthlandPositronics/Cogtrix/commit/bbf113d00238074218b8be83176045c28a3ba3a7))
* improve getting-started guide and document search provider setup ([16f6a92](https://github.com/NorthlandPositronics/Cogtrix/commit/16f6a9256aadd889b1ee92e25f9ec14c9ab47c42)), closes [#22](https://github.com/NorthlandPositronics/Cogtrix/issues/22)

## [0.1.1](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.0...v0.1.1) (2026-02-16)


### Features

* activity indicator (spinner) during LLM processing ([0d9b4ac](https://github.com/NorthlandPositronics/Cogtrix/commit/0d9b4ac3a2ecf944bf65bf2bf40fde1a221980a9))
* auto version bump and release on merge to main ([f420524](https://github.com/NorthlandPositronics/Cogtrix/commit/f420524b6ada84a26de3abb1ab3d854c5ad21c0e))
* double spinner phrases, gradient color, trailing space ([7cc2067](https://github.com/NorthlandPositronics/Cogtrix/commit/7cc2067f8317e0743bb2227d47d40721bf806dcd))
* expand spinner to 80 phrases with humorous tech messages ([fd9d569](https://github.com/NorthlandPositronics/Cogtrix/commit/fd9d5699b84c9a1f13c481dd4d22c64450c0c3cb))
* hybrid /think pipeline with category-aware prompts ([6308a86](https://github.com/NorthlandPositronics/Cogtrix/commit/6308a86f28e58c8d225a79fb7b461c17b5b40fe5))
* publish multi-arch Docker image to GHCR ([d4325e4](https://github.com/NorthlandPositronics/Cogtrix/commit/d4325e41254a128e4588fbc173ce6cdee638e611))
* publish multi-arch Docker image to GHCR on main push ([1a28ea6](https://github.com/NorthlandPositronics/Cogtrix/commit/1a28ea60166a037305c5ba19bb1478bbaac3de0a))
* randomize spinner phrase order on each start ([0e0b107](https://github.com/NorthlandPositronics/Cogtrix/commit/0e0b10778577dc59a367c4a21c8fe522ccb5de69))
* replace custom version-bump with release-please ([5790406](https://github.com/NorthlandPositronics/Cogtrix/commit/57904061c3816c6770de27847c37585616b1ea0e))
* sequential intro phrases then random fun phrases ([1f5bbf9](https://github.com/NorthlandPositronics/Cogtrix/commit/1f5bbf9d207be12872cf719216b245a03cfe9b88))
* show spinner during forced deep_think invocations ([6d6679e](https://github.com/NorthlandPositronics/Cogtrix/commit/6d6679ebd0bd503839a81c239dab8a94b6f2803f))


### Bug Fixes

* add nosec markers for Bandit false positives (B311, B110) ([739449b](https://github.com/NorthlandPositronics/Cogtrix/commit/739449b6b63af7feb87dc708790792716b6805aa))
* category-aware Stage 2 framing for /think pipeline ([875f085](https://github.com/NorthlandPositronics/Cogtrix/commit/875f08541712645131531f38c699716e8d83e7ca))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([3e21dd1](https://github.com/NorthlandPositronics/Cogtrix/commit/3e21dd16079c971d9dcb97334cefa32db126d8b1))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([93ac20a](https://github.com/NorthlandPositronics/Cogtrix/commit/93ac20aa415dcfcfd22c1690370a6085d8c78b7a))
* classifier label normalization and tool output error filter ([f87821f](https://github.com/NorthlandPositronics/Cogtrix/commit/f87821fe0fc8e2779684506f3caafc7317d2f4dd))
* detect unconfigured provider before LLM init ([8425b93](https://github.com/NorthlandPositronics/Cogtrix/commit/8425b9331a72ba46262c7ba560fbdf25038a3027))
* graceful error messages for common configuration issues ([a9922e3](https://github.com/NorthlandPositronics/Cogtrix/commit/a9922e30e08042bcf967686207aaf9943e048e11))
* graceful error messages for common configuration issues ([00e8128](https://github.com/NorthlandPositronics/Cogtrix/commit/00e81281ac993d366a2231f8f81f243bad0c6e5c))
* pass force as boolean in version-bump ref update ([d11d4dd](https://github.com/NorthlandPositronics/Cogtrix/commit/d11d4dd70051f9a2589a91aa61bda0bc69a896e4))
* pause spinner during deep_think progress output ([a8dd126](https://github.com/NorthlandPositronics/Cogtrix/commit/a8dd126d29843065ac127e067b2306d0d6f498be))
* prevent deep_think from producing meta-descriptions instead of answers ([a3fd699](https://github.com/NorthlandPositronics/Cogtrix/commit/a3fd699104d9584163b4f6bcf8eb0349dc9c85be))
* remove contradictory 'do not invent facts' from synthesis categories ([1f2e65e](https://github.com/NorthlandPositronics/Cogtrix/commit/1f2e65eecfafed94cabb24a1ea0b42cc86ba2de6))
* remove skip condition and add actions permission ([4caffd0](https://github.com/NorthlandPositronics/Cogtrix/commit/4caffd02e10ab3c4a6eefd68ed1cad89d05ec6a2))
* rename release-please config to expected filename ([0bb0a0c](https://github.com/NorthlandPositronics/Cogtrix/commit/0bb0a0cc61ea8f964841647153a5120d707cae86))
* show spinner during /think slash command ([bbb5b56](https://github.com/NorthlandPositronics/Cogtrix/commit/bbb5b564f55fb65864fa7cd3c83805c1eb8a10f1))
* spinner line not clearing between frames ([2236607](https://github.com/NorthlandPositronics/Cogtrix/commit/2236607f35450dd05494e941b392d905f38675ef))
* suppress primp native stderr leaking into spinner output ([2fa5531](https://github.com/NorthlandPositronics/Cogtrix/commit/2fa55310181a2261d373323c0417ac1c266f5a2d))
* suppress ruff F401 for TYPE_CHECKING-only import in safety.py ([d9ec409](https://github.com/NorthlandPositronics/Cogtrix/commit/d9ec409294dbac181673129e7ca9c448e1d1a0df))
* use native ARM64 Linux runner for Docker build ([d95b140](https://github.com/NorthlandPositronics/Cogtrix/commit/d95b1408da9f96bfedcc741fc49c1827540b1393))
* use temp files for GitHub API blob creation in version-bump ([a2bc546](https://github.com/NorthlandPositronics/Cogtrix/commit/a2bc5464e9fd3afff44352dbcf05018ff0b8e6d8))
