# Concurrency Policy

**Status:** Adopted 2026-05-29 (#1677). Migration to the centralized helper is incremental — see "Migration roadmap" below.

This document is the canonical reference for `ThreadPoolExecutor` usage in Cogtrix. Every new call site MUST conform to one of the patterns described here. Existing sites that pre-date this policy are catalogued and migrated incrementally.

---

## TL;DR

Use **one of three things**, in this preference order:

1. **`asyncio` primitives** — if the surrounding code is already `async`. Threads are not the right tool for async-native code paths.
2. **`src.concurrency.invoke_with_timeout(fn, *args, timeout=..., **kwargs)`** — the canonical replacement for the per-call `ThreadPoolExecutor(max_workers=1)` + `submit` + `result(timeout=...)` + `shutdown(wait=False)` dance. Uses a shared bounded pool. **Use this for any "call this with a hard timeout" pattern outside the agent-turn hot path.**
3. **One of the three pre-existing module-level shared pools**: `_get_tool_executor()` (parallel tool dispatch, 8 workers), `_get_llm_executor()` (LLM invokes on the turn hot path, 4 workers), `_get_compression_pool()` (background context compression, 4 workers). Reach for these only when you are operating inside the subsystem they belong to.

**Do NOT** spawn `ThreadPoolExecutor(max_workers=1)` directly. Every site that did this in the pre-#1677 codebase reproduced the same five-line dance and the same "do NOT use `with`" warning comment. The proliferation is the failure mode this policy exists to eliminate.

---

## Why this exists

The pre-#1677 audit found **20 distinct executor usage sites** across `src/`. Roughly:

| Pattern | Sites | Health |
|---|---|---|
| **Module-level shared bounded pools** | 3 (`_TOOL_EXECUTOR`, `_LLM_EXECUTOR`, `_COMPRESSION_POOL`) | Correct — keep |
| **Per-call `ThreadPoolExecutor(max_workers=1)` for timeout** | ~15 | Proliferation — migrate |
| **Parallel delegation pool (sized at runtime)** | 1 (`delegate.py:1308` — `min(len(tasks), 10)`) | Legitimate — keep, document |
| **Process pool (libxml2 isolation)** | 1 (`_web_search_extractor.py`) | Out of scope — different concurrency model |

The ~15 per-call sites all looked like:

```python
pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
try:
    future = pool.submit(fn, ...)
    return future.result(timeout=N)
except concurrent.futures.TimeoutError:
    future.cancel()
    pool.shutdown(wait=False)
    raise ...
finally:
    pool.shutdown(wait=False)
```

…with five nearly-identical copies of the "do NOT use `with`" warning comment scattered across them. Each copy already drifted in spelling and comment placement (#1133, #1134 closure comments noted this explicitly).

The **`with` footgun**: `ThreadPoolExecutor.__exit__` calls `shutdown(wait=True)`. If the submitted callable is hung inside a network or I/O syscall, `wait=True` blocks the calling thread until the syscall returns. For LLM invocations against an unresponsive provider this defeats the entire timeout machinery.

The shared-pool approach in `src.concurrency.invoke_with_timeout` sidesteps this by never owning a pool at the call site — the pool is module-level and atexit-shut-down with `wait=False, cancel_futures=True`.

---

## The four pools

### 1. `src.concurrency._INVOKE_POOL` — 8 workers, the new general-purpose pool

- **Purpose:** the standard timeout-bounded `fn(*args, **kwargs) → result` pattern outside the agent-turn hot path.
- **Access:** never reach into `_INVOKE_POOL` directly. Use `src.concurrency.invoke_with_timeout()`.
- **Sizing rationale:** absorbs concurrent setup-wizard + reflection + delegate + memory-summarization timeout-bounded calls without forcing them onto the hot-path `_LLM_EXECUTOR`.
- **Shutdown:** `atexit` register → `shutdown(wait=False, cancel_futures=True)`.

### 2. `src.orchestration.graph_runtime._TOOL_EXECUTOR` — 8 workers, parallel tool dispatch

- **Purpose:** fan-out parallel execution of tool calls within a single agent turn (`process_tools` node).
- **Access:** `_get_tool_executor()`.
- **Sizing rationale:** typical depth-6 web_search burst + delegate_parallel + http_get fan-outs.
- **Do not use for:** anything outside the agent-turn tool dispatch.

### 3. `src.orchestration.graph_runtime._LLM_EXECUTOR` — 4 workers, LLM invokes on the turn hot path

- **Purpose:** the LLM invoke inside `call_model` with retry + timeout. Backs the `_invoke_with_timeout` closure in `graph.py`.
- **Access:** `_get_llm_executor()`.
- **Sizing rationale:** small bounded pool — agent turns are typically sequential per session; concurrency comes from multiple sessions on the API surface.
- **Do not use for:** ancillary LLM calls outside the turn loop (setup wizard, reflection, delegate-target invocations) — those use `invoke_with_timeout`.

### 4. `src.orchestration.compression._COMPRESSION_POOL` — 4 workers, background context compression

- **Purpose:** the per-tool-message compression LLM calls during background warm-up.
- **Access:** `_get_compression_pool()`.
- **Sizing rationale:** background workload — bounded so a context burst doesn't starve foreground turns.
- **Do not use for:** anything outside `src/orchestration/compression.py`.

---

## Inventory (audit findings)

Inventory taken 2026-05-29 against `release/next`. Each row is a current call site. Status:
- **shared-pool** — already uses a module-level bounded pool (the good pattern).
- **per-call** — spawns a per-call `ThreadPoolExecutor(max_workers=1)` (proliferation, migration target).
- **runtime-sized** — sizes pool from task count (legitimate use case, document).
- **process-pool** — uses `ProcessPoolExecutor` instead (different concurrency model, out of scope).

| File | Line | Pattern | Status |
|---|---|---|---|
| `src/orchestration/graph_runtime.py` | 56, 75 | `_TOOL_EXECUTOR`, `_LLM_EXECUTOR` | shared-pool |
| `src/orchestration/compression.py` | 61 | `_COMPRESSION_POOL` | shared-pool |
| `src/orchestration/compression.py` | 126, 471 | per-call timeout | **migrated in #1903** |
| `src/orchestration/graph.py` | 960 | `_invoke_with_timeout` closure using `_LLM_EXECUTOR` | shared-pool |
| `src/orchestration/reflection_delegate.py` | 103 | per-call timeout | **migrated in #1903** |
| `src/orchestration/phases.py` | 264 | per-call timeout (force_delegation decomposer) | **migrated in #1903** |
| `src/orchestration/intent.py` | 1152, 1671 | per-call timeout | **migrated in #1903** |
| `src/orchestration/nodes/process_tools.py` | 199 | uses `_get_tool_executor` | shared-pool |
| `src/memory/manager.py` | 124 | `_SUMMARIZATION_POOL` | shared-pool |
| `src/memory/distillation.py` | 95 | per-call timeout | **migrated in #1903** |
| `src/memory/tier_cache.py` | 223 | per-call timeout | **migrated in #1903** |
| `src/memory/summarizer.py` | 310 | per-call timeout | **migrated in #1903** |
| `src/setup_wizard.py` | 158 | per-call timeout | **migrated in #1677** |
| `src/tools/delegate.py` | 991, 1159 | per-call timeout | **migrated in #1903** |
| `src/tools/delegate.py` | 1308 | `ThreadPoolExecutor(max_workers=min(len(tasks), 10))` | runtime-sized — keep |
| `src/tools/cron_tools.py` | 414 | per-call timeout | **migrated in #1903** |
| `src/tools/generate_tests.py` | 263 | per-call timeout | **migrated in #1903** |
| `src/tools/self_improve.py` | 389 | per-call timeout | **migrated in #1903** |
| `src/tools/web_search.py` | 186, 848 | asyncio-escape | **migrated in #1903** |
| `src/tools/_web_search_extractor.py` | — | `ProcessPoolExecutor` | process-pool — out of scope |
| `src/mcp_client.py` | (varies) | misc | reviewed; see below |

Total per-call sites pre-#1677: **15** (counting `compression.py` twice for its two distinct call sites). The first migrated in #1677 as proof of pattern; the remaining **14 migrated under the #1903 umbrella between 2026-05-29 and 2026-05-30** — see the per-row `**migrated in #1903**` markers above.

---

## Migration roadmap (historical)

> **Status: complete (2026-05-30).** Every row in the inventory above is either `shared-pool`, `runtime-sized — keep`, `process-pool — out of scope`, `**migrated in #1677**`, or `**migrated in #1903**`. Zero rows remain in the `per-call → migrate` state. The mechanical transformation documented below is preserved as reference for new code that the policy applies to.

Each per-call site was migrated in its own scoped PR. The mechanical transformation was the same for every site:

**Before:**
```python
pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
try:
    future = pool.submit(fn, *args)
    return future.result(timeout=N)
except concurrent.futures.TimeoutError:
    future.cancel()
    raise RuntimeError("…") from None
finally:
    pool.shutdown(wait=False)
```

**After:**
```python
from src.concurrency import invoke_with_timeout

try:
    return invoke_with_timeout(fn, *args, timeout=N)
except TimeoutError as exc:
    raise RuntimeError("…") from exc
```

The "do NOT use `with`" comment is removed — the helper documents the rationale in one place (this doc + `src/concurrency.py` module docstring).

### Follow-up tickets (closed)

The migration was tracked under umbrella issue #1903 and landed as one PR per site (14 PRs total: #1906–#1917, plus the #1677 proof-of-pattern). All closed.

---

## Anti-patterns

These shapes are **forbidden** in new code and should be migrated where found:

1. **`with ThreadPoolExecutor(...) as pool:`** for any callable that may hang. The context-manager exit blocks on `shutdown(wait=True)`.
2. **Per-call `ThreadPoolExecutor(max_workers=1)`.** Use `invoke_with_timeout` instead.
3. **Unbounded pool sizing.** `ThreadPoolExecutor()` with no `max_workers` uses `os.cpu_count() * 5` — fine for short bursts but unbounded under sustained load.
4. **Re-implementing the timeout dance.** Five copies pre-#1677 already drifted. Centralize.

---

## Open questions, not addressed by this policy

- **AsyncIO-native rewrite.** Several call sites that use threads today could be `async`-native. Migrating to `asyncio.wait_for` + native async LLM clients is a bigger architectural shift; this policy does not address it. If a site is being rewritten anyway, prefer the async path.
- **Per-session worker isolation.** Today the shared pools serve all sessions. If a single session monopolises pool capacity, other sessions queue. Per-session worker accounting is a separate design question (#TBD).
- **MCP client thread management.** `mcp_client.py` uses `concurrent.futures` but its threading model is shaped by the MCP protocol itself, not by the general concurrency-policy concern. Reviewed but excluded from this audit.

---

## Decision review

Revisit this policy:
- When a fourth shared pool is proposed.
- When the migration roadmap is complete (`per-call` count = 0 in the inventory above).
- When the `_INVOKE_POOL_WORKERS = 8` sizing demonstrably underprovisions or overprovisions on a production workload.

---

## References

- #1677 — this audit + policy.
- `src/orchestration/graph_runtime.py:23-28` — original deferral note that named `_invoke_with_timeout` as the future-PR target. This policy extracts it.
- `src/orchestration/compression.py:55-65`, `src/memory/manager.py:119-127` — pre-existing shared-pool patterns this policy formalises.
- PR #1109 (compress_to_tier), #1129 (generate_summary), #1709 (#1706 datetime tool), #1893 (#1175 ContradictionDetector) — recent PRs that touched timeout-bounded patterns and pre-date this policy.
