"""Extractor for the web_search tool (ADR-0056 stage 4).

Consumes the fetcher's ``FetchOutcome`` list and runs each fetched
HTML body through ``trafilatura`` to produce Markdown-like extracted
text. The output is what the synthesiser (stage 5, PR-D) sees per
source.

Policies (ADR-0056 reliability table):

* Per-page extract timeout 2s. ``trafilatura`` on a 10MB+ HTML page
  can take seconds even when we cap the output at 3000 chars; the
  timeout caps wall-clock loss when that happens.
* Raw-text fallback if trafilatura returns nothing.
* Per-source content cap 3000 chars. Derived from the synthesis
  prompt's token budget — see ``docs/optional/prompts/web-search-synthesis.md``.
* Empty / JS-only pages get marked ``low-yield`` so the synthesiser
  can ignore them.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import multiprocessing
import os
import re
import signal
import threading
import time
from dataclasses import dataclass
from typing import Literal

import trafilatura

from cogtrix_core.tools._web_search_fetcher import FetchOutcome

log = logging.getLogger("cogtrix")

ExtractionStatus = Literal[
    "extracted",
    "extracted-raw-fallback",
    "extracted-truncated",
    "low-yield",
    "extraction-timeout",
    "extraction-error",
    "no-content-to-extract",
    "snippet-only",
    "skipped",
]

_DEFAULT_PER_PAGE_TIMEOUT_S = 2.0
_DEFAULT_CHAR_CAP = 3000
_LOW_YIELD_THRESHOLD_CHARS = 200

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Process-pool extractor — libxml2 (which trafilatura uses through lxml)
# holds parser state in process-global C variables that ARE NOT thread-
# safe across concurrent invocations from Python threads. Two prior
# generations of fix attempted thread-level synchronisation:
#
#   PR #1707  threading.Lock() (semaphore-of-1) — crash-safe but
#             serialises EVERY extract process-wide. Cost: cogtrix-
#             quality corpus replay A01 / T01 took 360 s vs ~45 s
#             pre-#1707 — #1716.
#
#   PR #1730  threading.Semaphore(2) — relaxed parallelism, but
#             hit a real-user deadlock on a 6-URL fan-out (cogtrix58
#             ISTA-Dubai query, 2026-05-21). Reverted in PR #1732.
#
# Neither thread-level approach is both safe AND fast. The structural
# fix is PROCESS-level isolation: each ``trafilatura.extract`` call
# runs in a worker subprocess that has its own libxml2 instance. No
# shared C state across workers → no thread-safety crash AND no
# serialisation tax. Cost is the one-time subprocess startup
# (~500 ms per worker, paid lazily on first use and amortised across
# subsequent calls).
#
# Pool size: 4 workers. Empirically enough to absorb a depth-6
# web_search burst (6 pages, 4 in parallel + 2 queued). Higher
# counts add memory pressure (each worker imports trafilatura ≈
# 100 MB resident) without proportional speed-up because pages are
# I/O-bound on the parent's fetch step.
#
# A stuck worker in this design does NOT block others — it removes
# itself from the pool until process exit, but the remaining
# workers serve new tasks. That's strictly better than the
# thread-level deadlock that PR #1730 hit.
_PROCESS_POOL_MAX_WORKERS = 4

# ``max_tasks_per_child`` recycles workers after N successful tasks so an
# adversarial page that wedges trafilatura (forge audit C4, 2026-05-23)
# eventually loses its grip on a slot. 50 is conservative: an extract is
# typically <200 ms, so a fully-utilised worker turns over every ~10 s.
_PROCESS_POOL_MAX_TASKS_PER_CHILD = 50

# Pool rebuild policy (C4): when ``asyncio.wait_for`` times out, the
# subprocess is still running indefinitely — ``concurrent.futures`` cannot
# kill a worker mid-task. Track timeouts; once half the pool's effective
# capacity is suspected stuck, rebuild from scratch.
_STUCK_REBUILD_THRESHOLD = _PROCESS_POOL_MAX_WORKERS // 2 + 1  # =3 for max_workers=4

# Minimum interval between same-PID rebuilds (forge audit B3, 2026-05-23).
# Without rate-limiting, an attacker firing 3 stuck pages per cycle would
# trigger a pool rebuild every cycle. Combined with the ``cancel_futures``
# limitation (it cancels QUEUED futures only, not RUNNING workers wedged
# in libxml2) this would oscillate the pool indefinitely and leak workers
# at every rebuild — 4 fresh × N rebuilds × ~100 MB resident each.
# A 60s floor + SIGKILL of old workers (see ``_build_process_pool``) bounds
# the worst-case leak: even sustained attack costs 4 leaked workers per
# minute, not per request.
_MIN_REBUILD_INTERVAL_S = 60.0

_process_pool: concurrent.futures.ProcessPoolExecutor | None = None
_process_pool_pid: int | None = None  # PID that built ``_process_pool`` (C3)
_process_pool_stuck_count: int = 0
_process_pool_last_rebuild_at: float = 0.0  # monotonic time of last rebuild
_process_pool_lock = threading.Lock()


# Test override hook: when set (typically by a pytest fixture) the
# extractor uses this executor instead of spawning the production
# process pool. Lets tests run in-process with a thread pool so
# monkey-patches of ``trafilatura.extract`` are visible to workers.
# Production code never sets this — it stays None and the lazy
# process pool is used.
_executor_override: concurrent.futures.Executor | None = None


def _build_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    """Build a fresh process pool using the ``spawn`` start method.

    Forced ``spawn`` (forge audit C3, 2026-05-23): the default on Linux is
    ``fork``, which inherits the parent's libxml2 state. Under
    ``uvicorn --workers N`` the parent's pool would be half-inherited into
    each fork in an unusable state — submissions would hang forever.
    ``spawn`` gives every worker a clean Python interpreter with its own
    libxml2 instance.
    """
    ctx = multiprocessing.get_context("spawn")
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=_PROCESS_POOL_MAX_WORKERS,
        max_tasks_per_child=_PROCESS_POOL_MAX_TASKS_PER_CHILD,
        mp_context=ctx,
    )


def _kill_pool_workers(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    """SIGKILL every worker process owned by *pool* (forge audit B3, 2026-05-23).

    ``concurrent.futures.ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)``
    only cancels QUEUED futures — running workers wedged inside a C
    extension (libxml2 quadratic regex / billion-laughs HTML, etc.) keep
    running until process exit. Under sustained adversarial load this
    leaks ~100 MB resident per stuck worker, every rebuild. Sending
    SIGKILL to each worker PID bounds the leak: workers die immediately
    regardless of what they were doing in C land.

    Reaches into ``pool._processes`` (private attribute, available on
    CPython 3.7+; the dict maps pid → ``multiprocessing.Process``).
    Best-effort: if the attribute is missing or workers are already
    gone, swallow.
    """
    processes = getattr(pool, "_processes", None)
    if not processes:
        return
    for proc in list(processes.values()):
        try:
            pid = getattr(proc, "pid", None)
            if pid and proc.is_alive():
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError, ValueError):
            # Worker already died, or pid is None; nothing to do.
            continue
        except Exception:  # noqa: BLE001
            # Conservative — never let cleanup raise.
            continue


def _get_process_pool() -> concurrent.futures.Executor:
    """Lazy-init the trafilatura process pool.

    Returns the test-mode override executor if one is set; otherwise
    rebuilds the module-level ``ProcessPoolExecutor`` whenever:

    1. It hasn't been built yet (first use).
    2. The current PID differs from the PID that built it (forge audit C3
       — the parent's pool is not safe to inherit through ``os.fork()``;
       uvicorn multi-worker / gunicorn pre-fork would otherwise hang on
       every extraction).
    3. The accumulated ``_process_pool_stuck_count`` from timed-out workers
       reaches ``_STUCK_REBUILD_THRESHOLD`` (forge audit C4 — an adversarial
       page that wedges trafilatura would otherwise permanently consume a
       worker slot; rebuilding swaps the whole pool out for a fresh one).
    """
    if _executor_override is not None:
        return _executor_override

    global _process_pool, _process_pool_pid, _process_pool_stuck_count
    global _process_pool_last_rebuild_at
    current_pid = os.getpid()
    now = time.monotonic()

    # Determine whether a rebuild is warranted. The stuck-counter threshold
    # is rate-limited by ``_MIN_REBUILD_INTERVAL_S`` (forge audit B3,
    # 2026-05-23) to prevent oscillation under sustained adversarial load:
    # an attacker firing 3 stuck pages per cycle would otherwise rebuild
    # every cycle, leaking workers each time. The fork-PID-changed and
    # never-built paths bypass the rate limit because they have no choice
    # — the existing pool is unusable.
    need_first_build = _process_pool is None
    need_fork_rebuild = _process_pool is not None and _process_pool_pid != current_pid
    need_stuck_rebuild = (
        _process_pool is not None
        and _process_pool_pid == current_pid
        and _process_pool_stuck_count >= _STUCK_REBUILD_THRESHOLD
        and (now - _process_pool_last_rebuild_at) >= _MIN_REBUILD_INTERVAL_S
    )

    if not (need_first_build or need_fork_rebuild or need_stuck_rebuild):
        assert _process_pool is not None  # else need_first_build would be True
        return _process_pool

    with _process_pool_lock:
        # Re-evaluate under lock to defeat the double-checked race.
        now = time.monotonic()
        need_first_build = _process_pool is None
        need_fork_rebuild = _process_pool is not None and _process_pool_pid != current_pid
        need_stuck_rebuild = (
            _process_pool is not None
            and _process_pool_pid == current_pid
            and _process_pool_stuck_count >= _STUCK_REBUILD_THRESHOLD
            and (now - _process_pool_last_rebuild_at) >= _MIN_REBUILD_INTERVAL_S
        )
        if not (need_first_build or need_fork_rebuild or need_stuck_rebuild):
            assert _process_pool is not None  # else need_first_build would be True
            return _process_pool

        old_pool = _process_pool
        if old_pool is not None and _process_pool_pid == current_pid:
            # Same-PID rebuild path (stuck-worker recycle). SIGKILL the old
            # workers BEFORE shutdown so a wedged libxml2 worker can't
            # survive the rebuild and leak memory (forge audit B3).
            _kill_pool_workers(old_pool)
            try:
                old_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
        _process_pool = _build_process_pool()
        _process_pool_pid = current_pid
        _process_pool_stuck_count = 0
        _process_pool_last_rebuild_at = now
    return _process_pool


def _record_extraction_timeout() -> None:
    """Increment the stuck-worker counter (forge audit C4).

    Called from ``extract`` when ``asyncio.wait_for`` fires on a worker
    future. The next ``_get_process_pool`` call after the threshold rebuilds
    the pool from scratch.
    """
    global _process_pool_stuck_count
    with _process_pool_lock:
        _process_pool_stuck_count += 1


@atexit.register
def _shutdown_process_pool() -> None:
    """Best-effort pool shutdown on interpreter exit.

    ``wait=False`` because joining a worker stuck in libxml2 costs more
    than the leak — the OS reclaims the subprocesses on parent exit.
    """
    global _process_pool
    pool = _process_pool
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass


def _set_executor_override(executor: concurrent.futures.Executor | None) -> None:
    """Test hook — install an alternate executor (typically a
    ``ThreadPoolExecutor``) so pytest monkey-patches of
    ``trafilatura.extract`` are visible inside workers. Pass ``None``
    to clear the override and restore the production process pool.
    """
    global _executor_override
    _executor_override = executor


# Kept for backward compatibility with existing tests that import
# the symbol; the pool now provides isolation, so this lock is a
# no-op stand-in. Tests that exercise the lock release contract
# still pass because acquire()/release() on an unused lock is
# defined.
_LXML_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExtractedSource:
    """One stage-4 output per top-K ranked URL.

    ``extracted_text`` is None when no extraction was possible
    (snippet-only fetch outcome, skipped, or hard extraction error).
    ``status`` tells the formatter which annotation to attach.
    """

    fetch_outcome: FetchOutcome
    extracted_text: str | None
    status: ExtractionStatus


async def extract(
    fetched: list[FetchOutcome],
    *,
    per_page_timeout_s: float = _DEFAULT_PER_PAGE_TIMEOUT_S,
    char_cap: int = _DEFAULT_CHAR_CAP,
) -> list[ExtractedSource]:
    """Run trafilatura on each fetched body, **in parallel** via the
    module-level process pool.

    Order is preserved — the i-th input maps to the i-th output.
    Snippet-only / skipped outcomes pass through unchanged (we don't
    fabricate extracted text from a snippet).

    libxml2 thread-safety is sidestepped at the process level: each
    extract runs in a worker subprocess that has its own libxml2
    instance (see ``_get_process_pool``). Fan-out via
    ``asyncio.gather`` is preserved; the per-page timeout
    (``per_page_timeout_s``) caps the wait on each subprocess
    result.
    """
    if not fetched:
        return []

    async def _one(outcome: FetchOutcome) -> ExtractedSource:
        # Outcomes without a fetched body pass through.
        if outcome.status == "snippet-only":
            return ExtractedSource(outcome, None, "snippet-only")
        if outcome.status == "skipped":
            return ExtractedSource(outcome, None, "skipped")
        if outcome.fetch_result is None or not outcome.fetch_result.content:
            return ExtractedSource(outcome, None, "no-content-to-extract")

        body = outcome.fetch_result.content
        encoding = outcome.fetch_result.encoding or "utf-8"
        try:
            html_text = body.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            html_text = body.decode("utf-8", errors="replace")

        try:
            # Submit to the process pool: each worker has its own
            # libxml2, so concurrent submissions don't race. The
            # per-page timeout fires from the asyncio side; if it
            # trips, the worker subprocess keeps running but the
            # parent's future is cancelled. A stuck worker drops
            # out of the pool's effective capacity but doesn't
            # block other workers — strictly better than the
            # thread-level deadlock pattern PR #1730 hit.
            pool = _get_process_pool()
            future = pool.submit(_extract_sync, html_text, char_cap)
            extracted, fallback_used = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=per_page_timeout_s,
            )
        except TimeoutError:
            # The subprocess is still running; ``concurrent.futures`` cannot
            # kill it mid-task. Bump the stuck counter so the next
            # ``_get_process_pool`` call rebuilds the pool once enough
            # workers have been wedged (forge audit C4).
            _record_extraction_timeout()
            return ExtractedSource(outcome, None, "extraction-timeout")
        except BaseException as exc:  # noqa: BLE001
            log.debug("extraction raised for %s: %s", outcome.ranked.canonical_url, exc)
            return ExtractedSource(outcome, None, "extraction-error")

        if not extracted or len(extracted) < _LOW_YIELD_THRESHOLD_CHARS:
            return ExtractedSource(outcome, extracted or None, "low-yield")

        if len(extracted) > char_cap:
            extracted = extracted[:char_cap].rstrip() + "…"
            return ExtractedSource(outcome, extracted, "extracted-truncated")

        if fallback_used:
            return ExtractedSource(outcome, extracted, "extracted-raw-fallback")

        return ExtractedSource(outcome, extracted, "extracted")

    # Stage-4 work is CPU-bound (trafilatura). Run via asyncio.gather
    # which fans out across the process pool. No outer deadline here
    # — each call has its own per-page timeout.
    return await asyncio.gather(*(_one(o) for o in fetched))


def _extract_sync_locked(html_text: str, char_cap: int) -> tuple[str | None, bool]:
    """Backward-compat alias for ``_extract_sync``.

    The thread-lock semantics this name implied are gone — extraction
    now happens in a worker subprocess (see ``_get_process_pool``).
    The symbol is kept so existing test imports continue to work and
    any external callers find a no-op shim.
    """
    return _extract_sync(html_text, char_cap)


def _extract_sync(html_text: str, char_cap: int) -> tuple[str | None, bool]:
    """Trafilatura extract, with raw-text fallback.

    Returns ``(text, fallback_used)``. ``text`` is None when both
    trafilatura and the raw-text fallback yield nothing.
    """
    try:
        extracted = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=True,
            favor_recall=False,
            output_format="markdown",
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("trafilatura.extract raised: %s", exc)
        extracted = None

    if extracted:
        return extracted, False

    # Fallback — strip tags, collapse whitespace, return first
    # 1000 chars. Better than nothing for JS-shell pages that have
    # boilerplate-only HTML but still some text we can glean.
    raw = _HTML_TAG_RE.sub(" ", html_text)
    raw = _WHITESPACE_RE.sub(" ", raw).strip()
    if not raw:
        return None, True
    # Trim conservatively at the fallback path — raw text is noisier
    # than trafilatura's output, so 1000 chars vs the configured cap.
    return raw[: min(1000, char_cap)], True


__all__ = [
    "ExtractedSource",
    "ExtractionStatus",
    "extract",
]
