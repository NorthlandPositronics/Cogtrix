"""Centralized concurrency helpers — shared bounded pool + safe ``invoke_with_timeout``.

The codebase spawned ``ThreadPoolExecutor(max_workers=1)`` per call site for
LLM-invoke-with-timeout patterns (audited under #1677 — fifteen distinct sites
all reproducing the same five-line dance with the same "do NOT use ``with``"
warning comment). The proliferation creates:

- Unbounded thread growth under load (every concurrent call materialises a
  fresh executor).
- Five copies of the ``shutdown(wait=False)`` rationale that have already
  drifted in spelling and comment placement.
- No single place to tune timeout policy or worker limits — every adjustment
  is a multi-site patch.

This module provides the canonical implementation. New code SHOULD call
:func:`invoke_with_timeout`; existing call sites are migrated incrementally
per :doc:`docs/architecture/CONCURRENCY.md`.

## The ``with`` footgun (documented once, here)

``concurrent.futures.ThreadPoolExecutor.__exit__`` calls
``shutdown(wait=True)``.  If the submitted callable is hung inside a network
or I/O syscall, ``wait=True`` will block the entire calling thread until
that syscall returns.  For LLM invocations against an unresponsive provider
this is exactly the failure mode we are trying to avoid — the timeout
machinery never gets a chance to fire because the context-manager exit is
holding the caller hostage.

The shared executor here works around this by *never* leaving the pool: the
pool is module-level, lives for the process lifetime, and is shut down via
:func:`atexit` with ``wait=False, cancel_futures=True``.  Individual call
sites get timeout-bounded futures from :func:`invoke_with_timeout` and
never own a pool.

## Pool sizing

``_INVOKE_POOL_WORKERS = 8`` is sized to absorb a small-to-medium burst of
concurrent timeout-bounded calls (LLM invokes during setup wizard +
reflection + delegate + memory summarization paths can all overlap on a
busy session) without blocking new submissions when the orchestration
graph's own ``_LLM_EXECUTOR`` (4 workers) is fully utilised.  These two
pools are intentionally separate: ``_LLM_EXECUTOR`` is on the agent-turn
hot path and tunes for steady-state throughput; the invoke pool here is
for ancillary timeout-bounded calls outside the turn loop.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any

#: Worker cap for the shared invocation pool. Sized to absorb a setup-wizard
#: + reflection + delegate + memory-summarization burst without forcing
#: callers into the orchestration graph's ``_LLM_EXECUTOR`` (4 workers,
#: separate concern — agent-turn hot path). See module docstring.
_INVOKE_POOL_WORKERS = 8

_INVOKE_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_INVOKE_POOL_LOCK = threading.Lock()


def _get_invoke_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level invocation pool, creating it on first use.

    Lazy + double-checked-locking init mirrors the pattern in
    ``src/orchestration/graph_runtime.py`` and ``src/orchestration/compression.py``.
    Registered for shutdown at interpreter exit with ``wait=False`` so a
    hung worker thread cannot block process exit (``cancel_futures=True``
    additionally cancels queued-but-not-started work).
    """
    global _INVOKE_POOL
    if _INVOKE_POOL is None:
        with _INVOKE_POOL_LOCK:
            if _INVOKE_POOL is None:
                _INVOKE_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_INVOKE_POOL_WORKERS,
                    thread_name_prefix="invoke",
                )
                atexit.register(_INVOKE_POOL.shutdown, wait=False, cancel_futures=True)
    return _INVOKE_POOL


def invoke_with_timeout[T](
    fn: Callable[..., T],
    *args: Any,
    timeout: float,
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on the shared invocation pool with a hard timeout.

    The standard replacement for the ``ThreadPoolExecutor(max_workers=1)`` +
    ``submit`` + ``result(timeout=...)`` + ``shutdown(wait=False)`` dance that
    appeared in ~15 sites prior to #1677.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to ``fn``.
        timeout: Hard timeout in seconds.  Must be > 0.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        TimeoutError: If ``fn`` does not return within ``timeout`` seconds.
            The submitted future is cancelled before re-raising; the pool
            slot is reclaimed when the underlying OS thread completes
            (which may be later if the call is stuck inside a C-extension
            or syscall).
        Exception: Any exception raised by ``fn`` is re-raised unchanged.

    Concurrency notes:
        - ``timeout`` is wall-clock and includes time spent waiting for a
          free pool slot when the pool is saturated.  Callers needing
          "timeout after the call starts running" semantics should add
          their own backpressure handling upstream.
        - The pool is bounded at ``_INVOKE_POOL_WORKERS``; submissions
          beyond that limit queue.  Under sustained saturation the
          ``result(timeout=...)`` will fire before the call ever
          executes — which is the correct failure mode (caller sees a
          TimeoutError, not a hung future).
        - On Python 3.13+ ``future.cancel()`` after a TimeoutError can
          succeed if the work has not started yet; the OS thread is left
          to drain if it has.  Either way the pool slot reclaims itself
          when the thread returns.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    pool = _get_invoke_pool()
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"invoke_with_timeout: {getattr(fn, '__name__', repr(fn))!r} "
            f"did not return within {timeout}s"
        ) from exc


__all__ = ["invoke_with_timeout"]
