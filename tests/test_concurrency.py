"""Tests for cogtrix_core/concurrency.py — the centralized invoke_with_timeout helper.

The helper is the canonical replacement for the per-call
ThreadPoolExecutor(max_workers=1) + submit + result(timeout=...) +
shutdown(wait=False) dance that proliferated across ~15 sites pre-#1677.

These tests pin:

* The success path returns the callable's result unchanged.
* Args + kwargs forward correctly (since the helper signature
  interleaves a keyword-only ``timeout`` between them).
* A genuinely-slow callable raises ``TimeoutError`` (not
  ``concurrent.futures.TimeoutError`` — the helper re-raises the
  builtin so callers don't import ``concurrent.futures`` for the
  except clause).
* Exception propagation: a callable that raises forwards the
  exception unchanged.
* The ``timeout`` parameter rejects non-positive values at the
  boundary so a caller can't silently end up with no protection.
* Repeated calls share the same pool (lazy-init contract).

The "no thread leakage after timeout" property is structural — the
pool is module-level and bounded — and is asserted via the shared-pool
identity test rather than via thread counting (which would be racy).
"""

from __future__ import annotations

import time

import pytest

from cogtrix_core.concurrency import _get_invoke_pool, invoke_with_timeout

# ---------------------------------------------------------------------------
# Success / argument forwarding
# ---------------------------------------------------------------------------


class TestInvokeWithTimeoutSuccess:
    def test_returns_callable_result(self) -> None:
        result = invoke_with_timeout(lambda: 42, timeout=1.0)
        assert result == 42

    def test_forwards_positional_args(self) -> None:
        result = invoke_with_timeout(lambda a, b: a + b, 3, 4, timeout=1.0)
        assert result == 7

    def test_forwards_keyword_args(self) -> None:
        result = invoke_with_timeout(
            lambda a, *, multiplier: a * multiplier, 5, timeout=1.0, multiplier=3
        )
        assert result == 15

    def test_returns_none_when_callable_returns_none(self) -> None:
        result = invoke_with_timeout(lambda: None, timeout=1.0)
        assert result is None

    def test_returns_falsy_value_unmolested(self) -> None:
        """A returned ``0`` / ``False`` / ``""`` must not be confused with
        absence — the helper must not fall back to ``None`` on falsy
        results."""
        assert invoke_with_timeout(lambda: 0, timeout=1.0) == 0
        assert invoke_with_timeout(lambda: False, timeout=1.0) is False
        assert invoke_with_timeout(lambda: "", timeout=1.0) == ""


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


class TestInvokeWithTimeoutTimeout:
    def test_slow_callable_raises_timeout_error(self) -> None:
        """A callable that runs longer than the timeout must raise
        ``TimeoutError`` (the builtin, not ``concurrent.futures.TimeoutError``)
        so callers don't have to import the futures module for their
        except clause."""

        def slow() -> int:
            time.sleep(2.0)
            return 99

        with pytest.raises(TimeoutError):
            invoke_with_timeout(slow, timeout=0.1)

    def test_timeout_error_mentions_callable_name(self) -> None:
        """The error message must name the callable so an ops log
        without a stack trace can still be triaged."""

        def my_specific_fn() -> int:
            time.sleep(2.0)
            return 0

        with pytest.raises(TimeoutError, match="my_specific_fn"):
            invoke_with_timeout(my_specific_fn, timeout=0.1)

    def test_timeout_error_mentions_timeout_value(self) -> None:
        def slow() -> int:
            time.sleep(2.0)
            return 0

        with pytest.raises(TimeoutError, match="0.1"):
            invoke_with_timeout(slow, timeout=0.1)

    def test_timeout_error_chains_underlying_futures_timeout(self) -> None:
        """``raise ... from exc`` preserves the underlying
        ``concurrent.futures.TimeoutError`` as ``__cause__`` so deep
        debugging still has access to the original."""
        import concurrent.futures

        def slow() -> int:
            time.sleep(2.0)
            return 0

        try:
            invoke_with_timeout(slow, timeout=0.1)
        except TimeoutError as exc:
            assert isinstance(exc.__cause__, concurrent.futures.TimeoutError)
        else:
            pytest.fail("expected TimeoutError")


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestInvokeWithTimeoutExceptionPropagation:
    def test_runtime_error_propagates(self) -> None:
        def boom() -> int:
            raise RuntimeError("the callable raised")

        with pytest.raises(RuntimeError, match="the callable raised"):
            invoke_with_timeout(boom, timeout=1.0)

    def test_value_error_propagates(self) -> None:
        def boom() -> int:
            raise ValueError("specific")

        with pytest.raises(ValueError, match="specific"):
            invoke_with_timeout(boom, timeout=1.0)

    def test_custom_exception_propagates(self) -> None:
        class _MyError(Exception):
            pass

        def boom() -> int:
            raise _MyError("custom")

        with pytest.raises(_MyError, match="custom"):
            invoke_with_timeout(boom, timeout=1.0)


# ---------------------------------------------------------------------------
# Boundary conditions on the timeout argument
# ---------------------------------------------------------------------------


class TestInvokeWithTimeoutTimeoutValidation:
    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            invoke_with_timeout(lambda: None, timeout=0.0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            invoke_with_timeout(lambda: None, timeout=-1.0)

    def test_very_small_positive_timeout_accepted(self) -> None:
        """A tiny but positive timeout is valid; it might fire
        immediately, but that's a runtime decision, not a value
        rejection."""
        # 1ms — likely to time out on a slow CI worker, but acceptable
        # at the API boundary; just assert no ValueError.
        try:
            invoke_with_timeout(lambda: 1, timeout=0.001)
        except TimeoutError:
            pass  # acceptable — the call may legitimately time out
        # If a ValueError was raised, the test fails because pytest
        # propagates it out of this block.


# ---------------------------------------------------------------------------
# Shared-pool contract
# ---------------------------------------------------------------------------


class TestSharedPoolContract:
    def test_pool_is_shared_across_calls(self) -> None:
        """Repeated calls reuse the same pool instance — the whole
        point of the helper is to avoid spawning a fresh executor per
        call."""
        invoke_with_timeout(lambda: 1, timeout=1.0)
        pool_a = _get_invoke_pool()
        invoke_with_timeout(lambda: 2, timeout=1.0)
        pool_b = _get_invoke_pool()
        assert pool_a is pool_b

    def test_pool_is_bounded(self) -> None:
        """Pool size is bounded — without this guard, unbounded
        concurrent calls could spawn arbitrary OS threads."""
        from cogtrix_core.concurrency import _INVOKE_POOL_WORKERS

        pool = _get_invoke_pool()
        # ProcessPoolExecutor / ThreadPoolExecutor expose _max_workers.
        assert pool._max_workers == _INVOKE_POOL_WORKERS

    def test_pool_thread_name_prefix(self) -> None:
        """Threads in the shared pool should be identifiable in stack
        traces / debugging tools."""
        # Submit a job that captures its own thread name.
        import threading

        def get_thread_name() -> str:
            return threading.current_thread().name

        name = invoke_with_timeout(get_thread_name, timeout=1.0)
        assert name.startswith("invoke")
