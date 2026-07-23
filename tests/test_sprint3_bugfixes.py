"""Tests for Sprint 3 bug fixes: BUG-076, BUG-087, BUG-063/090, BUG-067/034."""

from __future__ import annotations

import threading
from collections import OrderedDict
from unittest.mock import patch


class TestCircuitBreakerLock:
    """BUG-076 + BUG-087 — circuit breaker lock refactor."""

    def test_check_availability_locked_returns_available_when_no_failures(self):
        from src.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

        breaker = ModelCircuitBreaker()
        with _circuit_breaker_lock:
            available, reason = breaker._check_availability_locked()
        assert available is True
        assert reason is None

    def test_check_availability_locked_returns_unavailable_when_tripped(self):
        import time

        from src.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

        breaker = ModelCircuitBreaker(
            is_unavailable=True,
            consecutive_failures=5,
            last_failure_time=time.time(),
            last_error="timeout",
        )
        with _circuit_breaker_lock:
            available, reason = breaker._check_availability_locked(cooldown=300.0)
        assert available is False
        assert reason is not None
        assert "timeout" in reason

    def test_check_availability_locked_resets_after_cooldown(self):
        import time

        from src.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

        # Set last_failure_time far in the past so cooldown has already elapsed
        breaker = ModelCircuitBreaker(
            is_unavailable=True,
            consecutive_failures=3,
            last_failure_time=time.time() - 400.0,
            last_error="old error",
        )
        with _circuit_breaker_lock:
            available, reason = breaker._check_availability_locked(cooldown=300.0)
        assert available is True
        assert reason is None
        assert breaker.is_unavailable is False
        assert breaker.consecutive_failures == 0

    def test_check_availability_public_method_still_works(self):
        from src.tools.delegate import ModelCircuitBreaker

        breaker = ModelCircuitBreaker()
        available, reason = breaker.check_availability()
        assert available is True
        assert reason is None

    def test_get_model_status_does_not_deadlock(self):
        """get_model_status must complete without deadlock."""
        from src.tools.delegate import _circuit_breaker_lock, _circuit_breakers, get_model_status

        with _circuit_breaker_lock:
            _circuit_breakers.clear()
        status = get_model_status()
        assert isinstance(status, dict)

    def test_get_model_status_uses_locked_helper(self):
        """Verify get_model_status calls _check_availability_locked (no reentrant lock)."""
        import inspect

        from src.tools.delegate import get_model_status

        source = inspect.getsource(get_model_status)
        # The old code called check_availability() (which acquires the lock itself)
        # The new code calls _check_availability_locked() directly under the lock
        assert (
            "_check_availability_locked" in source
        ), "get_model_status should call _check_availability_locked, not check_availability"


class TestEnsureLoopNoTOCTOU:
    """BUG-063/090 — _ensure_loop TOCTOU race eliminated."""

    def test_concurrent_ensure_loop_creates_exactly_one_loop(self):
        from src.mcp_client import MCPManager

        manager = MCPManager()
        loops_created: list = []

        original_new_event_loop = __import__("asyncio").new_event_loop

        def counting_new_event_loop():
            loop = original_new_event_loop()
            loops_created.append(loop)
            return loop

        barrier = threading.Barrier(2)

        def call_ensure_loop():
            barrier.wait()
            manager._ensure_loop()

        with patch("asyncio.new_event_loop", side_effect=counting_new_event_loop):
            t1 = threading.Thread(target=call_ensure_loop)
            t2 = threading.Thread(target=call_ensure_loop)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

        # Exactly one event loop should have been created
        assert len(loops_created) == 1
        # Clean up
        if manager._loop and not manager._loop.is_closed():
            manager._loop.call_soon_threadsafe(manager._loop.stop)
            if manager._thread:
                manager._thread.join(timeout=2)

    def test_ensure_loop_only_acquires_lock_first(self):
        """The outer unsynchronized check must not exist — all state guarded by lock."""
        import inspect

        from src.mcp_client import MCPManager

        source = inspect.getsource(MCPManager._ensure_loop)
        # Verify the unsynchronized outer check was removed.
        # In the fixed code there is only ONE "if self._loop is not None" check,
        # and it is INSIDE the "with self._loop_lock:" block.
        # Count occurrences of the guard to confirm only the inner one remains.
        guard_count = source.count("if self._loop is not None")
        assert (
            guard_count == 1
        ), f"Expected exactly 1 '_loop is not None' check (inside lock), found {guard_count}"
        # Confirm the lock acquisition appears before the guard in the source text
        lock_pos = source.find("with self._loop_lock:")
        guard_pos = source.find("if self._loop is not None")
        assert lock_pos != -1, "Lock acquisition not found in _ensure_loop"
        assert (
            lock_pos < guard_pos
        ), "Lock acquisition must appear before the guard check (guard must be inside the lock)"


class TestLRUCacheMerge:
    """BUG-067 + BUG-034 — LRU cache merge always updates MRU position."""

    def test_merge_updates_existing_key_to_mru_position(self):
        """A key present in persistent cache must be promoted to MRU after merge."""
        persistent: OrderedDict = OrderedDict()
        persistent["key_a"] = "val_a_old"
        persistent["key_b"] = "val_b"
        # key_a is at LRU position (first)

        local: OrderedDict = OrderedDict()
        local["key_a"] = "val_a_new"

        # Simulate the new merge logic (always update + move_to_end)
        for key, value in local.items():
            persistent[key] = value
            persistent.move_to_end(key)

        # key_a should now be at MRU position (last)
        keys = list(persistent.keys())
        assert keys[-1] == "key_a", f"Expected key_a at MRU position, got order: {keys}"
        assert persistent["key_a"] == "val_a_new"

    def test_merge_skipping_existing_loses_mru_position(self):
        """Demonstrate that the OLD (buggy) skip-if-present logic loses ordering."""
        persistent: OrderedDict = OrderedDict()
        persistent["key_a"] = "val_a_old"
        persistent["key_b"] = "val_b"

        local: OrderedDict = OrderedDict()
        local["key_a"] = "val_a_new"

        # Simulate the OLD (buggy) merge logic
        for key, value in local.items():
            if key not in persistent:
                persistent[key] = value
                persistent.move_to_end(key)

        # key_a stays at LRU (stale) position — this is the bug the fix addresses
        keys = list(persistent.keys())
        assert keys[0] == "key_a", "Old logic leaves key_a at stale LRU position"

    def test_runner_merge_uses_always_update_logic(self):
        """Inspect runner.py merge block to confirm 'if key not in' guard was removed."""
        import inspect

        from src.orchestration import runner

        source = inspect.getsource(runner.run_agent)
        assert (
            "if key not in _persistent_bound_cache" not in source
        ), "Old 'skip if present' guard still present in _persistent_bound_cache merge"
        assert (
            "if key not in _persistent_compression_cache" not in source
        ), "Old 'skip if present' guard still present in _persistent_compression_cache merge"
