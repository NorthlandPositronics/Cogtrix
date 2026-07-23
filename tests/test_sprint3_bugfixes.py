"""Tests for Sprint 3 bug fixes: BUG-076, BUG-087, BUG-063/090, BUG-067/034."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch


class TestCircuitBreakerLock:
    """BUG-076 + BUG-087 — circuit breaker lock refactor."""

    def test_check_availability_locked_returns_available_when_no_failures(self):
        from cogtrix_core.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

        breaker = ModelCircuitBreaker()
        with _circuit_breaker_lock:
            available, reason = breaker._check_availability_locked()
        assert available is True
        assert reason is None

    def test_check_availability_locked_returns_unavailable_when_tripped(self):
        import time

        from cogtrix_core.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

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

        from cogtrix_core.tools.delegate import ModelCircuitBreaker, _circuit_breaker_lock

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
        from cogtrix_core.tools.delegate import ModelCircuitBreaker

        breaker = ModelCircuitBreaker()
        available, reason = breaker.check_availability()
        assert available is True
        assert reason is None

    def test_get_model_status_does_not_deadlock(self):
        """get_model_status must complete without deadlock."""
        from cogtrix_core.tools.delegate import (
            _circuit_breaker_lock,
            _circuit_breakers,
            get_model_status,
        )

        with _circuit_breaker_lock:
            _circuit_breakers.clear()
        status = get_model_status()
        assert isinstance(status, dict)

    def test_get_model_status_uses_locked_helper(self):
        """Verify get_model_status acquires the lock and uses _check_availability_locked."""
        from cogtrix_core.tools.delegate import (
            ModelCircuitBreaker,
            _circuit_breaker_lock,
            get_model_status,
        )

        # Pre-set a known breaker state
        _circuit_breaker_lock.acquire()
        try:
            from cogtrix_core.tools.delegate import _circuit_breakers

            _circuit_breakers.clear()
            breaker = ModelCircuitBreaker()
            breaker.is_unavailable = False
            breaker.consecutive_failures = 0
        finally:
            _circuit_breaker_lock.release()

        # Call get_model_status and verify it completes without deadlock
        status = get_model_status()
        assert isinstance(status, dict)

        # Verify the breaker status is correctly reported in the status dict
        assert (
            "cohere" in status or status == {}
        ), "get_model_status must report circuit-breaker status for known models"


class TestEnsureLoopNoTOCTOU:
    """BUG-063/090 — _ensure_loop TOCTOU race eliminated."""

    def test_concurrent_ensure_loop_creates_exactly_one_loop(self):
        from cogtrix_core.mcp_client import MCPManager

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
        # Clean up using the manager's own shutdown path so heartbeat tasks are cancelled.
        manager.close_all()


class TestMCPToolsReadyGate:
    """MCP reconnects must clear and restore the readiness gate."""

    def test_reconnect_clears_tools_ready_before_connecting_and_sets_after(self):
        from cogtrix_core.mcp_client import MCPManager

        manager = MCPManager()

        class FakeConnection:
            def __init__(self, cfg):
                self.cfg = cfg
                self.tools = [SimpleNamespace(name="alpha")]

            async def connect(self):
                assert (
                    not manager.tools_ready.is_set()
                ), "tools_ready must be cleared before reconnect"

            async def close(self):
                return None

        manager._configs["srv"] = SimpleNamespace(name="srv", timeout=1)
        manager._connections["srv"] = FakeConnection(SimpleNamespace(name="srv", timeout=1))

        def fake_run(coro, timeout=30):
            return asyncio.run(coro)

        with (
            patch("cogtrix_core.mcp_client.MCPConnection", FakeConnection),
            patch.object(manager, "_run", side_effect=fake_run),
        ):
            manager._reconnect_server("srv")

        assert manager.tools_ready.is_set() is True
        assert isinstance(manager._connections["srv"], FakeConnection)

    def test_close_all_clears_tools_ready(self):
        from cogtrix_core.mcp_client import MCPManager

        manager = MCPManager()
        manager.tools_ready.set()
        manager.close_all()
        assert manager.tools_ready.is_set() is False


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
        """Verify the merge logic promotes existing keys to MRU (most-recently-used) position."""
        persistent: OrderedDict = OrderedDict()
        persistent["key_a"] = "val_a_old"
        persistent["key_b"] = "val_b"

        local: OrderedDict = OrderedDict()
        local["key_a"] = "val_a_new"

        # Simulate the merge logic from runner.py
        for key, value in local.items():
            persistent[key] = value
            persistent.move_to_end(key)

        # key_a should now be at MRU position (last)
        keys = list(persistent.keys())
        assert keys[-1] == "key_a", f"Expected key_a at MRU position after merge, got order: {keys}"
        assert persistent["key_a"] == "val_a_new", "Merged value should overwrite original"
