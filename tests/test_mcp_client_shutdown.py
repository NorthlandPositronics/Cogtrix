"""Regression tests for MCPManager shutdown ordering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cogtrix_core.mcp_client import MCPManager


class _FakeLoop:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.stopped = False

    def is_closed(self) -> bool:
        return False

    def stop(self) -> None:
        self.stopped = True

    def call_soon_threadsafe(self, callback, *args):
        name = getattr(callback, "__name__", repr(callback))
        self.calls.append(name)
        if name == "stop":
            self.stopped = True
        return None


class _FakeFuture:
    def __init__(self, side_effect=None) -> None:
        self._side_effect = side_effect

    def result(self, timeout=None):  # noqa: ARG002
        if self._side_effect is not None:
            raise self._side_effect
        return None


def _make_manager() -> MCPManager:
    manager = MCPManager()
    manager._loop = _FakeLoop()
    manager._thread = MagicMock()
    manager._thread.is_alive.return_value = False
    return manager


def test_close_all_clears_tools_ready_before_connection_close():
    manager = _make_manager()
    loop = manager._loop
    heartbeat_cancelled = []
    scheduled: list[str] = []

    class _FakeConnection:
        async def close(self) -> None:
            return None

    manager._connections["srv"] = _FakeConnection()
    manager._tool_server_map["tool"] = "srv"
    manager._heartbeat_task = SimpleNamespace(
        done=lambda: False, cancel=lambda: heartbeat_cancelled.append(True)
    )

    def fake_run_coroutine_threadsafe(coro, loop):  # noqa: ARG001
        assert not manager.tools_ready.is_set()
        coro_name = getattr(getattr(coro, "cr_code", None), "co_name", "<unknown>")
        scheduled.append(coro_name)
        heartbeat_cancelled.append("scheduled")
        coro.close()
        return _FakeFuture()

    with patch(
        "cogtrix_core.mcp_client.asyncio.run_coroutine_threadsafe",
        side_effect=fake_run_coroutine_threadsafe,
    ):
        manager.close_all()

    assert scheduled == ["_cancel_heartbeat", "_cancel_all"]
    assert manager._heartbeat_task is None
    assert manager.tools_ready.is_set() is False
    assert loop is not None
    assert loop.calls == ["stop"]


def test_close_all_stops_loop_after_pending_task_cancellation():
    manager = _make_manager()
    loop = manager._loop
    calls: list[str] = []
    manager._connections.clear()

    def fake_run_coroutine_threadsafe(coro, loop):  # noqa: ARG001
        calls.append("cancel_all")
        coro.close()
        return _FakeFuture()

    with patch(
        "cogtrix_core.mcp_client.asyncio.run_coroutine_threadsafe",
        side_effect=fake_run_coroutine_threadsafe,
    ):
        manager.close_all()

    assert calls == ["cancel_all"]
    assert loop is not None
    assert loop.calls == ["stop"]
    assert loop.stopped is True
    assert manager._thread is None


def test_close_all_stops_loop_even_if_task_cancellation_fails():
    manager = _make_manager()
    loop = manager._loop
    manager._connections.clear()

    def fake_run_coroutine_threadsafe(coro, loop):  # noqa: ARG001
        coro.close()
        return _FakeFuture(side_effect=RuntimeError("boom"))

    with patch(
        "cogtrix_core.mcp_client.asyncio.run_coroutine_threadsafe",
        side_effect=fake_run_coroutine_threadsafe,
    ):
        manager.close_all()

    assert loop is not None
    assert loop.stopped is True
    assert manager._thread is None
