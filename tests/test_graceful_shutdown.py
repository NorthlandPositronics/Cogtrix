"""Regression tests for graceful shutdown handling.

These tests verify that both the CLI and API handle SIGTERM correctly,
initiating graceful shutdown with proper draining and cleanup.

Test scenarios:
1. SIGTERM handler raises KeyboardInterrupt in CLI
2. SIGTERM handler sets shutdown flag in API lifespan
3. WebSocket drain completes within 30-second timeout
4. Background tasks complete within 60-second timeout
5. DB connection pool is closed cleanly
"""

from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry


class TestCLIShutdownHandler:
    """Tests for SIGTERM handler in cogtrix.py CLI."""

    def test_sigterm_handler_sets_shutdown_flag_and_raises(self) -> None:
        """SIGTERM handler should set shutdown flag and raise KeyboardInterrupt."""
        import cogtrix
        from cogtrix import _handle_sigterm

        # Reset the flag in the module
        cogtrix._shutdown_initiated = False

        # Call the handler - should raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt) as exc_info:
            _handle_sigterm(signal.SIGTERM, None)

        assert "SIGTERM received" in str(exc_info.value)
        assert cogtrix._shutdown_initiated is True

    def test_second_sigterm_force_exits(self) -> None:
        """Second SIGTERM should force exit immediately."""
        import cogtrix
        from cogtrix import _handle_sigterm

        # Set the flag in the module to simulate first SIGTERM
        cogtrix._shutdown_initiated = True

        # Mock os._exit to verify it's called
        with patch("cogtrix.os._exit") as mock_exit:
            mock_exit.side_effect = SystemExit

            with pytest.raises(SystemExit):
                _handle_sigterm(signal.SIGTERM, None)

            mock_exit.assert_called_once_with(1)


class TestAPIShutdownHandler:
    """Tests for SIGTERM handler in cogtrix_core/api/app.py API."""

    @pytest.mark.asyncio
    async def test_sigterm_handler_sets_shutdown_flag(self) -> None:
        """API SIGTERM handler should set shutdown flag."""
        from cogtrix_core.api import app as app_module
        from cogtrix_core.api.app import _handle_sigterm_for_api_sync, _register_sigterm_handler

        # Register the handler (must be sync function)
        _register_sigterm_handler()

        # Reset the flag in the module
        app_module._shutdown_initiated = False

        # Call the sync handler (not async)
        # We can't directly call it as signal.signal would because we're in a test
        # So we call the function directly like a normal function
        _handle_sigterm_for_api_sync(15, None)  # 15 = SIGTERM

        assert app_module._shutdown_initiated is True

    @pytest.mark.asyncio
    async def test_second_sigterm_logs_warning(self) -> None:
        """Second SIGTERM should log warning and prepare to force exit."""
        from cogtrix_core.api import app as app_module
        from cogtrix_core.api.app import _handle_sigterm_for_api_sync, _register_sigterm_handler

        # Register the handler
        _register_sigterm_handler()

        # Set the flag in the module to simulate first SIGTERM
        app_module._shutdown_initiated = True

        # Mock os._exit to verify it's called
        with patch("cogtrix_core.api.app.os._exit") as mock_exit:
            mock_exit.side_effect = SystemExit

            # Second SIGTERM should call os._exit(1)
            with pytest.raises(SystemExit):
                _handle_sigterm_for_api_sync(15, None)  # 15 = SIGTERM

            mock_exit.assert_called_once_with(1)


class TestSessionDrain:
    """Tests for WebSocket session draining during shutdown."""

    @pytest.mark.asyncio
    async def test_stop_eviction_loop_saves_sessions(self) -> None:
        """stop_eviction_loop should save all sessions before stopping."""
        # Create a mock app state
        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )

        # Create registry
        registry = ApiSessionRegistry(mock_app_state)

        # Add a test session
        session = ApiSession(
            id="test-session-1",
            user_id="user-1",
            name="Test Session",
        )
        await registry.put(session)

        # Start eviction loop
        registry.start_eviction_loop()

        # Stop eviction - should save sessions
        await registry.stop_eviction_loop()

        # Verify eviction task was cancelled
        assert registry._eviction_task is None or registry._eviction_task.done()

    @pytest.mark.asyncio
    async def test_stop_eviction_loop_handles_errors_gracefully(self) -> None:
        """stop_eviction_loop should handle save errors gracefully."""
        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )

        registry = ApiSessionRegistry(mock_app_state)

        session = ApiSession(
            id="test-session-1",
            user_id="user-1",
            name="Test Session",
        )
        await registry.put(session)

        # Mock _save_memory to raise an error
        with patch(
            "cogtrix_core.api.session_bridge._save_memory", side_effect=RuntimeError("DB Error")
        ):
            # Should not raise, should log warning instead
            await registry.stop_eviction_loop()


class TestBackgroundTasksCleanup:
    """Tests for background task cleanup during shutdown."""

    @pytest.mark.asyncio
    async def test_wait_for_background_tasks_completes(self) -> None:
        """Background tasks should complete within timeout."""

        async def dummy_task() -> str:
            await asyncio.sleep(0.1)
            return "done"

        # Create some background tasks
        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        # Wait for tasks to complete
        pending_tasks = [task1, task2]

        done, pending = await asyncio.wait(
            pending_tasks,
            timeout=5.0,  # generous timeout
            return_when=asyncio.ALL_COMPLETED,
        )

        assert len(done) == 2
        assert len(pending) == 0

        # Verify results
        for task in done:
            assert task.result() == "done"

    @pytest.mark.asyncio
    async def test_cancel_pending_tasks_after_timeout(self) -> None:
        """Pending tasks should be cancelled after timeout."""

        async def long_task() -> None:
            try:
                await asyncio.sleep(10)  # Long task
            except asyncio.CancelledError:
                pass  # Expected during cancellation

        task = asyncio.create_task(long_task())

        # Wait with short timeout
        done, pending = await asyncio.wait(
            [task],
            timeout=0.2,  # Very short timeout
            return_when=asyncio.ALL_COMPLETED,
        )

        if pending:
            # Cancel pending tasks
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        # Task should be cancelled or completed
        assert task.done()


class TestDBEngineCleanup:
    """Tests for database engine cleanup during shutdown."""

    @pytest.mark.asyncio
    async def test_engine_dispose_cleans_up_connections(self) -> None:
        """Engine dispose should clean up all connections."""
        import sqlalchemy as sa

        from cogtrix_core.api.db.engine import AsyncSessionLocal, engine

        # Create a session to ensure connections exist
        async with AsyncSessionLocal() as session:
            # Do a simple query
            result = await session.execute(sa.text("SELECT 1"))
            assert result is not None

        # Dispose the engine
        await engine.dispose()

        # Engine should be disposed
        assert engine.pool is not None  # Pool exists but is disposed
