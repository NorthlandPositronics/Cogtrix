"""Regression tests for wizard session lock initialisation (BUG-239).

Issue #1197 — ``_get_wizard_sessions_lock()`` had a TOCTOU race on lazy
initialisation, and ``_wizard_sessions`` dict was accessed without holding
the lock in multiple locations, risking corruption under concurrent wizard
operations.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.api.routes.config import _get_wizard_sessions_lock, _wizard_sessions


def test_wizard_sessions_lock_double_checked_locking() -> None:
    """Concurrent calls to ``_get_wizard_sessions_lock()`` must return the same ``asyncio.Lock``."""
    import cogtrix_core.api.routes.config as _config_mod

    original_lock = _config_mod._wizard_sessions_lock
    original_guard = _config_mod._wizard_sessions_lock_guard
    _config_mod._wizard_sessions_lock = None
    _config_mod._wizard_sessions_lock_guard = None

    try:
        locks: list[object] = []
        locks_lock = threading.Lock()
        barrier = threading.Barrier(10)

        def _worker() -> None:
            barrier.wait(timeout=2.0)
            lock = _get_wizard_sessions_lock()
            with locks_lock:
                locks.append(lock)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_worker) for _ in range(10)]
            for f in as_completed(futures):
                f.result(timeout=5.0)

        # All 10 workers must have obtained the *same* lock object.
        assert len(locks) == 10, f"Expected 10 lock references, got {len(locks)}"
        assert all(lock is locks[0] for lock in locks), (
            "_get_wizard_sessions_lock() returned different lock instances — "
            "TOCTOU race is not fixed"
        )
    finally:
        _config_mod._wizard_sessions_lock = original_lock
        _config_mod._wizard_sessions_lock_guard = original_guard


@pytest.mark.asyncio
async def test_wizard_sessions_dict_access_is_lock_protected() -> None:
    """Concurrent start/cancel/advance operations must not corrupt ``_wizard_sessions``."""
    # Isolate in place against any residual module-global state (#2247). This test's
    # helpers (_start/_cancel) and assertions use the module-level ``_wizard_sessions``
    # binding imported above, so swapping ``_config_mod._wizard_sessions`` to a fresh
    # dict would NOT isolate them — clear the live binding instead and restore it after.
    saved_sessions = dict(_wizard_sessions)
    _wizard_sessions.clear()

    try:
        # Start 5 wizard sessions concurrently.
        started_ids: list[str] = []
        started_lock = asyncio.Lock()

        async def _start() -> None:
            wid = f"wizard-{threading.current_thread().ident}-{asyncio.current_task().get_name()}"
            async with _get_wizard_sessions_lock():
                _wizard_sessions[wid] = {
                    "created_mono": 0.0,
                    "lock": asyncio.Lock(),
                }
            async with started_lock:
                started_ids.append(wid)

        await asyncio.gather(*[_start() for _ in range(5)])

        # All 5 sessions must be present.
        assert len(_wizard_sessions) == 5, f"Expected 5 sessions, got {len(_wizard_sessions)}"

        # Cancel 3 sessions concurrently.
        async def _cancel(wid: str) -> None:
            async with _get_wizard_sessions_lock():
                _wizard_sessions.pop(wid, None)

        to_cancel = started_ids[:3]
        await asyncio.gather(*[_cancel(wid) for wid in to_cancel])

        # Exactly 2 sessions must remain.
        assert len(_wizard_sessions) == 2, f"Expected 2 sessions, got {len(_wizard_sessions)}"

        # The remaining IDs must match.
        remaining = set(_wizard_sessions.keys())
        expected = set(started_ids[3:])
        assert remaining == expected, f"Remaining sessions mismatch: {remaining} != {expected}"
    finally:
        _wizard_sessions.clear()
        _wizard_sessions.update(saved_sessions)


@pytest.mark.asyncio
async def test_wizard_save_no_deadlock_when_caller_holds_sessions_lock() -> None:
    """_wizard_save must complete without deadlock when the caller already holds the sessions lock.

    Regression for BUG-239: ``asyncio.Lock`` is not reentrant.  If ``_wizard_save()``
    tried to acquire ``_get_wizard_sessions_lock()`` while ``advance_wizard()`` already
    held it, the wizard completion path would hang forever.
    """
    import cogtrix_core.api.routes.config as _config_mod

    wid = "test-wizard-deadlock"
    fake_msg = MagicMock()
    fake_msg.content = "```yaml\nproviders:\n  ollama:\n    model: qwen3.5:9b\n```"

    original_sessions = _config_mod._wizard_sessions
    _config_mod._wizard_sessions = {
        wid: {
            "created_mono": time.monotonic(),
            "step": 1,
            "messages": [fake_msg],
            "bootstrap_info": {"provider": "ollama", "model": "qwen3.5:9b"},
        }
    }

    try:
        async with _get_wizard_sessions_lock():
            with (
                patch.object(_config_mod, "_wizard_validate_and_write"),
                patch("cogtrix_core.config.load_config", return_value=MagicMock()),
            ):
                fake_request = MagicMock()
                # If _wizard_save re-acquires the lock, this will hang.
                await asyncio.wait_for(
                    _config_mod._wizard_save(wid, _config_mod._wizard_sessions[wid], fake_request),
                    timeout=2.0,
                )
        # Session must be cleaned up.
        assert wid not in _config_mod._wizard_sessions
    finally:
        _config_mod._wizard_sessions = original_sessions
