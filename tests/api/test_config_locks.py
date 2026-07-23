"""Regression tests for config.py module-level lock initialisation (BUG-237).

Issue #1196 — ``_get_provider_write_lock()`` had a TOCTOU race on lazy
initialisation: two concurrent admin requests could both see ``None``,
create different ``asyncio.Lock`` instances, and defeat the serialisation
guarantee that protects config file integrity.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from cogtrix_core.api.routes.config import _get_provider_write_lock


def test_provider_write_lock_double_checked_locking() -> None:
    """Concurrent calls to ``_get_provider_write_lock()`` must return the same ``asyncio.Lock``."""
    # Reset the module-level lock so we can observe the race window.
    global _provider_write_lock
    import cogtrix_core.api.routes.config as _config_mod

    original_lock = _config_mod._provider_write_lock
    _config_mod._provider_write_lock = None

    try:
        locks: list[object] = []
        locks_lock = threading.Lock()
        barrier = threading.Barrier(10)

        def _worker() -> None:
            barrier.wait(timeout=2.0)
            lock = _get_provider_write_lock()
            with locks_lock:
                locks.append(lock)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_worker) for _ in range(10)]
            for f in as_completed(futures):
                f.result(timeout=5.0)

        # All 10 workers must have obtained the *same* lock object.
        assert len(locks) == 10, f"Expected 10 lock references, got {len(locks)}"
        assert all(lock is locks[0] for lock in locks), (
            "_get_provider_write_lock() returned different lock instances — "
            "TOCTOU race is not fixed"
        )
    finally:
        _config_mod._provider_write_lock = original_lock
