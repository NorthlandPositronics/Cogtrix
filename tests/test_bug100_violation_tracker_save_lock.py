"""Regression tests for BUG-100: ViolationTracker._save() must call _save_snapshot
inside the lock.

Before the fix, _save() released the lock before calling _save_snapshot(). This created
a window where a concurrent record_violation() could write a newer snapshot, then _save()
would overwrite it with a stale one — losing the new violation from the persisted file.

After the fix, _save_snapshot() is called inside `with self._lock:`, matching the
pattern already used in record_violation() (BUG-095 fix).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from src.assistant.guardrails import ViolationTracker


def _make_tracker(persist_path: Path) -> ViolationTracker:
    cfg = {
        "auto_blacklist": {
            "enabled": True,
            "max_violations": 5,
            "window_minutes": 30,
        }
    }
    return ViolationTracker(cfg, persist_path=persist_path)


# ---------------------------------------------------------------------------
# Test 1: _save() includes all violations in the persisted snapshot
# ---------------------------------------------------------------------------
def test_save_persists_all_violations(tmp_path: Path) -> None:
    persist_file = tmp_path / "violations.json"
    tracker = _make_tracker(persist_file)

    tracker.record_violation("chat1")
    tracker.record_violation("chat2")

    tracker.save()

    assert persist_file.exists()
    data = json.loads(persist_file.read_text())
    assert "chat1" in data
    assert "chat2" in data


# ---------------------------------------------------------------------------
# Test 2: concurrent _save() and record_violation() — persisted file must
# always contain the new violation (not be overwritten by the stale snapshot)
# ---------------------------------------------------------------------------
def test_concurrent_save_and_record_violation_no_lost_write(tmp_path: Path) -> None:
    """Run _save() and record_violation() from two threads many times.

    If BUG-100 is present, _save() can overwrite the snapshot written by
    record_violation() — the new violation disappears from the persisted file.

    After the fix both operations hold the lock during _save_snapshot(), so only
    one can write at a time and the last writer wins — but that last writer always
    includes all violations that were recorded before it acquired the lock.
    """
    persist_file = tmp_path / "violations.json"
    tracker = _make_tracker(persist_file)

    errors: list[str] = []
    iterations = 100

    def do_record() -> None:
        for _ in range(iterations):
            try:
                tracker.record_violation("chatA")
                time.sleep(0)
            except Exception as exc:
                errors.append(f"record: {exc}")

    def do_save() -> None:
        for _ in range(iterations):
            try:
                tracker.save()
                time.sleep(0)
            except Exception as exc:
                errors.append(f"save: {exc}")

    t1 = threading.Thread(target=do_record)
    t2 = threading.Thread(target=do_save)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not errors, f"Concurrent access produced errors: {errors}"

    # The file must exist and be valid JSON.
    assert persist_file.exists(), "Persist file was not written"
    data = json.loads(persist_file.read_text())
    # After 100 record_violation calls, chatA must appear in the persisted state.
    assert "chatA" in data, (
        "chatA was not found in the persisted file — "
        "_save() may have overwritten the record_violation snapshot (BUG-100)."
    )


# ---------------------------------------------------------------------------
# Test 3: _save() with no persist_path is a no-op
# ---------------------------------------------------------------------------
def test_save_without_persist_path_is_noop() -> None:
    tracker = ViolationTracker({"auto_blacklist": {}}, persist_path=None)
    tracker.record_violation("chat1")
    tracker.save()  # must not raise


# ---------------------------------------------------------------------------
# Test 4: _save() must hold the lock during _save_snapshot (no lock released early)
#
# We verify this indirectly: patch _save_snapshot to check that _lock is held.
# ---------------------------------------------------------------------------
def test_save_holds_lock_during_save_snapshot(tmp_path: Path) -> None:
    persist_file = tmp_path / "violations.json"
    tracker = _make_tracker(persist_file)
    tracker.record_violation("chat1")

    lock_held_during_snapshot: list[bool] = []

    original_save_snapshot = tracker._save_snapshot

    def _patched_save_snapshot(data: dict) -> None:
        # Try to acquire the lock in a non-blocking way.
        # If _save() holds the lock, this will fail (return False).
        acquired = tracker._lock.acquire(blocking=False)
        if acquired:
            tracker._lock.release()
            lock_held_during_snapshot.append(False)  # lock was NOT held — bug
        else:
            lock_held_during_snapshot.append(True)  # lock WAS held — correct
        original_save_snapshot(data)

    tracker._save_snapshot = _patched_save_snapshot  # type: ignore[method-assign]
    tracker.save()

    assert lock_held_during_snapshot, "_save_snapshot was never called"
    assert all(
        lock_held_during_snapshot
    ), "_save() released the lock before calling _save_snapshot — BUG-100 has regressed."
