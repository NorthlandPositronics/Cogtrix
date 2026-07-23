"""Regression test for #2131 C5 — clear() must remove the per-session .lock and
_pending.json artifacts.

The flock guard file {id}.lock is O_CREAT'd on every save and was never removed;
the conversation _pending.json was removed by discard_prerecord()/update() but not
by clear(). A server cycling through many session ids leaked one of each per
cleared session.
"""

from __future__ import annotations

from pathlib import Path

from cogtrix_core.memory import modes  # noqa: F401 — triggers mode registration
from cogtrix_core.memory.json_store import JsonFileMemoryStore
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager


def test_delete_lock_removes_lock_file(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(base_dir=str(tmp_path))
    store.save_history("sess", [])
    lock_path = tmp_path / "sess.lock"
    assert lock_path.exists(), "save_history must create the .lock guard file"

    store.delete_lock("sess")
    assert not lock_path.exists(), "delete_lock must remove the .lock file"

    # Idempotent / missing-file safe.
    store.delete_lock("sess")
    store.delete_lock("never-saved")


def test_clear_removes_lock_and_pending(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(base_dir=str(tmp_path))
    mgr = ConversationMemoryManager(store, "sess")

    # Create the two leak-prone artifacts.
    store.save_history("sess", [])
    mgr.prerecord_user("hello")

    lock_path = tmp_path / "sess.lock"
    pending_path = tmp_path / "sess_pending.json"
    assert lock_path.exists()
    assert pending_path.exists()

    mgr.clear()

    assert not lock_path.exists(), "clear() must remove the .lock file (#2131 C5)"
    assert not pending_path.exists(), "clear() must remove _pending.json (#2131 C5)"
