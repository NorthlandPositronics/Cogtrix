"""Unit tests for src/memory/recall.py — SessionVectorStore."""

from __future__ import annotations

import threading

import pytest

from src.memory.recall import SessionVectorStore


class TestSessionVectorStoreLock:
    """BUG-045: SessionVectorStore must have a per-instance RLock."""

    def test_instance_has_lock_attribute(self):
        """SessionVectorStore.__init__ sets a _lock attribute."""
        store = SessionVectorStore("test-session-lock")
        assert hasattr(store, "_lock")

    def test_lock_is_rlock(self):
        """_lock is a threading.RLock (reentrant) instance."""
        store = SessionVectorStore("test-session-rlock")
        # threading.RLock() returns a _RLock instance; check via context manager protocol
        # and the canonical way to detect RLock vs Lock
        assert isinstance(store._lock, type(threading.RLock()))

    def test_each_instance_has_independent_lock(self):
        """Two separate SessionVectorStore instances have different lock objects."""
        store_a = SessionVectorStore("session-a")
        store_b = SessionVectorStore("session-b")
        assert store_a._lock is not store_b._lock


class TestSessionVectorStorePathGuard:
    """BUG-075: Path guard must use Path.relative_to() not startswith().

    The old guard used ``str(candidate).startswith(str(base_resolved))``.
    That check passes for a path like ``/tmp/sessions-evil/x`` when
    ``base_resolved`` is ``/tmp/sessions`` (no trailing separator).
    ``Path.relative_to()`` is the correct boundary-aware check.
    """

    def test_normal_session_id_accepted(self):
        store = SessionVectorStore("valid-session-123")
        assert store._index_dir is not None

    def test_session_index_dir_is_inside_storage_dir(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = str(Path(tmpdir) / "sessions")
            store = SessionVectorStore("my-session", storage_dir=storage)
            base = Path(storage).resolve()
            # relative_to raises ValueError if store._index_dir is not inside base
            store._index_dir.relative_to(base)

    def test_path_guard_uses_relative_to_not_startswith(self):
        """Confirm the guard raises for a path that would escape storage_dir.

        We patch _sanitize_session_id to return a crafted value that makes the
        resolved candidate land outside the storage directory, simulating a
        scenario where sanitization does not fully neutralize the input.
        """
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "sessions"
            storage.mkdir()
            with patch(
                "src.memory.recall._sanitize_session_id",
                return_value="..",
            ):
                with pytest.raises(ValueError, match="Path traversal"):
                    SessionVectorStore("anything", storage_dir=str(storage))

    def test_startswith_without_sep_would_have_allowed_sibling(self):
        """Demonstrate the original bug: startswith() passes for a sibling directory.

        /tmp/sessions-evil startswith /tmp/sessions => True (wrong!)
        Path.relative_to raises => correct.
        """
        from pathlib import Path

        base = Path("/tmp/sessions")
        sibling = Path("/tmp/sessions-evil/data")
        # The old check would have passed:
        assert str(sibling).startswith(str(base))
        # The new check raises:
        with pytest.raises(ValueError):
            sibling.relative_to(base)
