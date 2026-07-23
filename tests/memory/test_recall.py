"""Unit tests for src/memory/recall.py — SessionVectorStore."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# SessionVectorStore — functional coverage
# ---------------------------------------------------------------------------


class TestSessionVectorStoreNotReady:
    """Behaviour when configure() has not been called (not ready)."""

    def setup_method(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self.store = SessionVectorStore("test-notready", storage_dir=self._tmpdir)

    def test_ready_false_before_configure(self):
        assert self.store.ready is False

    def test_add_messages_noop_when_not_ready(self):
        """add_messages should return without error when not ready."""
        self.store.add_messages([{"type": "human", "content": "hi"}])
        # No exception and store unchanged
        assert self.store._vectorstore is None

    def test_recall_returns_empty_when_not_ready(self):
        result = self.store.recall("query")
        assert result == []

    def test_save_noop_when_no_vectorstore(self):
        """save() exits early when _vectorstore is None."""
        self.store.save()  # should not raise

    def test_clear_noop_when_no_index_dir(self):
        """clear() should not raise even when the index directory doesn't exist."""
        self.store.clear()


class TestSessionVectorStoreLoadOrReset:
    """Tests for _load_or_reset branches."""

    def setup_method(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self.store = SessionVectorStore("test-load", storage_dir=self._tmpdir)
        self.store._embedding_fn = MagicMock()
        self.store._embedding_model = "test-model"

    def test_configure_ready_when_no_meta_file(self):
        """configure() sets ready=True when no persisted index exists."""
        self.store.configure(self.store._embedding_fn, "test-model")
        assert self.store.ready is True

    def test_load_or_reset_with_corrupt_meta(self):
        """Corrupt meta.json causes reset and ready=True."""
        self.store._index_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.store._index_dir / "meta.json"
        meta_path.write_text("{bad json", encoding="utf-8")
        self.store._load_or_reset()
        assert self.store.ready is True
        assert self.store._vectorstore is None

    def test_load_or_reset_with_model_mismatch(self):
        """Different embedding model causes index discard and ready=True."""
        self.store._index_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.store._index_dir / "meta.json"
        meta_path.write_text(json.dumps({"embedding_model": "old-model"}), encoding="utf-8")
        self.store._embedding_model = "new-model"
        self.store._load_or_reset()
        assert self.store.ready is True
        # Index directory should have been removed
        assert not self.store._index_dir.exists()

    def test_load_or_reset_faiss_import_error(self):
        """When FAISS can't be imported, falls back to reset + ready=True."""
        import json as _json

        self.store._index_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.store._index_dir / "meta.json"
        meta_path.write_text(_json.dumps({"embedding_model": "test-model"}), encoding="utf-8")
        with patch(
            "src.memory.recall.FAISS",
            side_effect=Exception("FAISS unavailable"),
            create=True,
        ):
            with patch(
                "langchain_community.vectorstores.FAISS",
                side_effect=Exception("FAISS unavailable"),
                create=True,
            ):
                self.store._load_or_reset()
        # Should set ready=True despite failure
        assert self.store.ready is True

    def test_load_or_reset_without_meta_attempts_index_load(self):
        """Missing meta.json should still attempt to load an existing index."""
        self.store._index_dir.mkdir(parents=True, exist_ok=True)
        (self.store._index_dir / "index.faiss").write_bytes(b"index")
        mock_store = MagicMock()
        with patch("src.memory.recall.load_faiss_store_safe", return_value=mock_store) as mock_load:
            self.store._load_or_reset()
        mock_load.assert_called_once()
        assert self.store._vectorstore is mock_store
        assert self.store.ready is True

    def test_reset_index_removes_directory(self):
        """_reset_index() removes the index directory."""
        self.store._index_dir.mkdir(parents=True, exist_ok=True)
        self.store._reset_index()
        assert not self.store._index_dir.exists()
        assert self.store._vectorstore is None


class TestMessagesToTexts:
    """Tests for the _messages_to_texts static method."""

    def test_empty_list_returns_empty(self):
        result = SessionVectorStore._messages_to_texts([])
        assert result == []

    def test_human_only_flushed_at_end(self):
        msgs = [{"type": "human", "content": "hello"}]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert len(result) == 1
        assert "User: hello" in result[0]

    def test_human_ai_pair_grouped(self):
        msgs = [
            {"type": "human", "content": "question"},
            {"type": "ai", "content": "answer"},
        ]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert len(result) == 1
        assert "User: question" in result[0]
        assert "Assistant: answer" in result[0]

    def test_tool_messages_skipped(self):
        msgs = [
            {"type": "human", "content": "q"},
            {"type": "tool", "content": "tool output"},
            {"type": "ai", "content": "a"},
        ]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert len(result) == 1
        assert "tool output" not in result[0]

    def test_long_content_truncated(self):
        long = "x" * 2000
        msgs = [{"type": "human", "content": long}]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert len(result) == 1
        # Content is truncated with ellipsis marker
        assert "[...]" in result[0]
        assert len(result[0]) < 2000

    def test_list_content_joined(self):
        msgs = [{"type": "human", "content": ["part1", "part2"]}]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert "part1" in result[0]
        assert "part2" in result[0]

    def test_empty_content_skipped(self):
        msgs = [{"type": "human", "content": "   "}]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert result == []

    def test_message_objects_with_content_attr(self):
        """Objects with .content attribute are handled (e.g. LangChain messages)."""

        class FakeMsg:
            content = "fake message"

        result = SessionVectorStore._messages_to_texts([FakeMsg()])
        # role from class name
        assert len(result) == 1

    def test_multiple_exchanges(self):
        msgs = [
            {"type": "human", "content": "q1"},
            {"type": "ai", "content": "a1"},
            {"type": "human", "content": "q2"},
            {"type": "ai", "content": "a2"},
        ]
        result = SessionVectorStore._messages_to_texts(msgs)
        assert len(result) == 2
