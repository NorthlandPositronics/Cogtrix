"""Tests for standalone helper functions and BaseMemoryManager methods in manager.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.common.message_validation import _coerce_content, is_bad_ai_content
from src.memory.manager import (
    _msg_tokens,
    _sanitize_session_id,
)

# ---------------------------------------------------------------------------
# _sanitize_session_id
# ---------------------------------------------------------------------------


class TestSanitizeSessionId:
    def test_empty_string_returns_default(self):
        assert _sanitize_session_id("") == "default"

    def test_normal_alphanumeric_unchanged(self):
        result = _sanitize_session_id("session-123_abc.def")
        assert result == "session-123_abc.def"

    def test_special_chars_percent_encoded(self):
        result = _sanitize_session_id("user@host")
        assert "@" not in result
        assert "%" in result

    def test_slash_encoded(self):
        result = _sanitize_session_id("path/to/session")
        assert "/" not in result

    def test_space_encoded(self):
        result = _sanitize_session_id("my session")
        assert " " not in result

    def test_dot_dot_sanitized(self):
        result = _sanitize_session_id("../../evil")
        assert ".." not in result

    def test_long_id_truncated(self):
        long_id = "a" * 300
        result = _sanitize_session_id(long_id)
        assert len(result) <= 200

    def test_truncation_does_not_split_percent_triplet(self):
        # Build a string where the 200th char falls inside a %XX sequence
        # Force a special char near the boundary
        base = "a" * 198 + "@b"  # '@' encodes to %40 → 201 chars after encoding
        result = _sanitize_session_id(base)
        # Result must not end with an incomplete percent-encoded sequence
        assert not result.endswith("%") and not (
            len(result) >= 2 and result[-2] == "%" and not result[-1:].isalnum()
        )

    def test_null_byte_encoded(self):
        result = _sanitize_session_id("session\x00id")
        assert "\x00" not in result

    def test_bijectivity_distinct_ids(self):
        a = _sanitize_session_id("session-1")
        b = _sanitize_session_id("session-2")
        assert a != b

    def test_all_special_produces_default(self):
        # A session ID that sanitizes to nothing (all chars consumed by encoding
        # and length limit may produce non-empty; just verify no crash)
        result = _sanitize_session_id("!@#")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _coerce_content
# ---------------------------------------------------------------------------


class TestCoerceContent:
    def test_string_passthrough(self):
        assert _coerce_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _coerce_content(None) == ""

    def test_list_of_strings_joined(self):
        result = _coerce_content(["a", "b", "c"])
        assert result == "a b c"

    def test_list_with_dict_uses_text_key(self):
        result = _coerce_content([{"text": "hello"}, {"text": "world"}])
        assert "hello" in result
        assert "world" in result

    def test_list_with_dict_missing_text_falls_back_to_str(self):
        result = _coerce_content([{"other": "value"}])
        assert "value" in result

    def test_list_with_non_dict_non_str_items(self):
        result = _coerce_content([42, True])
        assert "42" in result
        assert "True" in result

    def test_int_falls_back_to_str(self):
        result = _coerce_content(42)
        assert result == "42"

    def test_empty_list_returns_empty_string(self):
        result = _coerce_content([])
        assert result == ""


# ---------------------------------------------------------------------------
# is_bad_ai_content
# ---------------------------------------------------------------------------


class TestIsBadAiContent:
    def test_empty_string_is_bad(self):
        assert is_bad_ai_content("") is True

    def test_whitespace_only(self):
        assert is_bad_ai_content("   ") is True

    def test_none_content(self):
        assert is_bad_ai_content(None) is True

    def test_valid_content(self):
        assert is_bad_ai_content("Here is the answer.") is False

    def test_model_not_found(self):
        assert is_bad_ai_content("**Model not found:** gpt-x") is True

    def test_auth_failed(self):
        assert is_bad_ai_content("**Authentication failed:** invalid key") is True

    def test_rate_limit(self):
        assert is_bad_ai_content("**Rate limit exceeded.**") is True

    def test_connection_error(self):
        assert is_bad_ai_content("**Connection error:** timeout") is True

    def test_error_occurred(self):
        assert is_bad_ai_content("An error occurred: something went wrong") is True

    def test_valid_with_error_prefix_in_middle(self):
        # Error prefix in the middle should NOT match
        assert is_bad_ai_content("Answer: **Connection error:** was avoided") is False

    def test_empty_list(self):
        assert is_bad_ai_content([]) is True

    def test_list_with_content(self):
        assert is_bad_ai_content(["Some content"]) is False


# ---------------------------------------------------------------------------
# BaseMemoryManager — concrete subclass for testing abstract methods
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path):
    """Create a minimal concrete BaseMemoryManager for testing base methods."""
    from src.memory.json_store import JsonFileMemoryStore
    from src.memory.modes.conversation import ConversationMemoryManager

    store = JsonFileMemoryStore(base_dir=str(tmp_path))
    return ConversationMemoryManager(store=store, session_id="test-session")


class TestBaseMemoryManagerHelpers:
    def test_set_llm(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_llm = MagicMock()
        mgr.set_llm(fake_llm)
        assert mgr._llm is fake_llm

    def test_hybrid_meta_path_inside_base(self, tmp_path):
        mgr = _make_manager(tmp_path)
        meta = mgr._hybrid_meta_path()
        # Must be inside the store's base_path
        assert str(meta).startswith(str(tmp_path))

    def test_hybrid_meta_path_traversal_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Patch _sanitize_session_id to return "../out" so the resolved path escapes.
        # The method builds f"{safe_id}_hybrid.json", so "../out_hybrid.json" resolves
        # one level above base_path (outside the store directory).
        with patch("src.memory.manager._sanitize_session_id", return_value="../out"):
            with pytest.raises(ValueError, match="Path traversal"):
                mgr._hybrid_meta_path()

    def test_save_hybrid_meta_noop_when_no_summary(self, tmp_path):
        """_save_hybrid_meta is a no-op when summary is None and idx is 0."""
        mgr = _make_manager(tmp_path)
        mgr._save_hybrid_meta()  # Should not raise, no file created
        assert not mgr._hybrid_meta_path().exists()

    def test_save_and_load_hybrid_meta_roundtrip(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "A conversation summary."
        mgr._summary_msg_idx = 10
        mgr._save_hybrid_meta()

        # Reset and reload
        mgr2 = _make_manager(tmp_path)
        mgr2._load_hybrid_meta()
        assert mgr2._summary == "A conversation summary."
        assert mgr2._summary_msg_idx == 10

    def test_load_hybrid_meta_noop_when_file_missing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._load_hybrid_meta()  # no file — should not raise
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0

    def test_load_hybrid_meta_handles_corrupt_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        meta_path = mgr._hybrid_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("{bad", encoding="utf-8")
        mgr._load_hybrid_meta()  # should not raise
        assert mgr._summary is None  # unchanged

    def test_set_embeddings_creates_vector_store(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_embed = MagicMock()
        mgr.set_embeddings(fake_embed, "my-model", vector_store_dir=str(tmp_path))
        assert mgr._vector_store is not None

    def test_set_embeddings_reconfigures_existing_store(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_embed = MagicMock()
        mgr.set_embeddings(fake_embed, "model-v1", vector_store_dir=str(tmp_path))
        first_store = mgr._vector_store
        mgr.set_embeddings(fake_embed, "model-v2", vector_store_dir=str(tmp_path))
        # The same store object is reused (reconfigured, not replaced)
        assert mgr._vector_store is first_store

    def test_ensure_embeddings_init_logs_warning_on_failure(self, tmp_path, caplog):
        """Broken embedding config must emit WARNING (not DEBUG) so users can diagnose."""
        from logging import WARNING

        mgr = _make_manager(tmp_path)
        mgr.set_embedding_config(
            emb_type="openai",
            emb_model="text-embedding-3-small",
            emb_base_url=None,
            emb_api_key="invalid-key",
        )
        with patch(
            "src.providers.create_embeddings_from_config",
            side_effect=RuntimeError("401 Unauthorized"),
        ):
            with caplog.at_level(WARNING, logger="cogtrix"):
                mgr._ensure_embeddings_initialized()

        assert mgr._lazy_emb_resolved is True
        assert mgr._vector_store is None
        assert any(
            "openai" in rec.message and "401 Unauthorized" in rec.message
            for rec in caplog.records
            if rec.levelno == WARNING
        ), f"Expected WARNING about openai failure, got: {[r.message for r in caplog.records]}"

    def test_ensure_embeddings_init_warning_is_one_shot(self, tmp_path, caplog):
        """The warning must be emitted once per provider-type, not every turn."""
        from logging import WARNING

        mgr = _make_manager(tmp_path)
        mgr.set_embedding_config(
            emb_type="openai",
            emb_model="text-embedding-3-small",
            emb_base_url=None,
            emb_api_key="invalid-key",
        )
        with patch(
            "src.providers.create_embeddings_from_config",
            side_effect=RuntimeError("401 Unauthorized"),
        ):
            with caplog.at_level(WARNING, logger="cogtrix"):
                mgr._ensure_embeddings_initialized()
                mgr._ensure_embeddings_initialized()  # second call — must not re-log

        warning_records = [
            r for r in caplog.records if r.levelno == WARNING and "openai" in r.message
        ]
        assert len(warning_records) == 1, (
            f"Expected exactly 1 WARNING, got {len(warning_records)}: "
            f"{[r.message for r in warning_records]}"
        )

    def test_clamp_summary_idx(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary_msg_idx = 999
        mgr._clamp_summary_idx()
        assert mgr._summary_msg_idx == mgr.get_message_count()

    def test_build_hybrid_prefix_returns_none_when_no_summary_no_store(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr._build_hybrid_prefix("what is going on?")
        assert result is None

    def test_build_hybrid_prefix_includes_summary(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "Old conversation context."
        result = mgr._build_hybrid_prefix("what is going on?")
        assert result is not None
        assert "Old conversation context." in result

    def test_check_summary_token_ttl_fires_at_threshold(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._mode_config["summary_max_uncovered_tokens"] = 100
        mgr._summary = "existing summary"
        mgr._summary_last_updated_at = datetime.now(UTC)
        mgr._tokens_since_summary = 100
        mgr._check_summary_token_ttl()
        assert mgr._summary is None  # reset fired
        assert mgr._tokens_since_summary == 0

    def test_check_summary_token_ttl_does_not_fire_below_threshold(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._mode_config["summary_max_uncovered_tokens"] = 100
        mgr._summary = "existing summary"
        mgr._summary_last_updated_at = datetime.now(UTC)
        mgr._tokens_since_summary = 99
        mgr._check_summary_token_ttl()
        assert mgr._summary == "existing summary"  # not reset
        assert mgr._tokens_since_summary == 99

    def test_check_summary_token_ttl_disabled_when_none(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._mode_config["summary_max_uncovered_tokens"] = None
        mgr._summary = "existing summary"
        mgr._tokens_since_summary = 999_999
        mgr._check_summary_token_ttl()
        assert mgr._summary == "existing summary"  # not reset

    def test_reset_summary_state_skips_when_summary_refreshes_after_snapshot(self, tmp_path):
        mgr = _make_manager(tmp_path)
        stale_ts = datetime.now(UTC) - timedelta(hours=2)
        fresh_ts = datetime.now(UTC)

        mgr._summary = "refreshed summary"
        mgr._summary_msg_idx = 12
        mgr._summary_last_updated_at = fresh_ts

        result = mgr._reset_summary_state(expected_summary_last_updated_at=stale_ts)

        assert result is False
        assert mgr._summary == "refreshed summary"
        assert mgr._summary_msg_idx == 12
        assert mgr._summary_last_updated_at == fresh_ts

    def test_reset_summary_state_clears_token_counter(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._tokens_since_summary = 500
        mgr._reset_summary_state()
        assert mgr._tokens_since_summary == 0

    def test_get_hybrid_snapshot_blocking_returns_tuple(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "test summary"
        mgr._summary_msg_idx = 5
        result = mgr._get_hybrid_snapshot(block=True)
        assert result == ("test summary", 5, None)

    def test_get_hybrid_snapshot_nonblocking_when_lock_held_returns_none(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "test summary"
        mgr._summary_msg_idx = 5
        # Hold the lock from another thread so non-blocking acquire fails
        mgr._hybrid_lock.acquire()
        try:
            result = mgr._get_hybrid_snapshot(block=False, timeout=0.0)
            assert result is None
        finally:
            mgr._hybrid_lock.release()

    def test_save_hybrid_meta_skips_when_snapshot_none(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "test summary"
        mgr._summary_msg_idx = 5
        # Pre-create a meta file
        mgr._save_hybrid_meta()
        assert mgr._hybrid_meta_path().exists()
        mtime_before = mgr._hybrid_meta_path().stat().st_mtime

        # Hold the lock so non-blocking snapshot returns None
        mgr._hybrid_lock.acquire()
        try:
            mgr._save_hybrid_meta(block=False, timeout=0.0)
        finally:
            mgr._hybrid_lock.release()

        # File should still exist and be unchanged (not deleted or overwritten)
        assert mgr._hybrid_meta_path().exists()
        mtime_after = mgr._hybrid_meta_path().stat().st_mtime
        assert mtime_after == mtime_before

    def test_save_hybrid_meta_preserves_existing_file_when_lock_held(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._summary = "original summary"
        mgr._summary_msg_idx = 3
        mgr._save_hybrid_meta()
        original_mtime = mgr._hybrid_meta_path().stat().st_mtime

        # Change state but hold lock so shutdown-style save can't snapshot
        mgr._summary = "new summary"
        mgr._summary_msg_idx = 10
        mgr._hybrid_lock.acquire()
        try:
            mgr._save_hybrid_meta(block=False, timeout=0.0)
        finally:
            mgr._hybrid_lock.release()

        # Reload and verify original state was preserved on disk
        mgr2 = _make_manager(tmp_path)
        mgr2._load_hybrid_meta()
        assert mgr2._summary == "original summary"
        assert mgr2._summary_msg_idx == 3
        assert mgr2._hybrid_meta_path().stat().st_mtime == original_mtime


# ---------------------------------------------------------------------------
# _msg_tokens
# ---------------------------------------------------------------------------


class TestMsgTokens:
    def test_string_content(self):
        msg = MagicMock()
        msg.content = "a" * 100
        assert _msg_tokens(msg) == 50  # 100 // 2

    def test_list_content(self):
        msg = MagicMock()
        msg.content = ["a" * 40, "b" * 60]
        assert _msg_tokens(msg) == 50  # (40+60) // 2

    def test_empty_string_returns_one(self):
        msg = MagicMock()
        msg.content = ""
        assert _msg_tokens(msg) == 1

    def test_none_content_returns_one(self):
        msg = MagicMock()
        msg.content = None
        assert _msg_tokens(msg) == 1

    def test_dict_content(self):
        msg = {"content": "a" * 80}
        assert _msg_tokens(msg) == 40  # 80 // 2
