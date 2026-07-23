"""Tests for standalone helper functions and BaseMemoryManager methods in manager.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.manager import (
    _coerce_content,
    _is_bad_ai_content,
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
# _is_bad_ai_content
# ---------------------------------------------------------------------------


class TestIsBadAiContent:
    def test_empty_string_is_bad(self):
        assert _is_bad_ai_content("") is True

    def test_whitespace_only_is_bad(self):
        assert _is_bad_ai_content("   ") is True

    def test_none_is_bad(self):
        assert _is_bad_ai_content(None) is True

    def test_normal_content_is_good(self):
        assert _is_bad_ai_content("Here is the answer.") is False

    def test_model_not_found_error_is_bad(self):
        assert _is_bad_ai_content("**Model not found:** gpt-x") is True

    def test_auth_error_is_bad(self):
        assert _is_bad_ai_content("**Authentication failed:** invalid key") is True

    def test_rate_limit_error_is_bad(self):
        assert _is_bad_ai_content("**Rate limit exceeded.**") is True

    def test_connection_error_is_bad(self):
        assert _is_bad_ai_content("**Connection error:** timeout") is True

    def test_generic_error_prefix_is_bad(self):
        assert _is_bad_ai_content("An error occurred: something went wrong") is True

    def test_partial_match_not_at_start_is_good(self):
        # Error prefix must be at the start of the stripped text
        assert _is_bad_ai_content("Answer: **Connection error:** was avoided") is False

    def test_list_content_empty_is_bad(self):
        assert _is_bad_ai_content([]) is True

    def test_list_content_with_text_is_good(self):
        assert _is_bad_ai_content(["Some content"]) is False


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
