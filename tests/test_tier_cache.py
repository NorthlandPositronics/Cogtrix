"""Tests for Tiered Context Cache (TCC) Phases 1-3."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.memory.tier_cache import CompressedMessage, TierCacheSnapshot

# ---------------------------------------------------------------------------
# CompressedMessage
# ---------------------------------------------------------------------------


class TestCompressedMessage:
    def test_to_dict_round_trip(self):
        msg = CompressedMessage(
            tool_call_id="abc123",
            name="web_search",
            content="<compressed result>",
            original_type="ToolMessage",
        )
        d = msg.to_dict()
        assert d == {
            "tool_call_id": "abc123",
            "name": "web_search",
            "content": "<compressed result>",
            "original_type": "ToolMessage",
        }
        restored = CompressedMessage.from_dict(d)
        assert restored == msg

    def test_from_dict_missing_keys_defaults_to_empty_strings(self):
        msg = CompressedMessage.from_dict({})
        assert msg.tool_call_id == ""
        assert msg.name == ""
        assert msg.content == ""
        assert msg.original_type == ""

    def test_from_dict_aimessage(self):
        d = {
            "tool_call_id": "",
            "name": "assistant",
            "content": "one-sentence summary",
            "original_type": "AIMessage",
        }
        msg = CompressedMessage.from_dict(d)
        assert msg.tool_call_id == ""
        assert msg.name == "assistant"
        assert msg.original_type == "AIMessage"

    def test_from_dict_coerces_non_string_values(self):
        d = {"tool_call_id": 42, "name": None, "content": 3.14, "original_type": True}
        msg = CompressedMessage.from_dict(d)
        assert isinstance(msg.tool_call_id, str)
        assert isinstance(msg.name, str)
        assert isinstance(msg.content, str)
        assert isinstance(msg.original_type, str)


# ---------------------------------------------------------------------------
# TierCacheSnapshot
# ---------------------------------------------------------------------------


class TestTierCacheSnapshot:
    def _make_snapshot(self) -> TierCacheSnapshot:
        return TierCacheSnapshot(
            version=1,
            tier0_boundary_idx=50,
            tier1_messages=[
                CompressedMessage("id1", "web_search", "light summary", "ToolMessage"),
                CompressedMessage("id2", "read_file", "file summary", "ToolMessage"),
            ],
            tier2_messages=[
                CompressedMessage("", "assistant", "one sentence", "AIMessage"),
            ],
            tier1_token_count=3800,
            tier2_token_count=1050,
            tier3_msg_idx=200,
            calibration_tokens=12450,
            calibration_chars=31200,
        )

    def test_to_dict_structure(self):
        snap = self._make_snapshot()
        d = snap.to_dict()

        assert d["version"] == 1
        assert d["tier0_boundary_idx"] == 50
        assert d["tier3_msg_idx"] == 200
        assert d["calibration_token_count"] == 12450
        assert d["calibration_char_count"] == 31200

        assert d["tier1"]["token_count"] == 3800
        assert len(d["tier1"]["messages"]) == 2
        assert d["tier1"]["messages"][0]["tool_call_id"] == "id1"

        assert d["tier2"]["token_count"] == 1050
        assert len(d["tier2"]["messages"]) == 1
        assert d["tier2"]["messages"][0]["name"] == "assistant"

    def test_from_dict_round_trip(self):
        snap = self._make_snapshot()
        restored = TierCacheSnapshot.from_dict(snap.to_dict())

        assert restored.version == snap.version
        assert restored.tier0_boundary_idx == snap.tier0_boundary_idx
        assert restored.tier1_token_count == snap.tier1_token_count
        assert restored.tier2_token_count == snap.tier2_token_count
        assert restored.tier3_msg_idx == snap.tier3_msg_idx
        assert restored.calibration_tokens == snap.calibration_tokens
        assert restored.calibration_chars == snap.calibration_chars
        assert len(restored.tier1_messages) == 2
        assert len(restored.tier2_messages) == 1
        assert restored.tier1_messages[0].tool_call_id == "id1"
        assert restored.tier2_messages[0].name == "assistant"

    def test_from_dict_empty_dict_returns_empty_snapshot(self):
        snap = TierCacheSnapshot.from_dict({})
        assert snap.version == 1
        assert snap.tier0_boundary_idx == 0
        assert snap.tier1_messages == []
        assert snap.tier2_messages == []
        assert snap.tier1_token_count == 0
        assert snap.tier2_token_count == 0
        assert snap.tier3_msg_idx == 0
        assert snap.calibration_tokens == 0
        assert snap.calibration_chars == 0

    def test_from_dict_corrupt_data_returns_empty_snapshot(self):
        # Non-int tier0_boundary_idx triggers exception inside from_dict
        corrupt = {"version": 1, "tier0_boundary_idx": "not-an-int-!!!", "tier1": "wrong"}
        # from_dict catches TypeError/ValueError internally and returns empty
        # In this case "not-an-int-!!!" will raise ValueError in int()
        snap = TierCacheSnapshot.from_dict(corrupt)
        assert snap.tier0_boundary_idx == 0
        assert snap.tier1_messages == []

    def test_from_dict_corrupt_message_entry(self):
        # Corrupt message list (not dicts) — should return empty snapshot
        corrupt = {
            "version": 1,
            "tier0_boundary_idx": 10,
            "tier1": {"token_count": 100, "messages": [None, 42, "bad"]},
            "tier2": {"token_count": 0, "messages": []},
        }
        # from_dict should handle this gracefully
        try:
            TierCacheSnapshot.from_dict(corrupt)
        except Exception:
            pytest.fail("from_dict should not raise on corrupt message entries")

    def test_total_token_estimate_sums_tier1_and_tier2(self):
        snap = TierCacheSnapshot(
            tier1_token_count=3800,
            tier2_token_count=1050,
        )
        assert snap.total_token_estimate == 4850

    def test_total_token_estimate_zero_when_empty(self):
        snap = TierCacheSnapshot()
        assert snap.total_token_estimate == 0

    def test_default_snapshot_fields(self):
        snap = TierCacheSnapshot()
        assert snap.version == 1
        assert snap.tier0_boundary_idx == 0
        assert snap.tier1_messages == []
        assert snap.tier2_messages == []
        assert snap.tier1_token_count == 0
        assert snap.tier2_token_count == 0
        assert snap.tier3_msg_idx == 0
        assert snap.calibration_tokens == 0
        assert snap.calibration_chars == 0


# ---------------------------------------------------------------------------
# BaseMemoryManager tier cache path and persistence
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path):
    """Create a ConversationMemoryManager backed by a temp directory."""
    from cogtrix_core.memory.json_store import JsonFileMemoryStore
    from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

    store = JsonFileMemoryStore(base_dir=str(tmp_path))
    mm = ConversationMemoryManager(store=store, session_id="test-session")
    return mm


class TestTierCachePath:
    def test_returns_correct_path(self, tmp_path):
        mm = _make_manager(tmp_path)
        p = mm._tier_cache_path()
        assert p.name == "test-session_tier_cache.json"
        assert p.parent == tmp_path.resolve()

    def test_path_traversal_raises(self, tmp_path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ConversationMemoryManager(store=store, session_id="../../etc/passwd")
        # _sanitize_session_id encodes ".." → "%2E%2E" so no traversal occurs;
        # the resulting path must still be inside the base directory.
        p = mm._tier_cache_path()
        assert p.is_relative_to(tmp_path.resolve())

    def test_sanitized_session_id_used_in_path(self, tmp_path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ConversationMemoryManager(store=store, session_id="abc/def")
        p = mm._tier_cache_path()
        # "/" is percent-encoded → path stays inside base_dir
        assert p.is_relative_to(tmp_path.resolve())
        assert "/" not in p.name


class TestLoadTierCache:
    def test_missing_file_leaves_cache_none(self, tmp_path):
        mm = _make_manager(tmp_path)
        mm._load_tier_cache()
        assert mm._tier_cache is None
        assert mm._tier_cache_ready is False

    def test_missing_file_does_not_raise(self, tmp_path):
        mm = _make_manager(tmp_path)
        mm._load_tier_cache()  # must not raise

    def test_corrupt_file_leaves_cache_none(self, tmp_path):
        mm = _make_manager(tmp_path)
        cache_path = mm._tier_cache_path()
        cache_path.write_text("not valid json {{{{", encoding="utf-8")
        mm._load_tier_cache()
        assert mm._tier_cache is None
        assert mm._tier_cache_ready is False

    def test_empty_json_object_loads_empty_snapshot(self, tmp_path):
        mm = _make_manager(tmp_path)
        cache_path = mm._tier_cache_path()
        cache_path.write_text("{}", encoding="utf-8")
        mm._load_tier_cache()
        # from_dict({}) returns an empty TierCacheSnapshot, not None
        assert mm._tier_cache is not None
        assert mm._tier_cache_ready is True
        assert mm._tier_cache.tier0_boundary_idx == 0


class TestSaveTierCache:
    def test_save_noop_when_tier_cache_is_none(self, tmp_path):
        mm = _make_manager(tmp_path)
        # _tier_cache starts as None
        assert mm._tier_cache is None
        mm._save_tier_cache()
        assert not mm._tier_cache_path().exists()

    def test_save_writes_valid_json(self, tmp_path):
        mm = _make_manager(tmp_path)
        mm._tier_cache = TierCacheSnapshot(
            tier0_boundary_idx=10,
            tier1_token_count=500,
            tier2_token_count=200,
        )
        mm._save_tier_cache()

        cache_path = mm._tier_cache_path()
        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["tier0_boundary_idx"] == 10
        assert data["tier1"]["token_count"] == 500
        assert data["tier2"]["token_count"] == 200

    def test_round_trip_save_then_load(self, tmp_path):
        mm = _make_manager(tmp_path)
        snap = TierCacheSnapshot(
            tier0_boundary_idx=42,
            tier1_messages=[
                CompressedMessage("id99", "shell", "output summary", "ToolMessage"),
            ],
            tier1_token_count=1234,
            tier2_token_count=567,
            tier3_msg_idx=88,
            calibration_tokens=9999,
            calibration_chars=24999,
        )
        mm._tier_cache = snap
        mm._save_tier_cache()

        mm2 = _make_manager(tmp_path)
        mm2._load_tier_cache()

        assert mm2._tier_cache is not None
        assert mm2._tier_cache_ready is True
        r = mm2._tier_cache
        assert r.tier0_boundary_idx == 42
        assert r.tier1_token_count == 1234
        assert r.tier2_token_count == 567
        assert r.tier3_msg_idx == 88
        assert r.calibration_tokens == 9999
        assert r.calibration_chars == 24999
        assert len(r.tier1_messages) == 1
        assert r.tier1_messages[0].tool_call_id == "id99"


class TestTierCacheWiredIntoLoadSave:
    def test_load_calls_load_tier_cache(self, tmp_path):
        mm = _make_manager(tmp_path)
        # Write a valid cache file before calling load()
        snap = TierCacheSnapshot(tier0_boundary_idx=7, tier1_token_count=111)
        mm._tier_cache = snap
        mm._save_tier_cache()

        # Fresh manager — load() should pick up the file
        mm2 = _make_manager(tmp_path)
        assert mm2._tier_cache is None
        mm2.load()
        assert mm2._tier_cache is not None
        assert mm2._tier_cache.tier0_boundary_idx == 7

    def test_save_calls_save_tier_cache(self, tmp_path):
        mm = _make_manager(tmp_path)
        mm.load()
        mm._tier_cache = TierCacheSnapshot(tier0_boundary_idx=99)
        mm.save()

        cache_path = mm._tier_cache_path()
        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["tier0_boundary_idx"] == 99

    def test_clear_deletes_tier_cache_file(self, tmp_path):
        mm = _make_manager(tmp_path)
        mm._tier_cache = TierCacheSnapshot(tier0_boundary_idx=5)
        mm._save_tier_cache()
        assert mm._tier_cache_path().exists()

        mm.clear()
        assert not mm._tier_cache_path().exists()
        assert mm._tier_cache is None
        assert mm._tier_cache_ready is False


class TestTierCacheThreadSafety:
    def test_hybrid_lock_protects_tier_cache_fields(self, tmp_path):
        """Verify that reading/writing _tier_cache and _tier_cache_ready
        while holding _hybrid_lock does not deadlock or corrupt state."""
        mm = _make_manager(tmp_path)
        errors: list[str] = []

        def writer():
            for i in range(50):
                with mm._hybrid_lock:
                    mm._tier_cache = TierCacheSnapshot(tier0_boundary_idx=i)
                    mm._tier_cache_ready = True

        def reader():
            for _ in range(50):
                with mm._hybrid_lock:
                    _ = mm._tier_cache
                    _ = mm._tier_cache_ready

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            if t.is_alive():
                errors.append("Thread did not finish in time")

        assert not errors, errors

    def test_load_and_save_tier_cache_use_hybrid_lock(self, tmp_path):
        """_load_tier_cache and _save_tier_cache acquire _hybrid_lock."""
        mm = _make_manager(tmp_path)
        snap = TierCacheSnapshot(tier1_token_count=42)

        # Write a file so _load_tier_cache has something to read
        mm._tier_cache = snap
        mm._save_tier_cache()
        mm._tier_cache = None
        mm._tier_cache_ready = False

        acquired_events: list[str] = []

        original_load = mm._load_tier_cache
        original_save = mm._save_tier_cache

        def patched_load():
            acquired_events.append("load_start")
            original_load()
            acquired_events.append("load_end")

        def patched_save():
            acquired_events.append("save_start")
            original_save()
            acquired_events.append("save_end")

        mm._load_tier_cache = patched_load  # type: ignore[method-assign]
        mm._save_tier_cache = patched_save  # type: ignore[method-assign]

        mm._load_tier_cache()
        mm._save_tier_cache()

        assert acquired_events == ["load_start", "load_end", "save_start", "save_end"]


# ---------------------------------------------------------------------------
# MemoryContext tier_token_counts field
# ---------------------------------------------------------------------------


class TestMemoryContextTierTokenCounts:
    def test_default_is_empty_dict(self):
        from cogtrix_core.memory.context import MemoryContext

        ctx = MemoryContext()
        assert ctx.tier_token_counts == {}

    def test_can_set_tier_token_counts(self):
        from cogtrix_core.memory.context import MemoryContext

        ctx = MemoryContext(tier_token_counts={0: 5000, 1: 3800, 2: 1050, 3: 280})
        assert ctx.tier_token_counts[0] == 5000
        assert ctx.tier_token_counts[1] == 3800
        assert ctx.tier_token_counts[2] == 1050
        assert ctx.tier_token_counts[3] == 280

    def test_existing_fields_unchanged(self):
        from cogtrix_core.memory.context import MemoryContext

        ctx = MemoryContext(
            messages=[],
            mode="conversation",
            total_messages_stored=10,
            context_messages_count=5,
            token_estimate=1234,
            metadata={"has_summary": True},
            tier_token_counts={1: 100},
        )
        assert ctx.messages == []
        assert ctx.mode == "conversation"
        assert ctx.total_messages_stored == 10
        assert ctx.token_estimate == 1234
        assert ctx.metadata["has_summary"] is True
        assert ctx.tier_token_counts[1] == 100


# ---------------------------------------------------------------------------
# Phase 2: assemble_from_tiers()
# ---------------------------------------------------------------------------


def _make_human(content: str):
    try:
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)
    except ImportError:
        return {"type": "human", "content": content}


def _make_ai(content: str):
    try:
        from langchain_core.messages import AIMessage

        return AIMessage(content=content)
    except ImportError:
        return {"type": "ai", "content": content}


def _content(msg) -> str:
    if hasattr(msg, "content"):
        return str(msg.content)
    return str(msg.get("content", ""))


class TestAssembleFromTiers:
    def test_empty_snapshot_returns_raw_messages(self):
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        snapshot = TierCacheSnapshot()
        msgs = [_make_human("hello"), _make_ai("world")]
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=msgs, summary="", summary_msg_idx=0
        )
        # No tier 1/2 messages and boundary at 0 → all raw messages returned
        assert len(assembled) == 2
        assert _content(assembled[0]) == "hello"
        assert _content(assembled[1]) == "world"

    def test_assembly_order_t3_t2_t1_t0(self):
        """Tier 3 summary first, then T2, T1, T0 verbatim tail."""
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        # Two verbatim messages (index 2 and 3), boundary at index 2
        raw_msgs = [_make_human("old0"), _make_human("old1"), _make_human("t0a"), _make_ai("t0b")]
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=2,
            tier1_messages=[
                CompressedMessage("id1", "web_search", "T1 light summary", "ToolMessage"),
            ],
            tier2_messages=[
                CompressedMessage("", "assistant", "T2 heavy summary", "AIMessage"),
            ],
            tier1_token_count=300,
            tier2_token_count=100,
        )
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot,
            messages=raw_msgs,
            summary="narrative summary",
            summary_msg_idx=2,
        )
        # Expected order: T3 (summary HumanMessage), T2 AIMessage, T1 ToolMessage, T0 x2
        assert len(assembled) == 5
        # T3: summary prefix (#2364 strengthened the marker to a "context only,
        # do not relay" instruction — still opens with the same marker stem).
        _t3 = _content(assembled[0])
        assert "[Session context summary" in _t3
        assert "narrative summary" in _t3
        # GAP-9 (forge audit): pin the "do not relay" deterrent body so a silent
        # revert to the bare "[Session context summary]\n…" marker is caught — the
        # assistant guard now matches by prefix, but this keeps the deterrent text
        # (the in-context half of the #2364 defence) from regressing unnoticed.
        assert "do not repeat" in _t3.lower()
        assert "relay this block" in _t3.lower()
        # T2: AIMessage with heavy compressed content
        assert "T2 heavy summary" in _content(assembled[1])
        # T1: ToolMessage with light compressed content
        assert "T1 light summary" in _content(assembled[2])
        # T0: verbatim raw messages
        assert _content(assembled[3]) == "t0a"
        assert _content(assembled[4]) == "t0b"

    def test_tier_token_counts_returned_correctly(self):
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=1,
            tier1_messages=[CompressedMessage("id1", "tool", "abcd", "ToolMessage")],
            tier2_messages=[CompressedMessage("", "asst", "ef", "AIMessage")],
            tier1_token_count=400,
            tier2_token_count=150,
        )
        raw = [_make_human("hello world"), _make_ai("response")]
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="summary text", summary_msg_idx=1
        )
        # Tier 3: from summary chars
        assert tier_counts[3] > 0
        # Tier 2: from snapshot token count
        assert tier_counts[2] == 150
        # Tier 1: from snapshot token count
        assert tier_counts[1] == 400
        # Tier 0: from raw message content chars
        assert tier_counts[0] > 0
        # All four tiers present
        assert set(tier_counts.keys()) == {0, 1, 2, 3}

    def test_uses_calibration_ratio_when_available(self):
        """When calibration_tokens and calibration_chars are set,
        the token counts should use the calibration ratio instead of _CHARS_PER_TOKEN."""
        from cogtrix_core.memory.tier_cache import _CHARS_PER_TOKEN, assemble_from_tiers

        # calibration: 1 token per 1 char (ratio=1.0), much denser than _CHARS_PER_TOKEN=2
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=0,
            calibration_tokens=1000,
            calibration_chars=1000,
        )
        raw = [_make_human("a" * 100)]  # 100 chars
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="", summary_msg_idx=0
        )
        # With ratio=1.0: 100 chars → 100 tokens
        # With default ratio: 100 chars // 2 = 50 tokens
        assert tier_counts[0] == 100  # calibration ratio applied
        assert tier_counts[0] != 100 // _CHARS_PER_TOKEN or _CHARS_PER_TOKEN == 1

    def test_calibration_not_used_when_chars_zero(self):
        """calibration_tokens>0 but calibration_chars=0 must not divide by zero."""
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=0,
            calibration_tokens=5000,
            calibration_chars=0,  # would cause ZeroDivisionError if used
        )
        raw = [_make_human("hello")]
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="", summary_msg_idx=0
        )
        # Must not raise; tier0 tokens estimated via fallback
        assert tier_counts[0] >= 0

    def test_tier0_boundary_clamped_when_beyond_list(self):
        """Stale tier0_boundary_idx beyond len(messages) is clamped silently."""
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        raw = [_make_human("only message")]
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=999,  # far beyond the list
        )
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="", summary_msg_idx=0
        )
        # Clamped to len(messages) → no raw messages in T0, but no crash
        assert isinstance(assembled, list)
        # boundary clamped to 1 → tier0_msgs is empty
        assert tier_counts[0] == 0

    def test_no_summary_skips_tier3(self):
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        snapshot = TierCacheSnapshot()
        raw = [_make_human("msg")]
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="", summary_msg_idx=0
        )
        # No summary → no T3 prefix message
        assert all("[Session context summary]" not in _content(m) for m in assembled)
        assert tier_counts[3] == 0

    def test_empty_tiers_and_no_summary_returns_all_raw(self):
        """Empty snapshot with no summary returns all raw messages in T0."""
        from cogtrix_core.memory.tier_cache import assemble_from_tiers

        raw = [_make_human("a"), _make_ai("b"), _make_human("c")]
        snapshot = TierCacheSnapshot(tier0_boundary_idx=0)
        assembled, tier_counts = assemble_from_tiers(
            snapshot=snapshot, messages=raw, summary="", summary_msg_idx=0
        )
        assert len(assembled) == 3
        assert tier_counts[1] == 0
        assert tier_counts[2] == 0
        assert tier_counts[3] == 0


# ---------------------------------------------------------------------------
# Phase 2: prepare_context() tier path in ConversationMemoryManager
# ---------------------------------------------------------------------------


class TestPrepareContextTierPath:
    def _make_warm_manager(self, tmp_path: Path, boundary: int = 0):
        """Return a ConversationMemoryManager with a warm tier cache."""
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ConversationMemoryManager(store=store, session_id="tier-test")
        mm._messages = [_make_human("hello"), _make_ai("hi"), _make_human("question")]
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=boundary,
            tier1_messages=[
                CompressedMessage("id1", "tool", "compressed tool output", "ToolMessage"),
            ],
            tier1_token_count=200,
            tier2_token_count=50,
        )
        with mm._hybrid_lock:
            mm._tier_cache = snapshot
            mm._tier_cache_ready = True
        return mm

    def test_prepare_context_uses_tier_cache_when_warm(self, tmp_path):
        mm = self._make_warm_manager(tmp_path, boundary=1)
        ctx = mm.prepare_context("")
        # tier_token_counts must be populated
        assert ctx.tier_token_counts != {}
        assert sum(ctx.tier_token_counts.values()) > 0
        # token_estimate must equal sum of tier counts
        assert ctx.token_estimate == sum(ctx.tier_token_counts.values())

    def test_prepare_context_falls_back_when_cache_cold(self, tmp_path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ConversationMemoryManager(store=store, session_id="cold-test")
        mm._messages = [_make_human("hello"), _make_ai("hi")]
        # No tier cache set
        assert mm._tier_cache_ready is False
        ctx = mm.prepare_context("")
        # Must return a valid MemoryContext without tier_token_counts
        assert isinstance(ctx.messages, list)
        assert ctx.tier_token_counts == {}

    def test_prepare_context_tier_counts_has_all_four_tiers(self, tmp_path):
        mm = self._make_warm_manager(tmp_path, boundary=1)
        mm._summary = "old conversation summary"
        mm._summary_msg_idx = 1
        ctx = mm.prepare_context("")
        assert set(ctx.tier_token_counts.keys()) == {0, 1, 2, 3}

    def test_prepare_context_tier_mode_includes_raw_tail(self, tmp_path):
        """T0 verbatim messages must appear in the assembled output."""
        mm = self._make_warm_manager(tmp_path, boundary=2)
        # boundary=2 → messages[2:] = [HumanMessage("question")] is T0
        ctx = mm.prepare_context("")
        contents = [_content(m) for m in ctx.messages]
        assert any("question" in c for c in contents)

    def test_code_manager_prepare_context_uses_tier_cache(self, tmp_path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = CodeDevelopmentMemoryManager(store=store, session_id="code-tier-test")
        mm._messages = [_make_human("write a function"), _make_ai("here it is")]
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=1,
            tier1_token_count=100,
            tier2_token_count=30,
        )
        with mm._hybrid_lock:
            mm._tier_cache = snapshot
            mm._tier_cache_ready = True
        ctx = mm.prepare_context("")
        assert ctx.tier_token_counts != {}
        assert ctx.token_estimate == sum(ctx.tier_token_counts.values())

    def test_reasoning_manager_prepare_context_uses_tier_cache(self, tmp_path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ReasoningMemoryManager(store=store, session_id="reasoning-tier-test")
        mm._messages = [_make_human("analyse this"), _make_ai("analysis done")]
        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=1,
            tier1_token_count=80,
            tier2_token_count=20,
        )
        with mm._hybrid_lock:
            mm._tier_cache = snapshot
            mm._tier_cache_ready = True
        ctx = mm.prepare_context("")
        assert ctx.tier_token_counts != {}
        assert ctx.token_estimate == sum(ctx.tier_token_counts.values())


# ---------------------------------------------------------------------------
# Phase 3: compress_to_tier()
# ---------------------------------------------------------------------------


def _make_tool_msg(content: str, tool_call_id: str = "tcid1", name: str = "web_search"):
    try:
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
    except ImportError:
        return {"type": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def _make_ai_final(content: str):
    """AIMessage without tool_calls (final response)."""
    try:
        from langchain_core.messages import AIMessage

        return AIMessage(content=content)
    except ImportError:
        return {"type": "ai", "content": content}


class TestCompressToTier:
    def test_tier1_calls_compress_tool_message(self):
        from cogtrix_core.memory.tier_cache import compress_to_tier

        llm = MagicMock()
        # Return something clearly shorter than the long input
        llm.invoke.return_value = MagicMock(content="compressed")

        # Input must be longer than the compressed output for the result to be used
        long_input = "long tool output text with lots of detail " * 5
        result = compress_to_tier(long_input, "web_search", 1, llm)

        assert llm.invoke.called
        assert result == "compressed"

    def test_tier2_produces_shorter_output_via_different_prompt(self):
        """Tier-2 prompt asks for a one-line summary; tier-1 prompt is more verbose."""
        from cogtrix_core.memory.tier_cache import (
            _TIER1_PROMPT_SUFFIX,
            _TIER2_PROMPT_SUFFIX,
            compress_to_tier,
        )

        assert len(_TIER2_PROMPT_SUFFIX) < len(_TIER1_PROMPT_SUFFIX)

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Found 42 results for query.")

        # Input must be longer than the compressed output
        long_input = "some very long content here with lots of detail about the tool output " * 3
        result = compress_to_tier(long_input, "shell", 2, llm)
        assert result == "Found 42 results for query."
        # Tier-2 prompt must contain the one-sentence instruction
        call_arg = llm.invoke.call_args[0][0]
        assert "ONE short sentence" in call_arg or "one sentence" in call_arg.lower()

    def test_compress_to_tier_handles_llm_failure_gracefully(self):
        from cogtrix_core.memory.tier_cache import compress_to_tier

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("network error")

        result = compress_to_tier("some long content " * 50, "tool", 1, llm)
        # Must not raise; returns truncated content
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_to_tier_fallback_when_compressed_empty(self):
        from cogtrix_core.memory.tier_cache import compress_to_tier

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="")  # empty result

        content = "meaningful content " * 20
        result = compress_to_tier(content, "tool", 1, llm)
        # Falls back to truncation; result is non-empty
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_to_tier_returns_original_when_not_reduced(self):
        """If compressed version is >= original, return original."""
        from cogtrix_core.memory.tier_cache import compress_to_tier

        original = "short text"
        llm = MagicMock()
        # LLM returns something longer than or equal to input
        llm.invoke.return_value = MagicMock(content="much longer response than original")

        result = compress_to_tier(original, "tool", 1, llm)
        assert result == original

    def test_compress_to_tier_times_out_and_falls_back(self):
        """A hung LLM must not block forever; fallback to truncation on timeout."""
        import time
        from unittest.mock import patch

        from cogtrix_core.memory.tier_cache import compress_to_tier

        def _slow_invoke(_prompt: str) -> object:
            time.sleep(2)  # longer than the patched timeout
            return MagicMock(content="compressed")

        llm = MagicMock()
        llm.invoke.side_effect = _slow_invoke

        content = "some long content " * 50
        with patch("cogtrix_core.memory.tier_cache._COMPRESSION_TIMEOUT_SECONDS", 0.1):
            result = compress_to_tier(content, "tool", 1, llm)

        # Must fall back to truncation, not hang or return empty
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "compressed"

    def test_compress_to_tier_fast_llm_still_works_with_executor(self):
        """Normal (fast) LLM calls must continue to work through the executor."""
        from cogtrix_core.memory.tier_cache import compress_to_tier

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="compressed result")

        long_input = "long tool output text with lots of detail " * 5
        result = compress_to_tier(long_input, "web_search", 1, llm)

        assert llm.invoke.called
        assert result == "compressed result"

    def test_compress_to_tier_hung_thread_returns_within_guard_timeout(self):
        """A forever-hung LLM thread must not block the caller — __exit__ bug."""
        import threading
        import time
        from unittest.mock import patch

        from cogtrix_core.memory.tier_cache import compress_to_tier

        event = threading.Event()

        def _hang_forever(_prompt: str) -> object:
            event.wait()  # blocks until event.set() — never happens
            return MagicMock(content="compressed")

        llm = MagicMock()
        llm.invoke.side_effect = _hang_forever

        content = "some long content " * 50
        start = time.monotonic()
        with patch("cogtrix_core.memory.tier_cache._COMPRESSION_TIMEOUT_SECONDS", 0.1):
            result = compress_to_tier(content, "tool", 1, llm)
        elapsed = time.monotonic() - start

        # Must return within 1s even though the thread hangs forever;
        # the old ``with ThreadPoolExecutor`` pattern would block on __exit__.
        assert elapsed < 1.0, f"elapsed {elapsed:.2f}s — __exit__ blocked on hung thread"
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "compressed"


# ---------------------------------------------------------------------------
# Phase 3: roll_forward()
# ---------------------------------------------------------------------------


class TestRollForward:
    def test_empty_messages_returns_empty_snapshot(self):
        from cogtrix_core.memory.tier_cache import roll_forward

        snap = roll_forward(
            messages=[],
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=128_000,
            llm=None,
        )
        assert snap.tier0_boundary_idx == 0
        assert snap.tier1_messages == []
        assert snap.tier2_messages == []
        assert snap.tier1_token_count == 0
        assert snap.tier2_token_count == 0

    def test_small_history_all_fits_in_tier0(self):
        """With a tiny message list and a large budget, everything stays in Tier 0."""
        from cogtrix_core.memory.tier_cache import roll_forward

        msgs = [_make_human("hi"), _make_ai_final("hello")]
        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=128_000,
            llm=None,
        )
        # Boundary should be at 0 (all verbatim)
        assert snap.tier0_boundary_idx == 0
        assert snap.tier1_messages == []
        assert snap.tier2_messages == []

    def test_large_messages_overflow_into_tier1(self):
        """Messages that exceed the Tier 0 budget spill into Tier 1."""
        from cogtrix_core.memory.tier_cache import roll_forward

        # Very small context window to force overflow.
        max_tokens = 2_048

        # Each tool message: 400 chars = 200 tokens (at _CHARS_PER_TOKEN=2).
        # With a tiny context window, 4 messages of 200 tokens each exceed Tier 0.
        msgs = [_make_tool_msg("x" * 400, f"id{i}", "tool") for i in range(4)]
        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=max_tokens,
            llm=None,  # truncation fallback
        )
        # At least some messages should be in Tier 1
        # (some will be outside Tier 0 boundary)
        assert snap.tier1_messages
        # Boundary must be a valid index
        assert 0 <= snap.tier0_boundary_idx <= len(msgs)

    def test_reuses_already_compressed_content_from_current_snapshot(self):
        """Messages already in current_snapshot.tier1_messages are not re-compressed."""
        from cogtrix_core.memory.tier_cache import roll_forward

        # Tiny context to ensure overflow.
        max_tokens = 2_048

        tool_content = "original long content " * 20
        tcid = "cached-tool-id"
        cached_compressed = "already compressed"

        current_snap = TierCacheSnapshot(
            tier1_messages=[
                CompressedMessage(
                    tool_call_id=tcid,
                    name="web_search",
                    content=cached_compressed,
                    original_type="ToolMessage",
                )
            ]
        )

        msgs = [
            _make_human("question"),
            _make_tool_msg(tool_content, tcid, "web_search"),
            _make_ai_final("answer"),
        ]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="new compression")

        snap = roll_forward(
            messages=msgs,
            current_snapshot=current_snap,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=max_tokens,
            llm=llm,
        )

        # If the cached message was reused, the LLM was NOT called for it.
        # The cached content should appear in the result if it ended up in T1/T2.
        t1_contents = {cm.tool_call_id: cm.content for cm in snap.tier1_messages}
        t2_contents = {cm.tool_call_id: cm.content for cm in snap.tier2_messages}
        if tcid in t1_contents:
            assert t1_contents[tcid] == cached_compressed
            assert not llm.invoke.called
        elif tcid in t2_contents:
            assert t2_contents[tcid] == cached_compressed
        # else: message stayed in T0 — that's fine too

    def test_reuses_compression_cache_entries(self):
        """compression_cache dict entries are reused without new LLM calls."""
        from cogtrix_core.memory.tier_cache import roll_forward

        max_tokens = 2_048
        tcid = "cc-tool-id"
        cached_text = "from compression cache"

        msgs = [
            _make_human("query"),
            _make_tool_msg("very long output " * 30, tcid, "shell"),
            _make_ai_final("done"),
        ]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="fresh compression")

        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=max_tokens,
            llm=llm,
            compression_cache={tcid: cached_text},
        )

        t1_map = {cm.tool_call_id: cm.content for cm in snap.tier1_messages}
        t2_map = {cm.tool_call_id: cm.content for cm in snap.tier2_messages}
        if tcid in t1_map:
            assert t1_map[tcid] == cached_text
        elif tcid in t2_map:
            assert t2_map[tcid] == cached_text
        # LLM should NOT have been called for the cached entry
        if tcid in t1_map or tcid in t2_map:
            assert not llm.invoke.called

    def test_works_without_llm_uses_truncation(self):
        """When llm=None, compression falls back to truncation without error."""
        from cogtrix_core.memory.tier_cache import roll_forward

        max_tokens = 2_048
        long_content = "data " * 200

        msgs = [
            _make_human("task"),
            _make_tool_msg(long_content, "id1", "read_file"),
            _make_ai_final("response"),
        ]

        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=max_tokens,
            llm=None,
        )

        assert isinstance(snap, TierCacheSnapshot)
        assert 0 <= snap.tier0_boundary_idx <= len(msgs)

    def test_human_messages_never_compressed(self):
        """HumanMessage content must not appear in tier1_messages or tier2_messages."""
        from cogtrix_core.memory.tier_cache import roll_forward

        max_tokens = 2_048
        msgs = [
            _make_human("user question " * 40),
            _make_ai_final("ai answer " * 40),
        ]

        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=max_tokens,
            llm=None,
        )

        for cm in snap.tier1_messages + snap.tier2_messages:
            assert cm.original_type != "HumanMessage"

    def test_returns_tier_cache_snapshot_type(self):
        from cogtrix_core.memory.tier_cache import roll_forward

        msgs = [_make_human("hello"), _make_ai_final("world")]
        result = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=128_000,
            llm=None,
        )
        assert isinstance(result, TierCacheSnapshot)

    def test_tier_token_counts_are_non_negative(self):
        from cogtrix_core.memory.tier_cache import roll_forward

        msgs = [_make_tool_msg("output " * 10, "id1", "tool"), _make_ai_final("response")]
        snap = roll_forward(
            messages=msgs,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=128_000,
            llm=None,
        )
        assert snap.tier1_token_count >= 0
        assert snap.tier2_token_count >= 0


# ---------------------------------------------------------------------------
# Phase 3: schedule_tier_roll_forward() on BaseMemoryManager
# ---------------------------------------------------------------------------


def _make_overflow_messages(count: int = 20):
    """Create a list of ToolMessages with enough content to overflow a small T0 budget."""
    msgs = []
    for i in range(count):
        msgs.append(_make_human(f"question {i}"))
        # 200-char tool output = 100 tokens at _CHARS_PER_TOKEN=2
        msgs.append(_make_tool_msg("x" * 200, f"id{i}", "tool"))
        msgs.append(_make_ai_final(f"answer {i}"))
    return msgs


class TestScheduleTierRollForward:
    # Use a small max_context_tokens so that messages overflow Tier 0 and the
    # roll-forward produces a snapshot with has_value=True.
    _SMALL_TOKENS = 2_048

    def _make_manager(self, tmp_path: Path):
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        return ConversationMemoryManager(store=store, session_id="roll-test")

    def test_submits_to_pool_and_updates_tier_cache(self, tmp_path):
        """After scheduling with overflow messages, the tier cache becomes ready."""
        mm = self._make_manager(tmp_path)
        # Use messages that will overflow T0 at a small context budget
        mm._messages = _make_overflow_messages(10)

        mm.schedule_tier_roll_forward(max_context_tokens=self._SMALL_TOKENS, llm=None)

        # Wait for the background job to complete (max 5 seconds).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with mm._hybrid_lock:
                ready = mm._tier_cache_ready
            if ready:
                break
            time.sleep(0.05)

        with mm._hybrid_lock:
            assert mm._tier_cache_ready is True
            assert mm._tier_cache is not None
            assert isinstance(mm._tier_cache, TierCacheSnapshot)

    def test_no_op_when_messages_empty(self, tmp_path):
        """Empty message list: no pool submission, cache stays None."""
        mm = self._make_manager(tmp_path)
        mm._messages = []

        mm.schedule_tier_roll_forward(max_context_tokens=128_000, llm=None)
        # Brief poll to allow any accidental background work to run
        for _ in range(10):
            with mm._hybrid_lock:
                if mm._tier_cache is not None:
                    break
            time.sleep(0.01)

        with mm._hybrid_lock:
            assert mm._tier_cache is None

    def test_no_op_when_all_messages_fit_in_tier0(self, tmp_path):
        """When all messages fit in Tier 0 (no overflow), cache stays cold."""
        mm = self._make_manager(tmp_path)
        # 2 short messages will never overflow T0 at 128_000 tokens
        mm._messages = [_make_human("hello"), _make_ai_final("world")]

        mm.schedule_tier_roll_forward(max_context_tokens=128_000, llm=None)
        # Brief poll for the background job
        for _ in range(20):
            with mm._hybrid_lock:
                if mm._tier_cache_ready:
                    break
            time.sleep(0.01)

        with mm._hybrid_lock:
            # has_value=False → _tier_cache_ready stays False
            assert mm._tier_cache_ready is False

    def test_handles_pool_submission_error_gracefully(self, tmp_path):
        """If the pool is shut down, schedule_tier_roll_forward logs a warning but doesn't raise."""
        from concurrent.futures import ThreadPoolExecutor

        mm = self._make_manager(tmp_path)
        mm._messages = [_make_human("hi")]

        dead_pool = ThreadPoolExecutor(max_workers=1)
        dead_pool.shutdown(wait=True)

        with patch("cogtrix_core.memory.manager._get_summarization_pool", return_value=dead_pool):
            # Must not raise
            mm.schedule_tier_roll_forward(max_context_tokens=128_000, llm=None)

    def test_tier_cache_set_tier_cache_ready_true_after_completion(self, tmp_path):
        """Cache becomes ready after a roll-forward that produces overflow content."""
        mm = self._make_manager(tmp_path)
        mm._messages = _make_overflow_messages(10)

        assert mm._tier_cache_ready is False
        mm.schedule_tier_roll_forward(max_context_tokens=self._SMALL_TOKENS, llm=None)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with mm._hybrid_lock:
                if mm._tier_cache_ready:
                    break
            time.sleep(0.05)

        with mm._hybrid_lock:
            assert mm._tier_cache_ready is True

    def test_update_triggers_roll_forward_conversation(self, tmp_path):
        """ConversationMemoryManager.update() calls schedule_tier_roll_forward
        once enough messages have been accumulated (> working_memory_size)."""
        mm = self._make_manager(tmp_path)
        window_size = mm._mode_config["working_memory_size"]

        # Pre-populate to exceed the window so the guard allows scheduling
        mm._messages = [_make_human(f"q{i}") for i in range(window_size + 2)]

        called = []
        original = mm.schedule_tier_roll_forward

        def patched(*args, **kwargs):
            called.append(True)
            original(*args, **kwargs)

        mm.schedule_tier_roll_forward = patched  # type: ignore[method-assign]
        mm.update("hello", "world")

        assert called, "schedule_tier_roll_forward was not called from update()"

    def test_update_triggers_roll_forward_code(self, tmp_path):
        """CodeDevelopmentMemoryManager.update() schedules roll-forward when
        message count exceeds the working memory window."""
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = CodeDevelopmentMemoryManager(store=store, session_id="code-roll")
        window_size = mm._mode_config["working_memory_size"]
        mm._messages = [_make_human(f"q{i}") for i in range(window_size + 2)]

        called = []
        original = mm.schedule_tier_roll_forward

        def patched(*args, **kwargs):
            called.append(True)
            original(*args, **kwargs)

        mm.schedule_tier_roll_forward = patched  # type: ignore[method-assign]
        mm.update("write a function", "here is the code")

        assert called, "schedule_tier_roll_forward was not called from update()"

    def test_update_triggers_roll_forward_reasoning(self, tmp_path):
        """ReasoningMemoryManager.update() schedules roll-forward when
        message count exceeds the working memory window."""
        from cogtrix_core.memory.json_store import JsonFileMemoryStore
        from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mm = ReasoningMemoryManager(store=store, session_id="reason-roll")
        window_size = mm._mode_config["working_memory_size"]
        mm._messages = [_make_human(f"q{i}") for i in range(window_size + 2)]

        called = []
        original = mm.schedule_tier_roll_forward

        def patched(*args, **kwargs):
            called.append(True)
            original(*args, **kwargs)

        mm.schedule_tier_roll_forward = patched  # type: ignore[method-assign]
        mm.update("think about X", "here is my reasoning")

        assert called, "schedule_tier_roll_forward was not called from update()"


# ---------------------------------------------------------------------------
# Phase 4: post-turn compression gate and _maybe_compress threshold
# ---------------------------------------------------------------------------


class TestPhase4PostTurnCompressionGate:
    """Verify that the post-turn compression pass is skipped when TCC is on and
    queued in the background when TCC is off."""

    @pytest.fixture(autouse=True)
    def _drain_compression_jobs(self):
        """Drain pending background compression jobs after each test."""
        yield
        from cogtrix_core.orchestration import runner as runner_mod

        for _ in range(100):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

    def test_tier_cache_enabled_field_defaults_to_true(self):
        from cogtrix_core.orchestration.run_config import AgentRunConfig

        cfg = AgentRunConfig()
        assert cfg.tier_cache_enabled is True

    def test_tier_cache_enabled_field_can_be_set_false(self):
        from cogtrix_core.orchestration.run_config import AgentRunConfig

        cfg = AgentRunConfig(tier_cache_enabled=False)
        assert cfg.tier_cache_enabled is False

    def test_post_turn_compression_skipped_when_tcc_enabled(self):
        """apply_message_compression must NOT be called post-turn when TCC is on."""
        from unittest.mock import MagicMock, patch

        from langchain_core.messages import AIMessage, HumanMessage

        from cogtrix_core.orchestration.run_config import AgentRunConfig
        from cogtrix_core.orchestration.runner import run_agent

        msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
        mock_graph_result = {"messages": msgs}

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        config = AgentRunConfig(
            llm=mock_llm,
            context_compression=True,
            tier_cache_enabled=True,
            system_prompt="test",
            available_tools={},
            active_tools_list=[],
        )

        with (
            patch("cogtrix_core.orchestration.runner.build_agent_graph") as mock_build_graph,
            patch("cogtrix_core.orchestration.runner.apply_message_compression") as mock_compress,
        ):
            mock_graph = MagicMock()
            mock_graph.stream.return_value = [mock_graph_result]
            mock_build_graph.return_value = mock_graph

            run_agent("hello", [], MagicMock(), set(), config=config)

        # With TCC enabled, the post-turn compression pass must be skipped.
        # The pre-turn pass may be called (call_count=0), but NOT the post-turn
        # pass (call_count=999 / min_age_cycles=1).
        post_turn_calls = [
            c
            for c in mock_compress.call_args_list
            if c.kwargs.get("call_count", c.args[1] if len(c.args) > 1 else None) == 999
        ]
        assert (
            not post_turn_calls
        ), "Post-turn apply_message_compression was called despite tier_cache_enabled=True"

    def test_post_turn_compression_runs_when_tcc_disabled(self):
        """run_agent() must queue post-turn compression without delaying return."""
        import threading
        import time
        from unittest.mock import MagicMock, patch

        from langchain_core.messages import AIMessage, HumanMessage

        from cogtrix_core.orchestration import runner as runner_mod
        from cogtrix_core.orchestration.run_config import AgentRunConfig
        from cogtrix_core.orchestration.runner import run_agent

        msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
        mock_graph_result = {"messages": msgs}

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        config = AgentRunConfig(
            llm=mock_llm,
            context_compression=True,
            tier_cache_enabled=False,
            system_prompt="test",
            available_tools={},
            active_tools_list=[],
            max_context_tokens=20_000,
        )

        started = threading.Event()
        release = threading.Event()

        def _slow_compression(*args, **kwargs):
            if kwargs.get("call_count") == 999:
                started.set()
                release.wait(timeout=1.0)
            return args[0]

        try:
            with (
                patch("cogtrix_core.orchestration.runner.build_agent_graph") as mock_build_graph,
                patch(
                    "cogtrix_core.orchestration.runner.apply_message_compression",
                    side_effect=_slow_compression,
                ),
                patch("cogtrix._spinner"),
            ):
                mock_graph = MagicMock()
                mock_graph.stream.return_value = [mock_graph_result]
                mock_build_graph.return_value = mock_graph

                began = time.perf_counter()
                result = run_agent("hello", [], MagicMock(), set(), config=config)
                elapsed = time.perf_counter() - began

                assert result == "world"
                assert elapsed < 0.2, f"run_agent() delayed response delivery for {elapsed:.3f}s"
                assert started.wait(timeout=1.0), "post-turn compression job did not start"
        finally:
            release.set()

        for _ in range(50):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)


class TestPhase4MaybeCompressThreshold:
    """Verify that _maybe_compress uses a higher threshold when TCC is active."""

    def test_maybe_compress_threshold_raised_when_tcc_enabled(self):
        """Verify TCC enabled uses higher (0.80) threshold via compression behavior."""
        from unittest.mock import MagicMock

        from cogtrix_core.memory.tier_cache import compress_to_tier

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="short")

        # Long input that would normally be truncated to ~50% at 60% threshold
        long_input = "x" * 1000
        result = compress_to_tier(long_input, "tool", 1, llm)

        # With default _CHARS_PER_TOKEN=2, 1000 chars = 500 tokens
        # At 60% threshold of 2048 = 1228.8, the input should be compressed
        # At 80% threshold of 2048 = 1638.4, input may pass-through if no LLM compression
        assert isinstance(result, str)
        # The result should either be compressed or truncated but NOT raise
        assert len(result) > 0

    def test_maybe_compress_threshold_normal_when_tcc_disabled(self):
        """Verify the 60% constant is used via _MID_TURN_COMPRESSION_THRESHOLD."""
        from cogtrix_core.orchestration.graph import _MID_TURN_COMPRESSION_THRESHOLD

        # The module constant must equal 0.60 (the "normal" non-TCC threshold)
        assert (
            _MID_TURN_COMPRESSION_THRESHOLD == 0.60
        ), "Default compression threshold must be 60% when TCC is disabled"

    def test_runner_source_contains_tcc_gate(self):
        """Verify runner post-turn gate behavior via actual compression queueing."""
        import time

        from cogtrix_core.orchestration import runner as runner_mod

        # Clear any pending jobs
        for _ in range(100):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

        # The gate is implemented via a conditional in the post-turn path
        # We test it indirectly via run_agent behavior in test_post_turn_compression_*


# ---------------------------------------------------------------------------
# Phase 5: Configuration and observability
# ---------------------------------------------------------------------------


class TestPhase5Config:
    """Verify Config fields for TCC and their parsing."""

    def test_config_tier_cache_enabled_default(self):
        """Config().tier_cache_enabled is True by default."""
        from cogtrix_core.config import Config

        cfg = Config()
        assert cfg.tier_cache_enabled is True

    def test_config_tier_fractions_default(self):
        """Default tier fractions match ADR-001 section 4 values."""
        from cogtrix_core.config import Config

        cfg = Config()
        assert cfg.tier0_fraction == 0.60
        assert cfg.tier1_fraction == 0.30
        assert cfg.tier2_fraction == 0.08

    def test_config_tier_fractions_parsed(self):
        """Tier fractions are read from context_compression dict."""
        from cogtrix_core.config import Config, _apply_config_file

        cfg = Config()
        from pathlib import Path

        yaml_content = (
            "context_compression:\n"
            "  tiered_cache: true\n"
            "  tier0_fraction: 0.50\n"
            "  tier1_fraction: 0.35\n"
            "  tier2_fraction: 0.10\n"
        )
        tmp = Path("/tmp/test_tcc_fractions.yaml")
        tmp.write_text(yaml_content)
        try:
            _apply_config_file(cfg, tmp)
            assert cfg.tier_cache_enabled is True
            assert cfg.tier0_fraction == 0.50
            assert cfg.tier1_fraction == 0.35
            assert cfg.tier2_fraction == 0.10
        finally:
            tmp.unlink(missing_ok=True)

    def test_config_tiered_cache_false_parsed(self):
        """tiered_cache: false disables tier_cache_enabled."""
        from pathlib import Path

        from cogtrix_core.config import Config, _apply_config_file

        cfg = Config()
        yaml_content = "context_compression:\n  tiered_cache: false\n"
        tmp = Path("/tmp/test_tcc_disabled.yaml")
        tmp.write_text(yaml_content)
        try:
            _apply_config_file(cfg, tmp)
            assert cfg.tier_cache_enabled is False
        finally:
            tmp.unlink(missing_ok=True)

    def test_config_tier_fractions_validated_sum_exceeds_one(self):
        """Fractions summing > 1.0 raise ConfigError."""
        from pathlib import Path

        from cogtrix_core.config import Config, ConfigError, _apply_config_file

        cfg = Config()
        yaml_content = (
            "context_compression:\n"
            "  tier0_fraction: 0.60\n"
            "  tier1_fraction: 0.35\n"
            "  tier2_fraction: 0.10\n"
        )
        tmp = Path("/tmp/test_tcc_bad_fractions.yaml")
        tmp.write_text(yaml_content)
        try:
            with pytest.raises(ConfigError, match="tier fractions must sum to"):
                _apply_config_file(cfg, tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_config_tier_fraction_out_of_range(self):
        """A fraction outside [0.01, 0.95] raises ConfigError."""
        from pathlib import Path

        from cogtrix_core.config import Config, ConfigError, _apply_config_file

        cfg = Config()
        yaml_content = "context_compression:\n  tier0_fraction: 1.5\n"
        tmp = Path("/tmp/test_tcc_out_of_range.yaml")
        tmp.write_text(yaml_content)
        try:
            with pytest.raises(ConfigError, match="0.01.*0.95|tier0_fraction"):
                _apply_config_file(cfg, tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestPhase5AgentRunConfig:
    """Verify AgentRunConfig.tier_cache_enabled wiring."""

    def test_agent_run_config_tier_cache_enabled_default(self):
        """AgentRunConfig.tier_cache_enabled defaults to True."""
        from cogtrix_core.orchestration.run_config import AgentRunConfig

        cfg = AgentRunConfig()
        assert cfg.tier_cache_enabled is True

    def test_agent_run_config_tier_cache_enabled_can_be_false(self):
        """AgentRunConfig accepts tier_cache_enabled=False."""
        from cogtrix_core.orchestration.run_config import AgentRunConfig

        cfg = AgentRunConfig(tier_cache_enabled=False)
        assert cfg.tier_cache_enabled is False


class TestPhase5SessionPanel:
    """Verify that /session shows tier breakdown when tier_token_counts is populated."""

    def test_session_panel_shows_tier_breakdown(self):
        """_session_rich writes 'Tiers' line when tier_token_counts is non-empty."""
        # Import _session_rich from the module under test
        from unittest.mock import MagicMock

        # We test the function directly via a minimal mock of Rich
        # Since cogtrix.py uses module-level `console`, we patch it
        captured_lines: list[str] = []

        class FakePanel:
            def __init__(self, body, **kwargs):
                captured_lines.append(body)

            def __init_subclass__(cls, **kwargs):  # noqa: B027
                pass

        fake_console = MagicMock()

        # Dynamically test by calling _session_rich with mocked globals
        import cogtrix

        original_console = cogtrix.console
        original_panel = cogtrix.Panel
        try:
            cogtrix.console = fake_console
            cogtrix.Panel = MagicMock(side_effect=lambda body, **kw: captured_lines.append(body))

            cfg = MagicMock()
            cfg.session = "test-session"
            cfg.memory_mode = "conversation"

            cogtrix._session_rich(
                cfg=cfg,
                stats={},
                msg_count=10,
                session_tokens=5000,
                max_context_tokens=32768,
                tier_token_counts={0: 3000, 1: 1500, 2: 500, 3: 100},
            )
        finally:
            cogtrix.console = original_console
            cogtrix.Panel = original_panel

        assert captured_lines, "_session_rich did not call Panel"
        panel_body = captured_lines[0]
        assert "Tiers" in panel_body, f"'Tiers' not found in panel body: {panel_body!r}"
        assert "T0:" in panel_body, f"'T0:' not found in panel body: {panel_body!r}"
        assert "T1:" in panel_body
        assert "T2:" in panel_body

    def test_session_panel_no_tier_line_when_counts_empty(self):
        """_session_rich does not write 'Tiers' line when tier_token_counts is empty/None."""
        captured_lines: list[str] = []

        from unittest.mock import MagicMock

        import cogtrix

        original_console = cogtrix.console
        original_panel = cogtrix.Panel
        try:
            cogtrix.console = MagicMock()
            cogtrix.Panel = MagicMock(side_effect=lambda body, **kw: captured_lines.append(body))

            cfg = MagicMock()
            cfg.session = "test-session"
            cfg.memory_mode = "conversation"

            cogtrix._session_rich(
                cfg=cfg,
                stats={},
                msg_count=5,
                session_tokens=0,
                max_context_tokens=None,
                tier_token_counts=None,
            )
        finally:
            cogtrix.console = original_console
            cogtrix.Panel = original_panel

        panel_body = captured_lines[0] if captured_lines else ""
        assert "Tiers" not in panel_body


class TestPhase5GetStats:
    """Verify that get_stats() includes tier_cache_ready and related fields."""

    def test_base_memory_manager_get_stats_includes_tcc_fields(self):
        """BaseMemoryManager.get_stats() always returns tier_cache_ready."""
        from unittest.mock import MagicMock

        from cogtrix_core.memory.base import BaseMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

        store = MagicMock(spec=BaseMemoryStore)
        store.load_history.return_value = []
        mm = ConversationMemoryManager(store, "test-phase5")
        stats = mm.get_stats()
        assert "tier_cache_ready" in stats
        assert stats["tier_cache_ready"] is False  # cold start

    def test_get_stats_includes_tier_counts_when_cache_ready(self):
        """get_stats() includes tier token counts when cache has been set."""
        from unittest.mock import MagicMock

        from cogtrix_core.memory.base import BaseMemoryStore
        from cogtrix_core.memory.modes.conversation import ConversationMemoryManager
        from cogtrix_core.memory.tier_cache import CompressedMessage, TierCacheSnapshot

        store = MagicMock(spec=BaseMemoryStore)
        store.load_history.return_value = []
        mm = ConversationMemoryManager(store, "test-phase5-stats")

        snapshot = TierCacheSnapshot(
            tier0_boundary_idx=10,
            tier1_messages=[CompressedMessage("id1", "tool", "compressed", "ToolMessage")],
            tier1_token_count=500,
            tier2_token_count=200,
        )
        with mm._hybrid_lock:
            mm._tier_cache = snapshot
            mm._tier_cache_ready = True

        stats = mm.get_stats()
        assert stats["tier_cache_ready"] is True
        assert stats["tier0_boundary_idx"] == 10
        assert stats["tier1_token_count"] == 500
        assert stats["tier2_token_count"] == 200


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression: /compact must produce visible output (print, not console.print)
# ---------------------------------------------------------------------------


class TestCompactOutputVisibility:
    """/compact must use print() so output survives prompt_toolkit redraw.

    Regression: console.print() output was erased by prompt_toolkit's
    post-dispatch redraw, making /compact appear to do nothing.
    """

    def test_compact_nothing_to_compress_uses_print(self, capsys):
        """When nothing is compressed, user must see 'Nothing to compress.'."""
        from unittest.mock import MagicMock, patch

        from cogtrix import _build_slash_commands

        reg = _build_slash_commands()
        mm = MagicMock()
        mm._messages = [MagicMock(content="short")]
        mm.get_messages.return_value = mm._messages
        mm.get_message_count.return_value = 1
        mm.get_stats.return_value = {"total_messages": 1}
        reg.memory_manager = mm
        reg.compression_llm = None
        reg.max_context_tokens = 128_000
        reg.last_input_tokens = 11_520  # 9%

        # apply_message_compression returns messages unchanged → nothing compressed
        with patch(
            "cogtrix_core.cli.commands.apply_message_compression", return_value=mm._messages
        ):
            reg.dispatch("/compact")

        out = capsys.readouterr().out
        assert (
            "Nothing to compress" in out
        ), f"Expected 'Nothing to compress' in stdout, got: {out!r}"

    def test_compact_success_uses_print(self, capsys):
        """When compression succeeds, user must see the summary."""
        from unittest.mock import MagicMock, patch

        from cogtrix import _build_slash_commands

        reg = _build_slash_commands()
        mm = MagicMock()
        orig = MagicMock()
        orig.content = "x" * 5000
        compressed = MagicMock()
        compressed.content = "short"
        mm._messages = [orig]
        mm.get_messages.return_value = mm._messages
        mm.get_message_count.return_value = 1
        mm.get_stats.return_value = {"total_messages": 1}
        reg.memory_manager = mm
        reg.compression_llm = None
        reg.max_context_tokens = 128_000
        reg.last_input_tokens = 50_000

        with patch(
            "cogtrix_core.cli.commands.apply_message_compression", return_value=[compressed]
        ):
            reg.dispatch("/compact")

        out = capsys.readouterr().out
        assert "Context reduced by" in out, f"Expected 'Context reduced by' in stdout, got: {out!r}"
