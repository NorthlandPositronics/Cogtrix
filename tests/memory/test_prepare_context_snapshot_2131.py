"""Regression test for #2131 C3 — prepare_context must read self._messages
under _mode_lock as a snapshot.

update() appends to self._messages under _mode_lock; prepare_context previously
read self._messages unlocked and passed the *live* list into assemble_from_tiers,
so a concurrent turn could mutate it mid-assembly (torn read / boundary
mismatch). The fix snapshots the list once under _mode_lock and uses the
snapshot for every read.
"""

from __future__ import annotations

import pytest

from cogtrix_core.memory import modes  # noqa: F401 — triggers mode registration
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager


class _MockStore:
    def __init__(self) -> None:
        self.data: dict = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


def test_tier_path_receives_message_snapshot_not_live_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = ConversationMemoryManager(_MockStore(), "test-2131-c3")
    # Isolate from hybrid-prefix internals.
    monkeypatch.setattr(mgr, "_build_hybrid_prefix", lambda _u: "")

    mgr._messages = ["m1", "m2", "m3"]
    # Activate the tier-cache path.
    mgr._tier_cache_ready = True
    mgr._tier_cache = {}  # truthy-enough: the code checks `is not None`

    captured: dict = {}

    def _fake_assemble(*, snapshot, messages, summary, summary_msg_idx):
        captured["messages"] = messages
        return ([], {})

    monkeypatch.setattr("cogtrix_core.memory.tier_cache.assemble_from_tiers", _fake_assemble)

    mgr.prepare_context("query")

    assert "messages" in captured, "tier path must call assemble_from_tiers"
    # Same contents, but a distinct object — i.e. a snapshot, not the live list.
    assert captured["messages"] == ["m1", "m2", "m3"]
    assert captured["messages"] is not mgr._messages, (
        "prepare_context must pass a snapshot of _messages (taken under _mode_lock), "
        "not the live list, to avoid a torn read against a concurrent update()"
    )
