"""Regression test for #2131 C4 — _run_slow_path must never rewind the summary
boundary.

The stuck-job fall-through can run two summarizers concurrently. A late-finishing
OLDER job (covering fewer messages) must not overwrite the newer summary or move
_summary_msg_idx backward — that would leave the messages between the two
boundaries uncovered. The writeback now applies summary + boundary together, and
only when the job extends coverage (unsummarized_end > current boundary).
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


def _make_manager(monkeypatch: pytest.MonkeyPatch, summary_text: str) -> ConversationMemoryManager:
    mgr = ConversationMemoryManager(_MockStore(), "test-2131-c4")
    mgr._vector_store = None
    mgr._ensure_embeddings_initialized = lambda: None  # type: ignore[method-assign]
    mgr._save_hybrid_meta = lambda *a, **k: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        "cogtrix_core.memory.summarizer.generate_summary",
        lambda llm, batch, before: summary_text,
    )
    return mgr


def test_stale_job_does_not_rewind_boundary_or_overwrite_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _make_manager(monkeypatch, summary_text="STALE summary (covers 0..20)")
    # A newer job already advanced coverage to 30.
    mgr._summary = "NEWER summary (covers 0..30)"
    mgr._summary_msg_idx = 30

    # A late-finishing older job tries to write back coverage only up to 20.
    mgr._run_slow_path(batch=["m1", "m2"], unsummarized_end=20)

    assert mgr._summary_msg_idx == 30, "stale job must not rewind the boundary"
    assert (
        mgr._summary == "NEWER summary (covers 0..30)"
    ), "stale job must not overwrite the summary"


def test_advancing_job_updates_boundary_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _make_manager(monkeypatch, summary_text="NEW summary (covers 0..25)")
    mgr._summary = None
    mgr._summary_msg_idx = 10

    mgr._run_slow_path(batch=["m1"], unsummarized_end=25)

    assert mgr._summary_msg_idx == 25, "an advancing job must move the boundary forward"
    assert mgr._summary == "NEW summary (covers 0..25)"


def test_equal_boundary_does_not_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job whose coverage equals the current boundary adds nothing — skip it
    (strictly-greater guard) so a re-run can't clobber the existing summary."""
    mgr = _make_manager(monkeypatch, summary_text="REDUNDANT")
    mgr._summary = "existing"
    mgr._summary_msg_idx = 15

    mgr._run_slow_path(batch=["m1"], unsummarized_end=15)

    assert mgr._summary_msg_idx == 15
    assert mgr._summary == "existing"
