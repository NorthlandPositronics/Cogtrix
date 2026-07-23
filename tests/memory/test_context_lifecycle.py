"""Integration tests for the context lifecycle memory flow."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from cogtrix_core.memory.facts import PersistentFactsStore
from cogtrix_core.memory.json_store import JsonFileMemoryStore
from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager


def _make_reasoning_manager(tmp_path: Path, session_id: str = "ctx-lifecycle"):
    store = JsonFileMemoryStore(base_dir=str(tmp_path / "history"))
    return ReasoningMemoryManager(
        store,
        session_id,
        {"summary_max_age_hours": 24, "working_memory_size": 4},
    )


class TestContextLifecycle:
    def test_summary_ttl_expiry_on_session_reload(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "ttl-expiry")
        mgr.update("hello", "hi")
        expired_at = datetime.now(UTC) - timedelta(hours=25)
        mgr._summary = "Expired summary"
        mgr._summary_msg_idx = 4
        mgr._summary_last_updated_at = expired_at
        mgr.save()

        meta_path = mgr._hybrid_meta_path()
        assert meta_path.exists()

        mgr2 = _make_reasoning_manager(tmp_path, "ttl-expiry")
        mgr2.load()

        assert mgr2._summary is None
        assert mgr2._summary_msg_idx == 0
        assert mgr2._summary_last_updated_at is None
        assert not meta_path.exists()

    def test_distillation_runs_before_reset(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "distill-enabled")
        mgr._summary = "Session summary with durable facts."
        mgr._summary_msg_idx = 8
        mgr._summary_last_updated_at = datetime.now(UTC)
        mgr._llm = MagicMock()

        with patch(
            "cogtrix_core.memory.distillation.distill_summary",
            return_value=[
                "PR #287 merged — SAML fix landed",
                "Issue #291 is the current QA gate",
            ],
        ) as distill_mock:
            mgr.reset_summary()

        distill_mock.assert_called_once_with(mgr._llm, "Session summary with durable facts.")

        facts_store = PersistentFactsStore(
            "distill-enabled", storage_dir=str(tmp_path / "memory" / "facts")
        )
        snapshot = facts_store.load()
        assert snapshot is not None
        assert snapshot.facts == [
            "PR #287 merged — SAML fix landed",
            "Issue #291 is the current QA gate",
        ]
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0

    def test_distillation_skipped_when_disabled(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "distill-disabled")
        mgr._mode_config["distill_on_expire"] = False
        mgr._summary = "Summary that should be cleared without distillation."
        mgr._summary_msg_idx = 6
        mgr._summary_last_updated_at = datetime.now(UTC)
        mgr._llm = MagicMock()

        with patch("cogtrix_core.memory.distillation.distill_summary") as distill_mock:
            mgr.reset_summary()

        distill_mock.assert_not_called()
        facts_store = PersistentFactsStore(
            "distill-disabled", storage_dir=str(tmp_path / "memory" / "facts")
        )
        assert facts_store.load() is None
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0

    def test_facts_injected_into_context_prefix(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "prefix-facts")
        facts_store = PersistentFactsStore(
            "prefix-facts", storage_dir=str(tmp_path / "memory" / "facts")
        )
        facts_store.save(
            [
                "PR #287 merged — SAML fix landed",
                "Phase 2.1 SSO is next priority",
            ],
            ttl_days=7,
        )

        prefix = mgr._build_hybrid_prefix("what should we do next?")

        assert prefix is not None
        assert "Persistent context from prior sessions:" in prefix
        assert "PR #287 merged — SAML fix landed" in prefix
        assert "Phase 2.1 SSO is next priority" in prefix

    def test_expired_facts_are_ignored(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "expired-facts")
        facts_store = PersistentFactsStore(
            "expired-facts", storage_dir=str(tmp_path / "memory" / "facts")
        )
        facts_store.save(["Stale institutional memory"], ttl_days=1)

        facts_path = facts_store._facts_path
        payload = json.loads(facts_path.read_text(encoding="utf-8"))
        payload["created_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        facts_path.write_text(json.dumps(payload), encoding="utf-8")

        prefix = mgr._build_hybrid_prefix("what should we do next?")

        assert prefix is None or "Stale institutional memory" not in prefix

    def test_full_context_lifecycle_round_trip(self, tmp_path):
        mgr = _make_reasoning_manager(tmp_path, "full-cycle")
        mgr._summary = "Long-lived summary."
        mgr._summary_msg_idx = 10
        mgr._summary_last_updated_at = datetime.now(UTC) - timedelta(hours=25)
        mgr._llm = MagicMock()

        with patch(
            "cogtrix_core.memory.distillation.distill_summary",
            return_value=["PR #287 merged — SAML fix landed"],
        ):
            mgr.reset_summary()

        mgr2 = _make_reasoning_manager(tmp_path, "full-cycle")
        mgr2.load()

        assert mgr2._summary is None
        assert mgr2._summary_msg_idx == 0

        facts_store = PersistentFactsStore(
            "full-cycle", storage_dir=str(tmp_path / "memory" / "facts")
        )
        snapshot = facts_store.load()
        assert snapshot is not None
        assert "PR #287 merged — SAML fix landed" in snapshot.facts
