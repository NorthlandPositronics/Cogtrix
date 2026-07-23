"""#2053 — WhatsApp duplicate reply after restart: persist seen-ids to disk.

The channel deduplicates already-answered messages with an in-memory ``_seen_ids``
set. On process/container restart that set is wiped, so a recently-received
message still inside the reactivation lookback window is re-fetched and answered
again — the contact gets a duplicate reply (incident #2057).

The fix persists processed message ids to ``whatsapp_seen_ids.json`` keyed by
WALL-CLOCK time, loads + prunes them on startup, and skips already-seen ids
across restarts. The in-memory monotonic map is kept for the in-process TTL
prune. These tests cover that contract — including the monotonic-vs-wall-clock
subtlety (a load-time prune against ``time.time()`` must not discard everything).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel
from cogtrix_core.tools._whatsapp_client import ChatOverview, Message, WahaClient


def _make_message(
    id: str = "msg-1",
    timestamp: int = 1000,
    from_number: str = "123@c.us",
    body: str = "Hello",
    from_me: bool = False,
) -> Message:
    return Message(
        id=id,
        timestamp=timestamp,
        from_number=from_number,
        body=body,
        from_me=from_me,
    )


def _make_overview(
    chat_id: str = "123@c.us",
    name: str = "Alice",
    last_message: Message | None = None,
) -> ChatOverview:
    return ChatOverview(id=chat_id, name=name, last_message=last_message, archived=False)


def _make_channel(seen_path: Path | None) -> WhatsAppChannel:
    cfg: dict = {"waha_url": "http://localhost:3000", "session": "default"}
    if seen_path is not None:
        cfg["seen_ids_path"] = str(seen_path)
    with patch.object(WahaClient, "__init__", lambda self, **kw: None):
        ch = WhatsAppChannel(cfg)
        ch._client = MagicMock(spec=WahaClient)
    ch._ensure_session = lambda: None  # type: ignore[method-assign]
    return ch


def _wire(ch: WhatsAppChannel, overview: ChatOverview, fetched: list[Message]) -> None:
    ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
    ch._client.get_chat_messages.return_value = fetched  # type: ignore[attr-defined]


class TestSeenPersistence:
    def test_message_not_reanswered_after_restart(self, tmp_path: Path) -> None:
        """A message answered before restart is NOT re-emitted after restart.

        Updated for #2057: the first poll of a genuine cold start (persistence
        configured, no prior seen-ids) now SEEDS watermarks and does not answer
        pre-existing backlog. The answered-once-then-not-again contract is
        exercised with a message that arrives *after* the cold-start seed.
        """
        seen_path = tmp_path / "whatsapp_seen_ids.json"

        ch1 = _make_channel(seen_path)
        backlog = _make_message(id="m-backlog", timestamp=int(time.time()), body="old")
        _wire(ch1, _make_overview(last_message=backlog), [backlog])
        assert ch1.poll() == [], "cold-start first poll must seed, not answer backlog"
        assert seen_path.exists(), "seed must persist a seen-ids file"

        # A genuinely-new message (after the seeded watermark) is answered once.
        msg = _make_message(id="m-restart", timestamp=int(time.time()) + 5, body="No rush")
        _wire(ch1, _make_overview(last_message=msg), [msg])
        first = ch1.poll()
        assert [r.message_id for r in first] == ["m-restart"]

        # Restart: a fresh channel loads the persisted state (now non-empty, so a
        # WARM restart — no re-seed). The watermark is gone (in-memory), so the same
        # message is re-fetched — but persistence must suppress the duplicate reply.
        ch2 = _make_channel(seen_path)
        _wire(ch2, _make_overview(last_message=msg), [msg])
        second = ch2.poll()
        assert second == [], "restart must not re-answer an already-seen message"

    def test_new_message_after_restart_is_answered(self, tmp_path: Path) -> None:
        """Persistence must not over-suppress: a genuinely new message still replies."""
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        old = _make_message(id="m-old", timestamp=int(time.time()), body="Earlier")

        # Cold start seeds (does not answer the backlog message).
        ch1 = _make_channel(seen_path)
        _wire(ch1, _make_overview(last_message=old), [old])
        assert ch1.poll() == []

        # Restart (warm — seed persisted), then a NEW message arrives.
        new = _make_message(id="m-new", timestamp=int(time.time()) + 10, body="Fresh")
        ch2 = _make_channel(seen_path)
        _wire(ch2, _make_overview(last_message=new), [old, new])
        result = ch2.poll()
        assert [r.message_id for r in result] == ["m-new"]

    def test_load_uses_wall_clock_not_monotonic(self, tmp_path: Path) -> None:
        """A persisted wall-clock ts within TTL survives the load-time prune.

        Regression for the monotonic-vs-wall-clock subtlety: if the load prune
        compared a *monotonic* value against ``time.time()`` (or vice versa),
        every entry would be discarded immediately. A recent wall-clock entry
        must be retained.
        """
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        recent_wall = time.time() - 60.0  # 1 min ago, well inside the 7-day TTL
        seen_path.write_text(json.dumps({"m-persisted": recent_wall}), encoding="utf-8")

        ch = _make_channel(seen_path)
        assert "m-persisted" in ch._seen_wall
        assert ch._seen_wall["m-persisted"] == recent_wall

    def test_stale_entries_pruned_on_load(self, tmp_path: Path) -> None:
        """Entries older than the persist TTL are dropped when loading."""
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        stale_wall = time.time() - (8 * 24 * 3600.0)  # 8 days ago > 7-day TTL
        fresh_wall = time.time() - 100.0
        seen_path.write_text(
            json.dumps({"m-stale": stale_wall, "m-fresh": fresh_wall}), encoding="utf-8"
        )

        ch = _make_channel(seen_path)
        assert "m-stale" not in ch._seen_wall
        assert "m-fresh" in ch._seen_wall

    def test_corrupt_file_is_ignored(self, tmp_path: Path) -> None:
        """A corrupt persistence file must not crash startup."""
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        seen_path.write_text("{not valid json", encoding="utf-8")

        ch = _make_channel(seen_path)  # must not raise
        assert ch._seen_wall == {}

    def test_persistence_disabled_without_path(self, tmp_path: Path) -> None:
        """No ``seen_ids_path`` → no disk I/O; cold-start behavior is unchanged."""
        ch = _make_channel(None)
        assert ch._seen_ids_path is None
        msg = _make_message(id="m-1", timestamp=int(time.time()), body="Hi")
        _wire(ch, _make_overview(last_message=msg), [msg])
        # First poll still answers (existing _REACTIVATION_LOOKBACK behavior intact).
        assert len(ch.poll()) == 1
        # Nothing written anywhere under tmp_path.
        assert not any(tmp_path.iterdir())
