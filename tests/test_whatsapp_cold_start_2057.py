"""#2057 — WhatsApp cold-start backlog guard.

The incident: after a container was replaced with a freshly-built image, the
assistant re-sent an already-answered short reply ("No rush, take your time.")
multiple times to a real contact. Root cause: on the FIRST boot of the new image
there was no persisted dedup state and no in-memory watermark, so the first poll
looked back over the reactivation window and re-answered pre-existing backlog the
previous (pre-fix) container had already handled.

The fix seeds per-chat watermarks from the overview on a genuine cold start
(persistence configured, no prior seen-ids on disk) and skips processing that
cycle, so only genuinely-new inbound messages — arriving after boot — get a reply.
A warm restart (seen-ids file present) keeps the normal path.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel
from cogtrix_core.tools._whatsapp_client import ChatOverview, Message, WahaClient


def _make_message(id: str, timestamp: int, body: str = "hi", from_me: bool = False) -> Message:
    return Message(id=id, timestamp=timestamp, from_number="123@c.us", body=body, from_me=from_me)


def _make_overview(last_message: Message | None) -> ChatOverview:
    return ChatOverview(id="123@c.us", name="Alice", last_message=last_message, archived=False)


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


class TestColdStartBacklogGuard:
    def test_cold_start_seeds_and_does_not_answer_backlog(self, tmp_path: Path) -> None:
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        ch = _make_channel(seen_path)
        assert ch._cold_start_seed_pending is True
        backlog = _make_message("m-backlog", int(time.time()), body="No rush, take your time.")
        _wire(ch, _make_overview(backlog), [backlog])

        result = ch.poll()

        assert result == [], "cold-start first poll must not re-answer pre-existing backlog"
        # The overview last-message is fetched to seed, but NOT the per-message history.
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]
        # Watermark seeded to the backlog message; it is marked seen and persisted.
        assert ch._chat_watermarks["123@c.us"] == backlog.timestamp
        assert "m-backlog" in ch._seen_wall
        assert seen_path.exists()
        assert ch._cold_start_seed_pending is False

    def test_cold_start_then_new_message_is_answered(self, tmp_path: Path) -> None:
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        ch = _make_channel(seen_path)
        backlog = _make_message("m-backlog", int(time.time()), body="old")
        _wire(ch, _make_overview(backlog), [backlog])
        assert ch.poll() == []  # seed

        # A message that arrives strictly after the seeded watermark is answered.
        new = _make_message("m-new", int(time.time()) + 30, body="Any update?")
        _wire(ch, _make_overview(new), [new])
        result = ch.poll()
        assert [r.message_id for r in result] == ["m-new"]

    def test_multiple_backlog_messages_none_answered(self, tmp_path: Path) -> None:
        """The incident shape: several recent inbound messages present at boot —
        none may be answered on the cold-start poll."""
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        ch = _make_channel(seen_path)
        newest = _make_message("m-3", int(time.time()), body="No rush, take your time.")
        _wire(ch, _make_overview(newest), [newest])
        assert ch.poll() == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_warm_restart_does_not_seed(self, tmp_path: Path) -> None:
        """A prior seen-ids file → warm restart → no cold-start seed; the normal
        reactivation path answers a genuinely-new message."""
        seen_path = tmp_path / "whatsapp_seen_ids.json"
        # Pre-seed a warm channel (persist one id), then simulate a restart.
        warm = _make_channel(seen_path)
        prior = _make_message("m-prior", int(time.time()), body="seed")
        _wire(warm, _make_overview(prior), [prior])
        warm.poll()  # cold-seed writes the file → next channel is warm

        ch = _make_channel(seen_path)
        assert ch._cold_start_seed_pending is False, "prior seen-ids ⇒ warm restart"
        new = _make_message("m-fresh", int(time.time()) + 20, body="hello?")
        _wire(ch, _make_overview(new), [new])
        assert [r.message_id for r in ch.poll()] == ["m-fresh"]

    def test_persistence_disabled_keeps_legacy_first_poll(self) -> None:
        """No seen_ids_path → no cross-restart dedup at all → cold-start seeding is
        disabled and the legacy reactivation-lookback first-poll behavior is kept."""
        ch = _make_channel(None)
        assert ch._cold_start_seed_pending is False
        msg = _make_message("m-1", int(time.time()), body="Hi")
        _wire(ch, _make_overview(msg), [msg])
        assert len(ch.poll()) == 1

    def test_new_built_double_resolves_flag_via_class_default(self) -> None:
        """A ``__new__``-built double (used by several channel tests) must resolve
        ``_cold_start_seed_pending`` from the class-level default — it never runs
        ``__init__`` — so ``poll()`` doesn't ``AttributeError``. Regression for the
        CI break where the flag was instance-only."""
        ch = WhatsAppChannel.__new__(WhatsAppChannel)
        assert ch._cold_start_seed_pending is False
