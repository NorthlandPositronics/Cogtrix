"""#2413 fix #2 — campaign stops following up an undeliverable (ack<2) thread.

When an outreach lands on a number that isn't on WhatsApp, every message stays at
ack=0 (never delivered) and the assistant kept generating follow-ups into the
void. The follow-up loop now checks the last outbound's delivery ack: if it's
still < 2 (never reached the recipient's device) at follow-up time, the thread is
flagged undeliverable, follow-ups stop, and the operator is alerted. Fail-open —
only a CONFIRMED non-delivery stops a campaign.
"""

from __future__ import annotations

import logging
import time
import uuid
from unittest.mock import MagicMock

from cogtrix_core.assistant.campaign import Campaign, CampaignManager, CampaignTarget
from cogtrix_core.tools._whatsapp_client import Message, WahaClient


def _campaign(target: CampaignTarget) -> Campaign:
    return Campaign(
        id=str(uuid.uuid4()),
        name="c",
        goal="g",
        instructions="i",
        targets=[target],
        status="active",
        max_follow_ups=3,
        follow_up_interval_hours=24.0,
    )


def _eligible_target(last_outbound_id: str | None = "m1") -> CampaignTarget:
    return CampaignTarget(
        contact_name="Alice",
        channel="whatsapp",
        chat_id="+111@c.us",
        status="active",
        last_outbound_at=time.time() - 90000,  # 25h ago > 24h interval
        last_outbound_id=last_outbound_id,
    )


def _mgr_with_ack(tmp_path, ack_value):
    mgr = CampaignManager(tmp_path / "campaigns.json")
    handler = MagicMock()
    handler.handle_outbound.return_value = ("follow-up", "m2")
    mgr.set_handler(handler)
    ch = MagicMock()
    ch.get_message_ack.return_value = ack_value
    mgr.set_channels({"whatsapp": ch})
    return mgr, handler, ch


class TestWahaClientGetMessageAck:
    def _client(self) -> WahaClient:
        return WahaClient(base_url="http://waha:3000", session="default")

    def test_finds_ack_by_id(self):
        c = self._client()
        c.get_chat_messages = MagicMock(  # type: ignore[method-assign]
            return_value=[
                Message(id="a", timestamp=1, from_number="1", from_me=True, ack=1),
                Message(id="m1", timestamp=2, from_number="1", from_me=True, ack=0),
            ]
        )
        assert c.get_message_ack("chat", "m1") == 0

    def test_missing_message_returns_none(self):
        c = self._client()
        c.get_chat_messages = MagicMock(return_value=[])  # type: ignore[method-assign]
        assert c.get_message_ack("chat", "m1") is None

    def test_fetch_error_returns_none(self):
        c = self._client()
        c.get_chat_messages = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("waha down")
        )
        assert c.get_message_ack("chat", "m1") is None


class TestCampaignNonDelivery:
    def test_undelivered_stops_followups_and_flags_target(self, tmp_path, caplog):
        mgr, handler, _ch = _mgr_with_ack(tmp_path, ack_value=0)  # PENDING = undelivered
        campaign = _campaign(_eligible_target())
        mgr.create(campaign)
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            mgr._process_follow_ups()
        handler.handle_outbound.assert_not_called()  # no follow-up into the void
        assert campaign.targets[0].status == "failed"
        assert "undeliverable" in (campaign.targets[0].completion_reason or "")
        assert any(
            "UNDELIVERED" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_delivered_still_follows_up(self, tmp_path):
        mgr, handler, _ch = _mgr_with_ack(tmp_path, ack_value=2)  # DEVICE = delivered
        campaign = _campaign(_eligible_target())
        mgr.create(campaign)
        mgr._process_follow_ups()
        handler.handle_outbound.assert_called_once()
        assert campaign.targets[0].status == "active"

    def test_read_still_follows_up(self, tmp_path):
        mgr, handler, _ch = _mgr_with_ack(tmp_path, ack_value=3)  # READ = delivered+read
        campaign = _campaign(_eligible_target())
        mgr.create(campaign)
        mgr._process_follow_ups()
        handler.handle_outbound.assert_called_once()

    def test_unverifiable_ack_fails_open(self, tmp_path):
        # ack None (can't determine) → do NOT stop the campaign; follow up normally.
        mgr, handler, _ch = _mgr_with_ack(tmp_path, ack_value=None)
        campaign = _campaign(_eligible_target())
        mgr.create(campaign)
        mgr._process_follow_ups()
        handler.handle_outbound.assert_called_once()

    def test_no_outbound_id_skips_ack_check(self, tmp_path):
        mgr, handler, ch = _mgr_with_ack(tmp_path, ack_value=0)
        campaign = _campaign(_eligible_target(last_outbound_id=None))
        mgr.create(campaign)
        mgr._process_follow_ups()
        handler.handle_outbound.assert_called_once()
        ch.get_message_ack.assert_not_called()

    def test_channel_without_ack_capability_fails_open(self, tmp_path):
        # A channel lacking get_message_ack (e.g. Telegram) → fail open, follow up.
        mgr = CampaignManager(tmp_path / "campaigns.json")
        handler = MagicMock()
        handler.handle_outbound.return_value = ("f", "m2")
        mgr.set_handler(handler)
        mgr.set_channels({"whatsapp": MagicMock(spec=["send"])})
        campaign = _campaign(_eligible_target())
        mgr.create(campaign)
        mgr._process_follow_ups()
        handler.handle_outbound.assert_called_once()
