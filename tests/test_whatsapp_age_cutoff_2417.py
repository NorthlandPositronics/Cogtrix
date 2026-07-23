"""#2417 — a never-answered inbound dropped by the age cutoff must be operator-visible.

A legitimate lead whose reply was suppressed/missed then ages out past
``ignore_older_than`` was silently skipped at DEBUG — "looks like the agent was
contacted and just went quiet". The channel now WARNs when it drops an UNSEEN
(never-processed) inbound, while an already-seen old message stays a quiet DEBUG
(dedup handles it — no log spam).
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel
from cogtrix_core.tools._whatsapp_client import ChatOverview, Message, WahaClient


def _make_channel() -> WhatsAppChannel:
    cfg = {"waha_url": "http://localhost:3000", "session": "default", "ignore_older_than": "30m"}
    with patch.object(WahaClient, "__init__", lambda self, **kw: None):
        ch = WhatsAppChannel(cfg)
        ch._client = MagicMock(spec=WahaClient)
    return ch


def _old_msg(mid: str, from_me: bool = False) -> Message:
    return Message(
        id=mid,
        timestamp=int(time.time()) - 3000,  # 50 min ago > 1800s cutoff
        from_number="971564016789@c.us",
        body="Property Finder passed on your enquiry … Hi",
        from_me=from_me,
    )


_CHAT = ChatOverview(id="971564016789@c.us", name="Lead", last_message=None, archived=False)


class TestAgeCutoffVisibility:
    def test_unseen_dropped_inbound_warns(self, caplog) -> None:
        ch = _make_channel()
        assert ch._ignore_older_than == 1800.0
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = ch._process_message(_old_msg("lead-1"), _CHAT, time.monotonic())
        assert result is None  # dropped
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("UNANSWERED" in r.getMessage() and "lead-1" in r.getMessage() for r in warnings)

    def test_already_seen_old_message_does_not_warn(self, caplog) -> None:
        ch = _make_channel()
        ch._seen_wall["seen-1"] = time.time()  # answered/processed already
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = ch._process_message(_old_msg("seen-1"), _CHAT, time.monotonic())
        assert result is None
        assert not [
            r for r in caplog.records if r.levelno == logging.WARNING
        ], "an already-seen old message must not warn (dedup handles it)"

    def test_from_me_old_message_does_not_warn(self, caplog) -> None:
        # Outbound (fromMe) is filtered before the age check — no lead lost.
        ch = _make_channel()
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = ch._process_message(_old_msg("out-1", from_me=True), _CHAT, time.monotonic())
        assert result is None
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_no_cutoff_configured_no_warn(self, caplog) -> None:
        ch = _make_channel()
        ch._ignore_older_than = None  # cutoff disabled → age branch inert
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            # A very old message no longer trips the (disabled) cutoff; it proceeds
            # past the age branch (seen-check then contact filter). No age warning.
            ch._process_message(_old_msg("x-1"), _CHAT, time.monotonic())
        assert not [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "UNANSWERED" in r.getMessage()
        ]
