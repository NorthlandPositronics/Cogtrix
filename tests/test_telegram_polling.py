"""Tests for the Telegram channel polling — ignore_older_than."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from cogtrix_core.assistant.channels.telegram import TelegramChannel
from cogtrix_core.tools._telegram_client import TelegramBotClient, TelegramMessage


def _make_tg_msg(
    message_id: int = 1,
    date: int = 0,
    chat_id: int = 100,
    text: str = "Hello",
    update_id: int = 1,
) -> TelegramMessage:
    return TelegramMessage(
        message_id=message_id,
        date=date,
        chat_id=chat_id,
        text=text,
        update_id=update_id,
    )


def _make_tg_channel(config: dict | None = None) -> TelegramChannel:
    cfg: dict = {"bot_token": "fake-token"}
    if config:
        cfg.update(config)
    ch = TelegramChannel(cfg, long_poll_timeout=0)
    ch._client = MagicMock(spec=TelegramBotClient)
    ch._seeded = True  # bypass cold-start drain loop in tests
    return ch


class TestTelegramIgnoreOlderThan:
    def test_old_message_skipped(self) -> None:
        ch = _make_tg_channel({"ignore_older_than": "1h"})
        old_ts = int(time.time()) - 7200
        msg = _make_tg_msg(date=old_ts, text="Old")
        ch._client.get_updates.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_recent_message_processed(self) -> None:
        ch = _make_tg_channel({"ignore_older_than": "1h"})
        recent_ts = int(time.time()) - 60
        msg = _make_tg_msg(date=recent_ts, text="Recent")
        ch._client.get_updates.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "Recent"

    def test_disabled_by_default(self) -> None:
        ch = _make_tg_channel()
        old_ts = int(time.time()) - 86400 * 30
        msg = _make_tg_msg(date=old_ts, text="Ancient")
        ch._client.get_updates.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1

    def test_update_id_still_advances(self) -> None:
        """Skipped old messages still advance _last_update_id."""
        ch = _make_tg_channel({"ignore_older_than": "1h"})
        old_ts = int(time.time()) - 7200
        msg = _make_tg_msg(date=old_ts, text="Old", update_id=42)
        ch._client.get_updates.return_value = [msg]  # type: ignore[attr-defined]

        ch.poll()

        assert ch._last_update_id == 42
