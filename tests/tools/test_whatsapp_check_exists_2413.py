"""#2413 — validate a number is on WhatsApp before a proactive send.

The assistant sent outreach (+ follow-ups) to a number not registered on
WhatsApp; every message sat at ack=0 (never delivered) with no signal, so it
kept following up into the void. ``whatsapp_send`` now checks WAHA's
``check-exists`` before an individual send and refuses (with a clear, don't-
follow-up message) when the number is explicitly not on WhatsApp.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import cogtrix_core.tools.whatsapp as wa
from cogtrix_core.tools._whatsapp_client import SendResult, WahaClient


class _Resp:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self) -> dict:
        return self._body


class TestWahaClientCheckExists:
    def _client(self) -> WahaClient:
        return WahaClient(base_url="http://waha:3000", session="default")

    def test_number_exists_true(self, monkeypatch) -> None:
        c = self._client()
        monkeypatch.setattr(
            "cogtrix_core.tools._whatsapp_client.requests.get",
            lambda *a, **k: _Resp(200, {"numberExists": True}),
        )
        assert c.check_exists("971554841810") is True

    def test_number_exists_false(self, monkeypatch) -> None:
        c = self._client()
        monkeypatch.setattr(
            "cogtrix_core.tools._whatsapp_client.requests.get",
            lambda *a, **k: _Resp(200, {"numberExists": False}),
        )
        # Accepts a @c.us chatId too (suffix stripped).
        assert c.check_exists("971502376812@c.us") is False

    def test_missing_field_is_none(self, monkeypatch) -> None:
        c = self._client()
        monkeypatch.setattr(
            "cogtrix_core.tools._whatsapp_client.requests.get", lambda *a, **k: _Resp(200, {})
        )
        assert c.check_exists("971502376812") is None

    def test_http_error_is_none(self, monkeypatch) -> None:
        c = self._client()
        monkeypatch.setattr(
            "cogtrix_core.tools._whatsapp_client.requests.get", lambda *a, **k: _Resp(500, {})
        )
        assert c.check_exists("971502376812") is None

    def test_exception_is_none(self, monkeypatch) -> None:
        c = self._client()

        def _boom(*a, **k):
            raise RuntimeError("waha down")

        monkeypatch.setattr("cogtrix_core.tools._whatsapp_client.requests.get", _boom)
        assert c.check_exists("971502376812") is None

    def test_empty_number_is_none(self, monkeypatch) -> None:
        c = self._client()
        # No request should be made for an empty number.
        monkeypatch.setattr(
            "cogtrix_core.tools._whatsapp_client.requests.get",
            lambda *a, **k: pytest.fail("should not call WAHA for empty number"),
        )
        assert c.check_exists("@c.us") is None


class TestWhatsappSendGate:
    @pytest.fixture(autouse=True)
    def _allow_and_unlimited(self, monkeypatch):
        monkeypatch.setattr(wa, "REQUESTS_AVAILABLE", True)
        monkeypatch.setattr(wa, "_check_contact", lambda _to: (True, ""))
        monkeypatch.setattr(wa, "_rate_limit_ok", lambda: True)
        monkeypatch.setattr(wa, "_record_send", lambda: None)

    def _fake_client(self, exists) -> MagicMock:
        client = MagicMock()
        client.check_exists.return_value = exists
        client.send_text.return_value = SendResult(ok=True, message_id="m1")
        return client

    def test_non_whatsapp_number_is_not_sent(self, monkeypatch) -> None:
        client = self._fake_client(exists=False)
        monkeypatch.setattr(wa, "_get_client", lambda: client)
        out = wa.whatsapp_send("971502376812", "Hi, still interested?")
        assert "Not sent" in out and "numberExists=false" in out
        client.send_text.assert_not_called()

    def test_valid_number_is_sent(self, monkeypatch) -> None:
        client = self._fake_client(exists=True)
        monkeypatch.setattr(wa, "_get_client", lambda: client)
        out = wa.whatsapp_send("971554841810", "Hello")
        assert "Message sent" in out
        client.send_text.assert_called_once()

    def test_unverifiable_number_fails_open(self, monkeypatch) -> None:
        # check_exists error (None) must NOT block a send (transient WAHA hiccup).
        client = self._fake_client(exists=None)
        monkeypatch.setattr(wa, "_get_client", lambda: client)
        out = wa.whatsapp_send("971554841810", "Hello")
        assert "Message sent" in out
        client.send_text.assert_called_once()

    def test_group_send_skips_check(self, monkeypatch) -> None:
        # Groups (@g.us) are not individual numbers — no existence check.
        client = self._fake_client(exists=False)
        monkeypatch.setattr(wa, "_get_client", lambda: client)
        out = wa.whatsapp_send("120363000000000000@g.us", "Team update")
        assert "Message sent" in out
        client.check_exists.assert_not_called()
        client.send_text.assert_called_once()
