"""#2229 — a persistently-failing WhatsApp poll must escalate, not stay silent.

``WahaClient.get_chats_overview`` (and the sibling ``get_messages``) caught every
exception, logged at DEBUG, and returned ``[]``. If the poll failed every cycle
(auth error, WAHA down, the NOWEB-store 400), the assistant received an empty
chat list forever — processing no messages — while looking healthy, with no
signal at normal verbosity and the actionable HTTP body thrown away.

The client now tracks consecutive failures per operation and, once the failure
is persistent, emits a single WARNING that surfaces the HTTP response body and a
NOWEB-store hint. The counter resets (and logs recovery) on the first success.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.tools._whatsapp_client import _POLL_FAILURE_ESCALATION_THRESHOLD, WahaClient

_NOWEB_BODY = "Enable NOWEB store to use this method when starting a new session."


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _HTTPErr(Exception):
    """Mimics requests.HTTPError — carries the server response (with a body)."""

    def __init__(self, msg: str, body: str) -> None:
        super().__init__(msg)
        self.response = _FakeResp(body)


def _client() -> WahaClient:
    return WahaClient(base_url="http://waha:3000", session="default")


class TestPollFailureEscalation:
    @patch("src.tools._whatsapp_client.requests")
    def test_persistent_failure_escalates_once_with_body_and_hint(
        self, mock_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="cogtrix")
        resp = MagicMock()
        resp.raise_for_status.side_effect = _HTTPErr("400 Client Error: Bad Request", _NOWEB_BODY)
        mock_requests.get.return_value = resp
        client = _client()

        for _ in range(_POLL_FAILURE_ESCALATION_THRESHOLD):
            assert client.get_chats_overview() == []

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "must escalate exactly once, not every cycle"
        msg = warnings[0].getMessage()
        assert "3 times in a row" in msg
        # The actionable HTTP body (the *reason*) is surfaced, not just str(exc).
        assert "Enable NOWEB store" in msg
        assert "enable the noweb store" in msg.lower()  # the hint
        assert client._consecutive_poll_failures["fetch chats overview"] == 3

    @patch("src.tools._whatsapp_client.requests")
    def test_below_threshold_stays_at_debug(
        self, mock_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="cogtrix")
        resp = MagicMock()
        resp.raise_for_status.side_effect = _HTTPErr("400", _NOWEB_BODY)
        mock_requests.get.return_value = resp
        client = _client()

        client.get_chats_overview()
        client.get_chats_overview()  # only 2 failures < threshold

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
        assert [r for r in caplog.records if r.levelno == logging.DEBUG]

    @patch("src.tools._whatsapp_client.requests")
    def test_success_resets_counter_and_logs_recovery(
        self, mock_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="cogtrix")
        fail = MagicMock()
        fail.raise_for_status.side_effect = _HTTPErr("400", _NOWEB_BODY)
        ok = MagicMock()
        ok.raise_for_status = MagicMock()
        ok.json.return_value = []
        client = _client()

        mock_requests.get.return_value = fail
        for _ in range(3):
            client.get_chats_overview()
        assert client._consecutive_poll_failures["fetch chats overview"] == 3

        mock_requests.get.return_value = ok
        assert client.get_chats_overview() == []
        assert client._consecutive_poll_failures["fetch chats overview"] == 0
        recovered = [
            r for r in caplog.records if r.levelno == logging.INFO and "recovered" in r.getMessage()
        ]
        assert len(recovered) == 1

    @patch("src.tools._whatsapp_client.requests")
    def test_connection_error_without_response_body(
        self, mock_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # requests.get itself raises (no HTTP response) → no body, no crash.
        caplog.set_level(logging.DEBUG, logger="cogtrix")
        mock_requests.get.side_effect = ConnectionError("waha unreachable")
        client = _client()

        for _ in range(3):
            assert client.get_chats_overview() == []

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "waha unreachable" in warnings[0].getMessage()

    @patch("src.tools._whatsapp_client.requests")
    def test_sibling_get_messages_also_escalates(
        self, mock_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="cogtrix")
        resp = MagicMock()
        resp.raise_for_status.side_effect = _HTTPErr("500 Server Error", "boom")
        mock_requests.get.return_value = resp
        client = _client()

        for _ in range(3):
            assert client.get_messages() == []

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "fetch messages" in warnings[0].getMessage()
