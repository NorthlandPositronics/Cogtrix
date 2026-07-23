"""#2136 F1/F2 — sync HTTP egress must verify the *connected* IP, not a URL string.

F1: ``_install_dns_pin_hook`` used to swallow a urllib3 import failure and let
``_follow_redirects`` issue an **un-pinned** request, reopening the
validate-then-connect TOCTOU (DNS rebinding). It now reports whether pinning is
active so the caller can fall back or fail closed.

F2: the old post-connection backstop compared ``response.url`` to the request
URL. With ``allow_redirects=False`` requests sets ``response.url == url``, so the
branch was dead code — and a rebind changes the connected *IP*, not the URL
string, so it could never observe the attack. The backstop now re-validates the
real peer IP read off the live socket.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import src.tools.http_request as hr
from src.tools.http_request import (
    _follow_redirects,
    _install_dns_pin_hook,
    _peer_ip_from_response,
)


def _resp(*, peer_ip: str | None, status_code: int = 200, headers: dict | None = None) -> MagicMock:
    """Build a fake streamed response whose live socket reports *peer_ip*.

    ``peer_ip=None`` models the "cannot determine the connected IP" case
    (socket released / urllib3 version skew) — the underlying socket is absent.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.close = MagicMock()
    if peer_ip is None:
        resp.raw._connection.sock = None
    else:
        sock = MagicMock()
        sock.getpeername.return_value = (peer_ip, 443)
        resp.raw._connection.sock = sock
    return resp


class TestPeerIpExtraction:
    def test_reads_peer_ip_off_socket(self) -> None:
        resp = _resp(peer_ip="93.184.216.34")
        assert _peer_ip_from_response(resp) == "93.184.216.34"

    def test_none_when_socket_absent(self) -> None:
        resp = _resp(peer_ip=None)
        assert _peer_ip_from_response(resp) is None

    def test_none_when_getpeername_raises(self) -> None:
        resp = MagicMock()
        resp.raw._connection.sock.getpeername.side_effect = OSError("not connected")
        assert _peer_ip_from_response(resp) is None

    def test_none_for_object_without_raw(self) -> None:
        obj = MagicMock()
        obj.raw = None
        assert _peer_ip_from_response(obj) is None


class TestInstallDnsPinHookContract:
    def test_returns_true_when_urllib3_present(self) -> None:
        # urllib3 is installed in the test env (requests depends on it).
        assert _install_dns_pin_hook() is True

    def test_returns_false_when_urllib3_connection_unimportable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the one-shot to re-run and make the import fail (version skew).
        monkeypatch.setattr(hr, "_dns_pin_installed", False)
        # Setting the module to None in sys.modules makes `import ...` raise.
        monkeypatch.setitem(sys.modules, "urllib3.util.connection", None)
        assert _install_dns_pin_hook() is False


class TestFollowRedirectsPeerVerification:
    def test_blocked_peer_ip_raises_rebinding(self) -> None:
        """The connected IP is private even though validation passed → block."""
        session = MagicMock()
        session.request.return_value = _resp(peer_ip="127.0.0.1", status_code=200)

        with patch("src.tools.http_request._validate_url", return_value=(True, "", "8.8.8.8")):
            with pytest.raises(ValueError, match="rebinding"):
                _follow_redirects(
                    session, "GET", "https://example.com/", pinned_ip="8.8.8.8", timeout=10
                )

    def test_unknown_peer_unpinned_fails_closed(self) -> None:
        """Peer IP unknown AND no DNS pin for the hop → refuse (fail closed)."""
        session = MagicMock()
        session.request.return_value = _resp(peer_ip=None, status_code=200)

        # pinned_ip=None → hop is not pinned; peer IP unreadable → must fail closed.
        with pytest.raises(ValueError, match="SSRF protection"):
            _follow_redirects(session, "GET", "https://example.com/", pinned_ip=None, timeout=10)

    def test_unknown_peer_but_pinned_proceeds(self) -> None:
        """Peer IP unknown but the hop was DNS-pinned to a validated IP → trust it."""
        session = MagicMock()
        final = _resp(peer_ip=None, status_code=200)
        session.request.return_value = final

        result = _follow_redirects(
            session, "GET", "https://example.com/", pinned_ip="93.184.216.34", timeout=10
        )
        assert result is final

    def test_public_peer_ip_allowed(self) -> None:
        """A genuinely public connected IP passes the backstop."""
        session = MagicMock()
        final = _resp(peer_ip="93.184.216.34", status_code=200)
        session.request.return_value = final

        result = _follow_redirects(
            session, "GET", "https://example.com/", pinned_ip="93.184.216.34", timeout=10
        )
        assert result is final
