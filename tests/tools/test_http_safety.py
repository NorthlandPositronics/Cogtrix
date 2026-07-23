"""Tests for cogtrix_core/tools/_http_safety.py — the shared SSRF / URL-safety
helpers used by both the sync ``http_get`` tool and the async
``_http_fetch`` primitive (ADR-0056 PR-A2).

Existing ``tests/tools/test_http_request.py`` already covers these
helpers indirectly (via http_get / http_post). This module pins them
directly so a regression on either the sync or async consumer is
traceable to a clear failure here.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from cogtrix_core.tools._http_safety import (
    _BLOCKED_HEADERS,
    _is_blocked_ip,
    _parse_headers,
    _validate_url,
)


class TestIsBlockedIp:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "127.0.0.2",
            "0.0.0.0",  # unspecified
            "::1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # AWS/GCP IMDS
            "169.254.1.1",
            "100.64.0.1",  # CGNAT
            "100.127.255.255",
            "224.0.0.1",  # multicast
            "::ffff:127.0.0.1",  # IPv6-mapped IPv4 loopback
            "fc00::1",  # IPv6 unique local
            "fe80::1",  # IPv6 link-local
        ],
    )
    def test_blocked(self, ip: str) -> None:
        assert _is_blocked_ip(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",  # example.com
            "2606:4700:4700::1111",  # cloudflare DNS over IPv6
        ],
    )
    def test_public(self, ip: str) -> None:
        assert _is_blocked_ip(ip) is False

    def test_not_an_ip_returns_false(self) -> None:
        assert _is_blocked_ip("example.com") is False
        assert _is_blocked_ip("") is False
        assert _is_blocked_ip("not-an-ip") is False

    @pytest.mark.parametrize(
        "host",
        [
            "2130706433",  # decimal 127.0.0.1
            "0x7f000001",  # hex 127.0.0.1
            "0177.0.0.1",  # octal-leading 127.0.0.1
            "127.1",  # short form 127.0.0.1
            "0x7f.1",  # mixed hex+short
            "3232235521",  # decimal 192.168.0.1 (private)
            "2852039166",  # decimal 169.254.169.254 (IMDS link-local)
        ],
    )
    def test_obfuscated_numeric_ipv4_is_blocked(self, host: str) -> None:
        """#2136 F3: decimal/hex/octal/short IPv4 forms (which ipaddress rejects)
        must still be recognized as their real (blocked) address via inet_aton."""
        assert _is_blocked_ip(host) is True

    def test_obfuscated_numeric_public_ipv4_not_blocked(self) -> None:
        # 134744072 == 8.8.8.8 (public) — canonicalized but not blocked.
        assert _is_blocked_ip("134744072") is False


class TestValidateUrl:
    def test_valid_https_url_returns_resolved_ip(self) -> None:
        """Mock DNS so the test doesn't actually hit the network."""
        with patch("cogtrix_core.tools._http_safety.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (None, None, None, "", ("93.184.216.34", 0)),
            ]
            ok, err, resolved = _validate_url("https://example.com/path")
        assert ok is True
        assert err == ""
        assert resolved == "93.184.216.34"

    def test_blocked_scheme(self) -> None:
        for url in [
            "ftp://example.com/",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<h1>x</h1>",
        ]:
            ok, err, _ = _validate_url(url)
            assert ok is False, url

    def test_no_scheme(self) -> None:
        ok, err, _ = _validate_url("example.com/path")
        assert ok is False
        assert "Invalid URL" in err

    def test_empty_url(self) -> None:
        ok, _err, _ = _validate_url("")
        assert ok is False

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "LOCALHOST",  # case-insensitive check
            "metadata.google.internal",
            "instance-data",
            "169.254.169.254",  # blocked-host literal
        ],
    )
    def test_blocked_internal_hosts(self, host: str) -> None:
        ok, err, _ = _validate_url(f"http://{host}/path")
        assert ok is False
        assert "localhost" in err or "private" in err or "internal" in err

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://192.168.0.1/",
            "http://[::1]/",
            "http://2130706433/",  # decimal-encoded 127.0.0.1
            "http://0x7f000001/",  # hex-encoded
        ],
    )
    def test_blocked_ip_literals(self, url: str) -> None:
        ok, err, _ = _validate_url(url)
        assert ok is False, url
        assert "private" in err.lower() or "reserved" in err.lower() or "internal" in err.lower()

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost./",  # trailing-dot FQDN form
            "http://LOCALHOST./",
            "http://169.254.169.254./",  # trailing-dot IMDS literal
            "http://metadata.google.internal./",
        ],
    )
    def test_trailing_dot_hosts_blocked(self, url: str) -> None:
        """#2136 F4: a trailing dot must not let an internal host/IP slip the
        pre-resolution name + IP-literal block (no DNS needed)."""
        ok, err, _ = _validate_url(url)
        assert ok is False, url
        assert "localhost" in err.lower() or "private" in err.lower() or "internal" in err.lower()

    def test_dns_returning_private_address_blocked(self) -> None:
        """DNS rebinding attempt: hostname resolves to a private IP → blocked."""
        with patch("cogtrix_core.tools._http_safety.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (None, None, None, "", ("10.0.0.5", 0)),
            ]
            ok, err, _ = _validate_url("https://attacker.example.com/")
        assert ok is False
        assert "private" in err.lower()

    def test_dns_with_mixed_public_and_private_blocked(self) -> None:
        """Even one private IP in the resolution set blocks the request."""
        with patch("cogtrix_core.tools._http_safety.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (None, None, None, "", ("8.8.8.8", 0)),
                (None, None, None, "", ("10.0.0.5", 0)),
            ]
            ok, err, _ = _validate_url("https://mixed.example.com/")
        assert ok is False

    def test_dns_failure_blocks_request(self) -> None:
        import socket as _socket

        with patch("cogtrix_core.tools._http_safety.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = _socket.gaierror("no such host")
            ok, err, _ = _validate_url("https://nx.example.com/")
        assert ok is False
        assert "DNS" in err


class TestParseHeaders:
    def test_empty_returns_empty_dict(self) -> None:
        headers, err = _parse_headers(None)
        assert headers == {}
        assert err is None

    def test_empty_string(self) -> None:
        headers, err = _parse_headers("")
        assert headers == {}
        assert err is None

    def test_well_formed_json(self) -> None:
        headers, err = _parse_headers(json.dumps({"Authorization": "Bearer abc"}))
        assert err is None
        assert headers == {"Authorization": "Bearer abc"}

    def test_invalid_json(self) -> None:
        headers, err = _parse_headers("not json")
        assert headers == {}
        assert err is not None
        assert "JSON" in err

    def test_non_object_json(self) -> None:
        headers, err = _parse_headers("[1, 2, 3]")
        assert headers == {}
        assert "object" in (err or "")

    @pytest.mark.parametrize(
        "blocked_name",
        sorted(_BLOCKED_HEADERS),
    )
    def test_blocked_headers_stripped(self, blocked_name: str) -> None:
        headers, _err = _parse_headers(json.dumps({blocked_name: "evil"}))
        assert blocked_name.lower() not in {k.lower() for k in headers}

    def test_blocked_headers_case_insensitive(self) -> None:
        """``HOST: x.com`` and ``host: x.com`` both stripped."""
        headers, _err = _parse_headers(json.dumps({"HOST": "evil.com"}))
        assert headers == {}

    def test_crlf_injection_stripped(self) -> None:
        """Carriage-return / line-feed in header values is sanitised."""
        headers, _err = _parse_headers(json.dumps({"X-Custom": "value\r\nInjected-Header: x"}))
        assert "\r" not in headers["X-Custom"]
        assert "\n" not in headers["X-Custom"]

    def test_crlf_in_header_name_stripped(self) -> None:
        headers, _err = _parse_headers(json.dumps({"X-Foo\r\nInjected": "value"}))
        # name had CR/LF stripped, but the resulting "X-FooInjected" is not blocked
        assert all("\r" not in k and "\n" not in k for k in headers)

    # ── Native dict path (Bug L follow-up, 2026-05-20) ──────────────────
    # LLMs frequently pass headers as dicts directly rather than as a
    # JSON-encoded string. Before this fix, dicts triggered a pydantic
    # ValidationError against the str-typed schema and were caught only
    # by the orchestration's error-wrapping path. The tool now accepts
    # dicts natively.

    def test_dict_headers_passthrough(self) -> None:
        headers, err = _parse_headers({"User-Agent": "Mozilla/5.0"})
        assert err is None
        assert headers == {"User-Agent": "Mozilla/5.0"}

    def test_dict_headers_strips_blocked(self) -> None:
        headers, err = _parse_headers({"Host": "evil.com", "X-Allowed": "ok"})
        assert err is None
        assert "Host" not in headers
        assert headers.get("X-Allowed") == "ok"

    def test_dict_headers_sanitises_crlf(self) -> None:
        headers, err = _parse_headers({"X-Custom": "value\r\nInjected: x"})
        assert err is None
        assert "\r" not in headers["X-Custom"]
        assert "\n" not in headers["X-Custom"]

    def test_dict_headers_coerces_non_string_values(self) -> None:
        # Pydantic dict[str, str] coercion rejects non-string values
        # upstream of _parse_headers; this is a defence-in-depth check
        # for any code path that bypasses pydantic.
        headers, err = _parse_headers({"X-Count": 42, "X-Flag": True})  # type: ignore[dict-item]
        assert err is None
        assert headers == {"X-Count": "42", "X-Flag": "True"}

    def test_non_dict_non_string_rejected(self) -> None:
        headers, err = _parse_headers(123)  # type: ignore[arg-type]
        assert headers == {}
        assert err is not None
        assert "dict" in err.lower() or "json string" in err.lower()
