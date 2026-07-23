"""Tests for the http_request tool — timeout clamping and SSRF protection."""

import json
import socket
from unittest.mock import MagicMock, patch

from src.tools.http_request import (
    _BLOCKED_HEADERS,
    _MAX_TIMEOUT,
    _dns_pins,
    _is_blocked_ip,
    _parse_headers,
    _pin_dns,
    _validate_url,
    http_get,
    http_post,
)


def _make_mock_response(status_code: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = "OK"
    resp.headers = {"Content-Type": "text/plain"}
    resp.content = text.encode()
    resp.text = text
    resp.encoding = "utf-8"
    resp.json.side_effect = ValueError("not json")
    resp.iter_content.return_value = iter([text.encode("utf-8")])
    return resp


class TestHttpGetTimeoutClamping:
    def test_large_timeout_is_clamped_to_max(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            result = http_get("https://example.com", timeout=999)

        assert len(captured) == 1
        assert captured[0] == _MAX_TIMEOUT
        assert "Error" not in result

    def test_zero_timeout_is_clamped_to_one(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            http_get("https://example.com", timeout=0)

        assert captured[0] == 1

    def test_negative_timeout_is_clamped_to_one(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            http_get("https://example.com", timeout=-5)

        assert captured[0] == 1

    def test_in_range_timeout_is_unchanged(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            http_get("https://example.com", timeout=30)

        assert captured[0] == 30


class TestHttpPostTimeoutClamping:
    def test_large_timeout_is_clamped_to_max(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            result = http_post("https://example.com", data='{"key": "value"}', timeout=86400)

        assert len(captured) == 1
        assert captured[0] == _MAX_TIMEOUT
        assert "Error" not in result

    def test_zero_timeout_is_clamped_to_one(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            http_post("https://example.com", data='{"key": "value"}', timeout=0)

        assert captured[0] == 1

    def test_in_range_timeout_is_unchanged(self) -> None:
        captured: list[int] = []

        def fake_follow_redirects(session, method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["timeout"])
            return _make_mock_response()

        with (
            patch("src.tools.http_request.REQUESTS_AVAILABLE", True),
            patch("src.tools.http_request._follow_redirects", side_effect=fake_follow_redirects),
            patch("src.tools.http_request._check_recent_failure", return_value=None),
        ):
            http_post("https://example.com", data='{"key": "value"}', timeout=60)

        assert captured[0] == 60


class TestIsBlockedIp:
    """Unit tests for _is_blocked_ip()."""

    def test_loopback_127_0_0_1(self) -> None:
        assert _is_blocked_ip("127.0.0.1") is True

    def test_loopback_127_0_0_2(self) -> None:
        assert _is_blocked_ip("127.0.0.2") is True

    def test_loopback_127_255_255_255(self) -> None:
        assert _is_blocked_ip("127.255.255.255") is True

    def test_loopback_ipv6(self) -> None:
        assert _is_blocked_ip("::1") is True

    def test_private_10_block(self) -> None:
        assert _is_blocked_ip("10.0.0.1") is True

    def test_private_172_16(self) -> None:
        assert _is_blocked_ip("172.16.0.1") is True

    def test_private_172_31(self) -> None:
        assert _is_blocked_ip("172.31.255.255") is True

    def test_private_192_168(self) -> None:
        assert _is_blocked_ip("192.168.1.1") is True

    def test_link_local_169_254(self) -> None:
        assert _is_blocked_ip("169.254.169.254") is True

    def test_link_local_ipv6(self) -> None:
        assert _is_blocked_ip("fe80::1") is True

    def test_unspecified_0_0_0_0(self) -> None:
        assert _is_blocked_ip("0.0.0.0") is True

    def test_public_ip_not_blocked(self) -> None:
        assert _is_blocked_ip("93.184.216.34") is False

    def test_invalid_string_returns_false(self) -> None:
        assert _is_blocked_ip("not-an-ip") is False


class TestValidateUrlSsrf:
    """Tests for SSRF-related URL validation in _validate_url()."""

    def test_public_https_url_is_valid(self) -> None:
        ok, err, _ip = _validate_url("https://example.com/path")
        assert ok is True
        assert err == ""

    def test_localhost_by_name_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://localhost/admin")
        assert ok is False

    def test_127_0_0_1_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://127.0.0.1/")
        assert ok is False

    def test_127_0_0_2_blocked(self) -> None:
        """BUG-052: 127.0.0.2 is a loopback address and must be blocked."""
        ok, _err, _ip = _validate_url("http://127.0.0.2/")
        assert ok is False

    def test_127_x_x_x_range_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://127.100.50.1/")
        assert ok is False

    def test_decimal_integer_ip_blocked(self) -> None:
        """BUG-053: 2130706433 == 127.0.0.1 as a decimal integer literal."""
        ok, _err, _ip = _validate_url("http://2130706433/")
        assert ok is False

    def test_hex_ip_blocked(self) -> None:
        """BUG-053: 0x7f000001 == 127.0.0.1 in hex notation."""
        ok, _err, _ip = _validate_url("http://0x7f000001/")
        assert ok is False

    def test_private_10_range_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://10.0.0.1/")
        assert ok is False

    def test_private_192_168_range_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://192.168.1.100/")
        assert ok is False

    def test_private_172_16_range_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://172.16.0.1/")
        assert ok is False

    def test_link_local_metadata_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://169.254.169.254/latest/meta-data/")
        assert ok is False

    def test_metadata_google_internal_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://metadata.google.internal/")
        assert ok is False

    def test_ipv6_loopback_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://[::1]/")
        assert ok is False

    def test_ipv6_link_local_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://[fe80::1]/")
        assert ok is False

    def test_unspecified_0_0_0_0_blocked(self) -> None:
        ok, _err, _ip = _validate_url("http://0.0.0.0/")
        assert ok is False

    def test_unsupported_scheme_blocked(self) -> None:
        ok, _err, _ip = _validate_url("ftp://example.com/")
        assert ok is False

    def test_dns_resolves_to_private_ip_blocked(self) -> None:
        """Hostname that DNS resolves to a private IP must be blocked."""
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
        with patch("src.tools.http_request.socket.getaddrinfo", return_value=fake_addrinfo):
            ok, _err, _ip = _validate_url("http://internal.corp.example/")
        assert ok is False

    def test_dns_resolves_to_public_ip_allowed(self) -> None:
        """Hostname that DNS resolves to a public IP must be allowed."""
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        with patch("src.tools.http_request.socket.getaddrinfo", return_value=fake_addrinfo):
            ok, err, _ip = _validate_url("http://example.com/")
        assert ok is True
        assert err == ""


class TestDnsPinning:
    """Tests for DNS pinning infrastructure (BUG-074)."""

    def test_validate_url_returns_resolved_ip(self) -> None:
        """_validate_url returns the first public IP as the third element."""
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        with patch("src.tools.http_request.socket.getaddrinfo", return_value=fake_addrinfo):
            ok, err, ip = _validate_url("https://example.com/")
        assert ok is True
        assert err == ""
        assert ip == "93.184.216.34"

    def test_validate_url_rejects_on_dns_failure(self) -> None:
        """_validate_url rejects URLs when DNS resolution fails (SEC-002)."""
        with patch(
            "src.tools.http_request.socket.getaddrinfo",
            side_effect=socket.gaierror("DNS failure"),
        ):
            ok, err, ip = _validate_url("https://example.com/")
        assert ok is False
        assert "DNS resolution failed" in err
        assert ip is None

    def test_pin_dns_sets_thread_local(self) -> None:
        """_pin_dns context manager sets the thread-local pin and clears it on exit."""
        hostname = "example.com"
        ip = "93.184.216.34"

        # Ensure no stale pin exists before the test
        if hasattr(_dns_pins, "map"):
            _dns_pins.map.pop(hostname, None)

        with _pin_dns(hostname, ip):
            assert getattr(_dns_pins, "map", {}).get(hostname) == ip

        assert getattr(_dns_pins, "map", {}).get(hostname) is None

    def test_follow_redirects_pins_dns_for_each_hop(self) -> None:
        """_follow_redirects uses the pinned IP for the initial request, then validates
        and re-pins for each redirect hop."""
        from src.tools.http_request import _follow_redirects

        public_ip = "93.184.216.34"
        pinned_calls: list[str] = []

        def fake_validate(url: str) -> tuple[bool, str, str | None]:
            pinned_calls.append(url)
            return True, "", public_ip

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://example.com/final"}
        redirect_response.close = MagicMock()

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {}
        final_response.close = MagicMock()

        session = MagicMock()
        session.request.side_effect = [redirect_response, final_response]

        with patch("src.tools.http_request._validate_url", side_effect=fake_validate):
            result = _follow_redirects(
                session,
                "GET",
                "https://example.com/start",
                pinned_ip=public_ip,
                timeout=10,
            )

        assert result is final_response
        # _validate_url must have been called for the redirect target
        assert any("final" in url for url in pinned_calls)


class TestParseHeaders:
    """Tests for _parse_headers — blocked-header stripping and CRLF sanitization."""

    def test_blocked_headers_set_is_defined(self) -> None:
        assert "host" in _BLOCKED_HEADERS
        assert "x-forwarded-for" in _BLOCKED_HEADERS
        assert "x-real-ip" in _BLOCKED_HEADERS
        assert "x-forwarded-host" in _BLOCKED_HEADERS
        assert "x-forwarded-proto" in _BLOCKED_HEADERS
        assert "x-forwarded-server" in _BLOCKED_HEADERS

    def test_host_header_stripped(self) -> None:
        headers_json = json.dumps({"Host": "evil.internal", "Accept": "text/plain"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert "Host" not in result
        assert "host" not in result
        assert result.get("Accept") == "text/plain"

    def test_x_forwarded_for_stripped(self) -> None:
        headers_json = json.dumps({"X-Forwarded-For": "127.0.0.1"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert not any(k.lower() == "x-forwarded-for" for k in result)

    def test_x_real_ip_stripped(self) -> None:
        headers_json = json.dumps({"X-Real-IP": "10.0.0.1"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert not any(k.lower() == "x-real-ip" for k in result)

    def test_x_forwarded_host_stripped(self) -> None:
        headers_json = json.dumps({"X-Forwarded-Host": "internal.host"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert not any(k.lower() == "x-forwarded-host" for k in result)

    def test_x_forwarded_proto_stripped(self) -> None:
        headers_json = json.dumps({"X-Forwarded-Proto": "https"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert not any(k.lower() == "x-forwarded-proto" for k in result)

    def test_x_forwarded_server_stripped(self) -> None:
        headers_json = json.dumps({"X-Forwarded-Server": "proxy.internal"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert not any(k.lower() == "x-forwarded-server" for k in result)

    def test_all_blocked_headers_stripped_at_once(self) -> None:
        payload = {
            "Host": "evil",
            "X-Forwarded-For": "127.0.0.1",
            "X-Real-IP": "10.0.0.1",
            "X-Forwarded-Host": "bad",
            "X-Forwarded-Proto": "http",
            "X-Forwarded-Server": "proxy",
            "Authorization": "Bearer token",
        }
        result, err = _parse_headers(json.dumps(payload))
        assert err is None
        lower_keys = {k.lower() for k in result}
        for blocked in _BLOCKED_HEADERS:
            assert blocked not in lower_keys
        assert "authorization" in lower_keys

    def test_crlf_stripped_from_header_value(self) -> None:
        headers_json = json.dumps({"X-Custom": "value\r\nX-Injected: bad"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert "\r" not in result.get("X-Custom", "")
        assert "\n" not in result.get("X-Custom", "")

    def test_cr_only_stripped_from_header_value(self) -> None:
        headers_json = json.dumps({"X-Custom": "val\rue"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert "\r" not in result.get("X-Custom", "")

    def test_lf_only_stripped_from_header_value(self) -> None:
        headers_json = json.dumps({"X-Custom": "val\nue"})
        result, err = _parse_headers(headers_json)
        assert err is None
        assert "\n" not in result.get("X-Custom", "")

    def test_none_returns_empty_dict(self) -> None:
        result, err = _parse_headers(None)
        assert result == {}
        assert err is None

    def test_invalid_json_returns_error(self) -> None:
        result, err = _parse_headers("{not valid json")
        assert result == {}
        assert err is not None
        assert "Invalid headers JSON" in err

    def test_non_object_json_returns_error(self) -> None:
        result, err = _parse_headers('["list", "not", "object"]')
        assert result == {}
        assert err is not None

    def test_allowed_headers_pass_through(self) -> None:
        headers_json = json.dumps(
            {"Authorization": "Bearer abc", "Content-Type": "application/json"}
        )
        result, err = _parse_headers(headers_json)
        assert err is None
        assert result["Authorization"] == "Bearer abc"
        assert result["Content-Type"] == "application/json"
