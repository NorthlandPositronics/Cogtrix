"""Tests for src/tools/error_sanitizer.py.

Issue #1454: network OSError subclasses (ConnectionResetError, ConnectionError,
TimeoutError) are covered by TestNetworkOSErrorSubclasses at the end of this file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.tools.error_sanitizer import (
    _google_http_category,
    sanitize_error,
    sanitize_file_error,
    sanitize_google_api_error,
    sanitize_http_error,
    sanitize_search_error,
    sanitize_shell_error,
)

# ── sanitize_error — generic fallback ─────────────────────────────────────────


class TestSanitizeErrorGenericFallback:
    def test_value_error_returns_invalid_data_format(self) -> None:
        assert sanitize_error(ValueError("something went wrong")) == "Invalid data format"

    def test_json_JSONDecodeError_returns_invalid_data_format(self) -> None:
        assert sanitize_error(json.JSONDecodeError("...", "", 0)) == "Invalid data format"

    def test_RuntimeError_returns_operation_failed(self) -> None:
        assert sanitize_error(RuntimeError("something broke")) == "Operation failed"

    def test_RuntimeError_with_api_key_not_configured(self) -> None:
        assert sanitize_error(RuntimeError("API key not configured")) == "API key not configured"

    def test_unknown_exception_returns_fallback(self) -> None:
        class WeirdError(Exception):
            pass

        assert sanitize_error(WeirdError("details")) == "Operation failed"
        assert (
            sanitize_error(WeirdError("details"), fallback="Custom fallback") == "Custom fallback"
        )


# ── sanitize_error — filesystem ───────────────────────────────────────────────


class TestSanitizeErrorFilesystem:
    def test_FileNotFoundError_returns_file_not_found(self) -> None:
        assert sanitize_error(FileNotFoundError("/path/to/file")) == "File not found"

    def test_PermissionError_returns_permission_denied(self) -> None:
        assert sanitize_error(PermissionError("/path")) == "Permission denied"

    def test_IsADirectoryError_returns_not_a_file(self) -> None:
        assert sanitize_error(IsADirectoryError("/path")) == "Not a file"

    def test_UnicodeDecodeError_returns_could_not_decode(self) -> None:
        assert sanitize_error(UnicodeDecodeError("utf-8", b"", 0, 1, "reason")) == (
            "Could not decode file with the given encoding"
        )

    def test_OSError_ENOSPC_returns_disk_full(self) -> None:
        exc = OSError(28, "No space left on device")
        assert sanitize_error(exc) == "Disk full"

    def test_OSError_generic_returns_filesystem_operation_failed(self) -> None:
        exc = OSError(99, "Something went wrong")
        assert sanitize_error(exc) == "Filesystem operation failed"

    def test_IOError_returns_filesystem_operation_failed(self) -> None:
        exc = OSError("broken pipe")
        assert sanitize_error(exc) == "Filesystem operation failed"


# ── sanitize_error — shell ─────────────────────────────────────────────────────


class TestSanitizeErrorShell:
    def test_CalledProcessError_returns_command_failed_with_exit_code(self) -> None:
        import subprocess

        exc = subprocess.CalledProcessError(127, ["ls", "-la"])
        result = sanitize_error(exc)
        assert result == "Command failed (exit code 127)"


# ── sanitize_error — requests (lazy import) ───────────────────────────────────


class TestSanitizeErrorRequests:
    def test_requests_Timeout_returns_request_timed_out(self) -> None:
        with patch.dict("sys.modules", {"requests.exceptions": MagicMock()}):
            import requests.exceptions

            exc = requests.exceptions.Timeout()
            assert sanitize_error(exc) == "Request timed out"

    def test_requests_ConnectionError_returns_connection_failed(self) -> None:
        with patch.dict("sys.modules", {"requests.exceptions": MagicMock()}):
            import requests.exceptions

            exc = requests.exceptions.ConnectionError()
            assert sanitize_error(exc) == "Connection failed"


# ── sanitize_http_error ────────────────────────────────────────────────────────


class TestSanitizeHttpError:
    def test_wraps_generic_result_as_request_failed(self) -> None:
        exc = RuntimeError("something broke")
        result = sanitize_http_error(exc)
        assert result == "Request failed"

    def test_preserves_specific_http_messages(self) -> None:
        exc = json.JSONDecodeError("...", "", 0)
        result = sanitize_http_error(exc)
        assert result == "Invalid data format"


# ── sanitize_shell_error ───────────────────────────────────────────────────────


class TestSanitizeShellError:
    def test_wraps_generic_result_as_command_execution_failed(self) -> None:
        exc = RuntimeError("something broke")
        result = sanitize_shell_error(exc)
        assert result == "Command execution failed"


# ── sanitize_file_error ────────────────────────────────────────────────────────


class TestSanitizeFileError:
    def test_wraps_generic_result_as_filesystem_operation_failed(self) -> None:
        exc = RuntimeError("something broke")
        result = sanitize_file_error(exc)
        assert result == "Filesystem operation failed"


# ── sanitize_search_error ──────────────────────────────────────────────────────


class TestSanitizeSearchError:
    def test_wraps_generic_result_as_search_request_failed(self) -> None:
        exc = RuntimeError("something broke")
        result = sanitize_search_error(exc)
        assert result == "Search request failed"

    def test_custom_context_in_message(self) -> None:
        exc = RuntimeError("something broke")
        result = sanitize_search_error(exc, context="WebSearch")
        assert result == "WebSearch request failed"


# ── _google_http_category ──────────────────────────────────────────────────────


class TestGoogleHttpCategory:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, "authentication"),
            (403, "rate-limit"),
            (404, "not-found"),
            (429, "rate-limit"),
            (400, "client-error"),
            (422, "client-error"),
            (499, "client-error"),
            (500, "server-error"),
            (502, "server-error"),
            (503, "server-error"),
            (600, "unknown"),
            (200, "unknown"),
            (301, "unknown"),
        ],
    )
    def test_category_classification(self, status: int, expected: str) -> None:
        assert _google_http_category(status) == expected


# ── sanitize_google_api_error — HttpError handling ─────────────────────────────


class TestSanitizeGoogleApiError:
    """Tests for googleapiclient.errors.HttpError sanitization.

    HttpError.str() embeds the full request URI (including key=... query
    parameter) which can expose Google API keys to the LLM. These tests verify
    that the sanitization strips API keys and returns category-hinted messages.
    """

    @classmethod
    def setup_class(cls) -> None:
        pytest.importorskip("googleapiclient")

    def _make_http_error(
        self,
        status: int,
        message: str,
        reason: str = "",
        uri: str | None = None,
    ):
        """Build a real googleapiclient.errors.HttpError for testing."""
        from googleapiclient.errors import HttpError

        resp = SimpleNamespace(status=status, reason=reason or message)
        body = json.dumps({"error": {"message": message, "code": status}}).encode()
        if uri:
            return HttpError(resp, body, uri=uri)
        return HttpError(resp, body)

    def test_403_rate_limit_returns_correct_message(self) -> None:
        exc = self._make_http_error(403, "Rate limit exceeded", reason="Forbidden")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 403 (rate-limit)"

    def test_401_authentication(self) -> None:
        exc = self._make_http_error(401, "Invalid credentials", reason="Unauthorized")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 401 (authentication)"

    def test_404_not_found(self) -> None:
        exc = self._make_http_error(404, "Calendar not found", reason="Not Found")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 404 (not-found)"

    def test_429_rate_limit(self) -> None:
        exc = self._make_http_error(429, "Quota exceeded", reason="Too Many Requests")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 429 (rate-limit)"

    def test_500_server_error(self) -> None:
        exc = self._make_http_error(500, "Internal server error", reason="Internal Server Error")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 500 (server-error)"

    def test_400_client_error(self) -> None:
        exc = self._make_http_error(400, "Bad request", reason="Bad Request")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 400 (client-error)"

    def test_no_status_code_falls_back_to_message(self) -> None:
        from googleapiclient.errors import HttpError

        resp = SimpleNamespace(status=None, reason="Error")  # type: ignore[assignment]
        body = json.dumps({"error": {"message": "generic error"}}).encode()
        exc = HttpError(resp, body)
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: request failed"

    def test_malformed_json_body_falls_back_safely(self) -> None:
        from googleapiclient.errors import HttpError

        resp = SimpleNamespace(status=403, reason="Forbidden")
        exc = HttpError(resp, b"not valid json at all")
        result = sanitize_google_api_error(exc)
        assert result == "Google API error: 403 (rate-limit)"

    def test_api_key_in_uri_parameter_is_not_in_result(self) -> None:
        """URI with key=... passed to HttpError must not appear in sanitized output."""
        exc = self._make_http_error(
            403,
            "Rate limit exceeded",
            reason="Forbidden",
            uri="https://www.googleapis.com/calendar/v3/calendars/primary/events?key=AIzaSyDANG3R0L0NG0AND0T0T0T0T0T0T0T0TG",
        )
        result = sanitize_google_api_error(exc)
        assert "key=AIzaSyDANG3R0L0NG0AND0T0T0T0T0T0T0TG" not in result
        assert "Google API error: 403 (rate-limit)" in result

    def test_api_key_in_message_body_is_stripped(self) -> None:
        """API key embedded in the error message body must be redacted."""
        from googleapiclient.errors import HttpError

        resp = SimpleNamespace(status=403, reason="Forbidden")
        body = json.dumps(
            {
                "error": {
                    "message": (
                        "The request URI was: "
                        "https://www.googleapis.com/calendar/v3/calendars/primary/events"
                        "?key=AIzaSyVeryLongFakeApiKeyHere123456789"
                    )
                }
            }
        ).encode()
        exc = HttpError(resp, body)
        result = sanitize_google_api_error(exc)
        assert "AIzaSyVeryLongFakeApiKeyHere123456789" not in result
        assert "Google API error: 403 (rate-limit)" in result

    def test_custom_service_name_in_message(self) -> None:
        exc = self._make_http_error(404, "not found", reason="Not Found")
        result = sanitize_google_api_error(exc, service="Google Calendar API")
        assert result == "Google Calendar API error: 404 (not-found)"

    def test_non_http_error_falls_through_to_sanitize_error(self) -> None:
        """Non-HttpError exceptions should be handled by the generic sanitizer."""
        exc = FileNotFoundError("/path/to/file")
        result = sanitize_google_api_error(exc)
        assert result == "File not found"

    def test_permission_error_in_google_context(self) -> None:
        exc = PermissionError("/path")
        result = sanitize_google_api_error(exc)
        assert result == "Permission denied"

    def test_http_error_str_does_leak_key_but_sanitize_does_not(self) -> None:
        """Verify the raw HttpError.str() would leak the key, but our sanitization doesn't."""
        exc = self._make_http_error(
            403,
            "Rate limit exceeded",
            reason="Forbidden",
            uri="https://www.googleapis.com/calendar/v3/calendars/primary/events?key=AIzaSyLEAKYLEAKYLEAK",
        )
        raw_str = str(exc)
        sanitized = sanitize_google_api_error(exc)
        # Raw str() leaks the key
        assert "key=AIzaSyLEAKYLEAKYLEAK" in raw_str
        # Sanitized output does NOT leak the key
        assert "key=" not in sanitized
        assert "AIzaSyLEAKYLEAKYLEAK" not in sanitized


# ── network OSError subclasses (issue #1454) ──────────────────────────────────


class TestNetworkOSErrorSubclasses:
    """
    Regression coverage for issue #1454.

    ConnectionResetError, ConnectionError, and TimeoutError are OSError
    subclasses. Without explicit handling they fall through to the OSError
    branch and receive "Filesystem operation failed" — semantically wrong
    and will cause the LLM to suggest filesystem remediation instead of
    network remediation.
    """

    def test_connection_reset_error_returns_network_message(self):
        """ConnectionResetError must not return filesystem fallback."""
        exc = ConnectionResetError("Connection reset by peer")
        result = sanitize_error(exc)
        assert (
            "Filesystem" not in result
        ), f"ConnectionResetError got filesystem message: {result!r}"
        assert "Filesystem operation failed" != result

    def test_connection_error_returns_network_message(self):
        """ConnectionError must not return filesystem fallback."""
        exc = ConnectionError("Connection refused")
        result = sanitize_error(exc)
        assert "Filesystem" not in result, f"ConnectionError got filesystem message: {result!r}"
        assert "Filesystem operation failed" != result

    def test_timeout_error_returns_network_message(self):
        """TimeoutError must not return filesystem fallback."""
        exc = TimeoutError("timed out")
        result = sanitize_error(exc)
        assert "Filesystem" not in result, f"TimeoutError got filesystem message: {result!r}"
        assert "Filesystem operation failed" != result

    def test_connection_reset_error_not_leaked(self):
        """Exception class name must not appear in sanitized output."""
        exc = ConnectionResetError("Connection reset by peer")
        result = sanitize_error(exc)
        assert "ConnectionResetError" not in result

    def test_connection_error_not_leaked(self):
        """Exception class name must not appear in sanitized output."""
        exc = ConnectionError("Connection refused")
        result = sanitize_error(exc)
        assert "ConnectionError" not in result

    def test_timeout_error_not_leaked(self):
        """Exception class name must not appear in sanitized output."""
        exc = TimeoutError("timed out")
        result = sanitize_error(exc)
        assert "TimeoutError" not in result

    def test_connection_reset_error_message_not_leaked(self):
        """Raw exception message must not appear in sanitized output."""
        exc = ConnectionResetError("reset by peer at 192.168.1.1:8080")
        result = sanitize_error(exc)
        assert "192.168.1.1" not in result
        assert "reset by peer" not in result

    def test_connection_error_message_not_leaked(self):
        """Raw exception message must not appear in sanitized output."""
        exc = ConnectionError("refused at 10.0.0.5:443")
        result = sanitize_error(exc)
        assert "10.0.0.5" not in result
        assert "refused" not in result

    def test_timeout_error_message_not_leaked(self):
        """Raw exception message must not appear in sanitized output."""
        exc = TimeoutError("connection timed out after 30s")
        result = sanitize_error(exc)
        assert "30s" not in result
        assert "timed out after" not in result
