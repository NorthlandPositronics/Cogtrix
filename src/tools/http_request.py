"""
HTTP request tool - Make HTTP GET and POST requests.
POST requests require user confirmation for safety.
"""

import html as _html_mod
import json
import logging
import re
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Any
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

# Try to import requests
try:
    import requests  # type: ignore[import-untyped]

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

# Re-exports below preserve the historical public surface of this
# module — tests/tools/test_http_request.py imports several of these
# names directly from here. The single source of truth lives in
# src/tools/_http_safety.py. # noqa: F401 markers keep ruff from
# pruning these "unused" imports.
from src.tools._http_safety import (  # noqa: F401
    _BLOCKED_HEADERS,
    _CGNAT_NETWORK,
    _MAX_RESPONSE_BYTES,
    _MAX_TIMEOUT,
    MAX_REDIRECTS,
    _is_blocked_ip,
    _parse_headers,
    _validate_url,
)
from src.tools.delegate import register_tool_categories
from src.tools.error_sanitizer import sanitize_error as _sanitize_error


class HttpGetInput(BaseModel):
    """Input schema for HTTP GET requests."""

    url: str = Field(description="The URL to request")
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            'Optional HTTP headers as an object, e.g. {"Authorization": "Bearer token"}. '
            "Legacy: a JSON-encoded string is also accepted."
        ),
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_chars: int = Field(
        default=10_000,
        description=(
            "Maximum characters of extracted text to return (default: 10,000). "
            "Increase for long documents; decrease to save context."
        ),
    )


class HttpPostInput(BaseModel):
    """Input schema for HTTP POST requests."""

    url: str = Field(description="The URL to request")
    data: str = Field(description='Request body as JSON string (e.g., \'{"key": "value"}\')')
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            'Optional HTTP headers as an object, e.g. {"Authorization": "Bearer token"}. '
            "Legacy: a JSON-encoded string is also accepted."
        ),
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


# ── DNS-pinned connections (BUG-074: eliminate TOCTOU) ──────────────
# urllib3 resolves hostnames inside create_connection().  We intercept
# that function and replace the hostname with a pre-validated IP so
# the same address used for SSRF checks is the one actually connected.
log = logging.getLogger("cogtrix")

_dns_pins: threading.local = threading.local()
_dns_pin_installed: bool = False
_dns_pin_lock = threading.Lock()


def _install_dns_pin_hook() -> bool:
    """Monkey-patch urllib3's create_connection once to honour thread-local pins.

    Returns ``True`` if the pin hook is installed (DNS-rebinding protection
    active), ``False`` if urllib3's connection module could not be imported
    (version skew). Callers MUST treat a ``False`` return as "pinning is
    unavailable" and fall back to post-connection peer-IP verification or fail
    closed (#2136 F1) — silently issuing an un-pinned request restores the
    validate-then-connect TOCTOU this hook exists to close (BUG-074).
    """
    global _dns_pin_installed
    if _dns_pin_installed:
        return True
    with _dns_pin_lock:
        if _dns_pin_installed:
            return True
        try:
            import urllib3.util.connection as _uc  # type: ignore[import-not-found]
        except ImportError:
            log.warning("DNS pin hook unavailable — SSRF rebinding protection degraded")
            return False
        _orig = _uc.create_connection

        def _pinned_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
            host, port = address
            pin_map = getattr(_dns_pins, "map", None)
            if pin_map and host in pin_map:
                address = (pin_map[host], port)
            return _orig(address, *args, **kwargs)

        _uc.create_connection = _pinned_create_connection  # type: ignore[assignment]
        _dns_pin_installed = True
        return True


def _peer_ip_from_response(response: "Any") -> str | None:
    """Best-effort extraction of the remote IP the response actually connected to.

    The post-connection SSRF backstop (#2136 F1/F2): the IP we *validated* must
    equal the IP we *connected to*. Reads the live socket's peer address off the
    underlying urllib3 connection (only reliably available while the response is
    still streaming, i.e. before the body is consumed). Returns the IP string, or
    ``None`` if it can't be determined — urllib3 internals vary across versions
    and the connection may already be released — in which case the caller decides
    whether DNS pinning alone is sufficient or it must fail closed.
    """
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    conn = getattr(raw, "_connection", None) or getattr(raw, "connection", None)
    sock = getattr(conn, "sock", None)
    if sock is None:
        return None
    try:
        peer = sock.getpeername()
    except (OSError, AttributeError):
        return None
    if isinstance(peer, (tuple, list)) and peer:
        return str(peer[0])
    return None


@contextmanager
def _pin_dns(hostname: str, ip: str):  # type: ignore[no-untyped-def]
    """Context manager: pin *hostname* -> *ip* for the calling thread."""
    pin_map = getattr(_dns_pins, "map", None)
    if pin_map is None:
        _dns_pins.map = {}
        pin_map = _dns_pins.map
    pin_map[hostname] = ip
    try:
        yield
    finally:
        pin_map.pop(hostname, None)


# NOTE: ``_is_blocked_ip``, ``_validate_url``, ``_parse_headers``, plus the
# constants ``_BLOCKED_HEADERS`` / ``_CGNAT_NETWORK`` / ``_MAX_TIMEOUT`` /
# ``_MAX_RESPONSE_BYTES`` / ``MAX_REDIRECTS`` are re-exported from
# ``src/tools/_http_safety.py``. See the import block at the top of this
# file. The single source of truth lives in that module so the async
# ``_http_fetch`` primitive (ADR-0056 stage 3) shares it.


# ── Recent failure tracker ──────────────────────────────────────────
# Prevents the model from retrying URLs that just timed out or refused
# connection, saving 30+ seconds per avoided retry.
_recent_failures: dict[str, float] = {}
_recent_failures_lock = threading.Lock()
_FAILURE_COOLDOWN = 60  # seconds
_RECENT_FAILURES_MAX = 1000
_FAILURE_EVICT_INTERVAL = 30.0  # seconds between stale-entry scans
_last_failure_evict: float = 0.0


def _check_recent_failure(url: str) -> str | None:
    """Return an error message if *url* failed recently, else ``None``."""
    with _recent_failures_lock:
        last_fail = _recent_failures.get(url)
        if last_fail is None:
            return None
        now = time.monotonic()
        if (now - last_fail) < _FAILURE_COOLDOWN:
            ago = int(now - last_fail)
            return (
                f"Error: This URL failed {ago}s ago (timeout/connection error). "
                "Try a different source or a web reader proxy like r.jina.ai."
            )
    return None


def _record_failure(url: str) -> None:
    """Record a failure timestamp for *url*."""
    global _last_failure_evict
    now = time.monotonic()
    with _recent_failures_lock:
        _recent_failures[url] = now
        if now - _last_failure_evict >= _FAILURE_EVICT_INTERVAL:
            _last_failure_evict = now
            stale = [k for k, v in _recent_failures.items() if (now - v) >= _FAILURE_COOLDOWN]
            for k in stale:
                del _recent_failures[k]
        while len(_recent_failures) > _RECENT_FAILURES_MAX:
            _recent_failures.pop(next(iter(_recent_failures)))


def _truncate_response(text: str, max_length: int = 10000) -> str:
    """Truncate long responses."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n\n... (truncated, {len(text)} total characters)"


def _read_bounded_response(response: "Any") -> tuple[bytes, bool]:
    """
    Read at most _MAX_RESPONSE_BYTES from a streaming response.

    Returns (raw_bytes, was_truncated).  The caller is responsible for
    closing the response connection.
    """
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_RESPONSE_BYTES:
                    break
    except Exception as exc:  # noqa: BLE001
        # Network error mid-stream — return what we have, marked truncated
        log.debug("Stream read interrupted: %s", exc)
        raw = b"".join(chunks)
        return raw[:_MAX_RESPONSE_BYTES], True
    raw = b"".join(chunks)
    if len(raw) > _MAX_RESPONSE_BYTES:
        return raw[:_MAX_RESPONSE_BYTES], True
    return raw, False


def _follow_redirects(
    session: "Any",
    method: str,
    url: str,
    *,
    pinned_ip: str | None = None,
    **kwargs: "Any",
) -> "Any":
    """
    Follow HTTP redirects manually, validating each redirect target against SSRF rules.
    Uses DNS pinning to prevent rebinding between validation and connection.

    Raises ValueError if a redirect target fails SSRF validation or the redirect
    limit is exceeded.
    """
    pinning_active = _install_dns_pin_hook()
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        hop_pinned = bool(pinning_active and pinned_ip and hostname)
        if hop_pinned:
            pin_ctx = _pin_dns(hostname, pinned_ip)  # type: ignore[arg-type]
        else:
            pin_ctx = nullcontext()

        with pin_ctx:
            response = session.request(method, url, allow_redirects=False, stream=True, **kwargs)

        # Post-connection SSRF re-validation (#2136 F1/F2). The old check compared
        # response.url to the request URL, but with allow_redirects=False requests
        # sets response.url == url, so it was dead code AND a rebind changes the
        # connected IP, not the URL string. Instead verify the IP we actually
        # connected to: it must not be a private/reserved address. If we can't
        # read the peer IP, only proceed when this hop was DNS-pinned to a
        # pre-validated IP; otherwise fail closed rather than trust an un-pinned,
        # unverifiable connection (DNS-rebinding TOCTOU).
        peer_ip = _peer_ip_from_response(response)
        if peer_ip is not None:
            if _is_blocked_ip(peer_ip):
                response.close()
                raise ValueError(
                    "Connection resolved to a private/reserved IP (DNS rebinding blocked)"
                )
        elif not hop_pinned:
            response.close()
            raise ValueError(
                "Unable to verify the connection target IP and DNS pinning is "
                "unavailable — refusing request (SSRF protection)"
            )

        if response.status_code not in (301, 302, 303, 307, 308):
            return response

        location = response.headers.get("Location", "")
        if not location:
            return response

        # Resolve relative Location URLs against the current request URL
        redirect_url = urljoin(url, location)

        is_valid, error, pinned_ip = _validate_url(redirect_url)
        if not is_valid:
            response.close()
            raise ValueError(f"Redirect to private/internal address blocked: {error}")

        response.close()
        url = redirect_url

        # RFC 7231 §6.4.4: 303 See Other requires the client to switch to GET
        # and drop the request body, regardless of the original method.
        if response.status_code == 303:
            method = "GET"
            kwargs.pop("json", None)
            kwargs.pop("data", None)

    response.close()
    raise ValueError(f"Too many redirects (limit: {MAX_REDIRECTS})")


_RE_SCRIPT = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_RE_STYLE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_SVG = re.compile(r"<svg[^>]*>.*?</svg>", re.DOTALL | re.IGNORECASE)
_RE_BLOCK_ELEMENT = re.compile(r"<(?:p|div|br|h[1-6]|li|tr|section|article)[^>]*>", re.IGNORECASE)
_RE_HTML_TAG = re.compile(  # codeql[py/bad-tag-filter] not a security sanitizer — LLM text extraction only, not XSS prevention
    r"<[^>]{0,2000}>"
)
_RE_INLINE_WS = re.compile(r"[^\S\n]+")
_RE_MULTI_BLANK = re.compile(r"\n\s*\n+")


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML, stripping tags and noise.

    Removes script/style blocks, HTML comments, and tags, then
    collapses whitespace into a clean text representation that is
    actually useful for an LLM (as opposed to raw markup).
    """
    text = _RE_SCRIPT.sub(" ", html)
    text = _RE_STYLE.sub(" ", text)
    text = _RE_COMMENT.sub(" ", text)
    text = _RE_SVG.sub(" ", text)
    text = _RE_BLOCK_ELEMENT.sub("\n", text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _html_mod.unescape(text)
    text = _RE_INLINE_WS.sub(" ", text)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def http_get(
    url: str,
    headers: dict[str, str] | str | None = None,
    timeout: int = 30,
    max_chars: int = 10_000,
) -> str:
    """
    Make an HTTP GET request.

    Args:
        url: The URL to request
        headers: Optional HTTP headers as a dict (preferred) or a
            JSON-encoded string (legacy shape kept for backward
            compatibility with older tool-call envelopes).
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

    timeout = min(max(1, timeout), _MAX_TIMEOUT)

    # Validate URL and resolve DNS once (pins the IP to prevent TOCTOU rebinding)
    _install_dns_pin_hook()
    is_valid, error, pinned_ip = _validate_url(url)
    if not is_valid:
        return f"Error: {error}"

    # Skip URLs that failed recently (timeout/connection)
    recent_err = _check_recent_failure(url)
    if recent_err:
        return recent_err

    # Parse headers
    parsed_headers, header_error = _parse_headers(headers)
    if header_error:
        return f"Error: {header_error}"

    try:
        with requests.Session() as session:
            response = _follow_redirects(
                session,
                "GET",
                url,
                pinned_ip=pinned_ip,
                headers=parsed_headers,
                timeout=timeout,
            )
            try:
                raw_bytes, body_truncated = _read_bounded_response(response)
            finally:
                response.close()

        encoding = response.encoding or "utf-8"
        text = raw_bytes.decode(encoding, errors="replace")

        # Build response info
        result = []
        result.append(f"Status: {response.status_code} {response.reason}")
        result.append(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        reported_bytes = len(raw_bytes)
        size_note = " (read limit reached)" if body_truncated else ""
        result.append(f"Content-Length: {reported_bytes} bytes{size_note}")
        result.append("")

        # Try to parse as JSON for better formatting
        try:
            json_data = json.loads(text)
            result.append("Response (JSON):")
            result.append(json.dumps(json_data, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, ValueError):
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                extracted = _extract_text_from_html(text)
                result.append("Response (text extracted from HTML):")
                result.append(_truncate_response(extracted, max_chars))
            else:
                result.append("Response:")
                result.append(_truncate_response(text, max_chars))

        return "\n".join(result)

    except ValueError as e:
        return f"Error: {_sanitize_error(e)}"
    except requests.exceptions.Timeout:
        _record_failure(url)
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        _record_failure(url)
        return f"Error: {_sanitize_error(e)}"
    except requests.exceptions.RequestException as e:
        return f"Error: {_sanitize_error(e)}"
    except Exception as e:  # noqa: BLE001
        log.debug("Unexpected error: %s", e, exc_info=True)
        return f"Error: {_sanitize_error(e)}"


def http_post(
    url: str,
    data: str,
    headers: dict[str, str] | str | None = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP POST request.
    WARNING: This sends data to external servers. Use with caution.

    Args:
        url: The URL to request
        data: Request body as JSON string
        headers: Optional HTTP headers as a dict (preferred) or a
            JSON-encoded string (legacy shape).
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

    timeout = min(max(1, timeout), _MAX_TIMEOUT)

    # Validate URL and resolve DNS once (pins the IP to prevent TOCTOU rebinding)
    _install_dns_pin_hook()
    is_valid, error, pinned_ip = _validate_url(url)
    if not is_valid:
        return f"Error: {error}"

    # Skip URLs that failed recently (timeout/connection)
    recent_err = _check_recent_failure(url)
    if recent_err:
        return recent_err

    # Parse headers
    parsed_headers, header_error = _parse_headers(headers)
    if header_error:
        return f"Error: {header_error}"

    # Parse request data
    try:
        json_data = json.loads(data)
    except json.JSONDecodeError as e:
        return f"Error: {_sanitize_error(e)}"

    # Set Content-Type if not provided
    if "Content-Type" not in parsed_headers and "content-type" not in parsed_headers:
        parsed_headers["Content-Type"] = "application/json"

    try:
        with requests.Session() as session:
            response = _follow_redirects(
                session,
                "POST",
                url,
                pinned_ip=pinned_ip,
                json=json_data,
                headers=parsed_headers,
                timeout=timeout,
            )
            try:
                raw_bytes, body_truncated = _read_bounded_response(response)
            finally:
                response.close()

        encoding = response.encoding or "utf-8"
        text = raw_bytes.decode(encoding, errors="replace")

        # Build response info
        result = []
        result.append(f"Status: {response.status_code} {response.reason}")
        result.append(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        reported_bytes = len(raw_bytes)
        size_note = " (read limit reached)" if body_truncated else ""
        result.append(f"Content-Length: {reported_bytes} bytes{size_note}")
        result.append("")

        # Try to parse as JSON for better formatting
        try:
            json_response = json.loads(text)
            result.append("Response (JSON):")
            result.append(json.dumps(json_response, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, ValueError):
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                extracted = _extract_text_from_html(text)
                result.append("Response (text extracted from HTML):")
                result.append(_truncate_response(extracted))
            else:
                result.append("Response:")
                result.append(_truncate_response(text))

        return "\n".join(result)

    except ValueError as e:
        return f"Error: {_sanitize_error(e)}"
    except requests.exceptions.Timeout:
        _record_failure(url)
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        _record_failure(url)
        return f"Error: {_sanitize_error(e)}"
    except requests.exceptions.RequestException as e:
        return f"Error: {_sanitize_error(e)}"
    except Exception as e:  # noqa: BLE001
        log.debug("Unexpected error: %s", e, exc_info=True)
        return f"Error: {_sanitize_error(e)}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "http_get",
        "description": (
            "Make an HTTP GET request to a URL. "
            "Returns status, headers, and response body (default: 10,000 chars of extracted text). "
            "Set max_chars higher for long documents, lower to save context budget. "
            "Use this to read full page content when a search snippet is too short. "
            "If the page does not contain the information you need, return to the "
            "search results and try another URL."
        ),
        "input_schema": HttpGetInput,
        "requires_confirmation": False,
        "function": http_get,
        "category": "readonly",
    },
    {
        "name": "http_post",
        "description": (
            "Make an HTTP POST request with JSON data. "
            "WARNING: This sends data to external servers."
        ),
        "input_schema": HttpPostInput,
        "requires_confirmation": True,  # Requires confirmation for safety
        "function": http_post,
        "category": "mutation",
    },
]

register_tool_categories({"http_get": "readonly", "http_post": "mutation"})

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "http_get",
    "http_post",
    "HttpGetInput",
    "HttpPostInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
