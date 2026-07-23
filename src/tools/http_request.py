"""
HTTP request tool - Make HTTP GET and POST requests.
POST requests require user confirmation for safety.
"""

import ipaddress
import json
import re
import socket
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

MAX_REDIRECTS = 5
_MAX_TIMEOUT = 120  # seconds
_MAX_RESPONSE_BYTES = 512_000  # 512 KB — more than enough for 10 K char truncation

# RFC 6598 Shared Address Space (CGNAT) — not classified as private by ipaddress module
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class HttpGetInput(BaseModel):
    """Input schema for HTTP GET requests."""

    url: str = Field(description="The URL to request")
    headers: str | None = Field(
        default=None,
        description='Optional headers as JSON string (e.g., \'{"Authorization": "Bearer token"}\')',
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


class HttpPostInput(BaseModel):
    """Input schema for HTTP POST requests."""

    url: str = Field(description="The URL to request")
    data: str = Field(description='Request body as JSON string (e.g., \'{"key": "value"}\')')
    headers: str | None = Field(
        default=None,
        description="Optional headers as JSON string",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


# ── DNS-pinned connections (BUG-074: eliminate TOCTOU) ──────────────
# urllib3 resolves hostnames inside create_connection().  We intercept
# that function and replace the hostname with a pre-validated IP so
# the same address used for SSRF checks is the one actually connected.
_dns_pins: threading.local = threading.local()
_dns_pin_installed: bool = False
_dns_pin_lock = threading.Lock()


def _install_dns_pin_hook() -> None:
    """Monkey-patch urllib3's create_connection once to honour thread-local pins."""
    global _dns_pin_installed
    if _dns_pin_installed:
        return
    with _dns_pin_lock:
        if _dns_pin_installed:
            return
        try:
            import urllib3.util.connection as _uc  # type: ignore[import-not-found]
        except ImportError:
            return
        _orig = _uc.create_connection

        def _pinned_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
            host, port = address
            pin_map = getattr(_dns_pins, "map", None)
            if pin_map and host in pin_map:
                address = (pin_map[host], port)
            return _orig(address, *args, **kwargs)

        _uc.create_connection = _pinned_create_connection  # type: ignore[assignment]
        _dns_pin_installed = True


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


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if ip_str represents a non-public IP address."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # Unwrap IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1) so that all
    # IPv4-space checks (CGNAT, loopback, private, …) apply correctly.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip in _CGNAT_NETWORK:
        return True
    return (
        ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified
    )


def _validate_url(url: str) -> tuple[bool, str, str | None]:
    """Validate URL for safety. Returns (is_valid, error, resolved_ip)."""
    try:
        parsed = urlparse(url)

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format", None

        # Only allow http and https
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}", None

        hostname = parsed.hostname or ""
        if not hostname:
            return False, "Invalid URL format", None

        # Defense-in-depth: block well-known internal hostnames by name
        blocked_hosts = {
            "localhost",
            "metadata.google.internal",
            "instance-data",
            "169.254.169.254",
        }  # nosec B104
        if hostname.lower() in blocked_hosts:
            return False, "Requests to localhost or internal hosts are not allowed", None

        # If the hostname is a raw IP literal (including decimal/hex/octal forms),
        # ipaddress.ip_address() will parse it directly — catches 2130706433,
        # 0x7f000001, 0177.0.0.1, 127.0.0.2, ::1, etc.
        try:
            if _is_blocked_ip(hostname):
                return (
                    False,
                    "Requests to localhost or private/reserved IP ranges are not allowed",
                    None,
                )
        except Exception:
            pass

        # Resolve hostname via DNS and check every returned address
        resolved_ip: str | None = None
        try:
            addrinfo = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for _family, _type, _proto, _canonname, sockaddr in addrinfo:
                ip_str = str(sockaddr[0])
                if _is_blocked_ip(ip_str):
                    return (
                        False,
                        "Requests to localhost or private/reserved IP ranges are not allowed",
                        None,
                    )
                if resolved_ip is None:
                    resolved_ip = ip_str
        except socket.gaierror:
            return False, "DNS resolution failed for hostname", None

        return True, "", resolved_ip
    except Exception as e:
        return False, f"URL validation error: {e}", None


def _parse_headers(headers_str: str | None) -> tuple[dict, str | None]:
    """Parse headers JSON string."""
    if not headers_str:
        return {}, None

    try:
        headers = json.loads(headers_str)
        if not isinstance(headers, dict):
            return {}, "Headers must be a JSON object"
        return headers, None
    except json.JSONDecodeError as e:
        return {}, f"Invalid headers JSON: {e}"


# ── Recent failure tracker ──────────────────────────────────────────
# Prevents the model from retrying URLs that just timed out or refused
# connection, saving 30+ seconds per avoided retry.
_recent_failures: dict[str, float] = {}
_recent_failures_lock = threading.Lock()
_FAILURE_COOLDOWN = 60  # seconds


def _check_recent_failure(url: str) -> str | None:
    """Return an error message if *url* failed recently, else ``None``."""
    with _recent_failures_lock:
        last_fail = _recent_failures.get(url)
        if last_fail is None:
            return None
        if (time.time() - last_fail) < _FAILURE_COOLDOWN:
            ago = int(time.time() - last_fail)
            return (
                f"Error: This URL failed {ago}s ago (timeout/connection error). "
                "Try a different source or a web reader proxy like r.jina.ai."
            )
    return None


def _record_failure(url: str) -> None:
    """Record a failure timestamp for *url*."""
    now = time.time()
    with _recent_failures_lock:
        _recent_failures[url] = now
        stale = [k for k, v in _recent_failures.items() if (now - v) >= _FAILURE_COOLDOWN]
        for k in stale:
            del _recent_failures[k]


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
    except Exception:
        # Network error mid-stream — return what we have, marked truncated
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
    _install_dns_pin_hook()
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        if pinned_ip and hostname:
            pin_ctx = _pin_dns(hostname, pinned_ip)
        else:
            pin_ctx = nullcontext()

        with pin_ctx:
            response = session.request(method, url, allow_redirects=False, stream=True, **kwargs)

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

    response.close()
    raise ValueError(f"Too many redirects (limit: {MAX_REDIRECTS})")


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML, stripping tags and noise.

    Removes script/style blocks, HTML comments, and tags, then
    collapses whitespace into a clean text representation that is
    actually useful for an LLM (as opposed to raw markup).
    """
    # Remove script and style elements entirely
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Remove SVG elements (often huge and useless)
    text = re.sub(r"<svg[^>]*>.*?</svg>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Insert newline before block-level elements for readability
    text = re.sub(
        r"<(?:p|div|br|h[1-6]|li|tr|section|article)[^>]*>", "\n", text, flags=re.IGNORECASE
    )
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    for entity, char in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&nbsp;", " "),
        ("&#x27;", "'"),
        ("&ndash;", "\u2013"),
        ("&mdash;", "\u2014"),
    ):
        text = text.replace(entity, char)
    # Collapse runs of whitespace (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def http_get(url: str, headers: str | None = None, timeout: int = 30) -> str:
    """
    Make an HTTP GET request.

    Args:
        url: The URL to request
        headers: Optional headers as JSON string
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install it with: pip install requests"

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
                result.append(_truncate_response(extracted))
            else:
                result.append("Response:")
                result.append(_truncate_response(text))

        return "\n".join(result)

    except ValueError as e:
        return f"Error: {e}"
    except requests.exceptions.Timeout:
        _record_failure(url)
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        _record_failure(url)
        return f"Error: Connection failed - {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except Exception as e:
        return f"Error: {e}"


def http_post(
    url: str,
    data: str,
    headers: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP POST request.
    WARNING: This sends data to external servers. Use with caution.

    Args:
        url: The URL to request
        data: Request body as JSON string
        headers: Optional headers as JSON string
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install it with: pip install requests"

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
        return f"Error: Invalid request data JSON: {e}"

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
        return f"Error: {e}"
    except requests.exceptions.Timeout:
        _record_failure(url)
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        _record_failure(url)
        return f"Error: Connection failed - {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except Exception as e:
        return f"Error: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "http_get",
        "description": (
            "Make an HTTP GET request to a URL. "
            "Returns status, headers, and response body. "
            "Use this to read full page content when a search snippet is too short. "
            "If the page does not contain the information you need, return to the "
            "search results and try another URL."
        ),
        "input_schema": HttpGetInput,
        "requires_confirmation": False,
        "function": http_get,
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
    },
]

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
