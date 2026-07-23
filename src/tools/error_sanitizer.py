"""
Shared error sanitization utilities for Cogtrix tools.

All tool functions return strings to the LLM. Raw exception messages can leak
internal library details, filesystem paths, IP addresses, and exception class
names. This module provides centralized sanitization so every tool returns safe,
user-comprehensible messages without exposing internal implementation details.

Usage:
    from src.tools.error_sanitizer import sanitize_error

    try:
        result = do_something()
    except Exception as e:
        return f"Error: {sanitize_error(e)}"
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Final

log: Final[logging.Logger] = logging.getLogger(__name__)


# ── Generic fallback ──────────────────────────────────────────────────────────


def sanitize_error(exc: BaseException, fallback: str = "Operation failed") -> str:
    """
    Return a safe, user-comprehensible error message with no library internals.

    This function handles the full exception hierarchy used across Cogtrix tools.
    It deliberately avoids exposing exception class names, library internals,
    filesystem paths, IP addresses, or connection-pool details.

    Args:
        exc: The caught exception.
        fallback: Message returned for completely unexpected exceptions.
                  Must be a safe, short string with no dynamic content.

    Returns:
        A safe error message safe to return to the LLM.
    """
    # ── HTTP / network layer (requests) ──────────────────────────────────────
    # These are the most common in network tools. Import lazily to avoid
    # hard dependency on requests being installed in every environment.
    try:
        import requests.exceptions

        if isinstance(exc, requests.exceptions.Timeout):
            return "Request timed out"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "Connection failed"
        if isinstance(exc, requests.exceptions.HTTPError):
            status = getattr(exc.response, "status_code", None)
            if status:
                return f"HTTP error {status}"
            return "HTTP error"
        if isinstance(exc, requests.exceptions.RequestException):
            return "Request failed"
    except ImportError:
        pass  # requests not available — continue through other branches

    # ── HTTP / network layer (urllib3) ───────────────────────────────────────
    try:
        import urllib3.exceptions

        if isinstance(exc, urllib3.exceptions.HTTPError):
            return "HTTP request failed"
    except ImportError:
        pass

    # ── Network OSError subclasses ────────────────────────────────────────────
    # These are OSError subclasses and would fall through to the generic OSError
    # branch below, receiving "Filesystem operation failed" — semantically wrong.
    # Check them before the filesystem branch so the LLM gets network-appropriate
    # remediation hints.
    if isinstance(exc, ConnectionResetError):
        return "Connection reset"
    if isinstance(exc, ConnectionError):
        return "Connection failed"
    if isinstance(exc, TimeoutError):
        return "Request timed out"

    # ── Filesystem errors ─────────────────────────────────────────────────────
    if isinstance(exc, (FileNotFoundError,)):
        return "File not found"
    if isinstance(exc, PermissionError):
        return "Permission denied"
    if isinstance(exc, IsADirectoryError):
        return "Not a file"
    if isinstance(exc, UnicodeDecodeError):
        return "Could not decode file with the given encoding"
    if isinstance(exc, OSError):
        # OSError covers a wide range: disk full, I/O errors, broken pipe, etc.
        # Map the more common ones; everything else gets the generic message.
        errno = getattr(exc, "errno", None)
        if errno == 28:  # ENOSPC
            return "Disk full"
        if errno == 2:  # ENOENT — already handled above, but belt-and-suspenders
            return "File not found"
        if errno == 13:  # EACCES — already handled above
            return "Permission denied"
        if errno == 21:  # EISDIR
            return "Not a file"
        return "Filesystem operation failed"
    if isinstance(exc, IOError):
        return "Filesystem operation failed"

    # ── Shell / subprocess errors ─────────────────────────────────────────────
    if isinstance(exc, subprocess.CalledProcessError):
        # The return code is safe to expose; the command output is not.
        return f"Command failed (exit code {exc.returncode})"

    # ── JSON parsing errors ───────────────────────────────────────────────────
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "Invalid data format"

    # ── Runtime / configuration errors ───────────────────────────────────────
    if isinstance(exc, RuntimeError):
        # Check for API key configuration errors — safe to tell the LLM the
        # key is missing; we never expose the actual key value.
        msg = str(exc).lower()
        if "api key" in msg and ("not configured" in msg or "not found" in msg):
            return "API key not configured"
        return "Operation failed"

    # ── Generic / completely unexpected ─────────────────────────────────────
    # Log full details server-side but never expose to the LLM.
    log.debug("Unexpected tool error: %s: %s", type(exc).__name__, exc, exc_info=True)
    return fallback


# ── Tool-specific convenience functions ───────────────────────────────────────


def sanitize_http_error(exc: BaseException) -> str:
    """Sanitize HTTP/network exceptions. Returns a short HTTP-specific message."""
    result = sanitize_error(exc)
    # If the generic sanitization returned something generic, make it HTTP-specific
    if result in ("Operation failed", "Request failed"):
        return "Request failed"
    return result


def sanitize_shell_error(exc: BaseException) -> str:
    """Sanitize shell/subprocess exceptions. Returns a safe shell-specific message."""
    result = sanitize_error(exc)
    if result in ("Operation failed",):
        return "Command execution failed"
    return result


def sanitize_file_error(exc: BaseException) -> str:
    """Sanitize filesystem exceptions. Returns a safe file-op-specific message."""
    result = sanitize_error(exc)
    if result in ("Operation failed",):
        return "Filesystem operation failed"
    return result


def sanitize_search_error(exc: BaseException, context: str = "Search") -> str:
    """Sanitize search API exceptions. Returns a safe search-specific message.

    Improves diagnostic detail (closes #1586) by:

    * Mapping DDGS / duckduckgo_search library exceptions
      (``RatelimitException``, ``DuckDuckGoSearchException``,
      ``TimeoutException``) onto specific categorical messages.  Without
      this mapping every DDG failure fell through to the generic
      "Search request failed" placeholder, which gave the operator no
      handle to diagnose whether it was a rate-limit, a captcha block,
      or a connection error.
    * Appending the exception class name when the generic-fallback path
      fires.  Class names are abstract type identifiers — they do not
      leak secrets, file paths, IPs, or library internals — but they
      give the operator one concrete piece of debugging signal when
      the categorical mapping doesn't recognise the failure.

    Args:
        exc: The caught exception.
        context: A short string identifying the search provider
            (e.g. ``"DuckDuckGo"``, ``"Tavily"``).  Used as a prefix
            in the returned message so operators can identify *which*
            provider failed.

    Returns:
        A safe categorical error message identifying the provider, the
        failure class, and (where available) a short categorical hint
        such as "rate-limited" or "bot-detection".
    """
    # ── DDGS / duckduckgo_search library-specific exceptions ─────────────
    #
    # The DDGS library raises its own exception hierarchy that the
    # generic ``sanitize_error`` doesn't know about.  Without this
    # specialised branch, every DDGS failure falls through to the
    # generic fallback and emerges as "Search request failed" — the
    # opacity that prompted #1586.
    exc_class = type(exc).__name__
    if exc_class == "RatelimitException":
        return f"{context} rate-limited (HTTP 429)"
    if exc_class == "TimeoutException":
        # DDGS uses this name; differs from built-in ``TimeoutError``
        return f"{context} request timed out"
    if exc_class == "DuckDuckGoSearchException":
        # Multiple root causes are wrapped in this single class.
        # Inspect the message body cautiously — it can contain HTML
        # snippets when DDG returns a "blocked" page, so we look for
        # category keywords rather than echoing the full string.
        msg = str(exc).lower()
        if "rate" in msg or "limit" in msg or "429" in msg:
            return f"{context} rate-limited"
        if "blocked" in msg or "captcha" in msg or "anomaly" in msg:
            return f"{context} blocked by source (likely bot detection)"
        if "timeout" in msg:
            return f"{context} request timed out"
        return f"{context} client error ({exc_class})"

    result = sanitize_error(exc)
    if result in ("Operation failed",):
        # Generic fallback fired — the sanitizer didn't recognise the
        # exception.  Append the class name so the operator gets one
        # diagnostic handle without us leaking message content.
        return f"{context} request failed ({exc_class})"
    return result


# ── Google API errors ──────────────────────────────────────────────────────────


def sanitize_google_api_error(exc: BaseException, service: str = "Google API") -> str:
    """
    Sanitize googleapiclient.errors.HttpError — strips API keys from request URIs.

    HttpError.str() embeds the full request URI (including key=... query parameter)
    which can expose Google API keys to the LLM. This function extracts the JSON error
    body instead, returning a category-hinted message with no credential leakage.

    Args:
        exc: The caught exception (expected to be HttpError or compatible).
        service: Service name for the error prefix (default "Google API").

    Returns:
        A sanitized error message safe to return to the LLM, e.g.
        "Google API error: 403 (rate-limit)".
    """
    try:
        import googleapiclient.errors  # pyright: ignore[reportMissingImports]

        if not isinstance(exc, googleapiclient.errors.HttpError):
            # Not an HttpError — fall through to generic sanitization
            return sanitize_error(exc)
    except ImportError:
        # googleapiclient not installed — fall through to generic sanitization
        return sanitize_error(exc)

    # Extract status code from the HTTP response
    status_code: int | None = getattr(getattr(exc, "resp", None), "status", None)

    # ── Strip API keys from the URI attribute first ─────────────────────────
    # HttpError stores the request URI in the uri attribute. str(exc) includes it,
    # and older versions of googleapiclient embed it in the JSON body "message" field.
    # Strip keys from the URI directly so the attribute cannot leak credentials even
    # if the JSON body message does not reference the URI.
    import re as _re

    uri: str | None = getattr(exc, "uri", None)
    sanitized_uri: str | None = None
    if uri:
        # Strip API key query parameters from the URI
        stripped_uri = _re.sub(r"key=[A-Za-z0-9_\-]{10,}", "[KEY_REDACTED]", uri)
        sanitized_uri = stripped_uri

    # Parse JSON error body to extract the API error message
    # Only extract the message when we have a status code (for category hint).
    # Without a status code, we use the safe fallback to avoid leaking content.
    message = "request failed"
    if status_code:
        try:
            content_bytes: bytes | None = getattr(exc, "content", None)
            if content_bytes:
                body = json.loads(content_bytes.decode("utf-8", errors="replace"))
                # googleapiclient errors follow {"error": {"message": "...", "code": N}}
                error_obj = body.get("error", {})
                raw_message = error_obj.get("message", "")
                if raw_message:
                    # Strip any request URIs that leaked into the message body
                    # URIs with API keys look like: "...the request URI was: https://...key=AIza..."
                    stripped = _re.sub(
                        r"https?://[^\s\"']+key=[^\s\"'>]+", "[URI_REDACTED]", raw_message
                    )
                    stripped = _re.sub(r"key=[A-Za-z0-9_\-]{10,}", "[KEY_REDACTED]", stripped)
                    # If the only content is the URI reference (no real error message),
                    # fall back to the sanitized URI to avoid leaking URI structure
                    if stripped and not stripped.startswith("The request URI was:"):
                        message = stripped
                    elif sanitized_uri and "The request URI was:" in raw_message:
                        # URI was embedded in the message — use the sanitized version
                        message = sanitized_uri
        except Exception:
            # JSON parse failed or unexpected structure — keep safe fallback
            pass

    # Classify the status code into a category hint for the LLM
    if status_code:
        category = _google_http_category(status_code)
        return f"{service} error: {status_code} ({category})"
    return f"{service} error: {message}"


def _google_http_category(status: int) -> str:
    """Classify an HTTP status code into a Google API error category."""
    if status == 401:
        return "authentication"
    if status == 403:
        # Google Calendar uses 403 for quota exceeded and rate-limit scenarios
        return "rate-limit"
    if status == 404:
        return "not-found"
    if status == 429:
        return "rate-limit"
    if 400 <= status < 500:
        return "client-error"
    if 500 <= status < 600:
        return "server-error"
    return "unknown"
