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
    """Sanitize search API exceptions. Returns a safe search-specific message."""
    result = sanitize_error(exc)
    if result in ("Operation failed",):
        return f"{context} request failed"
    return result
