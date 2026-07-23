"""Native-library coexistence guard for the agent process.

Bug D / cogtrix46 background: ``curl_cffi`` ships its own bundled
libcurl + BoringSSL. Python's stdlib ``ssl`` module links OpenSSL
statically into the interpreter. When both end up loaded into the
same process, their malloc / TLS state corrupt each other and the
process dies with a glibc heap abort (``"double free or corruption
(!prev)"``, ``"corrupted size vs. prev_size"``) on the first network
call — there is no clean Python traceback to debug from.

The agent's defence against this is to never import ``curl_cffi``
into the parent process: the DDG scraper runs in a subprocess
(``cogtrix_core/tools/web_search.py: _ddg_subprocess_call``) and the import
of ``curl_cffi`` is function-local inside the worker.

This module adds a *detection* layer on top of that defence. It
inspects ``sys.modules`` and reports back whether the two
incompatible TLS backends have both been pulled into the current
process. Callers use it to:

* Hard-fail a tool dispatch with a clear Python-level error before
  the glibc abort, rather than crashing silently.
* Surface a loud structured log line so operators see the cause
  instead of just ``"double free or corruption"`` in the terminal.
* Pin the safety invariant in regression tests
  (``tests/tools/test_native_safety.py``).
"""

from __future__ import annotations

import logging
import sys
import threading

_log = logging.getLogger("cogtrix")

# Modules whose presence in the parent process indicates the agent
# has pulled curl_cffi (libcurl + BoringSSL) alongside the stdlib
# OpenSSL bindings. We check the top-level module *and* the common
# submodule entry points so partial imports are still detected.
_CURL_CFFI_MODULES: frozenset[str] = frozenset(
    {
        "curl_cffi",
        "curl_cffi.requests",
        "curl_cffi.curl",
    }
)

# Modules that imply the process has OpenSSL loaded via Python's
# stdlib path. We do NOT check ``ssl`` directly because ssl is
# imported by far too many transitive paths (urllib, asyncio
# certificate handling) to be a useful signal. ``httpx`` and
# ``urllib3`` are the ones the agent actually pulls; their presence
# guarantees OpenSSL is in the malloc table.
_OPENSSL_MODULES: frozenset[str] = frozenset(
    {
        "httpx",
        "urllib3",
    }
)

# One-shot guard so a misconfigured environment doesn't spam the
# log on every tool dispatch.
_warning_emitted_lock = threading.Lock()
_warning_emitted = False


def detect_curl_cffi_openssl_coexistence() -> tuple[bool, list[str], list[str]]:
    """Return ``(coexistence_present, curl_cffi_hits, openssl_hits)``.

    A truthy first element means the parent process has both TLS
    backends loaded. The two lists are returned so callers can
    surface exactly which modules tripped the check.

    Pure inspection of ``sys.modules`` — never imports anything new,
    never raises.
    """
    curl_hits = sorted(m for m in _CURL_CFFI_MODULES if m in sys.modules)
    openssl_hits = sorted(m for m in _OPENSSL_MODULES if m in sys.modules)
    return (bool(curl_hits) and bool(openssl_hits), curl_hits, openssl_hits)


def warn_if_unsafe(context: str = "web_search") -> bool:
    """Emit a structured warning when curl_cffi + OpenSSL coexist.

    Returns True when a warning was emitted (or would have been —
    the one-shot guard suppresses duplicates). The intent is to give
    operators a Python-level message *before* the glibc abort lands
    on the next network call: when this fires, the next ``http_get``
    / LLM call / DDG fetch is racing the heap.

    *context* is a free-form string baked into the log line so the
    operator can tell which code path tripped the check.
    """
    global _warning_emitted
    coexists, curl_hits, openssl_hits = detect_curl_cffi_openssl_coexistence()
    if not coexists:
        return False

    with _warning_emitted_lock:
        already = _warning_emitted
        _warning_emitted = True

    if already:
        return True

    _log.warning(
        "NATIVE_TLS_COEXISTENCE detected (context=%s): "
        "curl_cffi modules=%s AND OpenSSL-binding modules=%s are BOTH "
        "loaded in this process. This is the documented Bug D / "
        "cogtrix46 condition — libcurl/BoringSSL (from curl_cffi) and "
        "OpenSSL (from httpx/urllib3) corrupt each other's malloc "
        "state and the process is likely to die with a glibc heap "
        "abort on the next TLS call. curl_cffi should be loaded "
        "ONLY in the DDG subprocess worker — see "
        "cogtrix_core/tools/web_search.py: _ddg_subprocess_call.",
        context,
        curl_hits,
        openssl_hits,
    )
    return True


def _reset_warning_emitted_for_tests() -> None:
    """Tests only — clear the one-shot flag between cases."""
    global _warning_emitted
    with _warning_emitted_lock:
        _warning_emitted = False


__all__ = [
    "detect_curl_cffi_openssl_coexistence",
    "warn_if_unsafe",
]
