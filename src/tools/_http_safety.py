"""Shared SSRF + URL-safety helpers for HTTP-fetch tools.

Extracted from ``src/tools/http_request.py`` (ADR-0056 PR-A2) so the
sync ``http_get`` / ``http_post`` tools and the new async ``_http_fetch``
primitive share a single source of truth for URL validation, IP-block
rules, and header sanitisation.

This module is **import-time safe**: no side effects, no monkey-patching,
no network calls. The urllib3 DNS-pin hook used by the sync path stays
in ``http_request.py`` because it is sync-stack-specific (httpx uses a
different connection model and needs its own DNS-pinning approach).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger("cogtrix")

MAX_REDIRECTS = 5
_MAX_TIMEOUT = 120  # seconds — hard ceiling per-call
_MAX_RESPONSE_BYTES = 512_000  # 512 KB — sync-tool response cap

# RFC 6598 Shared Address Space (CGNAT) — not classified as private by
# ipaddress module but reachable from many corporate networks and not
# safe to expose to LLM-driven fetches.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

_BLOCKED_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "x-forwarded-host",
        "x-forwarded-for",
        "x-real-ip",
        "x-forwarded-proto",
        "x-forwarded-server",
    }
)


def _numeric_ipv4_to_dotted(host: str) -> str | None:
    """Return the canonical dotted-quad for an obfuscated numeric IPv4 host.

    ``ipaddress.ip_address`` rejects (raises ``ValueError`` for) the decimal-int,
    hex, octal, and short numeric IPv4 forms — ``2130706433``, ``0x7f000001``,
    ``0177.0.0.1``, ``127.1`` — but ``socket.inet_aton`` accepts and normalizes
    them. Without this, ``_is_blocked_ip`` returned ``False`` for such hosts and
    the IP-literal SSRF check was skipped, leaving only the platform resolver to
    catch them (which musl/Alpine and some resolvers do not normalize the same
    way) — a real bypass (#2136 F3). Returns ``None`` for anything ``inet_aton``
    does not accept (ordinary hostnames raise ``OSError``).
    """
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except (OSError, UnicodeError):
        return None


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if *ip_str* represents a non-public IP address.

    Catches all the cases that matter for SSRF:
      - Loopback (127/8, ::1)
      - Private (10/8, 172.16/12, 192.168/16, fc00::/7)
      - Link-local (169.254/16 — AWS/GCP IMDS sit at 169.254.169.254)
      - CGNAT (100.64/10)
      - Reserved / unspecified / multicast
      - IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1)
      - Obfuscated numeric IPv4 forms (decimal/hex/octal/short) — #2136 F3
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # #2136 F3: ipaddress does NOT accept obfuscated numeric IPv4 forms;
        # recover the dotted-quad via inet_aton so the checks below still apply.
        # inet_ntoa always yields a valid dotted-quad, so ip_address won't raise.
        dotted = _numeric_ipv4_to_dotted(ip_str)
        if dotted is None:
            return False
        ip = ipaddress.ip_address(dotted)
    # Unwrap IPv6-mapped IPv4 addresses so IPv4-space checks apply.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip in _CGNAT_NETWORK:
        return True
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def _validate_url(url: str) -> tuple[bool, str, str | None]:
    """Validate *url* for safety. Returns ``(is_valid, error, resolved_ip)``.

    Steps:
      1. Parse the URL; reject empty / scheme-less / non-HTTP schemes.
      2. Reject well-known internal hostnames by name (defence in depth).
      3. Reject hostnames that ARE IP literals if the IP is non-public.
      4. Resolve DNS; reject if any returned IP is non-public.

    On success, ``resolved_ip`` is the first DNS-resolved IP — callers
    using DNS-pinning (sync path) pass this to their connection layer.
    """
    try:
        parsed = urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format", None

        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}", None

        hostname = parsed.hostname or ""
        if not hostname:
            return False, "Invalid URL format", None

        # #2136 F4: normalize a trailing dot (the FQDN root label) for the
        # name-blocklist and IP-literal checks so "localhost." /
        # "169.254.169.254." cannot slip the pre-resolution defence-in-depth
        # layer. Keep the original `hostname` for DNS resolution / pinning so
        # the pin key still matches what the HTTP client connects with.
        check_host = hostname.rstrip(".")

        # Defence-in-depth: name-based block list. The IP-level checks
        # below would catch the same hosts via DNS resolution, but a
        # name-based block prevents an attacker from injecting a host
        # alias even before resolution.
        blocked_hosts = {
            "localhost",
            "metadata.google.internal",
            "instance-data",
            "169.254.169.254",
        }  # nosec B104
        if check_host.lower() in blocked_hosts:
            return False, "Requests to localhost or internal hosts are not allowed", None

        # If the hostname is itself an IP literal we don't need a DNS round-trip.
        # _is_blocked_ip handles canonical IPv4/IPv6 literals AND obfuscated
        # numeric IPv4 forms (2130706433, 0x7f000001, 0177.0.0.1, 127.1) via
        # inet_aton — ipaddress alone rejects those, so they used to slip the
        # literal check and fall through to the resolver (#2136 F3).
        try:
            if _is_blocked_ip(check_host):
                return (
                    False,
                    "Requests to localhost or private/reserved IP ranges are not allowed",
                    None,
                )
        except Exception as exc:
            log.warning("IP validation failed for %s: %s — blocking request", hostname, exc)
            return False, f"IP validation error: {exc}", None

        # Resolve via DNS; reject if any returned A/AAAA record is
        # non-public. We return the *first* address so the caller can
        # DNS-pin the connection (sync path).
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


def _parse_headers(
    headers: dict[str, str] | str | None,
) -> tuple[dict[str, str], str | None]:
    """Parse and sanitise an HTTP headers value.

    Accepts either a native dict (the idiomatic LLM tool-call shape) or
    a JSON-encoded string (legacy shape kept for backward compatibility
    with model-generated tool calls that still emit the stringified
    form). Strips CR/LF (header injection prevention) and drops any
    header name in ``_BLOCKED_HEADERS`` so callers can't forge ``Host``
    / ``X-Forwarded-*`` headers that downstream servers might trust.
    """
    if headers is None or headers == "":
        return {}, None

    raw: dict[str, str] | None = None
    if isinstance(headers, dict):
        raw = headers  # type: ignore[assignment]
    elif isinstance(headers, str):
        try:
            decoded = json.loads(headers)
        except json.JSONDecodeError as e:
            return {}, f"Invalid headers JSON: {e}"
        if not isinstance(decoded, dict):
            return {}, "Headers must be a JSON object"
        raw = decoded
    else:
        return {}, f"Headers must be a dict or JSON string, got {type(headers).__name__}"

    sanitized: dict[str, str] = {}
    for k, v in raw.items():
        safe_key = str(k).replace("\r", "").replace("\n", "")
        safe_value = str(v).replace("\r", "").replace("\n", "")
        if safe_key.lower() not in _BLOCKED_HEADERS:
            sanitized[safe_key] = safe_value
    return sanitized, None


__all__ = [
    "MAX_REDIRECTS",
    "_BLOCKED_HEADERS",
    "_CGNAT_NETWORK",
    "_MAX_RESPONSE_BYTES",
    "_MAX_TIMEOUT",
    "_is_blocked_ip",
    "_parse_headers",
    "_validate_url",
]
