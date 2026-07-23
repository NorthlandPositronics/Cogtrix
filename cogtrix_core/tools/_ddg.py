"""DDG HTML scraping primitive — curl_cffi-based replacement for
``ddgs``/``primp``.

This module replaces the bug-prone ``ddgs`` library (whose embedded
``primp`` Rust HTTP client caused the heap-corruption crashes
documented in the cogtrix34/35/36 incident reports) with a small,
self-contained scraper that uses ``curl_cffi`` for browser-fingerprint
impersonation against ``html.duckduckgo.com``.

Why curl_cffi instead of plain ``httpx``: DDG actively fingerprints
TLS / HTTP-stack metadata and serves empty result pages to anything
that isn't a browser-shaped client. The probe documented in the PR
description showed plain httpx getting a 14 KB stub while curl_cffi
with ``impersonate="chrome120"`` returns the full 34 KB result page
(10 result anchors, real content).

Why this module is small and pure: the public entry points
(``fetch_ddg_html`` and ``parse_ddg_html``) are testable in isolation
without network or subprocess machinery. The orchestration that
spawns the subprocess sandbox around them lives in
``cogtrix_core/tools/web_search.py``.

The subprocess sandbox is kept as a safety net for the first
release of this change — curl_cffi has a better stability reputation
than primp, but it is still native code (patched libcurl + BoringSSL)
and we want one release of clean production data before retiring the
sandbox.
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse

# Browser-fingerprint impersonation profile passed to curl_cffi.
# Chrome 120 was chosen because it consistently returned full result
# pages in the probe (10 anchors, ~34 KB) across multiple test queries.
# If DDG starts blocking it, the fallback profiles probed-and-passed
# were chrome116, firefox133, safari17_2_ios — any of those works.
_IMPERSONATE = "chrome120"
_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

# Per-request budget. Conservative enough to fit inside the
# web_search stage-1 aggregator deadline (5 s) with margin for the
# parent-side subprocess overhead and JSON marshalling.
_DEFAULT_TIMEOUT_S = 10

# Result-anchor regex: matches the title <a> tag inside every result
# block. Groups are (redirect_href, title_html). The href is DDG's
# click-tracking redirect; ``_decode_redirect`` unwraps it.
_RESULT_TITLE_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
# Snippet regex: matches the visible snippet text. One match per result.
_SNIPPET_RE = re.compile(
    r'<a class="result__snippet"[^>]+href="[^"]+"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")

# Anti-scrape stub detection. When DDG fingerprint-blocks a client,
# it serves a ~14 KB page with the search-form chrome and no result
# blocks. We treat that as a fatal fetch error so the parent surfaces
# it as a provider failure (per stage-1 aggregator conventions) and
# the rest of web_search falls back to the keyed providers.
_STUB_PAGE_SIZE_THRESHOLD = 20_000
_RESULT_MARKER = 'class="result__a"'


class DDGFetchError(RuntimeError):
    """Raised when the DDG fetch couldn't produce a usable HTML page.

    Distinct subclass so callers can catch fetch failures specifically
    without swallowing other ``RuntimeError``s emitted by curl_cffi.
    """


def fetch_ddg_html(query: str, region: str = "wt-wt", num_results: int = 5) -> str:
    """Fetch the DDG ``/html/`` endpoint for *query* and return the
    response body as a string.

    Parameters
    ----------
    query:
        The search query. Must be non-empty.
    region:
        DDG region hint (e.g. ``"wt-wt"`` for worldwide, ``"us-en"``).
    num_results:
        Hint for how many results we want. DDG's HTML page returns up
        to 10 results regardless; the cap is enforced by the caller
        when slicing the parsed list.

    Raises
    ------
    DDGFetchError
        On non-200 responses, anti-scrape stub pages, or curl_cffi
        errors. Caller (the subprocess worker) re-emits these as JSON
        ``{"error": ...}`` so the parent can classify the failure.
    """
    # Local import — curl_cffi is heavy (libcurl + BoringSSL) and we
    # don't want to pay its import cost in code paths that don't reach
    # DDG (e.g. agent runs against Tavily-only configurations). The
    # subprocess imports it on every spawn, but that's the
    # subprocess's job.
    from curl_cffi import requests as cffi_requests  # noqa: PLC0415

    if not query or not query.strip():
        raise DDGFetchError("empty query")

    # ``num_results`` is forwarded as a query param for transparency
    # even though DDG ignores most counts on /html/. The slicing
    # happens client-side.
    _ = num_results

    try:
        response = cffi_requests.get(
            _HTML_ENDPOINT,
            params={"q": query, "kl": region},
            impersonate=_IMPERSONATE,
            timeout=_DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        raise DDGFetchError(f"curl_cffi raised: {type(exc).__name__}: {exc}") from exc

    if response.status_code != 200:
        raise DDGFetchError(f"HTTP {response.status_code}")

    body = response.text
    if len(body) < _STUB_PAGE_SIZE_THRESHOLD and _RESULT_MARKER not in body:
        raise DDGFetchError(
            f"DDG returned stub page ({len(body)} bytes, no result anchors) — "
            "likely fingerprint-blocked or rate-limited"
        )

    return body


def parse_ddg_html(html_text: str) -> list[dict[str, str]]:
    """Extract result records from a DDG HTML response.

    Returns a list of ``{"href": <real URL>, "title": <text>,
    "body": <snippet text>}`` dicts. Empty list when no results are
    present (caller treats that as a provider returning zero hits,
    not a failure).

    The output dict shape matches the field names ``ddgs`` produced,
    so downstream ``ProviderResult`` construction in
    ``cogtrix_core/tools/web_search.py::_search_async`` keeps working without
    field-name updates.
    """
    title_matches = _RESULT_TITLE_RE.findall(html_text)
    snippet_matches = _SNIPPET_RE.findall(html_text)

    results: list[dict[str, str]] = []
    for (href, title_html), snippet_html in zip(title_matches, snippet_matches, strict=False):
        real_url = _decode_redirect(href)
        if not real_url:
            continue
        results.append(
            {
                "href": real_url,
                "title": _strip_html(title_html).strip(),
                "body": _strip_html(snippet_html).strip(),
            }
        )
    return results


def _decode_redirect(href: str) -> str:
    """Decode DDG's click-tracking redirect to a clean target URL.

    DDG wraps result URLs in
    ``//duckduckgo.com/l/?uddg=<urlencoded-target>&rut=<hex>``. The
    ``uddg`` query param holds the real destination. We URL-decode it
    and return; the ``rut`` param is dropped (it's an analytics token,
    not part of the target).
    """
    if not href:
        return ""
    decoded_entities = _html.unescape(href)
    if decoded_entities.startswith("//"):
        decoded_entities = "https:" + decoded_entities
    try:
        parsed = urllib.parse.urlparse(decoded_entities)
    except ValueError:
        return ""
    qs = urllib.parse.parse_qs(parsed.query)
    # ``parse_qs`` already URL-decodes the value once.
    return qs.get("uddg", [""])[0]


def _strip_html(html_text: str) -> str:
    """Remove HTML tags and decode entities. Plain-text snippet output."""
    return _html.unescape(_TAG_RE.sub("", html_text))


__all__ = ["DDGFetchError", "fetch_ddg_html", "parse_ddg_html"]
