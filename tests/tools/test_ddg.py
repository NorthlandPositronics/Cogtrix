"""Tests for ``src/tools/_ddg.py`` — the curl_cffi-based DDG HTML
scraper that replaces the retired ``ddgs``/``primp`` stack
(Bug D phase 2).

Two layers under test:

* ``parse_ddg_html`` — pure parser, no network. Exercised with a
  real DDG response captured at branch-time
  (``tests/tools/fixtures/ddg_chrome120_soudal.html``) plus
  hand-crafted edge cases. The captured fixture provides regression
  coverage against subtle changes in DDG's HTML shape — if DDG
  reshapes its result block, this file will be the first to fail.

* ``fetch_ddg_html`` — the curl_cffi wrapper. Tested with
  ``curl_cffi.requests.get`` patched so we don't make live network
  calls in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.tools._ddg import (
    DDGFetchError,
    _decode_redirect,
    _strip_html,
    fetch_ddg_html,
    parse_ddg_html,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_REAL_DDG_HTML = (_FIXTURE_DIR / "ddg_chrome120_soudal.html").read_text(encoding="utf-8")


# ── parse_ddg_html — real fixture ─────────────────────────────────────


class TestParseRealFixture:
    """Pin behaviour against a real DDG response. If DDG changes its
    HTML structure these will be the first tests to fail and serve as
    a tripwire for the team."""

    def test_extracts_ten_results(self) -> None:
        """The captured Chrome-120 fixture has 10 result anchors —
        DDG's default page size."""
        results = parse_ddg_html(_REAL_DDG_HTML)
        assert len(results) == 10

    def test_every_result_has_title_url_body(self) -> None:
        results = parse_ddg_html(_REAL_DDG_HTML)
        for i, r in enumerate(results):
            assert r["title"], f"result {i} missing title: {r}"
            assert r["href"], f"result {i} missing href: {r}"
            assert r["body"], f"result {i} missing body: {r}"

    def test_urls_are_decoded_not_ddg_redirects(self) -> None:
        """The output URLs must be the real target URLs, NOT DDG's
        ``//duckduckgo.com/l/?uddg=...`` redirect wrappers."""
        results = parse_ddg_html(_REAL_DDG_HTML)
        for r in results:
            assert (
                "duckduckgo.com/l/" not in r["href"]
            ), f"redirect URL leaked into output: {r['href']}"
            assert r["href"].startswith("http"), f"non-http URL: {r['href']}"

    def test_query_relevance(self) -> None:
        """The fixture query was 'Soudal Fix All sealant' — at least
        a couple results should mention Soudal."""
        results = parse_ddg_html(_REAL_DDG_HTML)
        soudal_mentions = sum(1 for r in results if "soudal" in (r["title"] + r["body"]).lower())
        assert soudal_mentions >= 5, (
            f"only {soudal_mentions}/10 results mention 'soudal' — fixture "
            "may have drifted from the captured query"
        )


# ── parse_ddg_html — edge cases ──────────────────────────────────────


class TestParseEdgeCases:
    def test_empty_input_returns_empty_list(self) -> None:
        assert parse_ddg_html("") == []

    def test_no_result_anchors_returns_empty_list(self) -> None:
        """When DDG fingerprint-blocks us it returns the search-form
        chrome with no result blocks. Parser must return [] gracefully
        rather than raising."""
        html = "<html><body><h1>DuckDuckGo</h1>no results</body></html>"
        assert parse_ddg_html(html) == []

    def test_html_entity_decoding_in_title(self) -> None:
        """Titles with HTML entities (``&amp;``, ``&#x27;``) must be
        decoded to plain text."""
        html = (
            '<a rel="nofollow" class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
            "Bob &amp; Alice&#x27;s Page</a>"
            '<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
            "snippet</a>"
        )
        results = parse_ddg_html(html)
        assert len(results) == 1
        assert results[0]["title"] == "Bob & Alice's Page"

    def test_html_tags_stripped_from_snippet(self) -> None:
        """Snippets often contain ``<b>`` tags around the matched
        query terms. Output should be plain text."""
        html = (
            '<a rel="nofollow" class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
            "Title</a>"
            '<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
            "Lorem <b>ipsum</b> dolor <strong>sit</strong> amet</a>"
        )
        results = parse_ddg_html(html)
        assert len(results) == 1
        assert results[0]["body"] == "Lorem ipsum dolor sit amet"


# ── _decode_redirect ─────────────────────────────────────────────────


class TestDecodeRedirect:
    def test_decodes_ddg_redirect_url(self) -> None:
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com%2Fpage" "&amp;rut=deadbeef"
        assert _decode_redirect(href) == "https://www.example.com/page"

    def test_handles_html_entity_in_ampersand(self) -> None:
        """DDG emits ``&amp;`` between query params in the raw HTML.
        Decoder must unescape entities before parsing the query string."""
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F" "&amp;rut=xyz"
        result = _decode_redirect(href)
        # The trailing ``&rut=xyz`` segment must not bleed into the URL.
        assert result == "https://example.com/"

    def test_missing_uddg_param_returns_empty(self) -> None:
        """Defensive: if a result block somehow lacks the ``uddg`` param,
        the parser should return empty rather than fabricate a URL."""
        href = "//duckduckgo.com/l/?rut=xyz"
        assert _decode_redirect(href) == ""

    def test_empty_input(self) -> None:
        assert _decode_redirect("") == ""


# ── _strip_html ───────────────────────────────────────────────────────


class TestStripHtml:
    def test_removes_tags(self) -> None:
        assert _strip_html("Hello <b>world</b>") == "Hello world"

    def test_decodes_entities(self) -> None:
        assert _strip_html("Tom &amp; Jerry") == "Tom & Jerry"

    def test_combined(self) -> None:
        assert _strip_html("<i>café</i> &amp; <b>thé</b>") == "café & thé"


# ── fetch_ddg_html — curl_cffi mocked ────────────────────────────────


def _mock_curl_response(status: int, text: str) -> Any:
    """Build a minimal stand-in for curl_cffi's Response."""
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


class TestFetchDdgHtml:
    def test_returns_response_text_on_200(self) -> None:
        # 25 KB of arbitrary content + a result-anchor marker, to
        # pass both the size and structure checks.
        body = "x" * 25000 + 'class="result__a"' + "y" * 5000
        with patch(
            "curl_cffi.requests.get",
            return_value=_mock_curl_response(200, body),
        ):
            result = fetch_ddg_html("query", region="wt-wt", num_results=5)
        assert result == body

    def test_non_200_raises_ddg_fetch_error(self) -> None:
        with (
            patch(
                "curl_cffi.requests.get",
                return_value=_mock_curl_response(429, "Rate limited"),
            ),
            pytest.raises(DDGFetchError, match="HTTP 429"),
        ):
            fetch_ddg_html("query")

    def test_stub_page_raises_fingerprint_blocked(self) -> None:
        """When DDG fingerprint-blocks us it returns a ~14 KB page
        with no result anchors. Caller must surface this as a
        DDGFetchError so the aggregator knows DDG didn't respond."""
        stub_body = "x" * 14000  # below 20 KB threshold, no result marker
        with (
            patch(
                "curl_cffi.requests.get",
                return_value=_mock_curl_response(200, stub_body),
            ),
            pytest.raises(DDGFetchError, match="fingerprint-blocked"),
        ):
            fetch_ddg_html("query")

    def test_curl_cffi_exception_wrapped_as_fetch_error(self) -> None:
        """curl_cffi raises various transport errors. They all need
        to be wrapped as DDGFetchError so the subprocess worker's
        single ``except DDGFetchError`` catches everything."""
        with (
            patch(
                "curl_cffi.requests.get",
                side_effect=ConnectionError("network unreachable"),
            ),
            pytest.raises(DDGFetchError, match="ConnectionError"),
        ):
            fetch_ddg_html("query")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(DDGFetchError, match="empty query"):
            fetch_ddg_html("")
        with pytest.raises(DDGFetchError, match="empty query"):
            fetch_ddg_html("   ")
