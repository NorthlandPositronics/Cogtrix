"""Tests for src/orchestration/search_quality (#1593, Option B; #1603).

Covers:
- SearchQualityThresholds dataclass defaults.
- Configurable thresholds (min_url_count, min_content_chars).
- Observability: false-negative detection logging when >=1 URL
  is classified non-substantive.
- Error-prefix check fix: "Error searching" detected via `in` not
  startswith (actual format is "Tool failed: search_web - Error...").
- Format-agnostic URL detection: regex-based counting works regardless
  of whether providers prefix URLs with "URL: " (#1603).
- Edge-case coverage: 0 URLs, 1 URL with long content, 2+ URLs with
  short content, alternative provider formats.
"""

from __future__ import annotations

import logging

import pytest

from src.orchestration.search_quality import (
    SearchQualityThresholds,
    has_substantive_search_results,
)


def _ddg_payload(n_results: int, snippet_chars: int = 80) -> str:
    """Build a synthetic search_web ToolMessage payload mirroring live
    DDG / Tavily / Brave / Exa output format (URL: + Domain: + snippet
    lines per result)."""
    snippet = "x" * snippet_chars
    lines = [f"Search results for: synthetic query {n_results} results", ""]
    for i in range(1, n_results + 1):
        lines.extend(
            [
                f"{i}. Synthetic Product {i}",
                f"   URL: https://example-{i}.test/landing",
                f"   Domain: example-{i}.test",
                f"   {snippet}",
                "",
            ]
        )
    return "\n".join(lines)


class _FakeToolMessage:
    def __init__(self, content: str, tool_call_id: str = "t1", name: str = "search_web") -> None:
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = name


class _FakeAIMessage:
    def __init__(self, tool_calls: list[dict]) -> None:
        self.tool_calls = tool_calls


class TestSearchQualityThresholdsDefaults:
    def test_min_url_count_default(self):
        t = SearchQualityThresholds()
        assert t.min_url_count == 2

    def test_min_content_chars_default(self):
        t = SearchQualityThresholds()
        assert t.min_content_chars == 300

    def test_frozen(self):
        t = SearchQualityThresholds()
        with pytest.raises(AttributeError):  # frozen dataclass — assignment forbidden
            t.min_url_count = 3  # type: ignore[attr-defined]


class TestHasSubstantiveSearchResultsDefaults:
    def test_empty_messages(self):
        assert has_substantive_search_results([]) is False

    def test_single_url_not_substantive(self):
        """A single URL: line could be a sponsored slot; require >= 2."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(1)),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_two_url_results_substantive(self):
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(2)),
        ]
        assert has_substantive_search_results(msgs) is True

    def test_five_url_results_substantive(self):
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(5)),
        ]
        assert has_substantive_search_results(msgs) is True

    def test_error_wrapper_not_substantive(self):
        """Error messages must not count even if they happen to contain 'URL:'."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Error searching: request failed"),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_no_results_found_not_substantive(self):
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("No results found for query"),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_not_loaded_not_substantive(self):
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Page could not be loaded"),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_only_post_last_human_message_scoped(self):
        """Results before the last HumanMessage must not be considered."""
        from langchain_core.messages import HumanMessage

        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "old"}, "id": "t0"}]),
            _FakeToolMessage(_ddg_payload(3), tool_call_id="t0"),
            HumanMessage(content="new query"),
            _FakeAIMessage([{"name": "search_web", "args": {"query": "new"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(1), tool_call_id="t1"),
        ]
        assert has_substantive_search_results(msgs) is False


class TestConfigurableThresholds:
    def test_min_url_count_threshold(self):
        """min_url_count=3 requires 3+ URL: lines."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(2)),
        ]
        thresholds = SearchQualityThresholds(min_url_count=3)
        assert has_substantive_search_results(msgs, thresholds) is False

        msgs2 = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(3)),
        ]
        assert has_substantive_search_results(msgs2, thresholds) is True

    def test_min_content_chars_threshold(self):
        """min_content_chars=1000 requires >= 1000 chars."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(2, snippet_chars=50)),
        ]
        thresholds = SearchQualityThresholds(min_content_chars=1000)
        assert has_substantive_search_results(msgs, thresholds) is False

        msgs2 = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(2, snippet_chars=500)),
        ]
        assert has_substantive_search_results(msgs2, thresholds) is True

    def test_combined_thresholds(self):
        """Both thresholds must be met simultaneously."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(3, snippet_chars=50)),
        ]
        thresholds = SearchQualityThresholds(min_url_count=3, min_content_chars=500)
        assert has_substantive_search_results(msgs, thresholds) is False

        msgs2 = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(3, snippet_chars=200)),
        ]
        assert has_substantive_search_results(msgs2, thresholds) is True


class TestErrorPrefixCheckFix:
    """#1593: the previous startswith check was dead code because the actual
    error format from search_web is "Tool failed: search_web - Error searching..."."""

    def test_tool_failed_error_format_detected(self):
        """Actual error format from search_web - must be caught via `in`."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Tool failed: search_web - Error searching: rate limited"),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_error_in_middle_of_content_detected(self):
        """Error text appearing mid-content must also be caught."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Some results: URL: https://example.test - Error searching: blocked"),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_startswith_error_not_substantive(self):
        """Traditional startswith error format still caught."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Error searching: DuckDuckGo rate-limited (HTTP 429)"),
        ]
        assert has_substantive_search_results(msgs) is False


class TestFalseNegativeObservability:
    """When a ToolMessage contains >=1 URL: line but is classified non-substantive
    (too few URLs or too short), a warning must be logged (#1593)."""

    def test_warning_logged_when_too_few_urls(self, caplog: pytest.LogCaptureFixture):
        """URL count below threshold triggers false-negative warning."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(1)),
        ]
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = has_substantive_search_results(msgs)
        assert result is False
        assert any(
            "URL(s)" in record.message and "classified non-substantive" in record.message
            for record in caplog.records
        )

    def test_warning_logged_when_too_short(self, caplog: pytest.LogCaptureFixture):
        """Content length below threshold triggers false-negative warning."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(2, snippet_chars=10)),
        ]
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = has_substantive_search_results(msgs)
        assert result is False
        assert any("classified non-substantive" in record.message for record in caplog.records)

    def test_no_warning_when_substantive(self, caplog: pytest.LogCaptureFixture):
        """When result is substantive, no false-negative warning."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(3)),
        ]
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = has_substantive_search_results(msgs)
        assert result is True
        assert not any("classified non-substantive" in record.message for record in caplog.records)

    def test_no_warning_for_error_messages(self, caplog: pytest.LogCaptureFixture):
        """Error messages (no URL content) must not trigger false-negative warning."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage("Error searching: rate limited"),
        ]
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = has_substantive_search_results(msgs)
        assert result is False
        assert not any("classified non-substantive" in record.message for record in caplog.records)

    def test_warning_includes_url_count_and_threshold(self, caplog: pytest.LogCaptureFixture):
        """Warning message must include actual URL count and threshold."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(_ddg_payload(1)),
        ]
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            has_substantive_search_results(msgs)
        warning_msgs = [
            r.message for r in caplog.records if "classified non-substantive" in r.message
        ]
        assert len(warning_msgs) == 1
        assert "1 URL(s)" in warning_msgs[0] or "url_count" in warning_msgs[0].lower()


class TestFormatAgnosticUrlDetection:
    """#1603: URL detection must work regardless of provider output format.

    Current providers (DDG, Tavily, Brave, Exa, Google, SearXNG, SerpAPI)
    all emit ``"   URL: https://..."`` lines, but a future provider might
    use markdown links, JSON, or plain text.  The heuristic must not
    silently degrade when the format changes.
    """

    def _raw_url_payload(self, n_results: int, prefix: str = "", snippet: str = "") -> str:
        """Build a payload with raw URLs (no ``"URL: "`` prefix)."""
        lines = [f"Search results for: synthetic query {n_results} results", ""]
        for i in range(1, n_results + 1):
            lines.extend(
                [
                    f"{i}. Synthetic Product {i}",
                    f"{prefix}https://example-{i}.test/landing",
                    snippet
                    or f"A short snippet describing synthetic product {i} for the test fixture.",
                    "",
                ]
            )
        return "\n".join(lines)

    def test_raw_urls_without_url_prefix_are_counted(self):
        """Provider that emits raw URLs (no 'URL: ' prefix) still works."""
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(self._raw_url_payload(2, snippet="x" * 120)),
        ]
        assert has_substantive_search_results(msgs) is True

    def test_markdown_link_format_works(self):
        """Markdown-style links like [text](url) are detected."""
        content = (
            "Search results:\n\n"
            "1. [Product A](https://example-a.test)\n"
            "   Description of product A with extra padding to exceed the "
            "300-character threshold comfortably so the test focuses on URL "
            "detection rather than length.\n\n"
            "2. [Product B](https://example-b.test)\n"
            "   Description of product B with similar padding to ensure the "
            "overall message length is well above the minimum content requirement.\n"
        )
        assert len(content) >= 300
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is True

    def test_json_embedded_urls_work(self):
        """URLs inside JSON strings are detected."""
        content = (
            '{"results":['
            '{"title":"Product A","url":"https://example-a.test"},'
            '{"title":"Product B","url":"https://example-b.test"}'
            '],"total":2}'
        )
        # Pad to exceed 300 chars
        content = content + " " * 350
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is True

    def test_zero_urls_not_substantive(self):
        """A result with no URLs at all is never substantive."""
        content = (
            "Search results for: x\n\n"
            "1. Product A\n"
            "   No URL available for this result. "
            "Padding text to exceed 300 characters so the test verifies "
            "that URL count (zero) is the deciding factor, not content length.\n\n"
            "2. Product B\n"
            "   No URL available for this result either. "
            "More padding text to ensure the overall message is long enough.\n"
        )
        assert len(content) >= 300
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_one_url_with_long_content_not_substantive(self):
        """Exactly 1 URL + very long content is still not substantive.

        The URL count threshold (default 2) is independent of content
        length — a single rich result is treated as a sponsored slot.
        """
        content = (
            "Search results for: x\n\n"
            "1. Product A\n"
            "   URL: https://example-a.test\n"
            "   " + "x" * 800 + "\n"
        )
        assert content.count("https://") == 1
        assert len(content) >= 300
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_two_urls_with_short_content_not_substantive(self):
        """2+ URLs but very short content is not substantive.

        Both thresholds must be met simultaneously.
        """
        content = "1. A\nhttps://a.test\n\n2. B\nhttps://b.test\n"
        assert len(content) < 300
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_error_with_url_in_text_not_substantive(self):
        """Error messages containing a documentation URL must not count.

        The error guard (``"Error searching" in content``) runs before
        URL counting, so the URL inside the error text is ignored.
        """
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(
                "Tool failed: search_web - Error searching: rate limited. "
                "See https://docs.example.com/rate-limits for details."
            ),
        ]
        assert has_substantive_search_results(msgs) is False

    def test_trailing_punctuation_not_included_in_url(self):
        """Trailing punctuation (period, comma, paren) must not be counted
        as part of the URL."""
        content = (
            "Results:\n\n"
            "1. See https://example-a.test.\n"
            "   Description A with extensive padding text to ensure the overall "
            "message exceeds the 300-character threshold. This verifies that "
            "trailing punctuation is stripped from URL detection while still "
            "meeting the length requirement for substantive classification.\n\n"
            "2. Visit https://example-b.test, then read more.\n"
            "   Description B with additional padding to make sure the content "
            "is long enough for the heuristic to evaluate properly.\n"
        )
        assert len(content) >= 300
        msgs = [
            _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
            _FakeToolMessage(content),
        ]
        assert has_substantive_search_results(msgs) is True


class TestEdgeCaseRegressionCoverage:
    """Regression tests for the three brittleness concerns from #1603.

    - Concern 1 (string-format coupling): tested in
      TestFormatAgnosticUrlDetection above.
    - Concern 2 (dead prefix check): tested in TestErrorPrefixCheckFix above.
    - Concern 3 (hardcoded thresholds): tested in TestConfigurableThresholds above.

    This class adds cross-cutting edge cases that exercise all three
    fixes simultaneously.
    """

    def test_all_providers_format_substantive(self):
        """Verify substantive detection for every known provider format."""
        _pad = "\nPadding text to exceed 300 characters. " * 15
        formats = [
            # Standard "URL: " prefix (all current providers)
            (
                "DDG/Tavily/Brave/Exa/Google/SearXNG/SerpAPI format",
                "1. Result\n   URL: https://a.test\n   Snip\n\n"
                "2. Result\n   URL: https://b.test\n   Snip\n" + _pad,
            ),
            # Markdown links
            (
                "Markdown link format",
                "- [A](https://a.test)\n  Description A\n\n"
                "- [B](https://b.test)\n  Description B\n" + _pad,
            ),
            # Plain raw URLs
            (
                "Plain URL format",
                "Result 1: https://a.test\nDescription\n\n"
                "Result 2: https://b.test\nDescription\n" + _pad,
            ),
            # JSON with URLs
            (
                "JSON format",
                '{"r":[{"u":"https://a.test"},{"u":"https://b.test"}]}' + " " * 350,
            ),
        ]
        for name, content in formats:
            msgs = [
                _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
                _FakeToolMessage(content),
            ]
            assert (
                has_substantive_search_results(msgs) is True
            ), f"Provider format '{name}' should be substantive"

    def test_error_messages_all_formats_not_substantive(self):
        """Error messages must be rejected regardless of format."""
        error_contents = [
            "Error searching: DuckDuckGo rate-limited (HTTP 429)",
            "Tool failed: search_web - Error searching: blocked",
            "No results found for: query",
            "Tool 'search_web' is in the catalog but not loaded.",
        ]
        for content in error_contents:
            msgs = [
                _FakeAIMessage([{"name": "search_web", "args": {"query": "x"}, "id": "t1"}]),
                _FakeToolMessage(content),
            ]
            assert (
                has_substantive_search_results(msgs) is False
            ), f"Error content should not be substantive: {content[:60]}"

    def test_web_search_renamed_tool_is_recognised(self):
        """ADR-0056 PR-G regression. The tool was renamed
        ``search_web`` → ``web_search`` but ``search_quality.py``'s
        name filter was left as ``!= "search_web"``, which made
        every modern ``web_search`` ToolMessage classified as
        non-substantive. The downstream effect: ``call_model``'s
        thinking-break dispatch never hit the
        ``_effort_met and _has_results`` branch, every search loop
        fell into the ``honest refusal`` branch, and the agent
        emitted "I could not retrieve current data on this topic."
        even when ``web_search`` had returned multiple real
        sources (cogtrix62 turn 3 ScienceSoft reproducer,
        2026-05-22).

        This test pins that BOTH names are recognised. If a
        future renames the tool again without updating this
        list, the resulting silent regression trips here first.
        """
        substantive_payload = _ddg_payload(n_results=3, snippet_chars=120)

        # Both names must be recognised.
        for tool_name in ("search_web", "web_search"):
            msgs = [
                _FakeAIMessage([{"name": tool_name, "args": {"query": "ScienceSoft"}, "id": "t1"}]),
                _FakeToolMessage(substantive_payload, name=tool_name),
            ]
            assert has_substantive_search_results(msgs) is True, (
                f"Tool name {tool_name!r} must be recognised as a "
                "web-search ToolMessage; otherwise the thinking-break "
                "dispatch routes substantive results into the "
                "honest-refusal branch (cogtrix62 ScienceSoft regression)."
            )

        # An unrelated tool name must still be rejected — the fix
        # doesn't widen the filter to "any tool".
        msgs = [
            _FakeAIMessage([{"name": "calculator", "args": {}, "id": "t1"}]),
            _FakeToolMessage(substantive_payload, name="calculator"),
        ]
        assert has_substantive_search_results(msgs) is False, (
            "Non-search-tool ToolMessages must NOT be classified " "as substantive search results."
        )
