"""Tests for src/tools/_web_search_synthesiser.py — the
``synthesise()`` entry point + 4 post-call validators
(URL line-drop, citation-presence, schema check, length cap)
+ the two-tier primary/fallback retry policy.

The 11-test regression suite committed in
``docs/optional/prompts/web-search-synthesis.md`` "Unit tests" section.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.tools._http_fetch import FetchResult
from src.tools._web_search_aggregator import RankedResult
from src.tools._web_search_domain_class import DomainClass
from src.tools._web_search_extractor import ExtractedSource
from src.tools._web_search_fetcher import FetchOutcome
from src.tools._web_search_synthesiser import (
    SynthesisResult,
    _check_schema,
    _drop_url_lines,
    _enforce_citation_presence,
    _truncate_overlong,
    synthesise,
)

# ── Test fixtures ────────────────────────────────────────────────────


def _make_extract(
    url: str,
    title: str,
    text: str,
    *,
    domain_class: DomainClass = DomainClass.UNKNOWN,
    published_date: str | None = None,
) -> ExtractedSource:
    """Build an ExtractedSource for synthesiser tests."""
    rank = RankedResult(
        canonical_url=url,
        title=title,
        snippet=text[:200],
        published_date=published_date,
        domain_class=domain_class,
        score=1.0,
        providers=("ddg",),
    )
    fetch_result = FetchResult(
        url=url,
        status_code=200,
        content=text.encode(),
        encoding="utf-8",
        content_type="text/html",
        elapsed_ms=10,
        truncated=False,
        error=None,
    )
    outcome = FetchOutcome(
        ranked=rank,
        status="fetched",
        fetch_result=fetch_result,
        error=None,
    )
    return ExtractedSource(
        fetch_outcome=outcome,
        extracted_text=text,
        status="extracted",
    )


def _fake_llm(content: str) -> Any:
    """Build a mock LLM whose invoke returns the given content."""
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


# ── Unit validators ──────────────────────────────────────────────────


class TestUrlLineDrop:
    def test_drops_line_containing_https_url(self) -> None:
        text = "good line\nbad line at https://evil.example.com/x\ngood line"
        result = _drop_url_lines(text)
        assert "evil.example.com" not in result
        assert "good line" in result

    def test_drops_line_with_www_marker(self) -> None:
        text = "Fact A [①]\nSee www.example.com for more [②]\nFact B [③]"
        result = _drop_url_lines(text)
        assert "www.example.com" not in result
        assert "Fact A" in result
        assert "Fact B" in result

    def test_no_urls_passes_through(self) -> None:
        text = "## Key findings\n### A\nFact [①]"
        assert _drop_url_lines(text) == text


class TestCitationPresence:
    def test_well_cited_passes_through(self) -> None:
        text = "## Key findings\n" "### Topic\n" "First fact [①]\n" "Second fact [②]"
        out, status = _enforce_citation_presence(text)
        assert status == "ok"
        assert "First fact" in out
        assert "Second fact" in out

    def test_uncited_line_dropped_when_minority(self) -> None:
        """3 lines, 1 uncited → 33% drop → still ok status."""
        text = (
            "## Key findings\n"
            "### Topic\n"
            "First fact [①]\n"
            "Second fact [②]\n"
            "Uncited statement\n"
        )
        out, status = _enforce_citation_presence(text)
        assert status == "ok"
        assert "Uncited statement" not in out
        assert "First fact" in out

    def test_majority_uncited_returns_marker(self) -> None:
        """5 statements, 5 uncited → 100% drop → majority-uncited.

        The threshold was tightened from >50% to >80% after the
        cogtrix37 testing showed qwen3-coder and similar small models
        cite ~30–50% of lines but produce otherwise grounded prose.
        We still reject when >80% of lines lack citations because a
        synthesis with effectively zero citations is almost certainly
        hallucinated content disconnected from the input extracts.
        """
        text = (
            "## Key findings\n"
            "### Topic\n"
            "Uncited A\n"
            "Uncited B\n"
            "Uncited C\n"
            "Uncited D\n"
            "Uncited E\n"
        )
        _, status = _enforce_citation_presence(text)
        assert status == "majority-uncited"

    def test_partial_citation_passes_under_new_threshold(self) -> None:
        """4 statements, 3 uncited → 75% drop → ``ok`` under the
        small-model-friendly threshold (>80% rejects, ≤80% accepts).

        Pre-change (>50% threshold) this returned ``majority-uncited``
        and every qwen3-coder synthesis got dropped despite producing
        grounded prose; we'd then fall through to the Sources-only
        path even when the synthesis layer did its job.
        """
        text = (
            "## Key findings\n"
            "### Topic\n"
            "First fact [①]\n"
            "Uncited A\n"
            "Uncited B\n"
            "Uncited C\n"
        )
        out, status = _enforce_citation_presence(text)
        assert status == "ok", (
            "75% drop should be tolerated under the small-model-friendly "
            "threshold; only >80% drop counts as majority-uncited."
        )
        # Uncited lines still get dropped from the output even though
        # the synthesis as a whole is accepted.
        assert "First fact" in out
        assert "Uncited A" not in out
        assert "Uncited B" not in out
        assert "Uncited C" not in out

    def test_threshold_boundary_exactly_80pct_accepts(self) -> None:
        """5 statements, 4 uncited → 80% drop → ``ok`` (boundary is
        ``> 0.8``, so exactly 80% accepts)."""
        text = (
            "## Key findings\n"
            "### Topic\n"
            "Fact [①]\n"
            "Uncited A\n"
            "Uncited B\n"
            "Uncited C\n"
            "Uncited D\n"
        )
        _, status = _enforce_citation_presence(text)
        assert status == "ok"

    def test_lines_outside_key_findings_not_checked(self) -> None:
        """Lines under Disagreements / Gaps don't need terminal citation."""
        text = "## Key findings\n" "### Topic\n" "Fact [①]\n" "## Gaps\n" "- No coverage on X\n"
        out, status = _enforce_citation_presence(text)
        assert status == "ok"
        assert "No coverage" in out


class TestSchemaCheck:
    def test_normal_case_passes(self) -> None:
        text = (
            "## Key findings\n"
            "### Topic\n"
            "Fact [①]\n"
            "## Disagreements\n"
            "- A vs B\n"
            "## Gaps\n"
            "- Nothing on X\n"
        )
        assert _check_schema(text) is True

    def test_only_gaps_passes_rule_8(self) -> None:
        text = (
            "## Gaps\n"
            "- The retrieved sources do not contain information that answers the query.\n"
        )
        assert _check_schema(text) is True

    def test_disagreements_without_key_findings_fails(self) -> None:
        text = "## Disagreements\n- A vs B\n"
        assert _check_schema(text) is False

    def test_wrong_order_fails(self) -> None:
        """Disagreements before Key findings is invalid."""
        text = "## Disagreements\n" "- A vs B\n" "## Key findings\n" "### Topic\n" "Fact [①]\n"
        assert _check_schema(text) is False

    def test_gaps_before_disagreements_fails(self) -> None:
        text = "## Key findings\n### A\nFact [①]\n" "## Gaps\n- X\n" "## Disagreements\n- A vs B\n"
        assert _check_schema(text) is False

    def test_free_prose_no_headers_fails(self) -> None:
        text = "Just some free-form synthesis without any structure."
        assert _check_schema(text) is False


class TestLengthCap:
    def test_short_passes_through(self) -> None:
        text = "## Key findings\n### A\nShort fact [①]"
        assert _truncate_overlong(text) == text

    def test_over_hard_cap_truncated(self) -> None:
        words = ["word"] * 750  # over the hard 720 cap
        text = " ".join(words)
        result = _truncate_overlong(text)
        # Truncated at soft 600 cap with marker.
        assert "[synthesis truncated]" in result
        # Truncated body has ≤ 600 + a few marker words.
        assert len(result.split()) <= 605

    def test_between_caps_passes_through(self) -> None:
        """650 words is over the 600 soft cap but under the 720 hard
        cap — passes through without truncation."""
        words = ["word"] * 650
        text = " ".join(words)
        result = _truncate_overlong(text)
        assert "truncated" not in result


# ── End-to-end synthesis tests (11-test regression suite) ────────────


@pytest.mark.asyncio
async def test_passing_synthesis_two_agreeing_sources() -> None:
    """Two sources agree → cited synthesis passes through."""
    extracts = [
        _make_extract("https://example.com/a", "Doc A", "Fact text"),
        _make_extract("https://example.com/b", "Doc B", "Same fact"),
    ]
    expected = "## Key findings\n### Topic\nThe fact [①②]"
    llm = _fake_llm(expected)
    result = await synthesise(llm, extracts, "the query")
    assert result.text == expected
    assert result.model_used == "primary"


@pytest.mark.asyncio
async def test_synthesis_with_disagreement() -> None:
    extracts = [
        _make_extract("https://example.com/a", "A", "Released 25 September"),
        _make_extract("https://example.com/b", "B", "Released 2 October"),
    ]
    raw = (
        "## Key findings\n### Release\nVersion 18 shipped. [①②]\n"
        "## Disagreements\n- Release date. [①] says Sep 25; [②] says Oct 2.\n"
    )
    llm = _fake_llm(raw)
    result = await synthesise(llm, extracts, "release date")
    # Validator-normalised text strips trailing newline.
    assert result.text == raw.rstrip("\n")
    assert "Disagreements" in result.text


@pytest.mark.asyncio
async def test_empty_extracts_emits_gaps_only() -> None:
    extracts: list[ExtractedSource] = []
    raw = "## Gaps\n- The retrieved sources do not contain information that answers the query.\n"
    llm = _fake_llm(raw)
    result = await synthesise(llm, extracts, "query")
    # Validator-normalised text strips trailing newline.
    assert result.text == raw.rstrip("\n")


@pytest.mark.asyncio
async def test_url_in_synthesis_drops_whole_line() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    # Mock LLM emits a URL in violation of Rule 2.
    emitted = (
        "## Key findings\n### T\n"
        "Good fact [①]\n"
        "Visit https://evil.com for details [①]\n"
        "Another fact [①]"
    )
    llm = _fake_llm(emitted)
    result = await synthesise(llm, extracts, "query")
    assert result.text is not None
    assert "evil.com" not in result.text
    assert "Good fact" in result.text
    assert "Another fact" in result.text


@pytest.mark.asyncio
async def test_missing_citation_line_dropped() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    emitted = (
        "## Key findings\n### T\n"
        "Cited fact [①]\n"
        "Uncited statement that snuck in\n"
        "Another cited fact [①]"
    )
    llm = _fake_llm(emitted)
    result = await synthesise(llm, extracts, "query")
    assert result.text is not None
    assert "Uncited statement" not in result.text
    assert "Cited fact" in result.text


@pytest.mark.asyncio
async def test_majority_uncited_routes_to_fallback() -> None:
    """5 statements, 5 uncited → primary fails → fallback runs.

    Updated for the 0.8 threshold (was 0.5). A 75% drop now passes
    validation (small-model-friendly), so to exercise the
    fallback path the primary needs to produce essentially no
    citations at all.
    """
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    bad_primary = "## Key findings\n### T\nUncited A\nUncited B\nUncited C\nUncited D\nUncited E"
    good_fallback = "## Key findings\n### T\nCited fact [①]"
    primary = _fake_llm(bad_primary)
    fallback = _fake_llm(good_fallback)
    result = await synthesise(primary, extracts, "query", llm_fallback=fallback)
    assert result.text == good_fallback
    assert result.model_used == "fallback"


@pytest.mark.asyncio
async def test_overlong_synthesis_truncated() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    long_text = "## Key findings\n### T\n" + " ".join(["word"] * 750) + " [①]"
    llm = _fake_llm(long_text)
    result = await synthesise(llm, extracts, "query")
    assert result.text is not None
    assert "[synthesis truncated]" in result.text


@pytest.mark.asyncio
async def test_invalid_schema_routes_to_fallback() -> None:
    """Free-form prose with no headers → schema check fails → fallback."""
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    bad_primary = "Free-form synthesis prose without any structure."
    good_fallback = "## Key findings\n### T\nFact [①]"
    primary = _fake_llm(bad_primary)
    fallback = _fake_llm(good_fallback)
    result = await synthesise(primary, extracts, "query", llm_fallback=fallback)
    assert result.text == good_fallback
    assert result.model_used == "fallback"


@pytest.mark.asyncio
async def test_existing_summary_with_synthesis_purpose_raises() -> None:
    """generate_summary's ValueError surfaces through synthesise."""
    from langchain_core.messages import HumanMessage

    from src.memory.summarizer import generate_summary

    with pytest.raises(ValueError, match="existing_summary must be None"):
        generate_summary(
            _fake_llm("x"),
            [HumanMessage(content="x")],
            existing_summary="prior",
            purpose="web_search_synthesis",
        )


@pytest.mark.asyncio
async def test_purpose_default_unchanged() -> None:
    """Calling generate_summary without purpose= still hits the
    historical conversation path (verified end-to-end)."""
    from src.memory.summarizer import generate_summary

    msg = MagicMock()
    msg.content = "Hello"
    type(msg).__name__ = "HumanMessage"

    llm = _fake_llm("Summary text.")
    result = generate_summary(llm, [msg])
    assert result == "Summary text."
    # System message should be the conversation system prompt, not
    # the synthesis one.
    sys_content = llm.invoke.call_args[0][0][0].content
    assert "concise conversation summarizer" in sys_content
    assert "CITATION-CORRECTNESS" not in sys_content


@pytest.mark.asyncio
async def test_non_english_query_section_headers_stay_english() -> None:
    """Rule 11 says section headers stay English even when body is
    non-English. The schema check only requires the English headers."""
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    spanish_with_english_headers = "## Key findings\n" "### Tema\n" "Un hecho importante [①]\n"
    llm = _fake_llm(spanish_with_english_headers)
    result = await synthesise(llm, extracts, "consulta en español")
    assert result.text is not None
    assert "Un hecho importante" in result.text


# ── Smoke: primary success short-circuits before fallback ───────────


@pytest.mark.asyncio
async def test_primary_success_skips_fallback() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    good = "## Key findings\n### T\nFact [①]"
    primary = _fake_llm(good)
    fallback = _fake_llm("Fallback was called incorrectly")
    result = await synthesise(primary, extracts, "query", llm_fallback=fallback)
    assert result.model_used == "primary"
    fallback.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_both_fail_returns_synthesis_unavailable_marker() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    bad = "Free prose with no schema."
    primary = _fake_llm(bad)
    fallback = _fake_llm(bad)
    result = await synthesise(primary, extracts, "query", llm_fallback=fallback)
    assert result.text is None
    assert result.reason == "schema-invalid"
    assert isinstance(result, SynthesisResult)


@pytest.mark.asyncio
async def test_no_fallback_returns_none_on_primary_failure() -> None:
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    primary = _fake_llm("Free prose, no schema.")
    result = await synthesise(primary, extracts, "query")
    assert result.text is None
    assert result.reason == "schema-invalid"
    assert result.model_used is None


# ── Bug I: reason classification — distinguish timeout / exception /
#         empty-response so Coverage reads "failed (<real cause>)"
#         instead of always "failed (empty-response)".
# ── Bug J: log line accuracy — "trying fallback" must not fire when
#         no fallback is configured.


@pytest.mark.asyncio
async def test_primary_timeout_classified_as_timeout() -> None:
    """Bug I: when the underlying LLM call exceeds the per-attempt
    deadline, the reason on the result must be ``"timeout"``, not
    the legacy ``"empty-response"`` catch-all.

    Pre-fix, ``_attempt`` returned a bare ``str | None`` and
    ``_validate(None)`` collapsed timeout/exception/empty into
    ``"empty-response"``. Operators reading
    ``Synthesis: failed (empty-response)`` thought the LLM ran and
    returned nothing — but the real cause was a deadline miss.
    """
    extracts = [_make_extract("https://example.com/a", "A", "content")]

    class _SlowLLM:
        def invoke(self, prompt: Any) -> Any:
            # Block well past the test deadline. ``generate_summary``
            # runs this in a ThreadPoolExecutor with its own timeout
            # guard, but our outer ``asyncio.wait_for`` should fire
            # first and classify the cause as "timeout".
            import time as _time

            _time.sleep(2.0)
            return MagicMock(content="too late")

    result = await synthesise(_SlowLLM(), extracts, "query", deadline_s=0.3)
    assert result.text is None
    assert result.reason == "timeout"
    assert result.model_used is None


@pytest.mark.asyncio
async def test_primary_genuinely_empty_classified_as_empty_response() -> None:
    """Bug I: a genuine empty LLM response (call completed within
    budget, content is whitespace) still gets the ``"empty-response"``
    label — that classification is correct, we just don't want every
    failure to share it."""
    extracts = [_make_extract("https://example.com/a", "A", "content")]
    primary = _fake_llm("   \n  ")  # whitespace-only
    result = await synthesise(primary, extracts, "query")
    assert result.text is None
    assert result.reason == "empty-response"


@pytest.mark.asyncio
async def test_no_fallback_log_line_clarifies_when_fallback_absent(caplog) -> None:
    """Bug J: when ``llm_fallback`` is None, the synthesiser must NOT
    log "trying fallback" (which it then doesn't actually try). Log
    line should make the absence explicit instead."""
    import logging as _logging

    extracts = [_make_extract("https://example.com/a", "A", "content")]
    primary = _fake_llm("Free prose, no schema.")  # primary fails (schema)

    with caplog.at_level(_logging.INFO, logger="cogtrix"):
        result = await synthesise(primary, extracts, "query")

    assert result.text is None
    log_text = "\n".join(record.message for record in caplog.records)
    # The misleading "trying fallback" line must not appear when no
    # fallback is configured.
    assert "trying fallback" not in log_text
    # The accurate alternative line must appear.
    assert "no fallback configured" in log_text


@pytest.mark.asyncio
async def test_fallback_log_line_fires_only_when_fallback_runs(caplog) -> None:
    """Bug J: when a fallback IS configured, the "trying fallback"
    log line should fire (it accurately describes what's happening).
    Pair with the test above to pin both arms of the if-else."""
    import logging as _logging

    extracts = [_make_extract("https://example.com/a", "A", "content")]
    primary = _fake_llm("Free prose, no schema.")  # primary fails
    fallback = _fake_llm("## Key findings\n### T\nFact [①]")

    with caplog.at_level(_logging.INFO, logger="cogtrix"):
        result = await synthesise(primary, extracts, "query", llm_fallback=fallback)

    assert result.text is not None
    assert result.model_used == "fallback"
    log_text = "\n".join(record.message for record in caplog.records)
    assert "trying fallback" in log_text
    assert "no fallback configured" not in log_text


def test_primary_deadline_default_fits_typical_provider_latency() -> None:
    """Bug H: the default primary deadline must accommodate observed
    real-world latency on slow providers (qwen3-coder/spark took
    ~10s on a 4000-token synthesis prompt in cogtrix41c).

    The old 7s value was too tight; raising to 10s catches the
    common-case slow path without exceeding the 25s outer pipeline
    ceiling when combined with realistic stage 1-4 budgets."""
    from src.tools._web_search_synthesiser import _PRIMARY_DEADLINE_S

    assert _PRIMARY_DEADLINE_S >= 10, (
        f"Primary deadline {_PRIMARY_DEADLINE_S}s is too tight for "
        "observed slow-provider latency. cogtrix41c saw the synthesis "
        "LLM call take ~10s on a 4000-token prompt. Anything below 10s "
        "will frequently emit 'failed (timeout)' even when the LLM "
        "would have succeeded with a small extension."
    )


class TestSynthesisPromptAffiliationDisclaimer:
    """#1842 — the synthesiser's per-source block must also surface a
    content-declared affiliation disclaimer, so the synthesis itself never
    frames an unaffiliated source as official."""

    def _extract(self, url: str, text: str):
        from src.tools._http_fetch import FetchResult
        from src.tools._web_search_aggregator import RankedResult
        from src.tools._web_search_domain_class import DomainClass
        from src.tools._web_search_extractor import ExtractedSource
        from src.tools._web_search_fetcher import FetchOutcome

        rank = RankedResult(
            canonical_url=url,
            title="Kimi models",
            snippet="",
            published_date=None,
            domain_class=DomainClass.UNKNOWN,
            score=0.0,
            providers=("ddg",),
        )
        fr = FetchResult(
            url=url,
            status_code=200,
            content=text.encode(),
            encoding="utf-8",
            content_type="text/html",
            elapsed_ms=5,
            truncated=False,
            error=None,
        )
        outcome = FetchOutcome(ranked=rank, status="fetched", fetch_result=fr, error=None)
        return ExtractedSource(fetch_outcome=outcome, extracted_text=text, status="extracted")

    def test_disclaimer_surfaced_in_human_prompt(self) -> None:
        from src.tools._web_search_synthesiser import _format_human_prompt

        src = self._extract(
            "https://platform.kimi.ai/docs",
            "Kimi-AI.chat is an independent guide and is not affiliated with the "
            "official Kimi API Platform.",
        )
        msg = _format_human_prompt("kimi pricing", [src])
        assert "UNAFFILIATED" in msg.content
        assert "not" in msg.content.lower()

    def test_neutral_source_no_disclaimer_line(self) -> None:
        from src.tools._web_search_synthesiser import _format_human_prompt

        src = self._extract("https://example.com", "Neutral documentation content here.")
        msg = _format_human_prompt("q", [src])
        assert "UNAFFILIATED" not in msg.content
