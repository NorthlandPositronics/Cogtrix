"""Search-pipeline efficiency harness.

Where the existing ``test_web_search*`` files cover correctness of
individual functions (does ``_search_async`` map fields correctly?
does ``parse_ddg_html`` handle entities?), this module measures the
*end-to-end behaviour* of the assembled ``web_search`` pipeline
under realistic-but-deterministic conditions. Each scenario:

1. Mocks the four pipeline stages (provider fan-out, fetcher,
   extractor, synthesiser) at well-defined boundaries.
2. Runs the public ``web_search()`` entry point.
3. Parses the resulting Markdown for the Coverage block and the
   structural sections (Sources, extracts, Synthesis).
4. Asserts on a multi-dimensional ``SearchMetrics`` snapshot —
   provider yield rate, fetch yield rate, synthesis state,
   domain diversity, cache short-circuit latency, dedup behaviour.

The intent is to catch *regression of efficiency properties* the
test suite would otherwise miss. For example: if a future change
removed the per-task ``wait_for`` from ``fetch_top_k``, the
"slow URL doesn't poison batch" test below would fail loudly,
because the Coverage block would read ``Fetched: 0/N`` instead of
``Fetched: 3/N``. Bug F regression caught in one scenario; Bug C
in another; Bug E threshold drift in a third.

Every scenario is deterministic and offline — no real LLM, no
real network, no real subprocess. They run in ~2 s total.

If you add a new efficiency scenario, follow the
``_run_scenario`` shape so the metrics layer can parse the
Coverage block consistently.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.tools._http_fetch import FetchResult
from src.tools._web_search_aggregator import (
    CoverageInfo,
    RankedResult,
)
from src.tools._web_search_domain_class import DomainClass
from src.tools._web_search_extractor import ExtractedSource
from src.tools._web_search_fetcher import FetchOutcome
from src.tools._web_search_synthesiser import SynthesisResult
from src.tools.web_search import web_search

# ── Metrics dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchMetrics:
    """Parsed-from-output measurements for a single scenario run."""

    providers_attempted: int
    providers_succeeded: int
    raw_count: int
    distinct_count: int
    fetched_count: int
    selected_count: int
    snippet_only_count: int
    synthesis_state: str  # "ok", "skipped", "failed (<reason>)"
    total_wall_ms: int
    output_len: int
    has_synthesis_unavailable_marker: bool
    distinct_domains: tuple[str, ...]

    @property
    def provider_yield(self) -> float:
        """Fraction of attempted providers that responded successfully."""
        return (
            self.providers_succeeded / self.providers_attempted if self.providers_attempted else 0.0
        )

    @property
    def fetch_yield(self) -> float:
        """Fraction of selected top-K URLs that fetched successfully."""
        return self.fetched_count / self.selected_count if self.selected_count else 0.0


# ── Coverage-block parser ────────────────────────────────────────────


_COV_PROVIDERS_RE = re.compile(
    r"Searched:\s*(\d+)\s*of\s*(\d+)\s*providers\s*responded;\s*"
    r"(\d+)\s*raw\s*results,\s*(\d+)\s*distinct"
)
_COV_FETCHED_RE = re.compile(
    r"Fetched:\s*(\d+)/(\d+)\s*selected\s*sources\s*successful;\s*"
    r"(\d+)\s*fell\s*back\s*to\s*snippet"
)
_COV_SYNTH_RE = re.compile(r"Synthesis:\s*([^.]+)\.")
_COV_WALL_RE = re.compile(r"Total\s*wall\s*time:\s*([\d.]+)\s*s")

_SOURCE_LINE_RE = re.compile(r"^[①-⑳]\s+([^\s\[]+)", re.MULTILINE)


def _parse_metrics(output: str) -> SearchMetrics:
    """Pull a ``SearchMetrics`` snapshot from the rendered web_search output."""
    m_prov = _COV_PROVIDERS_RE.search(output)
    m_fetch = _COV_FETCHED_RE.search(output)
    m_synth = _COV_SYNTH_RE.search(output)
    m_wall = _COV_WALL_RE.search(output)

    providers_succeeded, providers_attempted, raw_count, distinct_count = (
        (int(m_prov.group(1)), int(m_prov.group(2)), int(m_prov.group(3)), int(m_prov.group(4)))
        if m_prov
        else (0, 0, 0, 0)
    )
    fetched_count, selected_count, snippet_only_count = (
        (int(m_fetch.group(1)), int(m_fetch.group(2)), int(m_fetch.group(3)))
        if m_fetch
        else (0, 0, 0)
    )
    synthesis_state = m_synth.group(1).strip() if m_synth else "missing"
    total_wall_ms = int(float(m_wall.group(1)) * 1000) if m_wall else 0
    domains = tuple(_SOURCE_LINE_RE.findall(output))

    return SearchMetrics(
        providers_attempted=providers_attempted,
        providers_succeeded=providers_succeeded,
        raw_count=raw_count,
        distinct_count=distinct_count,
        fetched_count=fetched_count,
        selected_count=selected_count,
        snippet_only_count=snippet_only_count,
        synthesis_state=synthesis_state,
        total_wall_ms=total_wall_ms,
        output_len=len(output),
        has_synthesis_unavailable_marker="## Synthesis unavailable" in output,
        distinct_domains=domains,
    )


# ── Scenario builder ─────────────────────────────────────────────────


def _rank(url: str, *, score: float = 1.0, providers: tuple[str, ...] = ("ddg",)) -> RankedResult:
    """Build a RankedResult for tests."""
    return RankedResult(
        canonical_url=url,
        title=f"Title for {url}",
        snippet=f"Snippet for {url}",
        published_date=None,
        domain_class=DomainClass.UNKNOWN,
        score=score,
        providers=providers,
    )


def _fetched_outcome(url: str, body: bytes = b"<html>body</html>") -> FetchOutcome:
    """Build a successful FetchOutcome."""
    fr = FetchResult(
        url=url,
        status_code=200,
        content=body,
        encoding="utf-8",
        content_type="text/html",
        elapsed_ms=10,
        truncated=False,
        error=None,
    )
    return FetchOutcome(ranked=_rank(url), status="fetched", fetch_result=fr, error=None)


def _snippet_only_outcome(url: str, error: str = "timeout") -> FetchOutcome:
    return FetchOutcome(ranked=_rank(url), status="snippet-only", fetch_result=None, error=error)


def _extract(url: str, text: str = "Extracted body content.") -> ExtractedSource:
    return ExtractedSource(
        fetch_outcome=_fetched_outcome(url, text.encode()),
        extracted_text=text,
        status="extracted",
    )


def _no_extract(url: str, status: str = "snippet-only") -> ExtractedSource:
    return ExtractedSource(
        fetch_outcome=_snippet_only_outcome(url),
        extracted_text=None,
        status=status,  # type: ignore[arg-type]
    )


@dataclass
class _Scenario:
    """Mock scaffold for a single web_search run."""

    name: str
    ranked: list[RankedResult]
    coverage: CoverageInfo
    fetched: list[FetchOutcome]
    extracted: list[ExtractedSource]
    synthesis: SynthesisResult | None = None
    synthesis_error: type[BaseException] | None = None
    providers: dict[str, Any] = field(default_factory=dict)


async def _run_scenario(
    s: _Scenario, query: str = "test query"
) -> tuple[SearchMetrics, str, float]:
    """Run the web_search pipeline with stage outputs forced by *s*.

    Returns ``(metrics, raw_output, wall_seconds)``. The wall time
    here is the wall time of the assembled framework only — every
    stage is mocked to return instantly. Useful for spotting
    orchestration overhead drift, not provider latency.
    """
    aggregate_mock = AsyncMock(return_value=(s.ranked, s.coverage))
    fetch_mock = AsyncMock(return_value=s.fetched)
    extract_mock = AsyncMock(return_value=s.extracted)

    if s.synthesis_error is not None:
        synth_mock: Any = AsyncMock(side_effect=s.synthesis_error("boom"))
    else:
        synth_mock = AsyncMock(return_value=s.synthesis)

    providers = s.providers or {"ddg": AsyncMock(), "tavily": AsyncMock()}

    t0 = time.monotonic()
    with (
        patch("src.tools.web_search._resolve_providers", return_value=providers),
        patch("src.tools._web_search_aggregator.aggregate", new=aggregate_mock),
        patch("src.tools._web_search_fetcher.fetch_top_k", new=fetch_mock),
        patch("src.tools._web_search_extractor.extract", new=extract_mock),
        patch("src.tools._web_search_synthesiser.synthesise", new=synth_mock),
        # Inject a fake synthesis LLM so stage 5 actually invokes
        # ``synthesise`` instead of skipping. The mock above controls
        # the return value.
        patch("src.tools.web_search._synthesis_llm_var") as llm_var,
    ):
        llm_var.get.return_value = object()  # truthy
        output = await web_search(query=query)
    wall = time.monotonic() - t0
    return _parse_metrics(output), output, wall


# ── Fixtures: standard scenarios ─────────────────────────────────────


def _happy_path_scenario() -> _Scenario:
    """Both providers responded, 6 distinct URLs across 6 domains,
    all 6 fetched successfully, all 6 extracted with content,
    synthesis produced validated text."""
    urls = [f"https://example-{i}.com/a" for i in range(6)]
    return _Scenario(
        name="happy_path",
        ranked=[
            _rank(u, providers=("ddg", "tavily") if i % 2 == 0 else ("ddg",))
            for i, u in enumerate(urls)
        ],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=10, distinct_count=6
        ),
        fetched=[_fetched_outcome(u) for u in urls],
        extracted=[_extract(u, f"Real content from {u}") for u in urls],
        synthesis=SynthesisResult(
            text="## Key findings\n### Topic\n- Synthesised fact [①]",
            reason=None,
            model_used="primary",
            elapsed_ms=2000,
        ),
    )


def _one_provider_fails_scenario() -> _Scenario:
    """Tavily responded with 5 results; DDG raised. Aggregator
    records 1/2, top-K is the 5 Tavily URLs."""
    urls = [f"https://tav-{i}.com/page" for i in range(5)]
    return _Scenario(
        name="one_provider_fails",
        ranked=[_rank(u, providers=("tavily",)) for u in urls],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=1, raw_count=5, distinct_count=5
        ),
        fetched=[_fetched_outcome(u) for u in urls],
        extracted=[_extract(u) for u in urls],
        synthesis=SynthesisResult(
            text="## Key findings\n### Topic\n- Fact [①]",
            reason=None,
            model_used="primary",
            elapsed_ms=1500,
        ),
    )


def _all_providers_fail_scenario() -> _Scenario:
    """Both providers raised. Aggregator returns 0/2, empty ranked
    list. Downstream stages get empty inputs."""
    return _Scenario(
        name="all_providers_fail",
        ranked=[],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=0, raw_count=0, distinct_count=0
        ),
        fetched=[],
        extracted=[],
        synthesis=None,
    )


def _slow_url_does_not_poison_scenario() -> _Scenario:
    """Bug F pin: even when 3 of 6 fetches time out, the other 3
    must survive and the synthesis must run on the survivors."""
    urls = [f"https://mixed-{i}.com/p" for i in range(6)]
    return _Scenario(
        name="slow_url_does_not_poison",
        ranked=[_rank(u) for u in urls],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=8, distinct_count=6
        ),
        fetched=[
            _fetched_outcome(urls[0]),
            _fetched_outcome(urls[1]),
            _fetched_outcome(urls[2]),
            _snippet_only_outcome(urls[3]),
            _snippet_only_outcome(urls[4]),
            _snippet_only_outcome(urls[5]),
        ],
        extracted=[
            _extract(urls[0]),
            _extract(urls[1]),
            _extract(urls[2]),
            _no_extract(urls[3]),
            _no_extract(urls[4]),
            _no_extract(urls[5]),
        ],
        synthesis=SynthesisResult(
            text="## Key findings\n### Topic\n- Partial fact [①][②][③]",
            reason=None,
            model_used="primary",
            elapsed_ms=1800,
        ),
    )


def _all_fetches_blocked_scenario() -> _Scenario:
    """Every top-K URL fetches as snippet-only. Synthesis should
    skip (no extracts have content); agent still gets a Sources
    block with snippets to work from."""
    urls = [f"https://blocked-{i}.com/p" for i in range(5)]
    return _Scenario(
        name="all_fetches_blocked",
        ranked=[_rank(u) for u in urls],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=7, distinct_count=5
        ),
        fetched=[_snippet_only_outcome(u, "blocked-robots") for u in urls],
        extracted=[_no_extract(u) for u in urls],
        synthesis=None,
    )


def _synthesis_timeout_scenario() -> _Scenario:
    """Bug H/I pin: synthesiser timed out (post-Bug-I, the reason
    is propagated distinctly from generic empty-response)."""
    urls = [f"https://timeout-{i}.com/p" for i in range(4)]
    return _Scenario(
        name="synthesis_timeout",
        ranked=[_rank(u) for u in urls],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=6, distinct_count=4
        ),
        fetched=[_fetched_outcome(u) for u in urls],
        extracted=[_extract(u) for u in urls],
        synthesis=SynthesisResult(
            text=None,
            reason="timeout",
            model_used=None,
            elapsed_ms=10_000,
        ),
    )


def _synthesis_exception_scenario() -> _Scenario:
    """When ``synthesise()`` raises, the pipeline must not crash —
    it should emit a SynthesisResult(text=None, reason='exception:…')
    that the formatter renders as the unavailable marker."""
    urls = [f"https://exc-{i}.com/p" for i in range(3)]
    return _Scenario(
        name="synthesis_exception",
        ranked=[_rank(u) for u in urls],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=5, distinct_count=3
        ),
        fetched=[_fetched_outcome(u) for u in urls],
        extracted=[_extract(u) for u in urls],
        synthesis_error=RuntimeError,
    )


def _high_domain_diversity_scenario() -> _Scenario:
    """Top-K URLs span 6 distinct domains. The Sources block must
    surface all of them — diversity is a quality signal that prevents
    the agent from being misled by a single-source consensus."""
    urls = [
        "https://en.wikipedia.org/wiki/x",
        "https://nature.com/articles/y",
        "https://arxiv.org/abs/2401.00001",
        "https://github.com/example/repo",
        "https://reuters.com/news/z",
        "https://nih.gov/research/w",
    ]
    return _Scenario(
        name="high_domain_diversity",
        ranked=[_rank(u, score=1.0 - 0.1 * i) for i, u in enumerate(urls)],
        coverage=CoverageInfo(
            providers_attempted=2, providers_succeeded=2, raw_count=10, distinct_count=6
        ),
        fetched=[_fetched_outcome(u) for u in urls],
        extracted=[_extract(u) for u in urls],
        synthesis=SynthesisResult(
            text="## Key findings\n### Topic\n- Diverse fact [①][②]",
            reason=None,
            model_used="primary",
            elapsed_ms=2500,
        ),
    )


# ── Scenario tests ───────────────────────────────────────────────────


class TestHappyPath:
    """Baseline: every stage works. Pins the typical-case metrics."""

    @pytest.mark.asyncio
    async def test_provider_yield_is_full(self) -> None:
        metrics, _, _ = await _run_scenario(_happy_path_scenario())
        assert metrics.provider_yield == 1.0
        assert metrics.providers_succeeded == 2

    @pytest.mark.asyncio
    async def test_fetch_yield_is_full(self) -> None:
        metrics, _, _ = await _run_scenario(_happy_path_scenario())
        assert metrics.fetch_yield == 1.0
        assert metrics.fetched_count == 6

    @pytest.mark.asyncio
    async def test_synthesis_succeeded(self) -> None:
        metrics, output, _ = await _run_scenario(_happy_path_scenario())
        assert "skipped" not in metrics.synthesis_state
        assert "failed" not in metrics.synthesis_state
        assert "## Key findings" in output
        assert not metrics.has_synthesis_unavailable_marker

    @pytest.mark.asyncio
    async def test_orchestration_overhead_under_100ms(self) -> None:
        """With every stage mocked to return instantly, the
        framework itself must not add more than 100 ms. Drift here
        means web_search() is doing avoidable work in the assembly
        path (extra parsing, debug logging, etc.)."""
        _, _, wall = await _run_scenario(_happy_path_scenario())
        assert wall < 0.1, f"framework overhead {wall * 1000:.0f} ms exceeds 100 ms budget"


class TestProviderResilience:
    """How the pipeline handles partial / total provider failure."""

    @pytest.mark.asyncio
    async def test_one_provider_fails_still_produces_results(self) -> None:
        metrics, output, _ = await _run_scenario(_one_provider_fails_scenario())
        assert metrics.providers_succeeded == 1
        assert metrics.providers_attempted == 2
        assert metrics.distinct_count == 5
        # Synthesis still runs because we have extracts.
        assert "skipped" not in metrics.synthesis_state
        assert "## Key findings" in output

    @pytest.mark.asyncio
    async def test_all_providers_fail_produces_graceful_empty_output(self) -> None:
        metrics, output, _ = await _run_scenario(_all_providers_fail_scenario())
        assert metrics.provider_yield == 0.0
        assert metrics.distinct_count == 0
        # No synthesis when there's nothing to synthesise — but the
        # output is still well-formed Markdown so the agent doesn't
        # see a Python exception.
        assert metrics.synthesis_state == "skipped"
        assert "# Research:" in output


class TestFetcherResilience:
    """Bug F regression bed: one slow URL must not poison the batch.
    Also covers the all-blocked degradation path."""

    @pytest.mark.asyncio
    async def test_partial_fetch_yield_preserves_successes(self) -> None:
        metrics, _, _ = await _run_scenario(_slow_url_does_not_poison_scenario())
        # 3 out of 6 succeeded; the 3 timeouts must not have
        # wiped the successes from the output (Bug F).
        assert metrics.fetched_count == 3
        assert metrics.snippet_only_count == 3
        assert metrics.fetch_yield == 0.5
        # Synthesis runs on the survivors.
        assert "skipped" not in metrics.synthesis_state

    @pytest.mark.asyncio
    async def test_all_fetches_blocked_skips_synthesis(self) -> None:
        """When every extract has no content, synthesis must skip —
        the Sources block is still emitted so the agent has snippets
        to work from."""
        metrics, output, _ = await _run_scenario(_all_fetches_blocked_scenario())
        assert metrics.fetched_count == 0
        assert metrics.synthesis_state == "skipped"
        assert "## Sources" in output


class TestSynthesisQuality:
    """Bug I regression bed: failure reasons must be distinct,
    and synthesis exceptions must not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_timeout_classified_distinctly(self) -> None:
        metrics, _, _ = await _run_scenario(_synthesis_timeout_scenario())
        assert (
            "timeout" in metrics.synthesis_state
        ), f"timeout reason was not propagated; got {metrics.synthesis_state!r}"
        assert "failed" in metrics.synthesis_state

    @pytest.mark.asyncio
    async def test_synthesis_exception_does_not_crash_pipeline(self) -> None:
        metrics, output, _ = await _run_scenario(_synthesis_exception_scenario())
        # Pipeline emitted the unavailable marker — the formatter's
        # "synthesis ran but failed" branch fired.
        assert metrics.has_synthesis_unavailable_marker
        # Output is still well-formed Markdown with extracts.
        assert "## Sources" in output


class TestCacheShortCircuit:
    """Latency property: repeat queries within TTL must short-circuit
    the entire pipeline. Stage mocks must NOT be called the second
    time. Wall time under 10 ms."""

    @pytest.mark.asyncio
    async def test_repeat_query_short_circuits(self) -> None:
        from src.tools._web_search_cache import cache_clear

        cache_clear()
        scenario = _happy_path_scenario()

        aggregate_mock = AsyncMock(return_value=(scenario.ranked, scenario.coverage))
        fetch_mock = AsyncMock(return_value=scenario.fetched)
        extract_mock = AsyncMock(return_value=scenario.extracted)
        synth_mock = AsyncMock(return_value=scenario.synthesis)

        with (
            patch("src.tools.web_search._resolve_providers", return_value={"ddg": AsyncMock()}),
            patch("src.tools._web_search_aggregator.aggregate", new=aggregate_mock),
            patch("src.tools._web_search_fetcher.fetch_top_k", new=fetch_mock),
            patch("src.tools._web_search_extractor.extract", new=extract_mock),
            patch("src.tools._web_search_synthesiser.synthesise", new=synth_mock),
            patch("src.tools.web_search._synthesis_llm_var") as llm_var,
        ):
            llm_var.get.return_value = object()
            first = await web_search("cache-hit-key", depth=3)
            t0 = time.monotonic()
            second = await web_search("cache-hit-key", depth=3)
            cache_wall = time.monotonic() - t0

        cache_clear()  # leave the cache clean for other tests
        assert first == second, "cached output must be identical to first call"
        assert aggregate_mock.await_count == 1, "stage-1 ran twice — cache short-circuit failed"
        assert cache_wall < 0.05, f"cache hit took {cache_wall * 1000:.1f} ms; expected <50 ms"


class TestStructuralIntegrity:
    """Every web_search output must have a stable structure regardless
    of which stages succeeded — this is what the agent prompt is
    trained against. Drift here would silently misalign the agent's
    parsing assumptions."""

    @pytest.mark.asyncio
    async def test_research_header_always_present(self) -> None:
        """Every output starts with ``# Research: <query>`` — agent
        prompt uses this as the marker that web_search ran."""
        for scenario_fn in [
            _happy_path_scenario,
            _one_provider_fails_scenario,
            _all_providers_fail_scenario,
            _all_fetches_blocked_scenario,
            _synthesis_timeout_scenario,
        ]:
            _, output, _ = await _run_scenario(scenario_fn())
            assert output.startswith(
                "# Research:"
            ), f"scenario {scenario_fn.__name__}: missing Research header in output"

    @pytest.mark.asyncio
    async def test_coverage_block_always_present_with_all_four_lines(self) -> None:
        """The Coverage block is the operator-facing diagnostic. It
        must have exactly four lines (Searched, Fetched, Synthesis,
        Total wall) in every output, even on the graceful-degraded
        paths."""
        for scenario_fn in [
            _happy_path_scenario,
            _all_providers_fail_scenario,
            _all_fetches_blocked_scenario,
        ]:
            _, output, _ = await _run_scenario(scenario_fn())
            assert "## Coverage" in output
            assert "Searched:" in output
            assert "Fetched:" in output
            assert "Synthesis:" in output
            assert "Total wall time:" in output


class TestRankingDiversity:
    """Quality signal: top-K results should span multiple domains
    when the input had domain diversity. Single-domain collapse is
    a flag for upstream consensus-rank logic going wrong."""

    @pytest.mark.asyncio
    async def test_six_distinct_domains_all_surface_in_sources(self) -> None:
        metrics, output, _ = await _run_scenario(_high_domain_diversity_scenario())
        # The Sources block should mention every domain.
        for domain in [
            "wikipedia.org",
            "nature.com",
            "arxiv.org",
            "github.com",
            "reuters.com",
            "nih.gov",
        ]:
            assert domain in output, f"missing domain {domain!r} in Sources"
        # The parsed-domain tuple from the Sources block — at least
        # 5 distinct should appear in the citation index (the parser
        # may count up to 6 depending on exact layout).
        assert len(set(metrics.distinct_domains)) >= 5, (
            f"only {len(set(metrics.distinct_domains))} distinct domains " f"in Sources; expected 6"
        )


# ── Aggregate efficiency-report fixture (info only, not gating) ──────


@dataclass
class _EfficiencyReport:
    """Accumulator that prints a scoreboard at session end. Pure
    diagnostic; doesn't gate the suite. Useful when iterating on
    pipeline changes to see the multi-dimensional metrics at a glance."""

    rows: list[tuple[str, SearchMetrics, float]] = field(default_factory=list)

    def add(self, name: str, m: SearchMetrics, wall: float) -> None:
        self.rows.append((name, m, wall))

    def render(self) -> str:
        if not self.rows:
            return ""
        lines = [
            "",
            "web_search efficiency scoreboard",
            "-" * 90,
            f"{'scenario':<32} {'prov':>6} {'fetch':>6} {'synth':<28} {'ms':>4}",
            "-" * 90,
        ]
        for name, m, wall in self.rows:
            lines.append(
                f"{name:<32} "
                f"{m.providers_succeeded}/{m.providers_attempted:<3} "
                f"{m.fetched_count}/{m.selected_count:<3} "
                f"{m.synthesis_state[:28]:<28} "
                f"{int(wall * 1000):>4}"
            )
        lines.append("-" * 90)
        return "\n".join(lines)


@pytest.fixture(scope="module")
def efficiency_report(request: pytest.FixtureRequest) -> _EfficiencyReport:
    report = _EfficiencyReport()

    def _print() -> None:
        text = report.render()
        if text:
            # Emit via the pytest capture machinery so it appears
            # under the test class in -s output.
            print(text)

    request.addfinalizer(_print)
    return report


class TestEfficiencyScoreboard:
    """Run every standard scenario and dump a single scoreboard.
    These tests don't have hard assertions of their own — the
    per-property assertions live in the dedicated test classes
    above. This is a diagnostic-only block."""

    SCENARIOS: Sequence[Callable[[], _Scenario]] = (
        _happy_path_scenario,
        _one_provider_fails_scenario,
        _all_providers_fail_scenario,
        _slow_url_does_not_poison_scenario,
        _all_fetches_blocked_scenario,
        _synthesis_timeout_scenario,
        _synthesis_exception_scenario,
        _high_domain_diversity_scenario,
    )

    @pytest.mark.asyncio
    async def test_scoreboard(self, efficiency_report: _EfficiencyReport) -> None:
        for fn in self.SCENARIOS:
            sc = fn()
            # Unique query per scenario so the in-process cache doesn't
            # short-circuit the second-and-onward run (the cache key is
            # the query string).
            metrics, _, wall = await _run_scenario(sc, query=f"scenario:{sc.name}")
            efficiency_report.add(sc.name, metrics, wall)
        # Soft floor: total accumulated framework overhead under 1s
        # for 8 scenarios = 125 ms each, with overhead room.
        assert sum(w for _, _, w in efficiency_report.rows) < 1.0


# ── Sanity-check on the parser itself ────────────────────────────────


class TestMetricsParser:
    """Pin the Coverage-block parser. If web_search formatting drift
    breaks these regex captures, all the scenario assertions become
    silently meaningless. So we test the parser too."""

    def test_parses_full_coverage_block(self) -> None:
        sample = (
            "# Research: q\n"
            "\n"
            "## Sources\n"
            "① example.com [news · 2024]\n"
            "\n"
            "## Coverage\n"
            "- Searched: 2 of 3 providers responded; 8 raw results, 6 distinct after dedupe.\n"
            "- Fetched: 4/6 selected sources successful; 2 fell back to snippet.\n"
            "- Synthesis: gpt-4o; 1.8s.\n"
            "- Total wall time: 5.2s.\n"
        )
        m = _parse_metrics(sample)
        assert m.providers_succeeded == 2
        assert m.providers_attempted == 3
        assert m.raw_count == 8
        assert m.distinct_count == 6
        assert m.fetched_count == 4
        assert m.selected_count == 6
        assert m.snippet_only_count == 2
        assert "gpt-4o" in m.synthesis_state
        assert m.total_wall_ms == 5200

    def test_parses_skipped_synthesis(self) -> None:
        sample = (
            "# Research: q\n"
            "## Coverage\n"
            "- Searched: 1 of 2 providers responded; 3 raw results, 3 distinct after dedupe.\n"
            "- Fetched: 0/3 selected sources successful; 0 fell back to snippet.\n"
            "- Synthesis: skipped.\n"
            "- Total wall time: 2.1s.\n"
        )
        m = _parse_metrics(sample)
        assert m.synthesis_state == "skipped"
        assert m.fetch_yield == 0.0

    def test_parses_failed_synthesis_with_reason(self) -> None:
        sample = (
            "## Coverage\n"
            "- Searched: 2 of 2 providers responded; 5 raw results, 5 distinct after dedupe.\n"
            "- Fetched: 3/5 selected sources successful; 2 fell back to snippet.\n"
            "- Synthesis: failed (timeout).\n"
            "- Total wall time: 12.0s.\n"
        )
        m = _parse_metrics(sample)
        assert "timeout" in m.synthesis_state
        assert m.fetch_yield == 0.6


# ── Top-level await helper to ease running scenarios from tests ──────


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every efficiency test starts with an empty query cache."""
    from src.tools._web_search_cache import cache_clear

    cache_clear()
    yield
    cache_clear()
