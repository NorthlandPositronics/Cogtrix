"""Tests for src/tools/_web_search_format.py (ADR-0056 PR-E stage 6)."""

from __future__ import annotations

from src.tools._http_fetch import FetchResult
from src.tools._web_search_aggregator import CoverageInfo, ProviderResult, RankedResult
from src.tools._web_search_domain_class import DomainClass
from src.tools._web_search_extractor import ExtractedSource
from src.tools._web_search_fetcher import FetchOutcome
from src.tools._web_search_format import FormatInput, format_output
from src.tools._web_search_synthesiser import SynthesisResult


def _rank(url: str, domain_class: DomainClass = DomainClass.UNKNOWN) -> RankedResult:
    return RankedResult(
        canonical_url=url,
        title=f"Title for {url}",
        snippet=f"Snippet for {url}",
        published_date=None,
        domain_class=domain_class,
        score=1.0,
        providers=("ddg",),
    )


def _extracted(url: str, text: str = "Extracted body content.") -> ExtractedSource:
    rank = _rank(url)
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
    outcome = FetchOutcome(ranked=rank, status="fetched", fetch_result=fetch_result, error=None)
    return ExtractedSource(fetch_outcome=outcome, extracted_text=text, status="extracted")


def _snippet_only(url: str) -> ExtractedSource:
    rank = _rank(url)
    outcome = FetchOutcome(
        ranked=rank, status="snippet-only", fetch_result=None, error="blocked-robots"
    )
    return ExtractedSource(fetch_outcome=outcome, extracted_text=None, status="snippet-only")


def _state(**overrides) -> FormatInput:
    extracted = overrides.pop("extracted", [_extracted("https://example.com/a")])
    ranked = overrides.pop("ranked", [es.fetch_outcome.ranked for es in extracted])
    fetched = overrides.pop("fetched", [es.fetch_outcome for es in extracted])
    coverage = overrides.pop(
        "coverage",
        CoverageInfo(providers_attempted=1, providers_succeeded=1, raw_count=1, distinct_count=1),
    )
    return FormatInput(
        query=overrides.pop("query", "test query"),
        ranked=ranked,
        fetched=fetched,
        extracted=extracted,
        snippet_only_tail=overrides.pop("snippet_only_tail", []),
        synthesis=overrides.pop("synthesis", None),
        coverage=coverage,
        total_wall_ms=overrides.pop("total_wall_ms", 1234),
        synthesis_model_name=overrides.pop("synthesis_model_name", None),
        compact=overrides.pop("compact", False),
        cache_hit=overrides.pop("cache_hit", False),
        cache_retrieved_at=overrides.pop("cache_retrieved_at", None),
        **overrides,
    )


# ── Schema basics ────────────────────────────────────────────────────


class TestSchema:
    def test_has_research_header(self) -> None:
        out = format_output(_state(query="kubernetes deployment"))
        assert out.startswith("# Research: kubernetes deployment")

    def test_has_sources_section(self) -> None:
        out = format_output(_state())
        assert "## Sources" in out

    def test_has_coverage_section(self) -> None:
        out = format_output(_state())
        assert "## Coverage" in out

    def test_no_synthesis_when_none(self) -> None:
        out = format_output(_state(synthesis=None))
        assert "## Key findings" not in out
        assert "## Synthesis unavailable" not in out


# ── Synthesis emission ───────────────────────────────────────────────


class TestSynthesisEmission:
    def test_validated_synthesis_text_emitted(self) -> None:
        syn = SynthesisResult(
            text="## Key findings\n### A\nFact [①]",
            reason=None,
            model_used="primary",
            elapsed_ms=1500,
        )
        out = format_output(_state(synthesis=syn))
        assert "## Key findings" in out
        assert "Fact [①]" in out

    def test_failed_synthesis_emits_unavailable_prefix(self) -> None:
        syn = SynthesisResult(text=None, reason="schema-invalid", model_used=None, elapsed_ms=8000)
        out = format_output(_state(synthesis=syn))
        assert "## Synthesis unavailable" in out


# ── Sources section ──────────────────────────────────────────────────


class TestSources:
    def test_citation_indices_used(self) -> None:
        out = format_output(
            _state(
                extracted=[
                    _extracted("https://example.com/a"),
                    _extracted("https://example.com/b"),
                ]
            )
        )
        assert "① example.com" in out
        assert "② example.com" in out

    def test_domain_class_in_source_tag(self) -> None:
        es = _extracted("https://en.wikipedia.org/wiki/Topic")
        # patch the domain_class on the ranked obj
        rank = es.fetch_outcome.ranked
        new_rank = RankedResult(
            canonical_url=rank.canonical_url,
            title=rank.title,
            snippet=rank.snippet,
            published_date=rank.published_date,
            domain_class=DomainClass.WIKI_ENCYCLOPEDIA,
            score=rank.score,
            providers=rank.providers,
        )
        new_outcome = FetchOutcome(
            ranked=new_rank,
            status=es.fetch_outcome.status,
            fetch_result=es.fetch_outcome.fetch_result,
            error=es.fetch_outcome.error,
        )
        new_es = ExtractedSource(
            fetch_outcome=new_outcome, extracted_text=es.extracted_text, status=es.status
        )
        out = format_output(_state(extracted=[new_es]))
        assert "wiki-encyclopedia" in out

    def test_url_present_in_sources(self) -> None:
        out = format_output(_state(extracted=[_extracted("https://example.com/article")]))
        assert "https://example.com/article" in out


# ── Per-source bodies ────────────────────────────────────────────────


class TestPerSourceBodies:
    def test_extracted_text_appears_in_non_compact(self) -> None:
        out = format_output(
            _state(extracted=[_extracted("https://example.com/a", "UNIQUE_BODY_MARKER")])
        )
        assert "UNIQUE_BODY_MARKER" in out

    def test_compact_drops_extract_bodies(self) -> None:
        out = format_output(
            _state(
                extracted=[_extracted("https://example.com/a", "UNIQUE_BODY_MARKER")],
                compact=True,
            )
        )
        assert "UNIQUE_BODY_MARKER" not in out
        # But the source itself is still in the Sources index.
        assert "https://example.com/a" in out

    def test_snippet_only_source_marked(self) -> None:
        out = format_output(_state(extracted=[_snippet_only("https://example.com/blocked")]))
        assert "(snippet-only)" in out
        assert "blocked-robots" in out


# ── Additional sources tail ──────────────────────────────────────────


class TestAdditionalSources:
    def test_tail_emitted_when_present(self) -> None:
        tail = [
            ProviderResult(
                provider="ddg",
                url="https://example.com/tail-a",
                title="Tail A",
                snippet="Tail snippet A",
                published_date=None,
            )
        ]
        out = format_output(_state(snippet_only_tail=tail))
        assert "## Additional sources" in out
        assert "https://example.com/tail-a" in out

    def test_tail_dropped_in_compact(self) -> None:
        tail = [
            ProviderResult(
                provider="ddg",
                url="https://example.com/tail-a",
                title="Tail A",
                snippet="Tail snippet A",
                published_date=None,
            )
        ]
        out = format_output(_state(snippet_only_tail=tail, compact=True))
        assert "## Additional sources" not in out


# ── Coverage block ───────────────────────────────────────────────────


class TestCoverage:
    def test_provider_counts_in_coverage(self) -> None:
        coverage = CoverageInfo(
            providers_attempted=4,
            providers_succeeded=3,
            raw_count=29,
            distinct_count=12,
            per_provider_failures={"google": "RuntimeError"},
        )
        out = format_output(_state(coverage=coverage))
        assert "3 of 4 providers responded" in out
        assert "29 raw results" in out
        assert "12 distinct" in out

    def test_fetch_counts_in_coverage(self) -> None:
        extracted = [
            _extracted("https://example.com/a"),
            _extracted("https://example.com/b"),
            _snippet_only("https://example.com/c"),
        ]
        out = format_output(_state(extracted=extracted))
        assert "Fetched: 2/3 selected" in out
        assert "1 fell back to snippet" in out

    def test_wall_time_in_coverage(self) -> None:
        out = format_output(_state(total_wall_ms=7800))
        assert "Total wall time: 7.8s" in out

    def test_cache_hit_short_circuits_coverage(self) -> None:
        out = format_output(_state(cache_hit=True, cache_retrieved_at="2026-05-19T10:00:00+00:00"))
        assert "Cache hit" in out
        assert "2026-05-19T10:00:00+00:00" in out
        # Other coverage lines absent on cache hit.
        assert "providers responded" not in out


# ── Citation cap ─────────────────────────────────────────────────────


class TestCitationCap:
    def test_over_twenty_sources_overflow_noted(self) -> None:
        extracted = [_extracted(f"https://example.com/p{i}") for i in range(25)]
        out = format_output(_state(extracted=extracted))
        # Coverage block notes the overflow.
        assert "5 more sources dropped" in out

    def test_first_twenty_use_circled_indices(self) -> None:
        extracted = [_extracted(f"https://example.com/p{i}") for i in range(20)]
        out = format_output(_state(extracted=extracted))
        # Index 20 (⑳) present.
        assert "⑳" in out


# ── Trailing newline ─────────────────────────────────────────────────


def test_output_ends_with_newline() -> None:
    out = format_output(_state())
    assert out.endswith("\n")
