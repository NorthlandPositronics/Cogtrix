"""Tests for cogtrix_core/tools/_web_search_aggregator.py — stage 1+2 fan-out
+ consensus rank (ADR-0056 PR-C)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

import pytest

from cogtrix_core.tools._web_search_aggregator import (
    CoverageInfo,
    ProviderResult,
    RankedResult,
    aggregate,
    canonicalise_url,
)
from cogtrix_core.tools._web_search_domain_class import DomainClass

# ── canonicalise_url ──────────────────────────────────────────────────


class TestCanonicaliseUrl:
    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("https://Example.com/Path/", "https://example.com/Path"),
            ("HTTP://www.Example.com/", "http://example.com/"),
            ("https://example.com/path#fragment", "https://example.com/path"),
            (
                "https://example.com/article?utm_source=newsletter&id=42",
                "https://example.com/article?id=42",
            ),
            (
                "https://example.com/?fbclid=abc&utm_campaign=x&q=hello",
                "https://example.com/?q=hello",
            ),
            # Empty path → "/" so both equivalent inputs map to the
            # same canonical key.
            ("https://example.com", "https://example.com/"),
            ("https://example.com/", "https://example.com/"),
            # Tracking-only query collapses to no query.
            ("https://example.com/p?utm_medium=x", "https://example.com/p"),
        ],
    )
    def test_canonical_forms(self, raw: str, canonical: str) -> None:
        assert canonicalise_url(raw) == canonical

    def test_two_phrasings_canonicalise_equal(self) -> None:
        a = canonicalise_url("https://WWW.Example.com/path/?utm_source=x#anchor")
        b = canonicalise_url("https://example.com/path")
        assert a == b

    def test_invalid_url_returned_unchanged(self) -> None:
        assert canonicalise_url("not a url") == "not a url"
        assert canonicalise_url("") == ""


# ── Provider fakes ────────────────────────────────────────────────────


def _fake_provider(
    name: str, results: Sequence[tuple[str, str, str, str | None]]
) -> Callable[[str], Awaitable[list[ProviderResult]]]:
    """Build an async callable returning canned ProviderResult list."""

    async def call(_query: str) -> list[ProviderResult]:
        return [
            ProviderResult(provider=name, url=url, title=title, snippet=snip, published_date=date)
            for (url, title, snip, date) in results
        ]

    return call


def _slow_provider(
    name: str, delay_s: float, results: Sequence[tuple[str, str, str, str | None]]
) -> Callable[[str], Awaitable[list[ProviderResult]]]:
    async def call(_query: str) -> list[ProviderResult]:
        await asyncio.sleep(delay_s)
        return [
            ProviderResult(provider=name, url=url, title=title, snippet=snip, published_date=date)
            for (url, title, snip, date) in results
        ]

    return call


def _failing_provider(
    exc: type[BaseException] = RuntimeError,
) -> Callable[[str], Awaitable[list[ProviderResult]]]:
    async def call(_query: str) -> list[ProviderResult]:
        raise exc("boom")

    return call


# ── Fan-out + ranking ────────────────────────────────────────────────


class TestAggregate:
    @pytest.mark.asyncio
    async def test_empty_providers(self) -> None:
        ranked, coverage = await aggregate("query", {})
        assert ranked == []
        assert coverage == CoverageInfo(0, 0, 0, 0, {})

    @pytest.mark.asyncio
    async def test_single_provider_single_result(self) -> None:
        providers = {
            "ddg": _fake_provider(
                "ddg",
                [("https://example.com/a", "Title A", "Body A", None)],
            )
        }
        ranked, coverage = await aggregate("query", providers)
        assert len(ranked) == 1
        assert ranked[0].canonical_url == "https://example.com/a"
        assert ranked[0].providers == ("ddg",)
        assert coverage.providers_succeeded == 1
        assert coverage.distinct_count == 1
        assert coverage.raw_count == 1

    @pytest.mark.asyncio
    async def test_consensus_boost_for_two_providers(self) -> None:
        """Same URL from two providers ranks above a single-source URL."""
        shared_url = "https://example.com/shared"
        providers = {
            "ddg": _fake_provider(
                "ddg",
                [
                    (shared_url, "Shared", "Body shared", None),
                    ("https://example.org/only-ddg", "Only", "OnlyDDG", None),
                ],
            ),
            "brave": _fake_provider(
                "brave",
                [(shared_url, "Shared", "Body shared", None)],
            ),
        }
        ranked, _coverage = await aggregate("query", providers)
        # Shared URL gets consensus=2; single-source gets 1.
        assert ranked[0].canonical_url == shared_url
        assert set(ranked[0].providers) == {"ddg", "brave"}

    @pytest.mark.asyncio
    async def test_canonicalisation_dedups_across_providers(self) -> None:
        """``https://www.example.com/?utm=x`` ≡ ``https://example.com``"""
        providers = {
            "ddg": _fake_provider(
                "ddg",
                [("https://www.example.com/?utm_source=newsletter", "T", "S", None)],
            ),
            "brave": _fake_provider(
                "brave",
                [("https://example.com/", "T", "S", None)],
            ),
        }
        ranked, coverage = await aggregate("query", providers)
        assert len(ranked) == 1
        assert coverage.distinct_count == 1
        assert set(ranked[0].providers) == {"ddg", "brave"}

    @pytest.mark.asyncio
    async def test_failing_provider_recorded_others_succeed(self) -> None:
        providers = {
            "ddg": _fake_provider("ddg", [("https://example.com/a", "T", "S", None)]),
            "brave": _failing_provider(RuntimeError),
        }
        ranked, coverage = await aggregate("query", providers)
        assert len(ranked) == 1
        assert coverage.providers_succeeded == 1
        assert coverage.providers_attempted == 2
        assert coverage.per_provider_failures == {"brave": "RuntimeError"}

    @pytest.mark.asyncio
    async def test_per_provider_truncation(self) -> None:
        """Provider returning 20 results gets capped at per_provider_max=15."""
        many = [(f"https://example.com/{i}", "T", "S", None) for i in range(20)]
        providers = {"ddg": _fake_provider("ddg", many)}
        _ranked, coverage = await aggregate("query", providers, per_provider_max=15, k=20)
        assert coverage.raw_count == 15

    @pytest.mark.asyncio
    async def test_top_k_cap(self) -> None:
        many = [(f"https://example{i}.com/", "T", "S", None) for i in range(15)]
        providers = {"ddg": _fake_provider("ddg", many)}
        ranked, _ = await aggregate("query", providers, k=6, cap_per_domain=2)
        assert len(ranked) == 6

    @pytest.mark.asyncio
    async def test_domain_diversity_cap(self) -> None:
        """No more than 2 results per registered domain in top-K."""
        same_domain = [(f"https://example.com/page-{i}", "T", "S", None) for i in range(5)]
        providers = {"ddg": _fake_provider("ddg", same_domain)}
        ranked, _ = await aggregate("query", providers, k=10, cap_per_domain=2)
        # All same domain; cap applies.
        assert len(ranked) == 2

    @pytest.mark.asyncio
    async def test_authority_bonus_lifts_wiki_over_blog(self) -> None:
        """Two URLs, identical otherwise; wikipedia.org outranks
        random blog domain via the authority bonus."""
        providers = {
            "ddg": _fake_provider(
                "ddg",
                [
                    ("https://random-blog.example.com/post", "T", "S", None),
                    ("https://en.wikipedia.org/wiki/Topic", "T", "S", None),
                ],
            )
        }
        ranked, _ = await aggregate("query", providers)
        assert ranked[0].canonical_url == "https://en.wikipedia.org/wiki/Topic"

    @pytest.mark.asyncio
    async def test_domain_class_attached(self) -> None:
        providers = {
            "ddg": _fake_provider("ddg", [("https://en.wikipedia.org/wiki/X", "T", "S", None)])
        }
        ranked, _ = await aggregate("query", providers)
        assert ranked[0].domain_class == DomainClass.WIKI_ENCYCLOPEDIA

    @pytest.mark.asyncio
    async def test_relevance_uses_query_tokens(self) -> None:
        """A result whose title contains the query tokens outranks
        an off-topic one (all else equal)."""
        providers = {
            "ddg": _fake_provider(
                "ddg",
                [
                    ("https://example.com/off-topic", "Cats", "Cat trivia", None),
                    ("https://example.org/on-topic", "Linkerd guide", "Linkerd 2.18", None),
                ],
            )
        }
        ranked, _ = await aggregate("query about Linkerd", providers)
        assert ranked[0].canonical_url == "https://example.org/on-topic"

    @pytest.mark.asyncio
    async def test_slow_provider_dropped_after_deadline(self) -> None:
        """Provider exceeding the deadline counts as a timeout failure."""
        providers = {
            "fast": _fake_provider("fast", [("https://example.com/a", "T", "S", None)]),
            "slow": _slow_provider("slow", 0.5, [("https://example.com/b", "T", "S", None)]),
        }
        ranked, coverage = await aggregate("query", providers, deadline_s=0.1)
        # Only the fast provider's result is present.
        urls = {r.canonical_url for r in ranked}
        assert "https://example.com/a" in urls
        assert "https://example.com/b" not in urls
        assert "slow" in coverage.per_provider_failures


# ── RankedResult contract ────────────────────────────────────────────


class TestRankedResultShape:
    @pytest.mark.asyncio
    async def test_picks_longest_snippet_as_representative(self) -> None:
        providers = {
            "ddg": _fake_provider("ddg", [("https://example.com/a", "Short", "tiny", None)]),
            "tavily": _fake_provider(
                "tavily",
                [
                    (
                        "https://example.com/a",
                        "Long",
                        "A much longer extract of the page that gives more signal.",
                        None,
                    )
                ],
            ),
        }
        ranked, _ = await aggregate("query", providers)
        assert len(ranked) == 1
        assert "longer extract" in ranked[0].snippet

    @pytest.mark.asyncio
    async def test_published_date_carried_through(self) -> None:
        providers = {
            "ddg": _fake_provider("ddg", [("https://example.com/x", "T", "S", None)]),
            "google": _fake_provider("google", [("https://example.com/x", "T", "S", "2026-05-01")]),
        }
        ranked, _ = await aggregate("query", providers)
        assert isinstance(ranked[0], RankedResult)
        assert ranked[0].published_date == "2026-05-01"
