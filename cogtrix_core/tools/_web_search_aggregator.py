"""Aggregator for the web_search tool (ADR-0056 stage 1 + 2).

Public surface:

* ``ProviderResult``  — canonical per-result shape every provider's
  ``_search_async`` emits (introduced in PR-B).
* ``RankedResult``    — a deduplicated result with score + domain class.
* ``CoverageInfo``    — operator-facing summary of the fan-out outcome.
* ``aggregate()``     — fan out across providers, dedup, rank, return.

This module does **not** implement the speculative-fetch / ranking-
freeze coordination from ADR-0056. That orchestration belongs in
``cogtrix_core/tools/web_search.py`` (PR-E) which interleaves the aggregator's
output with the fetcher's input. Here, ``aggregate()`` waits for every
provider (within the deadline) and returns a complete ranking.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from cogtrix_core.tools._web_search_domain_class import (
    DomainClass,
    authority_bonus,
    classify_domain,
)

log = logging.getLogger("cogtrix")

# Tracking parameters stripped during canonicalisation. Conservative
# list; extending is a one-line addition.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAM_EXACT: frozenset[str] = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "mc_eid",
        "mc_cid",
        "yclid",
        "_ga",
        "_gl",
        "ref_src",
        "ref_url",
        "icid",
    }
)

# Recency bonus window — results dated within this many days of "now"
# get a small boost. Conservative — we don't want stale news to win on
# a query about *current* affairs simply because it has multiple
# providers covering it.
_RECENCY_WINDOW_DAYS = 30


@dataclass(frozen=True)
class ProviderResult:
    """One search result from one provider (PR-B contract).

    Re-exported here so the rest of the pipeline imports from a single
    module. Identical to PR-B's shape.
    """

    provider: str
    url: str
    title: str
    snippet: str
    published_date: str | None = None


@dataclass(frozen=True)
class RankedResult:
    """A deduplicated result after stage-2 ranking."""

    canonical_url: str
    title: str
    snippet: str
    published_date: str | None
    domain_class: DomainClass
    score: float
    providers: tuple[str, ...]
    """Provider tags that surfaced this URL (post-canonicalisation)."""


@dataclass(frozen=True)
class CoverageInfo:
    """Operator-facing summary of one fan-out outcome."""

    providers_attempted: int
    providers_succeeded: int
    raw_count: int
    distinct_count: int
    per_provider_failures: dict[str, str] = field(default_factory=dict)


# ── URL canonicalisation ──────────────────────────────────────────────


def canonicalise_url(url: str) -> str:
    """Return a canonical form of *url* for dedup + ranking.

    Steps:
      * lowercase scheme + host
      * strip ``www.`` host prefix
      * strip trailing slash from path (but keep "/" if path is empty)
      * drop tracking query params (``utm_*`` + curated exact list)
      * drop fragment (``#section``) — different fragments aren't
        different documents for our purposes
      * preserve the remaining query, sorted for stable comparison

    Returns the input unchanged on parse failure so the caller can use
    the value for logging even if it's malformed.
    """
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return url
    if not parsed.scheme or not parsed.netloc:
        return url

    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    if parsed.query:
        keep_pairs = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if not k.startswith(_TRACKING_PARAM_PREFIXES) and k not in _TRACKING_PARAM_EXACT
        ]
        keep_pairs.sort()
        query = urlencode(keep_pairs)
    else:
        query = ""

    return urlunparse((scheme, host, path, parsed.params, query, ""))


# ── Snippet relevance ────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> set[str]:
    """Lowercase + alphanumeric-token tokeniser for relevance scoring."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _snippet_relevance(query: str, title: str, snippet: str) -> float:
    """Tiny relevance signal: fraction of query tokens that appear in
    title-or-snippet. Range [0, 1]. Stop-words intentionally not
    filtered — the ranker uses this as one term among several."""
    q_tokens = _tokenise(query)
    if not q_tokens:
        return 0.0
    text_tokens = _tokenise(title) | _tokenise(snippet)
    overlap = q_tokens & text_tokens
    return len(overlap) / len(q_tokens)


# ── Recency bonus ────────────────────────────────────────────────────


def _recency_bonus(published_date: str | None) -> float:
    """Tiny bonus for results dated within ``_RECENCY_WINDOW_DAYS``.

    Returns 0 when there's no date or it's malformed. We deliberately
    don't penalise old content — many high-quality docs are dated
    years ago and shouldn't be down-ranked relative to undated pages.
    """
    if not published_date:
        return 0.0
    try:
        from datetime import datetime

        # Accept YYYY-MM-DD or ISO 8601 with time.
        d = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta_days = (now - d).days
        if delta_days < 0:
            return 0.0  # date is in the future; ignore
        if delta_days > _RECENCY_WINDOW_DAYS:
            return 0.0
        # Linear from 0.3 (just published) to 0 (at the window edge).
        return 0.3 * (1.0 - delta_days / _RECENCY_WINDOW_DAYS)
    except (ValueError, TypeError):
        return 0.0


# ── Dedup + scoring core ──────────────────────────────────────────────


def _merge_into_groups(
    raw_results: list[ProviderResult],
) -> dict[str, list[ProviderResult]]:
    """Group provider results by canonical URL."""
    groups: dict[str, list[ProviderResult]] = {}
    for r in raw_results:
        if not r.url:
            continue
        key = canonicalise_url(r.url)
        groups.setdefault(key, []).append(r)
    return groups


def _score_group(query: str, canonical_url: str, group: list[ProviderResult]) -> float:
    """Combined stage-2 score for a deduplicated URL.

    Weights are illustrative for v1; ADR-0056 calibrates them after
    real-world measurement.
    """
    distinct_providers = len({r.provider for r in group})
    consensus = 1.0 * distinct_providers

    # Pick the longest title / snippet across this URL's provider
    # appearances for relevance scoring (more signal to match against).
    best_title = max((r.title for r in group), key=len, default="")
    best_snippet = max((r.snippet for r in group), key=len, default="")
    relevance = _snippet_relevance(query, best_title, best_snippet)

    domain_cls = classify_domain(canonical_url)
    authority = authority_bonus(domain_cls)

    # Use any non-empty published_date — providers don't always agree
    # but we trust whichever surfaced one.
    date = next((r.published_date for r in group if r.published_date), None)
    recency = _recency_bonus(date)

    return consensus + relevance + authority + recency


def _domain_diversity_filter(
    ranked: list[RankedResult], cap_per_domain: int = 2
) -> list[RankedResult]:
    """Cap at ``cap_per_domain`` results per registered domain.

    Preserves rank order; drops the lower-ranked results from any
    domain that exceeds the cap. Keeps top-K diverse so a single
    over-represented site doesn't crowd the synthesis stage.
    """
    counts: dict[str, int] = {}
    kept: list[RankedResult] = []
    for r in ranked:
        host = urlparse(r.canonical_url).netloc
        if counts.get(host, 0) >= cap_per_domain:
            continue
        counts[host] = counts.get(host, 0) + 1
        kept.append(r)
    return kept


# ── Public entry point ────────────────────────────────────────────────

# A provider callable is a coroutine returning ``list[ProviderResult]``.
# The aggregator only cares about the (query) signature for v1; per-
# provider region passthrough is deferred to PR-E.
Provider = Callable[[str], Awaitable[list[ProviderResult]]]


async def aggregate(
    query: str,
    providers: dict[str, Provider],
    *,
    deadline_s: float = 5.0,
    k: int = 6,
    per_provider_max: int = 15,
    cap_per_domain: int = 2,
) -> tuple[list[RankedResult], CoverageInfo]:
    """Fan out across providers, dedup, rank, return top-K.

    Parameters
    ----------
    query
        The user query (used both as the search input and as the
        relevance signal in stage 2).
    providers
        Mapping ``{provider_name: async_callable}``. Each callable
        receives the query and returns a list of ``ProviderResult``.
        Empty mapping → returns empty ranking with zero coverage.
    deadline_s
        Wall-clock budget for the fan-out (default 5s per ADR-0056).
        Providers exceeding this are dropped from the result set;
        their names appear in ``CoverageInfo.per_provider_failures``.
    k
        Top-K size returned to the caller. Domain-diversity filter
        runs before this cap.
    per_provider_max
        Per-provider truncation before merging (default 15 per ADR).
        Prevents one verbose provider from drowning out consensus
        signal.
    cap_per_domain
        Stage-2 domain-diversity cap (default 2 per ADR).

    Returns
    -------
    ``(top_k_ranked, coverage_info)``.
    """
    if not providers:
        return [], CoverageInfo(0, 0, 0, 0, {})

    async def _call(name: str, fn: Provider) -> tuple[str, list[ProviderResult] | BaseException]:
        try:
            return name, await fn(query)
        except BaseException as exc:  # noqa: BLE001 — provider failure is data
            return name, exc

    tasks = [asyncio.create_task(_call(n, fn)) for n, fn in providers.items()]
    try:
        completed_pairs = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=deadline_s,
        )
    except TimeoutError:
        completed_pairs = []
        for t in tasks:
            if t.done():
                try:
                    completed_pairs.append(t.result())
                except BaseException as exc:  # noqa: BLE001
                    # Should not happen because _call swallows; defensive.
                    completed_pairs.append(("?unknown", exc))
            else:
                t.cancel()

    per_provider_failures: dict[str, str] = {}
    raw_results: list[ProviderResult] = []
    succeeded = 0
    for name, payload in completed_pairs:
        if isinstance(payload, BaseException):
            per_provider_failures[name] = type(payload).__name__
            continue
        succeeded += 1
        # Truncate per-provider to prevent dominance.
        truncated = payload[:per_provider_max]
        raw_results.extend(truncated)

    # Providers that didn't return at all (timed out).
    for n in providers.keys():
        if n not in per_provider_failures and all(p[0] != n for p in completed_pairs):
            per_provider_failures[n] = "timeout"

    raw_count = len(raw_results)

    # Stage 2 — group by canonical URL, score, sort.
    groups = _merge_into_groups(raw_results)
    ranked: list[RankedResult] = []
    for canonical_url, group in groups.items():
        score = _score_group(query, canonical_url, group)
        # Pick title + snippet from the longest-snippet variant of this
        # URL group — better signal for the synthesis stage downstream.
        best = max(group, key=lambda r: len(r.snippet or ""))
        ranked.append(
            RankedResult(
                canonical_url=canonical_url,
                title=best.title,
                snippet=best.snippet,
                published_date=next((r.published_date for r in group if r.published_date), None),
                domain_class=classify_domain(canonical_url),
                score=score,
                providers=tuple(sorted({r.provider for r in group})),
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)

    # Stage 2 domain-diversity filter, then top-K.
    diverse = _domain_diversity_filter(ranked, cap_per_domain=cap_per_domain)
    top_k = diverse[:k]

    coverage = CoverageInfo(
        providers_attempted=len(providers),
        providers_succeeded=succeeded,
        raw_count=raw_count,
        distinct_count=len(groups),
        per_provider_failures=per_provider_failures,
    )
    return top_k, coverage


__all__ = [
    "CoverageInfo",
    "Provider",
    "ProviderResult",
    "RankedResult",
    "aggregate",
    "canonicalise_url",
]
