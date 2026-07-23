"""Output formatter for the web_search tool (ADR-0056 stage 6).

Consumes everything the upstream pipeline produced (the query, the
top-K aggregator results, the fetcher outcomes, the extractor's
output, the synthesiser's output, the coverage summary, and the total
wall-clock time) and emits the user-facing Markdown blob per the
ADR's "Output schema" section.

See ``docs/optional/adr/0056-web-search-tool.md`` — section "Output schema" —
for the canonical structure. This module mirrors that schema verbatim.

``compact=True`` drops the per-source extract bodies and the
"Additional sources" tail, producing the ~5KB short form documented in
the ADR (vs ~18KB for the full schema).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tools._web_search_aggregator import (
    CoverageInfo,
    ProviderResult,
    RankedResult,
)
from src.tools._web_search_domain_class import DomainClass, detect_affiliation_disclaimer
from src.tools._web_search_extractor import ExtractedSource
from src.tools._web_search_fetcher import FetchOutcome
from src.tools._web_search_synthesiser import SynthesisResult

# Citation symbols ① through ⑳ — same set the synthesiser uses.
_CIRCLED_INDICES = [chr(0x2460 + i) for i in range(20)]
_MAX_SOURCES_DISPLAYED = 20

_SYNTHESIS_UNAVAILABLE_PREFIX = (
    "## Synthesis unavailable\n"
    "The synthesis layer timed out. Source extracts are included below; "
    "please synthesise from them before responding to the user.\n"
)


@dataclass(frozen=True)
class FormatInput:
    """All the pipeline state the formatter needs.

    Bundled into a dataclass so the public ``web_search()`` entry
    point doesn't have to thread a dozen positional arguments
    through the call. The formatter reads only what's set on the
    object — missing fields fall back to safe defaults.
    """

    query: str
    ranked: list[RankedResult]
    fetched: list[FetchOutcome]
    extracted: list[ExtractedSource]
    snippet_only_tail: list[ProviderResult]
    """URLs that survived dedup but didn't make top-K — emitted as the
    "Additional sources" tail. Empty in compact mode."""
    synthesis: SynthesisResult | None
    coverage: CoverageInfo
    total_wall_ms: int
    synthesis_model_name: str | None = None
    compact: bool = False
    cache_hit: bool = False
    cache_retrieved_at: str | None = None


def format_output(state: FormatInput) -> str:
    """Render *state* as the ADR-0056 output Markdown blob."""
    out: list[str] = [f"# Research: {state.query}", ""]

    # Synthesis block — either the validated synthesis text or the
    # explicit "Synthesis unavailable" prefix that signals to the
    # calling agent that it must synthesise from the extracts below.
    if state.synthesis is not None and state.synthesis.text:
        out.append(state.synthesis.text.rstrip())
        out.append("")
    elif state.synthesis is not None:
        # Synthesis ran but failed all validation. Prefix with the
        # explicit agent-guidance marker per ADR-0056.
        out.append(_SYNTHESIS_UNAVAILABLE_PREFIX.rstrip())
        out.append("")

    # Sources index — flat list with citation indices, domain-class,
    # recency tag, title, URL. Capped at 20 entries per ADR.
    out.append("## Sources")
    sources_emitted, overflow = _emit_sources(out, state)
    out.append("")

    # Per-source extracts — only in non-compact mode.
    if not state.compact and state.extracted:
        for i, source in enumerate(state.extracted):
            if i >= len(_CIRCLED_INDICES):
                break
            _emit_extract_block(out, _CIRCLED_INDICES[i], source)

    # Additional sources tail — snippet-only URLs that didn't make
    # top-K. Skipped in compact mode.
    if not state.compact and state.snippet_only_tail:
        _emit_additional_sources(out, state.snippet_only_tail, start_index=sources_emitted)

    # Coverage block — 4-line operator summary.
    _emit_coverage(out, state, overflow=overflow)

    return "\n".join(out).rstrip() + "\n"


# ── Section emitters ─────────────────────────────────────────────────


def _emit_sources(out: list[str], state: FormatInput) -> tuple[int, int]:
    """Emit the Sources index block.

    Returns (sources_emitted_count, overflow_count) — the formatter
    uses overflow_count to mention truncation in the Coverage block
    if >20 sources would have appeared.
    """
    capacity = _MAX_SOURCES_DISPLAYED
    overflow = 0
    sources_emitted = 0

    primary = state.extracted or [
        # Use ranked directly when no extracts (snippet-only path).
        _adapt_ranked_to_source(r)
        for r in state.ranked
    ]

    for _i, source in enumerate(primary):
        if sources_emitted >= capacity:
            overflow += 1
            continue
        idx = _CIRCLED_INDICES[sources_emitted]
        ranked = (
            source.fetch_outcome.ranked
            if isinstance(source, ExtractedSource)
            else source  # _AdaptedRanked
        )
        recency = ranked.published_date or "undated"
        domain = _domain_of(ranked.canonical_url)
        title = ranked.title or "(no title)"
        out.append(f"{idx} {domain} [{ranked.domain_class} · {recency}]")
        out.append(f"   {title}")
        out.append(f"   {ranked.canonical_url}")
        # #1842: surface a content-declared affiliation disclaimer right
        # under the citation so the agent cannot present an
        # official-looking-but-unaffiliated source as authoritative.
        if isinstance(source, ExtractedSource):
            disclaimer = detect_affiliation_disclaimer(source.extracted_text or "")
            if disclaimer:
                out.append(
                    "   ⚠ self-identifies as UNAFFILIATED/UNOFFICIAL — not an official source"
                )
        sources_emitted += 1

    return sources_emitted, overflow


def _emit_extract_block(out: list[str], citation_idx: str, source: ExtractedSource) -> None:
    """Emit one per-source body block under the Sources index."""
    status = source.status
    if source.extracted_text:
        if status == "extracted-truncated":
            note = " (truncated)"
        elif status == "extracted-raw-fallback":
            note = " (raw-text fallback)"
        else:
            note = ""
        out.append(f"### {citation_idx}{note}")
        out.append(source.extracted_text.rstrip())
        out.append("")
    elif status in ("snippet-only", "skipped"):
        reason = source.fetch_outcome.error or status
        out.append(f"### {citation_idx} (snippet-only)")
        out.append(f"_Fetch did not yield extractable content ({reason})._")
        out.append("")


def _emit_additional_sources(
    out: list[str], tail: list[ProviderResult], *, start_index: int
) -> None:
    """Emit the "Additional sources" snippet-only tail."""
    capacity_left = _MAX_SOURCES_DISPLAYED - start_index
    if capacity_left <= 0 or not tail:
        return
    out.append("## Additional sources (snippet-only, not fetched)")
    for i, pr in enumerate(tail[:capacity_left]):
        idx_pos = start_index + i
        if idx_pos >= len(_CIRCLED_INDICES):
            break
        idx = _CIRCLED_INDICES[idx_pos]
        domain = _domain_of(pr.url)
        recency = pr.published_date or "undated"
        snippet = (pr.snippet or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        out.append(f"{idx} {domain} [{recency}] — {pr.url}")
        if snippet:
            out.append(f"   {snippet}")
    out.append("")


def _emit_coverage(out: list[str], state: FormatInput, *, overflow: int = 0) -> None:
    """Emit the 4-line Coverage block."""
    out.append("## Coverage")
    if state.cache_hit:
        ts = state.cache_retrieved_at or "(unknown time)"
        out.append(f"- Cache hit; original retrieved at {ts}.")
        out.append(f"- Total wall time: {state.total_wall_ms / 1000:.1f}s.")
        return

    # Line 1 — providers
    coverage = state.coverage
    out.append(
        f"- Searched: {coverage.providers_succeeded} of "
        f"{coverage.providers_attempted} providers responded; "
        f"{coverage.raw_count} raw results, {coverage.distinct_count} distinct "
        f"after dedupe."
    )
    # Line 2 — fetches
    fetched_count = sum(1 for f in state.fetched if f.status in ("fetched", "fetched-with-warning"))
    snippet_only_count = sum(1 for f in state.fetched if f.status == "snippet-only")
    selected = len(state.fetched)
    out.append(
        f"- Fetched: {fetched_count}/{selected} selected sources successful; "
        f"{snippet_only_count} fell back to snippet."
    )
    # Line 3 — synthesis
    if state.synthesis and state.synthesis.text:
        model_label = state.synthesis_model_name or state.synthesis.model_used or "primary"
        out.append(f"- Synthesis: {model_label}; {state.synthesis.elapsed_ms / 1000:.1f}s.")
    elif state.synthesis is not None:
        out.append(f"- Synthesis: failed ({state.synthesis.reason or 'unknown'}).")
    else:
        out.append("- Synthesis: skipped.")
    # Line 4 — wall time + overflow note
    line4 = f"- Total wall time: {state.total_wall_ms / 1000:.1f}s."
    if overflow:
        line4 += f"  +{overflow} more sources dropped from this view."
    out.append(line4)


# ── Helpers ──────────────────────────────────────────────────────────


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except (TypeError, ValueError):
        return "(unknown)"
    if host.startswith("www."):
        host = host[4:]
    return host or "(unknown)"


@dataclass(frozen=True)
class _AdaptedRanked:
    """Minimal shape adapter when there are no extracts to iterate.

    The Sources section can still be emitted from RankedResult alone.
    """

    canonical_url: str
    title: str
    published_date: str | None
    domain_class: DomainClass


def _adapt_ranked_to_source(r: RankedResult) -> _AdaptedRanked:
    return _AdaptedRanked(
        canonical_url=r.canonical_url,
        title=r.title,
        published_date=r.published_date,
        domain_class=r.domain_class,
    )


__all__ = ["FormatInput", "format_output"]
