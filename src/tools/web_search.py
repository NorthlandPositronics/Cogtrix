"""
DuckDuckGo Web Search — free, keyless web search.

DuckDuckGo is a privacy-focused search engine. Since Bug D
(heap-corruption aborts caused by the ``primp`` Rust HTTP client
embedded in the ``ddgs`` library), we no longer use ``ddgs``. We
scrape ``html.duckduckgo.com`` directly through ``curl_cffi``
(libcurl + BoringSSL + browser-fingerprint impersonation) — see
``src/tools/_ddg.py`` for the fetch and parser.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──q──→  │  curl_cffi      │──q──→  │  DuckDuckGo  │
    │          │        │  + HTML parser  │        │  html.duck…  │
    │          │←─────  │  (sandboxed in  │←─html  │              │
    │          │  list  │   subprocess)   │        │              │
    └──────────┘        └─────────────────┘        └──────────────┘

Backward-compatible exports:
    search_web   — General web search with snippets (sync shim).
    search_news  — Stub: DDG news scraping not yet ported off ddgs.

Subprocess sandbox: kept as a safety net for the first release of
the curl_cffi swap. curl_cffi has a better stability reputation
than primp, but it is still native code (patched libcurl); we want
one release of clean prod data before retiring the sandbox.
"""

import contextvars

# ``DDG_AVAILABLE`` reflects whether the curl_cffi scraper is
# importable, WITHOUT actually importing curl_cffi in the parent
# process. ``curl_cffi`` ships with BoringSSL as its TLS backend,
# and importing it into the agent process — which also has httpx
# (OpenSSL via cffi) and other native crypto deps loaded — produced
# a glibc ``corrupted size vs. prev_size`` heap abort during the
# cogtrix42 test run. The subprocess sandbox is the only place
# that actually needs curl_cffi, so we defer the import to that
# scope and check availability passively via ``find_spec``.
import importlib.util as _ddg_importlib_util
import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.concurrency import invoke_with_timeout
from src.tools._web_search_aggregator import ProviderResult
from src.tools.delegate import register_tool_categories
from src.tools.error_sanitizer import sanitize_search_error as _sanitize_search_error

DDG_AVAILABLE = _ddg_importlib_util.find_spec("curl_cffi") is not None

# Backwards-compatible alias for the old name. The aggregator and
# tests reference ``DDGS_AVAILABLE`` in a few places; retaining the
# alias keeps those untouched.
DDGS_AVAILABLE = DDG_AVAILABLE


# ── Stderr suppression context manager ──────────────────────────────
# Retained for back-compat; some legacy paths used it to mute primp's
# native stderr noise. curl_cffi doesn't write directly to fd 2, so
# this is effectively a no-op for current callers, but the context
# manager itself is harmless and tests reference it.

_stderr_lock = threading.Lock()


@contextmanager
def _suppress_native_stderr() -> Generator[None]:
    """Redirect fd 2 (stderr) to /dev/null for the duration of the block.

    Originally added to silence ``primp``'s native gzip warnings (the
    Rust HTTP client wrote directly to fd 2, bypassing Python's
    ``sys.stderr``). curl_cffi doesn't have this issue, but we keep
    the helper as a defensive no-op for any future native dep that
    misbehaves the same way.
    """
    try:
        with _stderr_lock:
            saved_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        with _stderr_lock:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def extract_domain(url: str) -> str:
    """Extract domain from URL for source tracking.

    Args:
        url: The URL to extract domain from

    Returns:
        Domain name without www. prefix, or "unknown" if extraction fails
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "unknown"
        domain = parsed.netloc
        # Remove www. prefix for consistency
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


class WebSearchInput(BaseModel):
    """Input schema for web search."""

    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return (max 10)")
    region: str = Field(
        default="wt-wt",
        description="Region for search (e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide)",
    )


class NewsSearchInput(BaseModel):
    """Input schema for news search."""

    query: str = Field(description="The news search query")
    num_results: int = Field(default=5, description="Number of results to return (max 10)")
    timelimit: str | None = Field(
        default="w",
        description="Time limit: 'd' (day), 'w' (week), 'm' (month)",
    )


def search_web(query: str, num_results: int = 5, region: str = "wt-wt") -> str:
    """Legacy sync DuckDuckGo search (retained for back-compat).

    Not in the agent catalogue since PR-G; kept importable for any
    external caller. Routes through the same curl_cffi scraper as
    the async ``_search_async`` path used by the agent.

    Returns
    -------
    Formatted human-readable string with titles/URLs/snippets, or an
    error message starting with ``"Error:"``.
    """
    if not DDG_AVAILABLE:
        return "Error: DuckDuckGo search not available. Run: uv add curl_cffi"

    if not query.strip():
        return "Error: Empty search query"

    num_results = min(max(1, num_results), 10)  # Clamp between 1 and 10

    # Bug D / cogtrix46 hardening (2026-05-20): dispatch through the
    # subprocess-isolated `_ddg_subprocess_call` path so curl_cffi is
    # never imported into the caller's process. The previous in-process
    # `fetch_ddg_html` call would pull libcurl + BoringSSL into a
    # process that also has httpx (Python stdlib OpenSSL) loaded — the
    # two TLS backends corrupt each other's malloc state and the
    # process dies with a glibc heap abort. Routing through the
    # subprocess sandbox is non-negotiable here.
    import asyncio

    try:
        try:
            raw = asyncio.run(_ddg_subprocess_call(query, region, num_results))
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            # Existing loop on the calling thread — escape onto the
            # centralized shared pool so we don't dead-lock the caller's
            # loop.  Migrated under #1903; the previous pattern was
            # ``with ThreadPoolExecutor(...) as pool: ... .result()``
            # which carries the shutdown(wait=True)-on-__exit__ footgun
            # the policy doc forbids.  30s is generous for a single DDG
            # provider call (web_search has a 25s internal budget).
            # ``invoke_with_timeout`` is generic over ``Callable[..., T]``;
            # pyright cannot propagate ``asyncio.run``'s own TypeVar through
            # the helper signature, so the call site narrows the result
            # explicitly to the shape ``_ddg_subprocess_call`` returns.
            raw = cast(
                list[dict[str, Any]],
                invoke_with_timeout(
                    asyncio.run,
                    _ddg_subprocess_call(query, region, num_results),
                    timeout=30,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        return f"Error searching: {_sanitize_search_error(exc, context='DuckDuckGo')}"

    results = raw[:num_results]
    if not results:
        return f"No results found for: {query}"

    output = [f"Search results for: {query}\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("href") or result.get("link") or "No URL"
        snippet = result.get("body") or result.get("snippet") or "No description"

        # Extract domain metadata for source tracking.
        domain = extract_domain(url)

        # Truncate long snippets — keep enough for the LLM to get the
        # key facts without having to fill gaps from memory.
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."

        output.append(f"{i}. {title}")
        output.append(f"   URL: {url}")
        output.append(f"   Domain: {domain}")
        output.append(f"   {snippet}")
        output.append("")

    return "\n".join(output)


# ── DDG subprocess isolation (Bug D safety net) ─────────────────────
#
# Phase 1 (PR #1686) introduced a subprocess sandbox around the
# ``ddgs``/``primp`` call to contain the heap-corruption aborts
# documented in the cogtrix34/35/36 incident reports. Phase 2 (this
# PR) replaces ``ddgs``/``primp`` itself with ``curl_cffi`` —
# libcurl + BoringSSL with browser-fingerprint impersonation — which
# has a substantially better stability reputation.
#
# We keep the subprocess sandbox as a safety net for the first
# release of the curl_cffi swap. curl_cffi is still native code; any
# heap bug there would have the same blast radius primp had. After
# a month of clean production data we can retire the sandbox and
# call curl_cffi directly from the asyncio event loop.

_DDG_SUBPROCESS_TIMEOUT_S = 10.0

# The worker script imports the project's ``_ddg`` module rather
# than inlining the fetch + parse logic. This keeps the worker
# minimal and lets ``tests/tools/test_ddg.py`` cover the parser /
# fetcher in isolation. The subprocess inherits the parent's
# Python executable (and therefore its sys.path), so ``from
# src.tools._ddg import …`` resolves correctly.
_DDG_WORKER_SCRIPT = r"""
import json
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")

try:
    from src.tools._ddg import DDGFetchError, fetch_ddg_html, parse_ddg_html
except ImportError as exc:
    sys.stdout.write(json.dumps({"error": f"ddg-module-import: {exc}"}))
    sys.exit(2)

if len(sys.argv) < 4:
    sys.stdout.write(json.dumps({"error": "missing-args"}))
    sys.exit(2)

query = sys.argv[1]
region = sys.argv[2]
try:
    num_results = int(sys.argv[3])
except ValueError:
    sys.stdout.write(json.dumps({"error": "bad-num-results"}))
    sys.exit(2)

try:
    body = fetch_ddg_html(query, region=region, num_results=num_results)
    results = parse_ddg_html(body)[:num_results]
    sys.stdout.write(json.dumps({"results": results}))
except DDGFetchError as exc:
    sys.stdout.write(json.dumps({"error": f"ddg-fetch: {exc}"}))
    sys.exit(1)
except Exception as exc:
    sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(1)
"""


async def _ddg_subprocess_call(query: str, region: str, num_results: int) -> list[dict[str, Any]]:
    """Run one DDG search in an isolated subprocess.

    Returns the raw result list. Raises ``RuntimeError`` on timeout,
    non-zero exit, malformed JSON, or worker-reported error — the
    stage-1 aggregator treats these as a per-provider failure and
    continues with the other providers.
    """
    import asyncio
    import json
    import sys as _sys

    proc = await asyncio.create_subprocess_exec(
        _sys.executable,
        "-c",
        _DDG_WORKER_SCRIPT,
        query,
        region,
        str(num_results),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_DDG_SUBPROCESS_TIMEOUT_S
        )
    except TimeoutError as exc:
        # Send SIGKILL — primp can be unresponsive to SIGTERM after
        # a heap corruption. Waiting for the kill prevents a zombie.
        proc.kill()
        await proc.wait()
        raise RuntimeError("DDG subprocess timed out") from exc

    if proc.returncode != 0:
        # Returncode -11 (SIGSEGV) / -6 (SIGABRT) is the heap-
        # corruption case we're guarding against — log it explicitly
        # so operators see "this is why ddgs went away" instead of a
        # silent fallback.
        log = logging.getLogger("cogtrix")
        if proc.returncode in (-6, -11):
            log.warning(
                "DDG subprocess aborted (signal %d) — primp heap corruption suspected",
                -proc.returncode,
            )
        else:
            log.debug("DDG subprocess exited %d", proc.returncode)
        raise RuntimeError(f"DDG subprocess exit {proc.returncode}")

    try:
        payload = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DDG subprocess invalid JSON: {exc}") from exc

    if "error" in payload:
        raise RuntimeError(f"DDG worker error: {payload['error']}")

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("DDG worker returned non-list results")
    return results


async def _search_async(
    query: str, num_results: int = 5, region: str = "wt-wt"
) -> list[ProviderResult]:
    """Async DDG search returning ProviderResult list (ADR-0056 PR-B).

    Runs the ``curl_cffi``-based DDG HTML scraper in a *subprocess*
    so any future native-code crash (e.g. in libcurl/BoringSSL)
    can't take down the agent. Raises on provider errors so the
    stage-1 aggregator (PR-C) can catch and sanitise per #1589
    conventions; returns ``[]`` when the provider reports zero
    results or the query is empty.
    """
    if not DDG_AVAILABLE:
        raise RuntimeError("DuckDuckGo scraper not installed (missing curl_cffi)")
    if not query.strip():
        return []

    num_results = min(max(1, num_results), 10)

    raw = await _ddg_subprocess_call(query, region, num_results)
    return [
        ProviderResult(
            provider="ddg",
            url=r.get("href") or r.get("link") or "",
            title=r.get("title") or "",
            snippet=r.get("body") or r.get("snippet") or "",
            published_date=None,
        )
        for r in raw
        if r.get("href") or r.get("link")
    ]


def search_news(query: str, num_results: int = 5, timelimit: str | None = "w") -> str:
    """Legacy sync DuckDuckGo news search — not currently supported.

    The ``ddgs``-based news search was removed when ``ddgs``/``primp``
    were retired (Bug D phase 2 / PR-port-from-ddgs). DDG's news
    endpoint has a separate HTML shape from ``/html/`` and we haven't
    ported the scraper across yet. ``search_news`` is not in the
    agent catalogue (removed in PR-G), so this only affects external
    importers — none exist in the codebase.

    Returns
    -------
    A user-facing error string. Same shape as the other failure
    paths so any rare external caller doesn't see a Python-level
    exception.
    """
    # Quiet ``_unused`` lints — these parameters define the public
    # signature even when the function is a stub.
    _ = (timelimit, num_results)
    if not query.strip():
        return "Error: Empty search query"
    return (
        "Error: DuckDuckGo news search is no longer supported. The "
        "ddgs/primp dependency was retired (Bug D). Use the `web_search` "
        "tool for research queries — it consults DuckDuckGo plus the "
        "configured keyed providers (Tavily, Brave, etc.) in parallel."
    )


# ─────────────────────────────────────────────────────────────────────
# ADR-0056 PR-E: public web_search() tool — multi-provider fan-out
# pipeline. The existing search_web / search_news tools stay alongside
# (deprecated in PR-F + PR-G); web_search is the new primary research
# entry point.
# ─────────────────────────────────────────────────────────────────────


class WebSearchToolInput(BaseModel):
    """Input schema for the universal web_search tool."""

    query: str = Field(
        description=(
            "The research query — keywords and proper nouns the search "
            "engine should match. KEEP IT TIGHT: include only terms the "
            "search index can rank pages on (product names, brands, "
            "locations, model numbers, dates).\n"
            "\n"
            "DO NOT put any of the following in a query:\n"
            "  • EXCLUSIONS — user constraints like 'except the Praterstern "
            "shop' or 'not from Amazon' are POST-FILTERS on the results, "
            "not search terms. Putting words like 'excluded', 'not', "
            "'except', 'without X' in the query makes DDG search for "
            "those literal words and corrupts the ranking. Drop the "
            "excluded entity from the query and apply the constraint "
            "yourself when reading the results.\n"
            "  • CONVERSATIONAL SCAFFOLDING — phrases like 'find me', "
            "'help me discover', 'I want to know about' are filler.\n"
            "  • ANSWER CONSTRAINTS — formats like 'in EUR', 'in markdown', "
            "'with citations' belong in your reasoning, not in the query.\n"
            "\n"
            "Examples (✓ = good, ✗ = wastes tokens or skews ranking):\n"
            "  ✓ 'Soudal Fix All Silirub 1GH-EJ4 Vienna retailer price'\n"
            "  ✗ 'find me Soudal Fix All Silirub 1GH-EJ4 in Vienna except "
            "the PowerTool shop by Praterstern' (the model uses 'except' "
            "/ 'PowerTool' / 'Praterstern' as RANKING TERMS — the opposite "
            "of the user's intent)."
        )
    )
    depth: int = Field(
        default=3,
        description=(
            "Top-K sources to fetch + extract (1-10). Higher = more breadth, "
            "longer wall time. Extraction runs in a ``ProcessPoolExecutor`` "
            "(PR #1716), so pages are parsed in true parallel; the in-process "
            "``_LXML_LOCK`` in ``_web_search_extractor.py`` is retained as an "
            "unused export for back-compat with callers that still imported it. "
            "depth=3 is therefore a latency default (cold-cache TTFB), not a "
            "serialisation workaround. Set depth explicitly to 5-10 when "
            "research breadth genuinely needs it (deep-research mode, "
            "competitor scans, etc.) and the latency hit is acceptable."
        ),
    )
    region: str = Field(
        default="wt-wt",
        description="Region hint for providers that accept one (e.g. ddg).",
    )
    compact: bool = Field(
        default=False,
        description="When True, drop per-source extracts and the Additional Sources tail. Saves ~13KB.",
    )


# Outer ceiling — wraps the entire pipeline so no single stage can
# blow past the budget.
#
# Stage budgets sum:
#   stage 1+2 aggregator: 5 s
#   stage 3 fetcher:      ~7 s (6 s deadline + 1 s per-task safety margin)
#   stage 4 extractor:    ~2 s (parallel per page)
#   stage 5 synthesiser:  7 s primary + 5 s fallback
#   stage 6 formatter:    <100 ms
#
# Worst case primary path: 5 + 7 + 2 + 7 = 21 s. Worst case with
# synthesis fallback: 21 + 5 = 26 s. The previous 15 s ceiling was
# under-budgeted: when stages 1-4 used most of their slots, the
# 15 s wait_for fired *during* stage 5, returning the "pipeline
# exceeded its 15s budget" fallback even though synthesis was
# actively making progress (Bug G from cogtrix39).
#
# 25 s gives synthesis-primary a fair chance even when prior stages
# took their full budgets; the fallback retry only fires when the
# primary failed validation, and that path is already best-effort.
# Cache-hit short-circuit at the top of web_search() means repeat
# queries pay zero of this budget.
_WEB_SEARCH_HARD_DEADLINE_S = 25.0

# Cache-hit marker — synthesised on the way out so the formatter can
# render the "Cache hit; original retrieved at …" Coverage line.
_CACHE_TIMESTAMP_KEY = "__cache_retrieved_at__"

# Stage-5 LLM injection. Per-run state lives in ContextVars so that
# concurrent sessions (multi-tenant API) don't bleed LLMs into each
# other. The single-tenant CLI is just the degenerate case where only
# one scope is ever active. Without an active LLM the pipeline still
# runs but skips synthesis (Sources-only output).
_synthesis_llm_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "web_search_synthesis_llm", default=None
)
_synthesis_fallback_llm_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "web_search_synthesis_fallback_llm", default=None
)


def set_synthesis_llm(llm: Any, fallback_llm: Any | None = None) -> None:
    """Set the stage-5 synthesiser LLM(s) in the current context.

    Affects only the current async context / thread. Used by tests and
    by callers that don't have a natural scope (the running event loop
    is the scope). For per-run scoping, prefer ``synthesis_llm_scope``.
    """
    _synthesis_llm_var.set(llm)
    _synthesis_fallback_llm_var.set(fallback_llm)


@contextmanager
def synthesis_llm_scope(
    llm: Any,
    fallback_llm: Any | None = None,
) -> Generator[None]:
    """Scope the synthesis LLM to a single agent run.

    Single-tenant and multi-tenant share this entry point: the
    orchestrator wraps each ``run_agent`` invocation in this scope, so
    web_search reads the right LLM for the currently-running session
    regardless of how many other sessions are active concurrently.

    The scope resets on exit, including on exception, so a crashed
    run never leaks its LLM into the next.
    """
    primary_token = _synthesis_llm_var.set(llm)
    fallback_token = _synthesis_fallback_llm_var.set(fallback_llm)
    try:
        yield
    finally:
        _synthesis_fallback_llm_var.reset(fallback_token)
        _synthesis_llm_var.reset(primary_token)


async def web_search(
    query: str,
    depth: int = 3,
    region: str = "wt-wt",
    compact: bool = False,
) -> str:
    """Run the full ADR-0056 web_search pipeline for *query*.

    Stages:
      0. Per-process query cache lookup (5min TTL).
      1+2. Multi-provider fan-out + consensus rank (aggregator).
      3. Speculative fetch of top-K (fetcher).
      4. trafilatura HTML→Markdown extract (extractor).
      5. LLM-as-synthesiser stage (synthesiser).
      6. Markdown format (formatter).

    Returns the assembled output as a single string. The hard outer
    deadline is 15s; on outer-deadline-hit the pipeline returns the
    "Synthesis unavailable" fallback shape with whatever stages
    completed in time.

    This function is the async entry point. The LangChain @tool wrapper
    below adapts it for the sync registry surface.
    """
    import asyncio
    import time
    from datetime import datetime

    from src.tools._web_search_cache import cache_get, cache_put

    started_at = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started_at) * 1000)

    if not query or not query.strip():
        return "Error: Empty search query"

    depth = max(1, min(depth, 10))

    # Stage 0 — cache.
    cached = cache_get(query)
    if cached is not None:
        return cached

    # Coordinate the rest of the pipeline under a hard outer deadline.
    try:
        result_text = await asyncio.wait_for(
            _run_pipeline(query, depth, region, compact),
            timeout=_WEB_SEARCH_HARD_DEADLINE_S,
        )
    except TimeoutError:
        # Outer ceiling fired. Emit a minimal Synthesis-unavailable
        # shape — no extracts, no sources, just the operator-facing
        # marker. The agent's system prompt rule (PR-F) handles this.
        result_text = (
            f"# Research: {query}\n\n"
            "## Synthesis unavailable\n"
            f"The web_search pipeline exceeded its "
            f"{_WEB_SEARCH_HARD_DEADLINE_S:.0f}s budget. No usable extracts "
            "were produced before the deadline.\n\n"
            "## Coverage\n"
            f"- Total wall time: {_elapsed_ms() / 1000:.1f}s (deadline hit).\n"
        )

    cache_put(query, result_text)
    # Annotate cache freshness so a follow-up call sees the timestamp
    # if it ever needs to render it. Done via a sentinel header so it
    # survives roundtrip without changing the visible body.
    _ = datetime.now(UTC).isoformat(timespec="seconds")  # reserved for future use
    return result_text


async def _run_pipeline(query: str, depth: int, region: str, compact: bool) -> str:
    """The inner pipeline body. Lives in its own function so the outer
    ``asyncio.wait_for`` wraps every stage uniformly.

    Stage 5 (synthesis) runs when an LLM has been injected via
    ``set_synthesis_llm``. Without an injected LLM the synthesis step
    is skipped and the formatter emits Sources + extracts under the
    "Synthesis unavailable" prefix — backward-compatible with the v1
    deferred shape.
    """
    import time

    from src.tools._web_search_aggregator import aggregate
    from src.tools._web_search_extractor import extract
    from src.tools._web_search_fetcher import fetch_top_k
    from src.tools._web_search_format import FormatInput, format_output
    from src.tools._web_search_synthesiser import SynthesisResult, synthesise

    del region  # v1 fan-out doesn't pass region per-provider; signature parity only

    pipeline_started = time.monotonic()

    # Stage 1+2 — aggregator.
    providers = _resolve_providers()
    ranked, coverage = await aggregate(query, providers, k=depth)

    # Stage 3 — fetch top-K.
    fetched = await fetch_top_k(ranked)

    # Stage 4 — extract.
    extracted = await extract(fetched)

    # Stage 5 — synthesise. Runs when an LLM has been injected AND at
    # least one extract has usable content. When skipped intentionally
    # (no LLM, or every extract is snippet-only) we leave synthesis
    # None → Coverage reads "Synthesis: skipped". When the synthesiser
    # raises we emit SynthesisResult(text=None, reason="exception") so
    # the formatter renders the explicit "Synthesis unavailable"
    # marker — the agent prompt treats that as a load-bearing signal.
    synthesis: SynthesisResult | None = None
    synthesis_model_name: str | None = None
    llm = _synthesis_llm_var.get()
    fallback_llm = _synthesis_fallback_llm_var.get()
    if llm is not None and any(e.extracted_text for e in extracted):
        try:
            synthesis = await synthesise(
                llm,
                list(extracted),
                query,
                llm_fallback=fallback_llm,
            )
            synthesis_model_name = _model_name_for(llm, fallback_llm, synthesis)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("cogtrix").warning("web_search synthesis raised: %s", exc)
            synthesis = SynthesisResult(
                text=None,
                reason=f"exception:{type(exc).__name__}",
                model_used=None,
                elapsed_ms=0,
            )

    # Stage 6 — format.
    state = FormatInput(
        query=query,
        ranked=list(ranked),
        fetched=list(fetched),
        extracted=list(extracted),
        snippet_only_tail=[],  # see _build_snippet_only_tail note below
        synthesis=synthesis,
        coverage=coverage,
        total_wall_ms=int((time.monotonic() - pipeline_started) * 1000),
        synthesis_model_name=synthesis_model_name,
        compact=compact,
    )
    return format_output(state)


def _model_name_for(
    primary: Any,
    fallback: Any,
    result: Any,
) -> str | None:
    """Resolve a display name for the model that produced *result*.

    Used by the formatter's Coverage line. Falls back to the class
    name when the LangChain ``model_name`` / ``model`` attribute isn't
    set (e.g. fake test models).
    """
    if result is None or getattr(result, "model_used", None) is None:
        return None
    chosen = fallback if result.model_used == "fallback" else primary
    if chosen is None:
        return None
    for attr in ("model_name", "model", "name"):
        val = getattr(chosen, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(chosen).__name__


def _resolve_providers() -> dict:
    """Build the ``{name: async_callable}`` mapping for the aggregator.

    Each provider's ``_search_async`` is included if its module reports
    ``is_configured()`` truthy (or has no such gate — e.g. DDG). When
    no providers are configured, returns an empty dict; the aggregator
    short-circuits.
    """
    providers: dict = {}

    # DDG (no API key required, but the ddgs package must be installed).
    if DDGS_AVAILABLE:
        from src.tools import web_search as _self

        providers["ddg"] = _self._search_async

    # The premium providers. Wrap each import in try/except so missing
    # SDKs / missing keys never crash the resolver.
    for module_name, attr in (
        ("tavily_search", "_search_async"),
        ("brave_search", "_search_async"),
        ("google_search", "_search_async"),
        ("exa_search", "_search_async"),
        ("serpapi_search", "_search_async"),
        ("searxng_search", "_search_async"),
    ):
        try:
            module = __import__(f"src.tools.{module_name}", fromlist=[attr])
            is_configured = getattr(module, "is_configured", None)
            if is_configured is None or is_configured():
                providers[module_name.replace("_search", "")] = getattr(module, attr)
        except (ImportError, AttributeError):
            # Module not importable (extra not installed) or missing
            # `_search_async` — silently skip. The aggregator records
            # absences via per_provider_failures only when the call
            # itself failed, not when the provider was never wired in.
            continue

    return providers


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "web_search",
        "description": (
            "Universal web research tool — searches multiple providers in "
            "parallel, fetches top results, extracts page content, and "
            "synthesises a topic-organised picture with explicit citations "
            "and disagreements.\n"
            "\n"
            "Replaces piecewise search_web + http_get + manual synthesis. "
            "Returns Markdown with sections:\n"
            "  ## Key findings — synthesised facts with ①②③ citations\n"
            "  ## Disagreements — explicit when sources contradict\n"
            "  ## Gaps — what the search couldn't answer\n"
            "  ## Sources — flat URL index\n"
            "  ## Coverage — per-provider + per-fetch outcomes\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You need to answer a research question with cited sources\n"
            "- Single-provider search has failed or returned conflicting results\n"
            "- The user explicitly asks for multi-source synthesis\n"
            "\n"
            "QUERY CONSTRUCTION:\n"
            "- Treat the query as keywords the index ranks pages on, NOT a "
            "natural-language request.\n"
            "- USER EXCLUSIONS ARE POST-FILTERS, NOT QUERY TERMS. When the "
            "user says 'except the X store' or 'not from Y', drop X / Y "
            "from the query and apply the exclusion when you summarise "
            "the results. Putting 'excluded' / 'except' / 'not X' in the "
            "query makes the search engine RANK pages by those words — "
            "the opposite of the user's intent.\n"
            "- Names that may not exist (a specific shop, an SKU code) — "
            "search for the broader category first, then verify the "
            "specific name appears in the results before treating it as "
            "real."
        ),
        "input_schema": WebSearchToolInput,
        "requires_confirmation": False,
        "function": lambda **kwargs: _sync_web_search(**kwargs),
        "category": "readonly",
    },
    # search_web and search_news removed from the agent catalogue in
    # PR-G — web_search supersedes them. The sync functions stay in
    # this module for internal use (_resolve_providers reaches the
    # DDG provider via web_search._search_async).
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]


register_tool_categories({"web_search": "readonly"})


def _sync_web_search(
    query: str,
    depth: int = 3,
    region: str = "wt-wt",
    compact: bool = False,
) -> str:
    """LangChain @tool-style sync wrapper around the async web_search.

    The tool registry hands us a sync callable; we adapt the async
    pipeline via ``asyncio.run`` (or run in the existing event loop
    when called from one).
    """
    import asyncio

    # Bug D / cogtrix46 hardening: surface a Python-level warning when
    # the parent process has accidentally loaded both curl_cffi and an
    # OpenSSL-binding module (httpx / urllib3). The actual prevention
    # is the subprocess sandbox; this is a *diagnostic* layer so the
    # next glibc heap abort comes with an attributable log line.
    from src.tools._native_safety import warn_if_unsafe

    warn_if_unsafe(context="web_search")

    coro = web_search(query=query, depth=depth, region=region, compact=compact)
    try:
        # Fast path — no running loop. Create one and run the pipeline.
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" in str(exc):
            # We're inside a running loop already. Escape onto the
            # centralized shared pool so the sync surface works from
            # inside async hosts.  Migrated under #1903; the previous
            # ``with ThreadPoolExecutor(...) as pool: .result()`` shape
            # carried the shutdown(wait=True)-on-__exit__ footgun the
            # policy doc forbids.  60s is generous for the full
            # web_search pipeline (its internal asyncio.wait_for budget
            # is 25s); if the outer timer fires the caller gets a
            # graceful error string rather than an unhandled exception.
            try:
                # See comment at site 1: explicit narrow to ``str`` because the
                # helper's TypeVar can't propagate through ``asyncio.run``.
                return cast(str, invoke_with_timeout(asyncio.run, coro, timeout=60))
            except TimeoutError as t_exc:
                return f"Error searching: {_sanitize_search_error(t_exc, context='web_search')}"
        raise


__all__ = [
    "web_search",
    "search_web",
    "search_news",
    "set_synthesis_llm",
    "WebSearchToolInput",
    "WebSearchInput",
    "NewsSearchInput",
    "extract_domain",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
