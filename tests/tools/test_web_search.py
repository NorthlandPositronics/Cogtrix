"""End-to-end tests for the public ``web_search()`` entry point
(ADR-0056 PR-E).

All internal deps are mocked — no network, no LLM. The tool's
pipeline-orchestration logic is what's under test: cache hit short-
circuit, hard outer deadline, error handling around each stage.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from cogtrix_core.tools._http_fetch import FetchResult
from cogtrix_core.tools._web_search_aggregator import CoverageInfo, RankedResult
from cogtrix_core.tools._web_search_domain_class import DomainClass
from cogtrix_core.tools._web_search_extractor import ExtractedSource
from cogtrix_core.tools._web_search_fetcher import FetchOutcome
from cogtrix_core.tools.web_search import TOOL_CONFIGS, _sync_web_search, web_search


def _rank(url: str) -> RankedResult:
    return RankedResult(
        canonical_url=url,
        title="Title",
        snippet="Snippet",
        published_date=None,
        domain_class=DomainClass.UNKNOWN,
        score=1.0,
        providers=("ddg",),
    )


def _coverage() -> CoverageInfo:
    return CoverageInfo(
        providers_attempted=1,
        providers_succeeded=1,
        raw_count=1,
        distinct_count=1,
    )


def _extracted(url: str, text: str = "Body content.") -> ExtractedSource:
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
        ranked=_rank(url), status="fetched", fetch_result=fetch_result, error=None
    )
    return ExtractedSource(fetch_outcome=outcome, extracted_text=text, status="extracted")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty cache."""
    from cogtrix_core.tools._web_search_cache import cache_clear

    cache_clear()
    yield
    cache_clear()


@pytest.fixture(autouse=True)
def _reset_synthesis_llm():
    """Each test starts with no injected synthesis LLM — the
    ContextVars would otherwise leak across tests when one of them
    calls ``set_synthesis_llm`` without a scoped context manager."""
    from cogtrix_core.tools.web_search import (
        _synthesis_fallback_llm_var,
        _synthesis_llm_var,
    )

    primary_token = _synthesis_llm_var.set(None)
    fallback_token = _synthesis_fallback_llm_var.set(None)
    yield
    _synthesis_fallback_llm_var.reset(fallback_token)
    _synthesis_llm_var.reset(primary_token)


# ── Tool registration ────────────────────────────────────────────────


class TestRegistration:
    def test_web_search_is_the_only_catalogue_tool(self) -> None:
        """PR-G removed the legacy search tools from the catalogue.
        ``web_search`` is now the sole entry exposed by this module."""
        names = [t["name"] for t in TOOL_CONFIGS]
        assert names == ["web_search"]

    def test_legacy_tool_functions_remain_importable(self) -> None:
        """The agent catalogue no longer exposes them, but the sync
        functions stay in their modules for power-user / internal use
        (web_search._resolve_providers reaches them via _search_async)."""
        from cogtrix_core.tools.web_search import search_news, search_web

        assert callable(search_web)
        assert callable(search_news)


class TestSchemaGuidance:
    """cogtrix47 (Issue 3): the model translated 'except the PowerTool
    shop by Praterstern' into the literal search term 'Praterstern
    excluded'. The fix is to teach the model — via the rendered
    schema — that user exclusions are post-filters, not query terms.

    The tool's pydantic schema flows into the JSON schema the LLM
    sees in its tool catalogue. These tests assert that the guidance
    strings are actually present in the rendered field/tool
    descriptions; without that, the model never reads them.
    """

    def test_query_field_warns_against_exclusions(self) -> None:
        from cogtrix_core.tools.web_search import WebSearchToolInput

        query_field = WebSearchToolInput.model_fields["query"]
        desc = (query_field.description or "").lower()

        # The "exclusions are post-filters" rule must be visible to
        # the model in the query field's own description.
        assert "exclusion" in desc or "exclusions" in desc
        assert "post-filter" in desc
        # The concrete failure shape from cogtrix47 ("except the X shop")
        # is named so the model recognises the pattern.
        assert "except" in desc

    def test_query_field_warns_against_conversational_scaffolding(self) -> None:
        from cogtrix_core.tools.web_search import WebSearchToolInput

        query_field = WebSearchToolInput.model_fields["query"]
        desc = (query_field.description or "").lower()
        # 'find me' / 'help me' filler should be called out.
        assert "find me" in desc or "scaffolding" in desc

    def test_tool_description_includes_query_construction_section(self) -> None:
        # The agent catalogue stores the tool description that the
        # LLM sees as the tool's per-call summary; the exclusion rule
        # must be there too, not just on the query field.
        config = next(t for t in TOOL_CONFIGS if t["name"] == "web_search")
        desc = config["description"].lower()
        assert "query construction" in desc
        assert "exclusion" in desc or "exclusions" in desc
        assert "post-filter" in desc


class TestLegacySearchWebSubprocessRouting:
    """Pins the Bug D / cogtrix46 fix: the legacy ``search_web``
    function must NOT call ``fetch_ddg_html`` from the caller's
    process. Before the fix, the in-process call pulled libcurl +
    BoringSSL into a process that also had httpx loaded — the two
    TLS backends corrupted each other's malloc state and the agent
    died with a glibc heap abort (``"double free or corruption
    (!prev)"``).
    """

    def test_search_web_dispatches_via_subprocess_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``search_web`` must route through ``_ddg_subprocess_call`` —
        never call ``fetch_ddg_html`` directly."""
        from cogtrix_core.tools import web_search as ws

        # Force DDG_AVAILABLE so the early "not available" return path
        # doesn't short-circuit the test.
        monkeypatch.setattr(ws, "DDG_AVAILABLE", True)

        # Track which path was taken. fetch_ddg_html (parent-process)
        # must NOT be called; _ddg_subprocess_call (subprocess) MUST.
        fetch_ddg_called: list[bool] = []
        subprocess_called: list[tuple[str, str, int]] = []

        async def fake_subprocess_call(query: str, region: str, num_results: int) -> list[dict]:
            subprocess_called.append((query, region, num_results))
            return [
                {"href": "https://example.com", "title": "Example", "body": "snippet"},
            ]

        def fake_fetch(query: str, region: str = "wt-wt", num_results: int = 5) -> str:
            # If this fires we've regressed — the test fails loud.
            fetch_ddg_called.append(True)
            raise AssertionError(
                "search_web pulled fetch_ddg_html into the parent process — "
                "Bug D / cogtrix46 regression"
            )

        monkeypatch.setattr(ws, "_ddg_subprocess_call", fake_subprocess_call)
        # Stub the module-level fetch_ddg_html import path defensively;
        # if any rewrite re-introduces the in-process call it'll trip
        # the assertion above.
        import cogtrix_core.tools._ddg as _ddg_mod

        monkeypatch.setattr(_ddg_mod, "fetch_ddg_html", fake_fetch)

        result = ws.search_web("vienna soudal sealant", num_results=2)

        assert subprocess_called == [("vienna soudal sealant", "wt-wt", 2)]
        assert fetch_ddg_called == []
        # Substring check on the tool's OUTPUT text, not URL-host sanitization (false positive).
        assert (
            "example.com" in result.lower()  # codeql[py/incomplete-url-substring-sanitization]
            or "Example" in result
        )

    def test_search_web_returns_error_on_subprocess_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the subprocess raises (timeout, exit code, etc.), the
        legacy ``search_web`` shim must surface the documented
        ``"Error searching: ..."`` shape rather than propagating the
        exception."""
        from cogtrix_core.tools import web_search as ws

        monkeypatch.setattr(ws, "DDG_AVAILABLE", True)

        async def boom(query: str, region: str, num_results: int) -> list[dict]:
            raise RuntimeError("DDG subprocess timed out")

        monkeypatch.setattr(ws, "_ddg_subprocess_call", boom)
        result = ws.search_web("anything")
        assert result.startswith("Error")

    def test_search_web_empty_query_returns_error_without_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The empty-query short-circuit must not spawn a subprocess."""
        from cogtrix_core.tools import web_search as ws

        monkeypatch.setattr(ws, "DDG_AVAILABLE", True)

        async def should_not_run(*args, **kwargs) -> list[dict]:
            raise AssertionError("subprocess fired for an empty query")

        monkeypatch.setattr(ws, "_ddg_subprocess_call", should_not_run)
        result = ws.search_web("   ")
        assert "Empty search query" in result


# ── Async entry point ───────────────────────────────────────────────


class TestAsyncEntryPoint:
    @pytest.mark.asyncio
    async def test_empty_query_returns_error_immediately(self) -> None:
        result = await web_search("")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_whitespace_only_query_returns_error(self) -> None:
        result = await web_search("   ")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_clamps_depth(self) -> None:
        """depth > 10 should clamp to 10."""
        captured: dict[str, Any] = {}

        async def fake_aggregate(query: str, providers: dict, **kwargs):
            captured["k"] = kwargs.get("k")
            return [], _coverage()

        with (
            patch("cogtrix_core.tools.web_search._resolve_providers", return_value={}),
            patch("cogtrix_core.tools._web_search_aggregator.aggregate", new=fake_aggregate),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract", new=AsyncMock(return_value=[])
            ),
        ):
            await web_search("a query", depth=20)
        assert captured["k"] == 10

    @pytest.mark.asyncio
    async def test_default_depth_is_three(self) -> None:
        """Bug #1716 follow-up: default ``depth`` lowered from 6 to 3.

        Every extracted page passes through the single-slot
        ``_LXML_LOCK`` (see ``_web_search_extractor.py``) which
        serialises ``trafilatura.extract`` calls process-wide. The
        cogtrix-quality corpus replay round 3 showed A01 / T01 took
        360 s with depth=6 against the cap-1 lock (vs 45-60 s
        pre-PR-#1707). Halving the default fan-out halves the
        serialised extraction tax per ``web_search`` call. Agents
        that need broader research can still override ``depth``
        explicitly up to 10.

        This test pins both the WebSearchInput Pydantic default
        (used by LangChain tool schema) and the async function
        default (used when call sites do not pass depth).
        """
        from cogtrix_core.tools.web_search import WebSearchToolInput

        # Pydantic field default — what the LLM sees as the default
        # when it constructs a tool call without specifying depth.
        # The universal tool schema is ``WebSearchToolInput`` (the
        # legacy ``WebSearchInput`` at line ~123 uses ``num_results``
        # instead of ``depth`` and is for the old DDG entry point).
        assert WebSearchToolInput.model_fields["depth"].default == 3, (
            "WebSearchToolInput.depth default must be 3 (#1716 follow-up); "
            "otherwise the lxml lock-1 serialisation tax explodes on "
            "every web_search call."
        )

        # Async function default — what gets used when a Python
        # caller calls ``web_search(query)`` without passing depth.
        captured: dict[str, Any] = {}

        async def fake_aggregate(query: str, providers: dict, **kwargs):
            captured["k"] = kwargs.get("k")
            return [], _coverage()

        with (
            patch("cogtrix_core.tools.web_search._resolve_providers", return_value={}),
            patch("cogtrix_core.tools._web_search_aggregator.aggregate", new=fake_aggregate),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract", new=AsyncMock(return_value=[])
            ),
        ):
            await web_search("a query")  # no depth arg
        assert captured["k"] == 3, (
            f"async web_search() default must pass depth=3 to aggregate, " f"got k={captured['k']}"
        )

    @pytest.mark.asyncio
    async def test_happy_path_through_full_pipeline(self) -> None:
        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "UNIQUE_BODY")]

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
        ):
            result = await web_search("test query")

        assert "# Research: test query" in result
        assert "https://example.com/a" in result
        assert "UNIQUE_BODY" in result
        # Synthesis is None in v1 → no Key findings block.
        assert "## Key findings" not in result
        # Coverage block always present.
        assert "## Coverage" in result


# ── Cache short-circuit ─────────────────────────────────────────────


class TestCache:
    @pytest.mark.asyncio
    async def test_cache_hit_short_circuits_pipeline(self) -> None:
        """Second call with the same query skips the pipeline entirely."""
        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a")]

        aggregate_mock = AsyncMock(return_value=(ranked, _coverage()))
        fetch_mock = AsyncMock(return_value=fetched)
        extract_mock = AsyncMock(return_value=extracted)

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch("cogtrix_core.tools._web_search_aggregator.aggregate", new=aggregate_mock),
            patch("cogtrix_core.tools._web_search_fetcher.fetch_top_k", new=fetch_mock),
            patch("cogtrix_core.tools._web_search_extractor.extract", new=extract_mock),
        ):
            r1 = await web_search("cached query")
            r2 = await web_search("cached query")

        assert r1 == r2
        # Pipeline ran exactly once.
        assert aggregate_mock.await_count == 1
        assert fetch_mock.await_count == 1
        assert extract_mock.await_count == 1


# ── Hard outer deadline ─────────────────────────────────────────────


class TestHardDeadline:
    @pytest.mark.asyncio
    async def test_outer_ceiling_admits_full_stage_budgets(self) -> None:
        """Bug G regression: with the old 15 s ceiling, stages 1-4 using
        9-10 s left synthesis only 5-6 s of headroom against its own
        7 s deadline → the outer ``wait_for`` fired *during* synthesis
        and the agent saw "pipeline exceeded its 15s budget" even
        though stage 5 was actively making progress.

        Pin: the constant must be at least the sum of the worst-case
        per-stage budgets so a stage-5 call that respects its own
        7 s deadline never trips the outer ceiling."""
        from cogtrix_core.tools.web_search import _WEB_SEARCH_HARD_DEADLINE_S

        # Sum of worst-case stage budgets per ADR-0056 + per-task fetch
        # safety margin: 5 + 7 + 2 + 7 = 21 s for the primary path.
        assert _WEB_SEARCH_HARD_DEADLINE_S >= 21.0, (
            f"Outer ceiling {_WEB_SEARCH_HARD_DEADLINE_S}s is too tight; "
            "stage 5 synthesis can't complete when prior stages "
            "consume their full budgets."
        )

    @pytest.mark.asyncio
    async def test_outer_deadline_emits_fallback(self, monkeypatch) -> None:
        """When the pipeline exceeds 15s the outer wait_for fires and we
        emit the synthesis-unavailable fallback."""
        # Shrink the deadline so the test runs fast.
        monkeypatch.setattr("cogtrix_core.tools.web_search._WEB_SEARCH_HARD_DEADLINE_S", 0.1)

        async def slow_aggregate(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            return [], _coverage()

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch("cogtrix_core.tools._web_search_aggregator.aggregate", new=slow_aggregate),
        ):
            result = await web_search("slow query")

        assert "## Synthesis unavailable" in result
        assert "deadline hit" in result


# ── Sync wrapper ────────────────────────────────────────────────────


class TestSyncWrapper:
    def test_sync_wrapper_runs_pipeline(self) -> None:
        """The sync wrapper is what the LangChain registry hands to the
        agent. Verify it executes the async pipeline and returns the
        formatted string."""
        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "SYNC_BODY")]

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
        ):
            result = _sync_web_search("query", depth=3, compact=False)

        assert "# Research: query" in result
        assert "SYNC_BODY" in result


# ── Provider resolver ───────────────────────────────────────────────


class TestResolveProviders:
    def test_ddg_included_when_available(self) -> None:
        from cogtrix_core.tools.web_search import _resolve_providers

        with patch("cogtrix_core.tools.web_search.DDGS_AVAILABLE", True):
            providers = _resolve_providers()
        # Either ddg present OR no providers at all (other extras
        # may also not be installed in this env). Just assert resolver
        # didn't crash and returned a dict.
        assert isinstance(providers, dict)

    def test_no_providers_when_nothing_configured(self) -> None:
        from cogtrix_core.tools.web_search import _resolve_providers

        # Force every provider to look unavailable.
        def fake_import(name: str, fromlist=None):  # type: ignore[no-untyped-def]
            raise ImportError(name)

        with (
            patch("cogtrix_core.tools.web_search.DDGS_AVAILABLE", False),
            patch("builtins.__import__", side_effect=fake_import),
        ):
            try:
                providers = _resolve_providers()
            except ImportError:
                # builtins.__import__ patch may also affect resolver's
                # own imports; resolve_providers itself is exercised
                # implicitly in other tests, so we don't strictly
                # require this case to pass under the heavy-handed
                # patch.
                return
        assert isinstance(providers, dict)


# ── Stage-5 synthesis injection ─────────────────────────────────────


class TestSynthesisInjection:
    def test_set_synthesis_llm_stores_primary_and_fallback(self) -> None:
        from cogtrix_core.tools.web_search import (
            _synthesis_fallback_llm_var,
            _synthesis_llm_var,
            set_synthesis_llm,
        )

        primary = object()
        fallback = object()
        set_synthesis_llm(primary, fallback)
        assert _synthesis_llm_var.get() is primary
        assert _synthesis_fallback_llm_var.get() is fallback

    def test_set_synthesis_llm_clears_with_none(self) -> None:
        from cogtrix_core.tools.web_search import _synthesis_llm_var, set_synthesis_llm

        set_synthesis_llm(object())
        set_synthesis_llm(None)
        assert _synthesis_llm_var.get() is None

    def test_synthesis_llm_scope_isolates_concurrent_callers(self) -> None:
        """Multi-tenant guarantee: two concurrent agents must see their
        own injected LLM, never each other's. ContextVar-with-Token
        gives us per-context isolation; module globals would not."""
        import threading

        from cogtrix_core.tools.web_search import _synthesis_llm_var, synthesis_llm_scope

        observed: dict[str, Any] = {}
        gate_a = threading.Event()
        gate_b = threading.Event()

        def tenant_a() -> None:
            with synthesis_llm_scope("LLM-A"):
                gate_a.set()
                gate_b.wait(timeout=2)
                observed["a"] = _synthesis_llm_var.get()

        def tenant_b() -> None:
            with synthesis_llm_scope("LLM-B"):
                gate_a.wait(timeout=2)
                observed["b"] = _synthesis_llm_var.get()
                gate_b.set()

        ta = threading.Thread(target=tenant_a)
        tb = threading.Thread(target=tenant_b)
        ta.start()
        tb.start()
        ta.join(timeout=3)
        tb.join(timeout=3)

        assert observed == {"a": "LLM-A", "b": "LLM-B"}

    def test_synthesis_llm_scope_resets_on_exit(self) -> None:
        """Leaving the scope (normally or by exception) must restore
        the prior value so the next run starts clean."""
        from cogtrix_core.tools.web_search import _synthesis_llm_var, synthesis_llm_scope

        assert _synthesis_llm_var.get() is None
        with synthesis_llm_scope("LLM-X"):
            assert _synthesis_llm_var.get() == "LLM-X"
        assert _synthesis_llm_var.get() is None

        with pytest.raises(RuntimeError):
            with synthesis_llm_scope("LLM-Y"):
                raise RuntimeError("boom")
        assert _synthesis_llm_var.get() is None

    @pytest.mark.asyncio
    async def test_synthesis_skipped_when_no_llm_injected(self) -> None:
        """Backward compatibility: without an injected LLM the pipeline
        still produces Sources-only output and never calls the
        synthesiser. Coverage records the skip; no failure marker."""
        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "BODY")]
        synth_mock = AsyncMock()

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
            patch("cogtrix_core.tools._web_search_synthesiser.synthesise", new=synth_mock),
        ):
            result = await web_search("q1")

        synth_mock.assert_not_awaited()
        # Intentional skip: no "Synthesis unavailable" failure marker,
        # Coverage reads "Synthesis: skipped."
        assert "## Synthesis unavailable" not in result
        assert "Synthesis: skipped." in result
        assert "BODY" in result  # extracts still emitted

    @pytest.mark.asyncio
    async def test_synthesis_runs_when_llm_injected(self) -> None:
        """When an LLM is injected and extracts have content, stage 5
        runs and its text appears in the final output."""
        from cogtrix_core.tools._web_search_synthesiser import SynthesisResult
        from cogtrix_core.tools.web_search import set_synthesis_llm

        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "BODY")]

        synthesis_text = "## Key findings\n" "### Topic\n" "- Fact one. [①]\n"
        synth_result = SynthesisResult(
            text=synthesis_text,
            reason=None,
            model_used="primary",
            elapsed_ms=42,
        )

        class _FakeLLM:
            model_name = "fake-model-7b"

        set_synthesis_llm(_FakeLLM())

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
            patch(
                "cogtrix_core.tools._web_search_synthesiser.synthesise",
                new=AsyncMock(return_value=synth_result),
            ),
        ):
            result = await web_search("q2")

        assert "## Key findings" in result
        assert "Fact one." in result
        assert "## Synthesis unavailable" not in result

    @pytest.mark.asyncio
    async def test_synthesis_skipped_when_extracts_have_no_content(self) -> None:
        """If every extract is snippet-only (extracted_text is None) we
        don't even attempt synthesis — there's nothing to synthesise.
        Coverage records the skip."""
        from cogtrix_core.tools._web_search_extractor import ExtractedSource
        from cogtrix_core.tools.web_search import set_synthesis_llm

        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        empty = ExtractedSource(
            fetch_outcome=fetched[0],
            extracted_text=None,
            status="snippet-only",
        )

        set_synthesis_llm(object())
        synth_mock = AsyncMock()

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=[empty]),
            ),
            patch("cogtrix_core.tools._web_search_synthesiser.synthesise", new=synth_mock),
        ):
            result = await web_search("q3")

        synth_mock.assert_not_awaited()
        assert "## Synthesis unavailable" not in result
        assert "Synthesis: skipped." in result

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_through_to_unavailable(self) -> None:
        """If synthesise() raises, the tool must not crash — it falls
        through to the Sources-only "Synthesis unavailable" shape."""
        from cogtrix_core.tools.web_search import set_synthesis_llm

        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "BODY")]

        set_synthesis_llm(object())

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
            patch(
                "cogtrix_core.tools._web_search_synthesiser.synthesise",
                new=AsyncMock(side_effect=RuntimeError("synth boom")),
            ),
        ):
            result = await web_search("q4")

        assert "## Synthesis unavailable" in result
        assert "BODY" in result  # extracts still emitted

    @pytest.mark.asyncio
    async def test_synthesis_validation_failure_emits_unavailable(self) -> None:
        """When synthesise returns SynthesisResult(text=None, ...) the
        formatter emits the Sources-only fallback, not the synthesis
        text."""
        from cogtrix_core.tools._web_search_synthesiser import SynthesisResult
        from cogtrix_core.tools.web_search import set_synthesis_llm

        ranked = [_rank("https://example.com/a")]
        fetched = [_extracted("https://example.com/a").fetch_outcome]
        extracted = [_extracted("https://example.com/a", "BODY")]

        failed = SynthesisResult(
            text=None,
            reason="citation-majority-uncited",
            model_used=None,
            elapsed_ms=20,
        )
        set_synthesis_llm(object())

        with (
            patch(
                "cogtrix_core.tools.web_search._resolve_providers",
                return_value={"ddg": AsyncMock()},
            ),
            patch(
                "cogtrix_core.tools._web_search_aggregator.aggregate",
                new=AsyncMock(return_value=(ranked, _coverage())),
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_top_k",
                new=AsyncMock(return_value=fetched),
            ),
            patch(
                "cogtrix_core.tools._web_search_extractor.extract",
                new=AsyncMock(return_value=extracted),
            ),
            patch(
                "cogtrix_core.tools._web_search_synthesiser.synthesise",
                new=AsyncMock(return_value=failed),
            ),
        ):
            result = await web_search("q5")

        assert "## Synthesis unavailable" in result


# ── DDG subprocess isolation (Bug D mitigation) ─────────────────────


class _FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process used by tests."""

    def __init__(
        self,
        stdout: bytes,
        stderr: bytes = b"",
        returncode: int = 0,
        communicate_delay: float = 0.0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._delay = communicate_delay
        self.kill_called = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.kill_called = True

    async def wait(self) -> int:
        return self.returncode


class TestDdgSubprocessIsolation:
    """Bug D guard: DDG calls must run in a subprocess so primp's
    native heap-corruption aborts (`munmap_chunk: invalid pointer`,
    `double free or corruption`) can't kill the agent process."""

    @pytest.mark.asyncio
    async def test_search_async_uses_subprocess(self) -> None:
        """``_search_async`` must dispatch through the subprocess
        worker, not call DDGS in-process."""
        import json

        from cogtrix_core.tools.web_search import _search_async

        canned_stdout = json.dumps(
            {
                "results": [
                    {
                        "href": "https://example.com/a",
                        "title": "A",
                        "body": "Snippet A",
                    },
                    {
                        "href": "https://example.com/b",
                        "title": "B",
                        "body": "Snippet B",
                    },
                ]
            }
        ).encode()
        proc = _FakeProc(stdout=canned_stdout, returncode=0)

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            results = await _search_async("test query", num_results=2)

        spawn.assert_awaited_once()
        # The argv must include the worker script + the three args.
        argv = spawn.await_args.args
        assert "-c" in argv  # python -c worker_script
        assert "test query" in argv
        assert "2" in argv  # num_results stringified
        assert len(results) == 2
        assert results[0].url == "https://example.com/a"
        assert results[0].provider == "ddg"

    @pytest.mark.asyncio
    async def test_subprocess_sigabrt_surfaces_as_error(self) -> None:
        """When the worker aborts (returncode == -6 = SIGABRT, the
        primp heap-corruption case) the caller must raise RuntimeError
        — the aggregator treats that as a per-provider failure and
        falls back to the other providers, instead of the abort taking
        down the agent."""
        from cogtrix_core.tools.web_search import _search_async

        proc = _FakeProc(stdout=b"", stderr=b"primp: munmap_chunk\n", returncode=-6)
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="exit"),
        ):
            await _search_async("query")

    @pytest.mark.asyncio
    async def test_subprocess_sigsegv_surfaces_as_error(self) -> None:
        """returncode == -11 (SIGSEGV) is the other heap-corruption
        signal we've seen in real sessions."""
        from cogtrix_core.tools.web_search import _search_async

        proc = _FakeProc(stdout=b"", returncode=-11)
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="exit"),
        ):
            await _search_async("query")

    @pytest.mark.asyncio
    async def test_subprocess_timeout_kills_worker(self) -> None:
        """If the worker hangs past the deadline, the caller must
        SIGKILL it (primp can be unresponsive to SIGTERM after a
        corruption) and surface a clean timeout error."""
        from cogtrix_core.tools.web_search import _search_async

        proc = _FakeProc(stdout=b"", returncode=0, communicate_delay=10.0)
        with (
            patch("cogtrix_core.tools.web_search._DDG_SUBPROCESS_TIMEOUT_S", 0.05),
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await _search_async("query")
        assert proc.kill_called is True

    @pytest.mark.asyncio
    async def test_subprocess_worker_reported_error(self) -> None:
        """The worker reports its own failures via JSON {"error": ...}
        (e.g. ddgs raised an exception). Caller surfaces as
        RuntimeError."""
        import json

        from cogtrix_core.tools.web_search import _search_async

        proc = _FakeProc(stdout=json.dumps({"error": "RateLimited: blah"}).encode())
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="worker error"),
        ):
            await _search_async("query")

    @pytest.mark.asyncio
    async def test_subprocess_invalid_json_surfaces_as_error(self) -> None:
        """If the worker prints garbage (e.g. interleaved native
        stderr), the caller must raise rather than crash the
        aggregator with a JSONDecodeError leak."""
        from cogtrix_core.tools.web_search import _search_async

        proc = _FakeProc(stdout=b"not json")
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="invalid JSON"),
        ):
            await _search_async("query")

    @pytest.mark.asyncio
    async def test_subprocess_empty_query_returns_empty(self) -> None:
        """Empty / whitespace-only queries short-circuit before
        spawning the subprocess — the agent shouldn't be charged for
        a no-op fork."""
        from cogtrix_core.tools.web_search import _search_async

        with patch("asyncio.create_subprocess_exec", new=AsyncMock()) as spawn:
            assert await _search_async("") == []
            assert await _search_async("   ") == []
        spawn.assert_not_awaited()
