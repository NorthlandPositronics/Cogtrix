"""Tests for src/tools/_web_search_extractor.py — stage 4 trafilatura
wrapper (ADR-0056 PR-C)."""

from __future__ import annotations

import concurrent.futures

import pytest

from src.tools._http_fetch import FetchResult
from src.tools._web_search_aggregator import RankedResult
from src.tools._web_search_domain_class import DomainClass
from src.tools._web_search_extractor import ExtractedSource, extract
from src.tools._web_search_fetcher import FetchOutcome


def _rank(url: str = "https://example.com/a") -> RankedResult:
    return RankedResult(
        canonical_url=url,
        title="T",
        snippet="S",
        published_date=None,
        domain_class=DomainClass.UNKNOWN,
        score=1.0,
        providers=("ddg",),
    )


def _fetched(html: bytes, url: str = "https://example.com/a") -> FetchOutcome:
    fetch_result = FetchResult(
        url=url,
        status_code=200,
        content=html,
        encoding="utf-8",
        content_type="text/html",
        elapsed_ms=10,
        truncated=False,
        error=None,
    )
    return FetchOutcome(
        ranked=_rank(url),
        status="fetched",
        fetch_result=fetch_result,
        error=None,
    )


def _snippet_only(url: str = "https://example.com/a") -> FetchOutcome:
    return FetchOutcome(
        ranked=_rank(url),
        status="snippet-only",
        fetch_result=None,
        error="blocked-robots",
    )


# A representative well-formed article body — long enough that
# trafilatura returns substantive text, structured well so it parses
# cleanly.
_WELL_FORMED_HTML = (
    "<!DOCTYPE html><html><head><title>Test page</title></head><body>"
    "<header><nav>Site Nav</nav></header>"
    "<main>"
    "<h1>The Main Article</h1>"
    "<p>"
    + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 30)
    + "</p>"
    + "<p>"
    + ("Another paragraph with substantive content for the synthesiser. " * 20)
    + "</p>"
    + "</main>"
    "<footer>Site Footer</footer>"
    "</body></html>"
).encode()

_EMPTY_BODY_HTML = (
    b"<!DOCTYPE html><html><head><title>JS shell</title></head>"
    b"<body><div id='root'></div></body></html>"
)


class TestExtract:
    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        assert await extract([]) == []

    @pytest.mark.asyncio
    async def test_well_formed_html_extracted(self) -> None:
        outcomes = await extract([_fetched(_WELL_FORMED_HTML)])
        assert len(outcomes) == 1
        result = outcomes[0]
        assert isinstance(result, ExtractedSource)
        assert result.status in ("extracted", "extracted-truncated")
        assert result.extracted_text is not None
        # Substantive text recovered from the article body.
        assert "Lorem ipsum" in result.extracted_text
        # Boilerplate stripped — footer should not appear.
        assert "Site Footer" not in result.extracted_text

    @pytest.mark.asyncio
    async def test_char_cap_enforced(self) -> None:
        outcomes = await extract([_fetched(_WELL_FORMED_HTML)], char_cap=500)
        assert outcomes[0].extracted_text is not None
        # Truncation marker appended.
        assert outcomes[0].status == "extracted-truncated"
        assert len(outcomes[0].extracted_text) <= 600  # cap + small slack

    @pytest.mark.asyncio
    async def test_js_shell_marked_low_yield(self) -> None:
        outcomes = await extract([_fetched(_EMPTY_BODY_HTML)])
        assert outcomes[0].status == "low-yield"

    @pytest.mark.asyncio
    async def test_snippet_only_passes_through(self) -> None:
        outcomes = await extract([_snippet_only()])
        assert outcomes[0].status == "snippet-only"
        assert outcomes[0].extracted_text is None

    @pytest.mark.asyncio
    async def test_skipped_passes_through(self) -> None:
        skipped = FetchOutcome(
            ranked=_rank(),
            status="skipped",
            fetch_result=None,
            error="timeout",
        )
        outcomes = await extract([skipped])
        assert outcomes[0].status == "skipped"

    @pytest.mark.asyncio
    async def test_order_preserved(self) -> None:
        ranks = [
            _fetched(_WELL_FORMED_HTML, "https://example.com/a"),
            _snippet_only("https://example.com/b"),
            _fetched(_WELL_FORMED_HTML, "https://example.com/c"),
        ]
        outcomes = await extract(ranks)
        assert outcomes[0].fetch_outcome.ranked.canonical_url == "https://example.com/a"
        assert outcomes[1].fetch_outcome.ranked.canonical_url == "https://example.com/b"
        assert outcomes[2].fetch_outcome.ranked.canonical_url == "https://example.com/c"

    @pytest.mark.asyncio
    async def test_empty_content_returns_no_content_to_extract(self) -> None:
        empty_fetch = FetchResult(
            url="https://example.com/a",
            status_code=200,
            content=b"",
            encoding="utf-8",
            content_type="text/html",
            elapsed_ms=10,
            truncated=False,
            error=None,
        )
        outcome = FetchOutcome(
            ranked=_rank(),
            status="fetched-with-warning",
            fetch_result=empty_fetch,
            error="empty-body",
        )
        outcomes = await extract([outcome])
        assert outcomes[0].status == "no-content-to-extract"
        assert outcomes[0].extracted_text is None

    @pytest.mark.asyncio
    async def test_raw_text_fallback_on_unparseable_html(self) -> None:
        """trafilatura returns nothing for non-HTML bodies; the raw-text
        fallback should at least extract the visible characters."""
        # Plain text inside an html shell that trafilatura usually
        # rejects (no semantic structure).
        bare = (
            b"<html><body>"
            + (
                "Just some sentences without any structural HTML "
                "at all that trafilatura would normally extract. " * 5
            ).encode()
            + b"</body></html>"
        )
        outcomes = await extract([_fetched(bare)])
        # Either trafilatura succeeded, or the raw-fallback kicked in.
        # Both produce extracted text or low-yield depending on
        # trafilatura's threshold tuning. Just assert we didn't crash
        # and got *some* status that downstream knows how to handle.
        assert outcomes[0].status in (
            "extracted",
            "extracted-truncated",
            "extracted-raw-fallback",
            "low-yield",
        )


# ────────────────────────────────────────────────────────────────────
# Regression — Bug #1703: trafilatura.extract serialisation
# ────────────────────────────────────────────────────────────────────


class TestLxmlProcessPool:
    """libxml2 (via lxml, via trafilatura) is not thread-safe across
    concurrent parser invocations. Thread-level synchronisation was
    tried (PR #1707 lock-1, PR #1730 semaphore-2) and both failed —
    cap-1 was crash-safe but serialised every extract (#1716 latency
    tax), cap-2 hit a real-user deadlock (cogtrix58 ISTA-Dubai).

    The structural fix is PROCESS-level isolation: each
    ``trafilatura.extract`` runs in a worker subprocess with its own
    libxml2 instance. Tests below pin the parallel-via-pool contract.

    The tests install a ``ThreadPoolExecutor`` override via
    ``_set_executor_override`` so pytest monkey-patches of
    ``trafilatura.extract`` are visible to workers. In production the
    override is None and the real ``ProcessPoolExecutor`` is used —
    its safety contract is verified by integration replays
    (5×{A01, A03, C03, T01}), not by these unit tests.
    """

    @pytest.fixture(autouse=True)
    def _executor_override(self):
        """Swap in a ThreadPoolExecutor so monkey-patches reach the
        workers. Cleared after each test."""
        from src.tools._web_search_extractor import _set_executor_override

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        _set_executor_override(executor)
        yield
        _set_executor_override(None)
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_extracts_run_in_parallel_via_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """6 pages × 100 ms stub delay must complete in < 6 × stub_delay
        because the executor (size 4) runs them in parallel. Below
        the parallel upper bound (~2 × stub_delay for the second
        wave of 2) we'd be back to thread-level serialisation —
        i.e., the bug PR #1707 used to mitigate."""
        import time

        import trafilatura  # type: ignore[import-not-found]

        stub_delay = 0.10  # 100 ms per page

        def slow_extract(*args: object, **kwargs: object) -> str:
            time.sleep(stub_delay)
            return "extracted content " * 30  # ~600 chars

        monkeypatch.setattr(trafilatura, "extract", slow_extract)

        fetched = [_fetched(_WELL_FORMED_HTML, f"https://example.com/page{i}") for i in range(6)]

        t0 = time.monotonic()
        outcomes = await extract(fetched)
        elapsed = time.monotonic() - t0

        assert len(outcomes) == 6
        for o in outcomes:
            assert o.status in ("extracted", "extracted-truncated")
            assert o.extracted_text is not None

        # Parallel via pool-of-4: 6 pages → 2 waves (4 + 2) ≈ 2 × stub_delay.
        # The old lock-1 pattern would have taken 6 × stub_delay.
        # Bound is generous; we just need to rule out the lock-1
        # regression that #1716 closed.
        assert elapsed < 5 * stub_delay, (
            f"6 extracts took {elapsed:.3f}s — expected < {5 * stub_delay:.3f}s "
            f"under pool-of-4 parallelism. The lock-1 serialisation tax "
            f"(#1716) is back — the process pool isn't running extracts in "
            f"parallel."
        )

    @pytest.mark.asyncio
    async def test_extractor_exception_does_not_break_subsequent_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker subprocess (or thread) that raises must not leave the
        executor in a broken state. Subsequent submissions still
        succeed."""
        import trafilatura  # type: ignore[import-not-found]

        call_count = {"n": 0}

        def sometimes_raise(*args: object, **kwargs: object) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated libxml2 internal error")
            return "extracted body " * 40

        monkeypatch.setattr(trafilatura, "extract", sometimes_raise)

        fetched = [_fetched(_WELL_FORMED_HTML, f"https://example.com/page{i}") for i in range(2)]
        outcomes = await extract(fetched)

        # ``_extract_sync`` catches the trafilatura exception and
        # falls through to its raw-text fallback path, so neither
        # outcome surfaces as ``extraction-error`` here. What matters
        # for this test is that BOTH calls completed (the executor
        # didn't deadlock after the first one raised), and at least
        # one used the fallback path because trafilatura raised.
        assert len(outcomes) == 2
        statuses = {o.status for o in outcomes}
        assert "extracted-raw-fallback" in statuses, (
            f"expected at least one raw-fallback outcome from the call "
            f"where trafilatura raised; got {statuses!r}"
        )
        # All outcomes must have completed — no None / hung futures.
        for o in outcomes:
            assert o.extracted_text is not None or o.status in (
                "low-yield",
                "extraction-error",
            ), f"outcome left without text or known terminal status: {o.status}"

    @pytest.mark.asyncio
    async def test_executor_override_hook_is_respected(self) -> None:
        """The test-mode override hook must be honored — if the
        production code ignored it, the monkey-patches in the other
        tests in this class would be invisible to workers and the
        suite would silently fail."""
        from src.tools._web_search_extractor import (
            _get_process_pool,
            _set_executor_override,
        )

        # The autouse fixture installed a ThreadPoolExecutor — it must
        # be visible via _get_process_pool().
        pool = _get_process_pool()
        assert isinstance(pool, concurrent.futures.ThreadPoolExecutor), (
            f"override hook ignored — got {type(pool).__name__} not " f"ThreadPoolExecutor"
        )

        # Clearing the override must restore the lazy process pool.
        _set_executor_override(None)
        try:
            pool2 = _get_process_pool()
            assert isinstance(pool2, concurrent.futures.ProcessPoolExecutor), (
                f"override clear didn't restore process pool — got " f"{type(pool2).__name__}"
            )
        finally:
            # Restore the test fixture's thread pool so the
            # autouse teardown doesn't see a None override mid-test.
            _set_executor_override(concurrent.futures.ThreadPoolExecutor(max_workers=4))
