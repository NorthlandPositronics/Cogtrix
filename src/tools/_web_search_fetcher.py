"""Fetcher for the web_search tool (ADR-0056 stage 3).

Takes the aggregator's ``RankedResult`` list and fetches each URL in
parallel via the async fetch primitive from PR-A2. Returns one
``FetchOutcome`` per ranked input — successful fetches carry the
``FetchResult`` body; failed fetches carry their failure category so
the formatter can emit ``Status: snippet-only (fetch failed: …)``.

No re-ranking happens here. The aggregator already picked the
top-K; the fetcher just runs them through the policy stack
(robots.txt, per-host spacing, size cap, redirect handling) and
collects outcomes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import httpx

from src.tools._http_fetch import BROWSER_HEADERS, FetchResult, fetch_async
from src.tools._web_search_aggregator import RankedResult

log = logging.getLogger("cogtrix")

FetchStatus = Literal[
    "fetched",
    "fetched-with-warning",
    "snippet-only",
    "skipped",
]


@dataclass(frozen=True)
class FetchOutcome:
    """One ranked URL's fetch result.

    ``status`` is the formatter-facing category:

    * ``"fetched"`` — body present, HTTP 2xx, ready for extraction.
    * ``"fetched-with-warning"`` — body present but truncated or low
      yield; extraction can still try.
    * ``"snippet-only"`` — fetch failed (any reason); only the
      aggregator's snippet is available.
    * ``"skipped"`` — the deadline expired before this URL could be
      attempted.

    ``fetch_result`` is None when ``status`` is ``"snippet-only"`` or
    ``"skipped"``; otherwise it carries the raw bytes + headers from
    ``_http_fetch.fetch_async``.
    """

    ranked: RankedResult
    status: FetchStatus
    fetch_result: FetchResult | None
    error: str | None = None
    """Short machine-readable failure category from FetchResult.error
    when status is "snippet-only"; None on success."""


async def fetch_top_k(
    ranked: list[RankedResult],
    *,
    deadline_s: float = 6.0,
    client: httpx.AsyncClient | None = None,
) -> list[FetchOutcome]:
    """Fetch every URL in *ranked* in parallel, honouring *deadline_s*.

    Outcomes are returned in the **same order** as the input ranking
    so the formatter can pair them positionally.

    Parameters
    ----------
    ranked
        Top-K from the aggregator. Empty list → returns ``[]``.
    deadline_s
        Wall-clock budget for the whole fetch stage. Default 6s per
        ADR-0056. URLs that haven't started by deadline get status
        ``"skipped"``; URLs in flight at deadline get cancelled and
        recorded as ``"snippet-only"`` with error ``"timeout"``.
    client
        Optional pre-built ``httpx.AsyncClient`` to share across
        fetches. When None we build one configured per ADR-0056
        (HTTP/2, no follow-redirects, our User-Agent) for the
        duration of this call.
    """
    if not ranked:
        return []

    owns_client = client is None
    if client is None:
        # HTTP/1.1 only — the h2 extra adds an extra dep without
        # material latency win for stage-3 fetches.
        client = httpx.AsyncClient(
            follow_redirects=False,
            headers=BROWSER_HEADERS,
        )

    # Each task gets its own bounded deadline — ``fetch_async`` already
    # tracks ``deadline_at = started_at + deadline_s`` internally, so a
    # small extra safety margin in the outer ``wait_for`` is enough.
    # ``deadline_s + 1.0`` is generous; if ``fetch_async`` is honouring
    # its own deadline the wait_for never fires and the worst case is
    # a single slow URL that costs us ~deadline_s + 1s.
    #
    # CRITICAL: do NOT wrap the whole ``asyncio.gather`` in a
    # ``wait_for(timeout=deadline_s)``. That was the pre-fix shape and
    # it caused Bug F: when one URL exceeded the deadline the outer
    # TimeoutError fired *the gather*, cancelling every sibling task —
    # including ones that had already produced a successful result —
    # and the except clause replaced every outcome with
    # ``status="skipped", error="timeout"``. Real-world impact: 0/N
    # fetched whenever the slowest URL was slow enough, even when 5/6
    # would have succeeded otherwise. Per-task ``wait_for`` keeps the
    # slow URL from poisoning the batch.
    _per_task_deadline_s = deadline_s + 1.0

    async def _one(r: RankedResult) -> FetchOutcome:
        try:
            result = await asyncio.wait_for(
                fetch_async(
                    r.canonical_url,
                    deadline_s=deadline_s,
                    client=client,
                ),
                timeout=_per_task_deadline_s,
            )
        except TimeoutError:
            log.debug("fetch hard-deadline hit for %s", r.canonical_url)
            return FetchOutcome(
                ranked=r,
                status="snippet-only",
                fetch_result=None,
                error="timeout",
            )
        except BaseException as exc:  # noqa: BLE001
            log.debug("fetch raised for %s: %s", r.canonical_url, exc)
            return FetchOutcome(
                ranked=r,
                status="snippet-only",
                fetch_result=None,
                error=type(exc).__name__,
            )
        return _classify(r, result)

    try:
        # No outer wait_for: each task is individually bounded above.
        # ``return_exceptions=True`` is belt-and-suspenders — _one
        # already catches everything, but if a future change forgets
        # that contract we don't want a single rogue exception to
        # cancel the whole gather.
        outcomes_or_exc = await asyncio.gather(*(_one(r) for r in ranked), return_exceptions=True)
        outcomes: list[FetchOutcome] = []
        for r, item in zip(ranked, outcomes_or_exc, strict=True):
            if isinstance(item, BaseException):
                outcomes.append(
                    FetchOutcome(
                        ranked=r,
                        status="snippet-only",
                        fetch_result=None,
                        error=type(item).__name__,
                    )
                )
            else:
                outcomes.append(item)
        return outcomes
    finally:
        if owns_client:
            await client.aclose()


def _classify(ranked: RankedResult, result: FetchResult) -> FetchOutcome:
    """Map a ``FetchResult`` from ``_http_fetch.fetch_async`` to a
    formatter-facing ``FetchOutcome``."""
    if result.error is None and result.status_code and 200 <= result.status_code < 300:
        if result.content is None or not result.content:
            return FetchOutcome(
                ranked=ranked,
                status="fetched-with-warning",
                fetch_result=result,
                error="empty-body",
            )
        if result.truncated:
            return FetchOutcome(
                ranked=ranked, status="fetched-with-warning", fetch_result=result, error=None
            )
        return FetchOutcome(ranked=ranked, status="fetched", fetch_result=result, error=None)

    # Any error → snippet-only with the FetchResult's error category.
    return FetchOutcome(
        ranked=ranked,
        status="snippet-only",
        fetch_result=None,
        error=result.error or "unknown",
    )


__all__ = ["FetchOutcome", "FetchStatus", "fetch_top_k"]
