"""
Search quality heuristics for orchestration.

Extracted from call_model.py as part of #1593 (Option B). The
_has_substantive_search_results heuristic previously lived inline in
call_model.py — this module gives it a dedicated home with:

- A typed threshold contract (SearchQualityThresholds).
- Configurable thresholds via cogtrix.yaml.
- Observability logging for false-negative detection.
- A fixed error-prefix check (previously dead code — the actual error
  format from search_web is "Tool failed: search_web - Error searching..."
  which does not start with "Error searching").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("cogtrix")

# Matches http:// or https:// URLs in text.  Format-agnostic — works
# regardless of whether the provider prefixes URLs with "URL: " or
# embeds them raw in markdown, JSON, etc. (#1603)
_URL_RE = re.compile(r"https?://[^\s)\"'<>]+")


@dataclass(frozen=True)
class SearchQualityThresholds:
    """Tunable thresholds for the substantive-search-results heuristic.

    Defaults mirror the hardcoded values that were previously inline in
    call_model.py.  Override via cogtrix.yaml ``search_quality`` block.
    """

    #: Minimum number of distinct http/https URLs in a search_web
    #: ToolMessage for it to be considered substantive.  A single URL
    #: could be a sponsored slot; two indicates real results.
    #: Detection is format-agnostic (regex) — not tied to the ``"URL: "``
    #: prefix used by current providers (#1603).
    min_url_count: int = 2

    #: Minimum character length for a search Web ToolMessage to be
    #: considered substantive.  Defensive lower bound — a real
    #: DDG/Tavily 2-result payload easily exceeds 600 chars.
    min_content_chars: int = 300


def has_substantive_search_results(
    messages: list[Any],
    thresholds: SearchQualityThresholds | None = None,
) -> bool:
    """Return True if the current turn's search results contained real content.

    Used by the thinking-break dispatch to distinguish two failure modes
    that look superficially alike:

    1. **Effort spent, results empty** — the agent searched 3+ times and
       got nothing useful back (every search returned an error, a
       blocked-page wrapper, or an empty result list).  An honest
       refusal is the right behaviour here (#1520).
    2. **Effort spent, results rich** — the agent searched 3+ times and
       the search backend returned real product names, real URLs, real
       snippets.  Refusing in this case is *laziness*, not honesty —
       the agent has material to synthesise from and should produce a
       structured answer.  Closes #1585.

    Heuristic — a ``search_web`` ToolMessage's content is considered
    substantive iff:

    - It is **not** an error wrapper (``"Error searching"`` appears
      anywhere in the content — note: the actual error format from
      search_web is ``"Tool failed: search_web - Error searching..."``,
      which is why we use ``in`` rather than ``startswith``).
    - It is **not** the empty-result placeholder (``"No results found"``).
    - It contains at least ``thresholds.min_url_count`` http/https URLs
      (detected via regex, so the heuristic is not tied to the
      ``"URL: "`` prefix used by current providers — #1603).
    - Its content length is at least ``thresholds.min_content_chars``.

    Effort scope matches ``_compute_search_effort`` — we look only at the
    current user turn (post the last ``HumanMessage``).

    Observability: when a ToolMessage contains ≥1 URL but is classified
    as non-substantive (too few URLs or too short), a warning is logged.
    This helps detect false negatives when provider formats change.
    """
    if thresholds is None:
        thresholds = SearchQualityThresholds()

    # Scope to current turn (messages after the last HumanMessage).
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:  # pragma: no cover — defensive, langchain always present at runtime

        class _HumanPlaceholder:
            pass

        HumanMessage = _HumanPlaceholder

    last_human_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    scope = messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages

    # ADR-0056 PR-G renamed ``search_web`` → ``web_search``. The
    # straggler ``!= "search_web"`` filter that was left behind here
    # made this heuristic always return False on current sessions
    # because no ToolMessage has the old name. That broke the
    # thinking-break dispatch: the ``_effort_met and _has_results``
    # branch never fired, every search loop fell into the
    # "honest refusal" branch, and the agent emitted "I could not
    # retrieve current data on this topic." even when web_search had
    # returned real sources (cogtrix62 turn 3 ScienceSoft reproducer,
    # 2026-05-22). We accept both names so the heuristic survives
    # any future tool-rename without a silent regression.
    _WEB_SEARCH_NAMES = ("search_web", "web_search")
    for msg in scope:
        if not hasattr(msg, "tool_call_id"):
            continue
        if getattr(msg, "name", None) not in _WEB_SEARCH_NAMES:
            continue
        content = str(getattr(msg, "content", "") or "")

        # Error / empty-result guards.
        # Note: we use `in` rather than `startswith` because the actual error
        # format from search_web is "Tool failed: search_web - Error searching..."
        # which does not start with "Error searching".  The previous
        # startswith check was dead code (issue #1593).
        if "Error searching" in content or content.startswith("No results found"):
            continue
        if "not loaded" in content.lower():
            continue

        url_count = len(_URL_RE.findall(content))
        url_threshold_met = url_count >= thresholds.min_url_count
        length_threshold_met = len(content) >= thresholds.min_content_chars

        if url_threshold_met and length_threshold_met:
            return True

        # Observability: log a warning when we see ≥1 URL but
        # classify the result as non-substantive (threshold not met).
        # This catches provider-format drift before it causes silent
        # regressions (#1593, #1603).
        if url_count >= 1:
            _log.warning(
                "Search quality heuristic: ToolMessage has %d URL(s) "
                "(threshold: %d) and %d chars (threshold: %d) — classified non-substantive. "
                "If this is unexpected, the provider format may have changed. "
                "First 120 chars: %r",
                url_count,
                thresholds.min_url_count,
                len(content),
                thresholds.min_content_chars,
                content[:120],
            )

    return False


# Backward-compatibility alias — existing call_model.py imports pin to this name.
# When call_model.py is fully migrated this alias can be removed.
_has_substantive_search_results = has_substantive_search_results
