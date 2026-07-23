"""Per-run runtime state + thread-pool executors for the orchestration graph.

Extracted from ``cogtrix_core/orchestration/graph.py`` as part of the /forge A1.4
extraction (2026-05-23). Pure structural move — no semantic change.

Contents:

* :class:`PerRunState` — dataclass holding every mutable counter, cache,
  and tracking state for a single agent turn. Scalar fields are
  intentionally list-wrapped so closures over node-builders see
  mutations without rebinding; ``_reset_for_new_run()`` in graph.py
  copies fresh values into the existing instance in-place so those
  closure references stay valid across turns.
* :func:`_get_tool_executor` — lazy-init shared ``ThreadPoolExecutor``
  for parallel tool dispatch (sized at ``_PARALLEL_TOOL_WORKERS=8``).
* :func:`_get_llm_executor` — lazy-init shared ``ThreadPoolExecutor``
  for LLM invocation (sized at ``_LLM_EXECUTOR_WORKERS=4``).

Both executors register ``atexit`` shutdowns so process exit cleans up
without blocking. ``cancel_futures=True`` so any in-flight work is
cancelled rather than waited on.

The architect-flagged ``_invoke_with_timeout`` extraction is
deliberately deferred — that function is currently a closure inside
``build_agent_graph`` capturing the bound LLM + per-run state. Pulling
it out cleanly requires threading those references through an
explicit parameter list, which is more risk than this pure-move PR
should carry. Future PR.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_PARALLEL_TOOL_WORKERS = 8

_TOOL_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_TOOL_EXECUTOR_LOCK = threading.Lock()

_LLM_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_LLM_EXECUTOR_LOCK = threading.Lock()
_LLM_EXECUTOR_WORKERS = 4


def _get_tool_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level parallel tool executor, creating it on first use."""
    global _TOOL_EXECUTOR
    if _TOOL_EXECUTOR is None:
        with _TOOL_EXECUTOR_LOCK:
            if _TOOL_EXECUTOR is None:
                _TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_PARALLEL_TOOL_WORKERS,
                    thread_name_prefix="tool",
                )
                atexit.register(_TOOL_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _TOOL_EXECUTOR


def _get_llm_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level LLM executor, creating it on first use.

    Use a shared bounded pool instead of creating a fresh
    ThreadPoolExecutor per LLM call to avoid thread leakage when
    calls time out and the underlying OS thread stays blocked in I/O.
    """
    global _LLM_EXECUTOR
    if _LLM_EXECUTOR is None:
        with _LLM_EXECUTOR_LOCK:
            if _LLM_EXECUTOR is None:
                _LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_LLM_EXECUTOR_WORKERS,
                    thread_name_prefix="llm",
                )
                atexit.register(_LLM_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _LLM_EXECUTOR


@dataclass
class PerRunState:
    """All mutable per-run state for a compiled agent graph.

    Scalar fields are list-wrapped so that closures mutated inside
    graph nodes see the current value without rebinding the closure.
    A fresh instance is created by ``_reset_for_new_run()`` between
    agent turns, which guarantees every counter and collection is
    zeroed — a new field added here is automatically reset.
    """

    # Retry / loop counters
    phantom_count: list[int] = field(default_factory=lambda: [0])
    fabrication_count: list[int] = field(default_factory=lambda: [0])
    action_intent_count: list[int] = field(default_factory=lambda: [0])
    # Bug L: unverified-claim guard (see cogtrix_core/orchestration/verification.py).
    unverified_claim_count: list[int] = field(default_factory=lambda: [0])
    # cogtrix47 Issues 5+6: unverified-entity guard.
    unverified_entity_count: list[int] = field(default_factory=lambda: [0])
    # #1841: output-fidelity guard (fabricated/misattributed quotes).
    unsupported_quote_count: list[int] = field(default_factory=lambda: [0])
    # #1843: version-scope-collapse guard (parent/series status mis-scoped
    # onto a specific child/newer version ID).
    version_scope_count: list[int] = field(default_factory=lambda: [0])
    # #1860: attributed-prose-claim guard (a source/authority is credited
    # for content not in any tool result this turn).
    unsupported_attribution_count: list[int] = field(default_factory=lambda: [0])
    # #1988 (post-mortem #1987 Cluster A): entity-owner mismatch guard
    # (<entity-id> + <stakeholder-name> co-mentioned in the response
    # without that pair appearing in any tool result or system prompt
    # this turn — agent stitched a plausible-but-wrong owner onto a
    # structured ID).
    entity_owner_mismatch_count: list[int] = field(default_factory=lambda: [0])
    # #2015 (post-mortem #2006 Cluster A root-cause): corpus-aware
    # attribution mismatch guard.  Stricter than the entity-owner sibling
    # above — uses a curated ``{entity_id → {valid_owner_names}}`` index
    # supplied by the caller (only the PM harness today) so it catches
    # cases where the wrongly-attached stakeholder name DOES co-occur
    # in retrieved chunks (which fools the grounding-based detector)
    # but is still attached to the wrong entity per the corpus.  Only
    # runs when ``build_agent_graph`` is given a non-None
    # ``corpus_attribution_detector`` callable.
    corpus_attribution_mismatch_count: list[int] = field(default_factory=lambda: [0])
    # #1989 (post-mortem #1987 Cluster C): topic-substitution guard.
    # User asks about subject X (e.g. "CompactSync codebase tech debt");
    # agent silently retitles to in-corpus subject Y ("Project Nimbus
    # Technical Debt Risks") and answers Y instead of acknowledging
    # out-of-scope.  Counter bounds retries.
    topic_substitution_count: list[int] = field(default_factory=lambda: [0])
    # #1713: sycophantic-prefix guard ("You're right" / "I apologize"
    # validation prefix preceding a final answer). Bounded retries.
    sycophancy_count: list[int] = field(default_factory=lambda: [0])
    # #1869: fabricated-action-success-without-tool-call guard. Sibling
    # to `fabrication_count` (which counts the tool-error-precursor case);
    # this one counts the "no tool call at all" case. Bounded retries.
    fabricated_action_count: list[int] = field(default_factory=lambda: [0])
    # #1871: fabricated-tool-error-quote guard. Polarity sibling of the
    # #1869 counter — counts cases where the model invents a verbatim
    # quoted error string that does not appear in any ToolMessage this
    # turn. Bounded retries.
    fabricated_quote_count: list[int] = field(default_factory=lambda: [0])
    # #1868: non-canonical GitHub-fork-recommendation guard. Catches
    # responses that recommend a project + cite a non-canonical
    # github.com/<owner>/<repo> URL with authoritative recommendation
    # framing.
    noncanonical_fork_count: list[int] = field(default_factory=lambda: [0])
    # #1943 PR #4: synthesis-after-eviction guard. Counts how many times
    # the model emitted a substantive response after a ``context_evicted``
    # marker without grounding or acknowledging the eviction.  Bounded
    # retries — see ``_MAX_SYNTHESIS_AFTER_EVICTION_RETRIES``.
    synthesis_after_eviction_count: list[int] = field(default_factory=lambda: [0])
    # #1960 follow-up: per-turn recovery-firing budget.  Counts every
    # ``handle_*`` decision from ``route_after_model`` since the most
    # recent ``HumanMessage``.  When the count exceeds the cap, the
    # router short-circuits to END so a runaway recovery cascade can't
    # exhaust the scenario timeout.  ``recovery_firings_turn_marker``
    # stores the index of the HumanMessage that opened the current
    # turn — when route_after_model sees a NEWER HumanMessage, it
    # resets the counter.  Independent safety net at the routing
    # layer; works regardless of which detector misfires.
    recovery_firings_this_turn: list[int] = field(default_factory=lambda: [0])
    recovery_firings_turn_marker: list[int] = field(default_factory=lambda: [-1])
    # #1964 Item D — observability for the cascade budget.  Per-turn
    # ordered history of which ``handle_*`` decisions fired since the
    # most recent ``HumanMessage``.  When the budget kicks in, the
    # router logs this list so an operator can see "in this turn:
    # detector X fired first, then Y, then budget killed at Z".
    # Helps diagnose cascade pathologies in production rather than
    # just CI.  Same reset semantics as the counter: cleared when
    # ``recovery_firings_turn_marker`` advances.
    recovery_firings_history: list[list[str]] = field(default_factory=lambda: [[]])
    incompleteness_nudge_given: list[int] = field(default_factory=lambda: [0])
    expansion_count: list[int] = field(default_factory=lambda: [0])
    auto_expansion_count: list[int] = field(default_factory=lambda: [0])
    call_count: list[int] = field(default_factory=lambda: [0])
    last_input_tokens: list[int] = field(default_factory=lambda: [0])
    request_tools_noop_count: list[int] = field(default_factory=lambda: [0])

    # Tool tracking
    tool_version: list[int] = field(default_factory=lambda: [0])
    last_tool_version: list[int] = field(default_factory=lambda: [-1])
    tool_call_history: OrderedDict[str, str] = field(default_factory=OrderedDict)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    # #2319: cumulative count of duplicate tool calls served from cache this turn.
    # The consecutive-error stuck-break misses a loop that alternates a failing
    # call with a *successful* re-read (the qwen patch-anchor loop) — those reset
    # the error streak. Duplicate hits do catch it, so once a model keeps
    # re-issuing identical calls the "do not repeat" note escalates to a forced
    # strategy change.
    duplicate_hit_count: list[int] = field(default_factory=lambda: [0])
    # #2269: tools the model has PINNED for this task via
    # request_tools(keep_loaded=[...]). A pinned tool is exempt from the fixed
    # per-tool hard cap — instead it gets the recursion-aware ceiling (same as
    # retrieval), so a long legitimate task (e.g. a multi-step deploy hammering
    # execute_shell_command) isn't cut off mid-run, while a non-converging model
    # still can't loop to the recursion limit. Bounded to _MAX_PINNED_TOOLS per
    # run; per-run (reset with this instance each turn), never persisted.
    pinned_tools: set[str] = field(default_factory=set)
    # #2213: tools that hit their per-tool budget ceiling THIS turn. call_model
    # filters these out of bind_tools so the LLM stops seeing/calling them (can't
    # burn recursion), but — unlike the old session-scoped deny — this is per-run:
    # cleared by _reset_for_new_run, so the tool is available again next turn. This
    # is what makes the budget a per-turn guard, not a session-wide kill of a tool
    # that's legitimately reused across turns (esp. execute_shell_command).
    budget_stopped_tools: set[str] = field(default_factory=set)

    # Reflection / health-check pacing
    last_reflection_at: list[int] = field(default_factory=lambda: [0])
    last_tool_health_check_at: list[int] = field(default_factory=lambda: [0])

    # Stuck-detection state
    stuck_threshold_calibrated: list[bool] = field(default_factory=lambda: [False])
    stuck_no_checkpoint_threshold: list[int] = field(default_factory=lambda: [15])
    consecutive_errors: list[int] = field(default_factory=lambda: [0])
    force_thinking_break: list[bool] = field(default_factory=lambda: [False])
    consecutive_identical_error_count: list[int] = field(default_factory=lambda: [0])
    last_identical_error_signature: list[tuple[str, str] | None] = field(
        default_factory=lambda: [None]
    )

    # Checkpoint pacing
    last_checkpoint_count: list[int] = field(default_factory=lambda: [0])
    rounds_since_checkpoint: list[int] = field(default_factory=lambda: [0])
    calls_since_last_checkpoint: list[int] = field(default_factory=lambda: [0])

    # File-write tracking
    same_file_writes: dict[str, int] = field(default_factory=dict)

    # Action-tier consecutive-call cap (Bug F #1712). Counts the
    # consecutive emissions of each action-tier tool name across rounds
    # within a single agent turn. When the same action-tier tool is
    # emitted more than ``MAX_CONSECUTIVE_ACTION_CALLS`` times in a row
    # (counting parallel-batched calls in the same AIMessage), the
    # dispatcher returns a cap-hit ToolMessage instead of executing.
    # Reset whenever a different tool is emitted.
    action_tier_consecutive_calls: dict[str, int] = field(default_factory=dict)
    last_action_tier_tool: list[str | None] = field(default_factory=lambda: [None])

    # Cache / lookup state
    bound_cache: OrderedDict = field(default_factory=OrderedDict)
    compression_cache: dict[str, str] = field(default_factory=dict)
    tool_lookup: dict[str, Any] = field(default_factory=dict)
    active_names: set[str] = field(default_factory=set)
    tool_catalog: dict[str, str] = field(default_factory=dict)
    available_tools_ref: list[dict] = field(default_factory=list)


__all__ = [
    "_LLM_EXECUTOR",
    "_LLM_EXECUTOR_LOCK",
    "_LLM_EXECUTOR_WORKERS",
    "_PARALLEL_TOOL_WORKERS",
    "_TOOL_EXECUTOR",
    "_TOOL_EXECUTOR_LOCK",
    "PerRunState",
    "_get_llm_executor",
    "_get_tool_executor",
]
