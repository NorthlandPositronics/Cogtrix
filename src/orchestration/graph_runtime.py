"""Per-run runtime state + thread-pool executors for the orchestration graph.

Extracted from ``src/orchestration/graph.py`` as part of the /forge A1.4
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
    # Bug L: unverified-claim guard (see src/orchestration/verification.py).
    unverified_claim_count: list[int] = field(default_factory=lambda: [0])
    # cogtrix47 Issues 5+6: unverified-entity guard.
    unverified_entity_count: list[int] = field(default_factory=lambda: [0])
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
