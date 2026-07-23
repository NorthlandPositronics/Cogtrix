"""Dedup + TOCTOU-safe single tool-call executor.

Extracted from ``src.orchestration.graph._invoke_one`` in the /forge A4
refactor (2026-05-24).  The function lived as a closure inside
``build_agent_graph`` and was passed by reference to
``build_process_tools_node`` as ``_invoke_one``.  The class form keeps the
exact behaviour while removing ~225 lines from ``graph.py`` and giving the
BUG-1293 TOCTOU fix a stable, documented home.

Dominant behavioural invariants the class MUST preserve:

* **TOCTOU guard (BUG-1293).** The "write history → pop event → signal"
  sequence at the end of a successful invocation runs inside a single
  ``self._history_lock`` block; do not split it.
* **Sentinel cleanup.** ``_pending_events.pop(call_key, None)`` happens on
  the success arm AND on both exception arms (``UserCancelledRun`` and
  generic ``Exception``); missing any one of them leaves a permanent
  30-second wait for the next duplicate caller.
* **No caching of per-run state.** Always read ``self._per_run_state[0]``
  fresh — ``graph.py``'s ``_reset_for_new_run`` mutates the underlying
  ``PerRunState`` instance fields in-place (the list cell is stable, but
  individual list/dict/set fields may be reassigned via ``setattr``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage
from opentelemetry.trace import Status, StatusCode

from src.agent.safety import UserCancelledRun
from src.api.telemetry import start_span
from src.logging_config import get_logger
from src.orchestration.compression import truncate_tool_output
from src.orchestration.tool_arg_correction import detect_url_tool_misuse
from src.orchestration.tool_message_kinds import COGTRIX_KIND_KEY, KIND_TOOL_MISUSE_REDIRECT

# Module-level cap on a single tool message before it lands in history.
# Moved here from graph.py during /forge A4 because the only consumer is
# this module; graph.py re-exports the name for back-compat with anyone
# who imported it from the old location.  The other budget knobs
# (_TOOL_BUDGET_HARD/SOFT, _MAX_TOOL_CALL_HISTORY, exempt sets) are still
# injected via the constructor because tests and eval scenarios may want
# to override them per-graph.
_HISTORY_TOOL_MESSAGE_CAP_CHARS = 30_000

# #2319: after this many cumulative duplicate-call cache hits in a turn, the soft
# "do not repeat" note escalates to a forced strategy change. The consecutive-error
# stuck-break (graph._STUCK_THRESHOLD) misses loops that alternate a failing call
# with a *successful* re-read (the qwen patch-anchor loop) — those reset the error
# streak — but duplicate hits catch them.
_DUPLICATE_ESCALATION_THRESHOLD = 3


def duplicate_call_banner(hit_count: int) -> str:
    """Prefix banner for a duplicate tool call served from cache.

    For the first couple of repeats it is the soft "do not repeat" note; once the
    model keeps re-issuing identical calls (``hit_count`` past the threshold) it
    becomes a forceful redirect that names the loop and offers a concrete escape
    (use ``write_file`` instead of re-patching, or finish) — #2319.
    """
    if hit_count < _DUPLICATE_ESCALATION_THRESHOLD:
        return "[Duplicate call — returning cached result. Do NOT repeat this call.]\n\n"
    return (
        f"[You have repeated identical tool calls {hit_count} times this turn and are making "
        "no progress — you are stuck in a loop. STOP retrying the same call. Do something "
        "DIFFERENT: re-read the file to see its exact current contents, then make the change "
        "with write_file (the full new file content) instead of re-attempting the same patch; "
        "or, if you genuinely cannot proceed, give your best final answer now.]\n\n"
    )


# Fallback recursion budget used to scale the retrieval-tool ceiling when the
# run config doesn't carry ``recursion_limit``. Mirrors
# ``graph.DEFAULT_RECURSION_LIMIT`` (kept as a literal to avoid a circular
# import: graph.py imports this module).
_RECURSION_LIMIT_FALLBACK = 90


class DedupedToolInvoker:
    """Execute a single tool call with cross-thread dedup + TOCTOU safety.

    Encapsulates the BUG-1293 fix: atomic check-and-reserve of the cache
    slot so parallel duplicate tool calls invoke the underlying tool only
    once.  Threads that arrive while another thread is executing block on
    a ``threading.Event`` until the result is stored, then return the
    cached result.

    Mutates ``per_run_state`` (``tool_call_history``, ``tool_call_counts``,
    ``tool_lookup``, ``active_names``, ``tool_version[0]``) and
    ``pending_events`` under the appropriate locks.  Aliasing note:
    ``per_run_state`` is passed as the 1-element list (not the
    ``PerRunState`` instance) because ``graph.py``'s
    ``_reset_for_new_run`` performs in-place field reassignment via
    ``setattr(_per_run_state[0], _f.name, _new)``; the list-cell remains
    stable but individual fields rotate, so every access goes through
    ``self._per_run_state[0]`` and we never cache the instance.
    """

    def __init__(
        self,
        *,
        per_run_state: list[Any],  # list[PerRunState] — 1-element list, see docstring.
        history_lock: threading.Lock,
        tool_budget_lock: threading.Lock,
        bound_cache_lock: threading.Lock,
        pending_events: dict[str, threading.Event],
        active_tools_list: list[Any],
        session_state: Any,
        tool_call_guard: Callable[..., Any] | None,
        tool_call_key: Callable[[dict], str | None],
        check_duplicate: Callable[..., ToolMessage | None],
        correct_tool_args: Callable[..., dict],
        safe_tool_name: Callable[[Any], str],
        max_tool_call_history: int,
        tool_budget_hard: int,
        tool_budget_soft: int,
        tool_budget_hard_exempt: frozenset[str] | set[str],
        tool_budget_soft_exempt: frozenset[str] | set[str],
        tool_budget_retrieval_tools: frozenset[str] | set[str] = frozenset(),
        tool_budget_retrieval_ceiling_divisor: int = 3,
    ) -> None:
        self._per_run_state = per_run_state
        self._history_lock = history_lock
        self._tool_budget_lock = tool_budget_lock
        self._bound_cache_lock = bound_cache_lock
        self._pending_events = pending_events
        self._active_tools_list = active_tools_list
        self._session_state = session_state
        self._tool_call_guard = tool_call_guard
        self._tool_call_key = tool_call_key
        self._check_duplicate = check_duplicate
        self._correct_tool_args = correct_tool_args
        self._safe_tool_name = safe_tool_name
        self._max_tool_call_history = max_tool_call_history
        self._tool_budget_hard = tool_budget_hard
        self._tool_budget_soft = tool_budget_soft
        self._tool_budget_hard_exempt = tool_budget_hard_exempt
        self._tool_budget_soft_exempt = tool_budget_soft_exempt
        self._tool_budget_retrieval_tools = tool_budget_retrieval_tools
        self._tool_budget_retrieval_ceiling_divisor = max(1, tool_budget_retrieval_ceiling_divisor)

    def _cap_history_tool_content(self, content: str) -> str:
        """Cap tool output before it is stored in message history."""
        if len(content) <= _HISTORY_TOOL_MESSAGE_CAP_CHARS:
            return content
        return truncate_tool_output(content, _HISTORY_TOOL_MESSAGE_CAP_CHARS)

    def _effective_recursion_limit(self, run_config: Any) -> int:
        """Best-effort read of the live LangGraph recursion budget from the run
        config, used to scale the retrieval-tool hard ceiling (#2014).

        LangGraph puts ``recursion_limit`` at the top level of the RunnableConfig
        it threads into nodes. Falls back to ``_RECURSION_LIMIT_FALLBACK`` when
        the config is absent or doesn't carry a positive int (e.g. unit tests
        that invoke ``invoke_one`` with ``run_config=None``).
        """
        if isinstance(run_config, dict):
            rl = run_config.get("recursion_limit")
            if isinstance(rl, int) and not isinstance(rl, bool) and rl > 0:
                return rl
        return _RECURSION_LIMIT_FALLBACK

    def _release_sentinel(self, call_key: str | None) -> None:
        """Pop and signal the pending-event sentinel reserved for ``call_key``.

        Must run on EVERY return path taken after the sentinel is reserved in
        ``invoke_one`` — including the early-return arms (denial, hard-budget
        cap, tool-call-guard block), not just the success and exception arms.
        Skipping it strands a duplicate caller blocked on the Event: it waits
        the full 30s timeout and then re-executes the tool, defeating the
        BUG-1293 dedup guarantee. See the module docstring's "Sentinel
        cleanup" invariant. Idempotent — a no-op if the slot is already gone.
        """
        if call_key is None:
            return
        with self._history_lock:
            _event = self._pending_events.pop(call_key, None)
        if _event is not None:
            _event.set()

    def invoke_one(self, call: dict, run_config: Any) -> Any:
        """Execute a single tool call already in tool_lookup. Returns ToolMessage."""
        call_key = self._tool_call_key(call)
        dup = self._check_duplicate(call, key=call_key)
        if dup is not None:
            return dup

        # ── TOCTOU guard (BUG-1293) ───────────────────────────────────────
        # Atomically check-and-reserve the cache slot so that parallel
        # duplicate tool calls invoke the tool only once.  Threads that
        # arrive while another thread is executing block on an Event until
        # the result is stored, then return the cached result.
        if call_key is not None:
            with self._history_lock:
                cached = self._per_run_state[0].tool_call_history.get(call_key)
                if cached is not None:
                    self._per_run_state[0].tool_call_history.move_to_end(call_key)
                    self._per_run_state[0].duplicate_hit_count[0] += 1
                    return ToolMessage(
                        content=(
                            duplicate_call_banner(self._per_run_state[0].duplicate_hit_count[0])
                            + cached
                        ),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                if call_key in self._pending_events:
                    _wait_event = self._pending_events[call_key]
                else:
                    _wait_event = None
                    self._pending_events[call_key] = threading.Event()
            if _wait_event is not None:
                _wait_event.wait(timeout=30.0)
                with self._history_lock:
                    cached = self._per_run_state[0].tool_call_history.get(call_key)
                    if cached is not None:
                        self._per_run_state[0].tool_call_history.move_to_end(call_key)
                        self._per_run_state[0].duplicate_hit_count[0] += 1
                        return ToolMessage(
                            content=(
                                duplicate_call_banner(self._per_run_state[0].duplicate_hit_count[0])
                                + cached
                            ),
                            tool_call_id=call["id"],
                            name=call["name"],
                        )
                # Should not reach here, but fall through to execute if it does
                _log = get_logger()
                _log.warning(
                    "TOCTOU wait timed out for %s — falling through to execute",
                    call_key,
                )
        # ───────────────────────────────────────────────────────────────────

        tool_name = call["name"]

        # ── Denial enforcement (#2070 / #2050 hardening) ───────────────────
        # is_denied() is the single source of truth for blocked tools (the API
        # ``api_dangerous_tools`` deny, ``/tools disable``, deny-all, budget
        # deny). Enforce it here at the execution chokepoint so a denial holds
        # regardless of how a tool reached the active set — e.g. PATCH
        # /sessions/{id}/tools ``load``, which bypasses the activation-time
        # gates, or any future activation path. For API sessions the safety
        # wrapper's is_denied check is skipped (no_confirm=True), so this is the
        # only guaranteed enforcement point.
        if self._session_state is not None and self._session_state.is_denied(tool_name):
            self._release_sentinel(call_key)
            return ToolMessage(
                content=f"Tool '{self._safe_tool_name(tool_name)}' is disabled and cannot be used.",
                tool_call_id=call["id"],
                name=tool_name,
            )

        # ── url-fetch misuse redirect (#2293) ──────────────────────────────
        # A url-fetch tool (http_get/http_post) called with a search query and no
        # URL is a web_search confusion. Short-circuit with an actionable redirect
        # BEFORE the budget/correction/invoke machinery — the call did NOT execute,
        # so it must not consume the per-tool budget, and the model gets clear
        # guidance instead of a cryptic Pydantic ``url Field required`` it loops on.
        _misuse = detect_url_tool_misuse(tool_name, call.get("args", {}))
        if _misuse is not None:
            self._release_sentinel(call_key)
            get_logger().info(
                "url-fetch tool '%s' called as a search (no url) — redirecting to web_search (#2293)",
                tool_name,
            )
            return ToolMessage(
                content=_misuse,
                tool_call_id=call["id"],
                name=tool_name,
                additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_MISUSE_REDIRECT},
            )

        # ── Per-tool call budget ──────────────────────────────────────────
        # Prevents runaway tool loops. Three tiers (#2014):
        #   * retrieval/search tools — historically *fully* exempt from the
        #     fixed cap ("research needs many progressive searches"), which let
        #     a non-converging model call e.g. search_web unbounded until the
        #     LangGraph recursion limit (GraphRecursionError / wasted budget).
        #     They now get a RECURSION-AWARE ceiling: a fraction of the live
        #     recursion budget, so it scales with the task (eval ~20,
        #     COMPLEX_ACTION ~100) and always fires before the recursion limit,
        #     forcing an honest synthesis/refusal instead of a crash.
        #   * other exempt tools (control + action: request_tools, shell, …) —
        #     still not counted (their ceilings are tracked in #2213 Layer 2).
        #   * everything else — the fixed hard cap.
        _is_retrieval = tool_name in self._tool_budget_retrieval_tools
        if _is_retrieval:
            _effective_hard = max(
                self._tool_budget_hard,
                self._effective_recursion_limit(run_config)
                // self._tool_budget_retrieval_ceiling_divisor,
            )
            _budgeted = True
        elif tool_name in self._tool_budget_hard_exempt:
            _budgeted = False
            _effective_hard = 0  # unused
        else:
            _effective_hard = self._tool_budget_hard
            _budgeted = True

        if _budgeted:
            _hard_capped = False
            # Critical section: protect compound read-increment-write on
            # _per_run_state[0].tool_call_counts and concurrent removal from active_tools_list
            with self._tool_budget_lock:
                count = self._per_run_state[0].tool_call_counts.get(tool_name, 0) + 1
                self._per_run_state[0].tool_call_counts[tool_name] = count
                if count > _effective_hard:
                    # Remove from active set AND add to denials so the model
                    # can't re-load it via request_tools(add=[...]).
                    # Also remove from active_tools_list so bind_tools stops
                    # advertising the disabled tool to the LLM, and so
                    # _reset_for_new_run doesn't silently re-enable it by
                    # rebuilding _per_run_state[0].tool_lookup from the stale list (root cause
                    # of the "Tool names must be unique" 400 on re-add).
                    self._per_run_state[0].tool_lookup.pop(tool_name, None)
                    self._per_run_state[0].active_names.discard(tool_name)
                    if self._session_state is not None:
                        self._session_state.deny_tool(tool_name)
                    _disabled_obj = next(
                        (t for t in self._active_tools_list if getattr(t, "name", "") == tool_name),
                        None,
                    )
                    if _disabled_obj is not None:
                        with self._bound_cache_lock:
                            try:
                                self._active_tools_list.remove(_disabled_obj)
                            except ValueError:
                                pass  # already removed by a concurrent invocation
                    self._per_run_state[0].tool_version[0] += 1  # force bind_tools refresh
                    _hard_capped = True
            if _hard_capped:
                # Release the TOCTOU sentinel OUTSIDE the budget lock — calling
                # _release_sentinel (which takes _history_lock) here avoids
                # introducing a tool_budget_lock → history_lock ordering.
                self._release_sentinel(call_key)
                return ToolMessage(
                    content=(
                        f"Tool '{self._safe_tool_name(tool_name)}' has been disabled after {_effective_hard} calls "
                        f"and is no longer available. Please synthesize your findings into a "
                        f"final response now using the data you already have."
                    ),
                    tool_call_id=call["id"],
                    name=tool_name,
                )

        tool_input = {**call, "type": "tool_call"}

        if self._tool_call_guard is not None:
            _guard_result = self._tool_call_guard(tool_name, call.get("args", {}))
            if hasattr(_guard_result, "is_safe") and not _guard_result.is_safe:
                log = get_logger()
                log.warning(
                    "Tool call blocked [%s]: %s — %s",
                    getattr(_guard_result, "guard_name", ""),
                    tool_name,
                    getattr(_guard_result, "reason", ""),
                )
                self._release_sentinel(call_key)
                return ToolMessage(
                    content=(
                        f"Tool call blocked by security policy: "
                        f"{getattr(_guard_result, 'reason', 'blocked')}"
                    ),
                    tool_call_id=call["id"],
                    name=tool_name,
                )
        try:
            with self._tool_budget_lock:
                tool = self._per_run_state[0].tool_lookup.get(tool_name)
            if tool is None:
                return ToolMessage(
                    content=f"Tool '{self._safe_tool_name(tool_name)}' is no longer active.",
                    tool_call_id=call["id"],
                    name=tool_name,
                )
            _corrected = self._correct_tool_args(tool, call.get("args", {}))
            _corrected_input = {**tool_input, "args": _corrected}
            _tool_t0 = time.monotonic()
            with start_span(
                "src.orchestration.graph",
                "tool.call",
                attributes={"tool.name": tool_name},
            ) as _tool_span:
                try:
                    result = tool.invoke(_corrected_input, run_config)
                except Exception as exc:
                    _tool_span.record_exception(exc)
                    _tool_span.set_attribute("tool.status", "error")
                    _tool_span.set_attribute(
                        "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                    )
                    _tool_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

                # Soft budget nudge: after N calls to the same tool, hint to synthesize.
                with self._tool_budget_lock:
                    _cnt = self._per_run_state[0].tool_call_counts.get(tool_name, 0)
                _nudge = ""
                if (
                    _cnt >= self._tool_budget_soft
                    and tool_name not in self._tool_budget_soft_exempt
                ):
                    _nudge = (
                        f"\n\n[Note: You have called {tool_name} {_cnt} times this turn. "
                        "You likely have enough data — please synthesize your findings "
                        "into a complete response now rather than searching further.]"
                    )

                if isinstance(result, ToolMessage):
                    content = result.content if isinstance(result.content, str) else ""
                    if _nudge:
                        content += _nudge
                    content = self._cap_history_tool_content(content)
                    if call_key is not None:
                        # Inlined from _store_call_result() so the history write
                        # and Event signalling happen atomically under _history_lock.
                        # Splitting them re-introduces the TOCTOU race (BUG-1293).
                        with self._history_lock:
                            self._per_run_state[0].tool_call_history[call_key] = content[:500]
                            self._per_run_state[0].tool_call_history.move_to_end(call_key)
                            if (
                                len(self._per_run_state[0].tool_call_history)
                                > self._max_tool_call_history
                            ):
                                self._per_run_state[0].tool_call_history.popitem(last=False)
                            _event = self._pending_events.pop(call_key, None)
                        if _event is not None:
                            _event.set()
                    result.content = content
                    _tool_span.set_attribute("tool.status", "success")
                    _tool_span.set_attribute(
                        "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                    )
                    _tool_span.set_status(Status(StatusCode.OK))
                    return result
                text = str(result) if result is not None else ""
                text = self._cap_history_tool_content(text)
                if call_key is not None:
                    # Inlined from _store_call_result() so the history write
                    # and Event signalling happen atomically under _history_lock.
                    # Splitting them re-introduces the TOCTOU race (BUG-1293).
                    with self._history_lock:
                        self._per_run_state[0].tool_call_history[call_key] = text[:500]
                        self._per_run_state[0].tool_call_history.move_to_end(call_key)
                        if (
                            len(self._per_run_state[0].tool_call_history)
                            > self._max_tool_call_history
                        ):
                            self._per_run_state[0].tool_call_history.popitem(last=False)
                        _event = self._pending_events.pop(call_key, None)
                    if _event is not None:
                        _event.set()
                _tool_span.set_attribute("tool.status", "success")
                _tool_span.set_attribute(
                    "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                )
                _tool_span.set_status(Status(StatusCode.OK))
                return ToolMessage(
                    content=text,
                    tool_call_id=call["id"],
                    name=tool_name,
                )
        except UserCancelledRun:
            if call_key is not None:
                with self._history_lock:
                    _event = self._pending_events.pop(call_key, None)
                if _event is not None:
                    _event.set()
            raise
        except Exception as exc:
            if call_key is not None:
                with self._history_lock:
                    _event = self._pending_events.pop(call_key, None)
                if _event is not None:
                    _event.set()
            log = get_logger()
            log.warning("Tool %s raised: %s", tool_name, exc, exc_info=True)
            return ToolMessage(
                content=self._cap_history_tool_content(
                    f"Error executing {self._safe_tool_name(tool_name)}: {exc}"
                ),
                tool_call_id=call["id"],
                name=tool_name,
            )


__all__ = [
    "DedupedToolInvoker",
    "_HISTORY_TOOL_MESSAGE_CAP_CHARS",
    "duplicate_call_banner",
]
