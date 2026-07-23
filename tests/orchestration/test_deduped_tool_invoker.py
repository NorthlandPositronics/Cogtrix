"""Unit tests for src.orchestration.deduped_tool_invoker.DedupedToolInvoker.

These tests exercise the class extracted from ``graph._invoke_one`` in
the /forge A4 refactor (2026-05-24).  They run at unit scope: no graph
build, no LLM, no LangGraph compile cycle.  Each test wires the
constructor by hand with minimal mocks so the dedup + TOCTOU + budget
logic can be verified without dragging the whole ``build_agent_graph``
construction in.

The full integration coverage of these invariants still lives in
``tests/test_agent_graph.py::TestDuplicateToolCallDetection``; this file
is the unit-level safety net for the extraction itself.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage

from cogtrix_core.orchestration.deduped_tool_invoker import DedupedToolInvoker
from cogtrix_core.orchestration.graph_runtime import PerRunState
from cogtrix_core.orchestration.session_state import SessionState


def _make_per_run_state(tool: Any, tool_name: str = "echo_tool") -> PerRunState:
    """Build a PerRunState pre-populated with a single registered tool."""
    state = PerRunState()
    state.tool_lookup[tool_name] = tool
    state.active_names.add(tool_name)
    return state


def _make_invoker(
    *,
    tool: Any,
    tool_name: str = "echo_tool",
    per_run_state: PerRunState | None = None,
    pending_events: dict[str, threading.Event] | None = None,
    active_tools_list: list[Any] | None = None,
    session_state: SessionState | None = None,
    tool_budget_hard: int = 8,
    tool_budget_soft: int = 5,
    tool_budget_hard_exempt: set[str] | None = None,
    tool_budget_soft_exempt: set[str] | None = None,
    tool_budget_retrieval_tools: set[str] | None = None,
    tool_budget_retrieval_ceiling_divisor: int = 3,
    tool_budget_action_tools: set[str] | None = None,
    tool_budget_action_ceiling_divisor: int = 3,
    tool_call_guard: Any = None,
    resolve_tool_category: Any = None,
    safe_tool_name: Any = None,
) -> tuple[DedupedToolInvoker, PerRunState, dict[str, threading.Event], list[Any]]:
    """Construct a DedupedToolInvoker with sensible defaults."""
    per_run_state = (
        per_run_state if per_run_state is not None else _make_per_run_state(tool, tool_name)
    )
    pending_events = pending_events if pending_events is not None else {}
    active_tools_list = active_tools_list if active_tools_list is not None else [tool]
    session_state = session_state if session_state is not None else SessionState()
    tool_budget_hard_exempt = (
        tool_budget_hard_exempt if tool_budget_hard_exempt is not None else set()
    )
    tool_budget_soft_exempt = (
        tool_budget_soft_exempt if tool_budget_soft_exempt is not None else set()
    )

    def _key(call: dict) -> str | None:
        # Mirror graph._tool_call_key for the test surface: name + sorted args.
        import json

        if call.get("name") in tool_budget_hard_exempt and call["name"] == "request_tools":
            # Mirror dedup exemption — keep tight scope but allow opt-out.
            return None
        try:
            args_json = json.dumps(call.get("args", {}), sort_keys=True)
        except (TypeError, ValueError):
            return None
        return f"{call['name']}:{args_json}"

    def _check_dup(call: dict, key: str | None = None) -> ToolMessage | None:
        # No-op pre-check: real TOCTOU window is inside invoke_one's locked block.
        return None

    def _correct(_tool: Any, args: dict) -> dict:
        return args

    def _safe(name: Any, max_len: int = 80) -> str:
        return str(name)[:max_len]

    _safe_fn = safe_tool_name if safe_tool_name is not None else _safe

    invoker = DedupedToolInvoker(
        per_run_state=[per_run_state],
        history_lock=threading.Lock(),
        tool_budget_lock=threading.Lock(),
        bound_cache_lock=threading.Lock(),
        pending_events=pending_events,
        active_tools_list=active_tools_list,
        session_state=session_state,
        tool_call_guard=tool_call_guard,
        tool_call_key=_key,
        check_duplicate=_check_dup,
        correct_tool_args=_correct,
        safe_tool_name=_safe_fn,
        max_tool_call_history=256,
        tool_budget_hard=tool_budget_hard,
        tool_budget_soft=tool_budget_soft,
        tool_budget_hard_exempt=tool_budget_hard_exempt,
        tool_budget_soft_exempt=tool_budget_soft_exempt,
        tool_budget_retrieval_tools=(
            tool_budget_retrieval_tools if tool_budget_retrieval_tools is not None else set()
        ),
        tool_budget_retrieval_ceiling_divisor=tool_budget_retrieval_ceiling_divisor,
        tool_budget_action_tools=(
            tool_budget_action_tools if tool_budget_action_tools is not None else set()
        ),
        tool_budget_action_ceiling_divisor=tool_budget_action_ceiling_divisor,
        resolve_tool_category=resolve_tool_category,
    )
    return invoker, per_run_state, pending_events, active_tools_list


class TestConstructorWiringSmoke:
    """A4 sanity: the extracted class can be built and invoked end-to-end."""

    def test_constructor_wiring_smoke(self):
        """Build an invoker with all mocks, call invoke_one, assert ToolMessage shape."""
        tool = MagicMock()
        tool.name = "echo_tool"
        tool.invoke.return_value = ToolMessage(content="world", tool_call_id="c1", name="echo_tool")

        invoker, state, _, _ = _make_invoker(tool=tool)

        call = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        result = invoker.invoke_one(call, None)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "c1"
        assert result.name == "echo_tool"
        assert result.content == "world"
        assert tool.invoke.call_count == 1
        # History was written so a follow-up dup-call would short-circuit.
        assert len(state.tool_call_history) == 1
        # tool_call_counts incremented for the non-exempt name.
        assert state.tool_call_counts.get("echo_tool") == 1


class TestDedupCache:
    """Cached results short-circuit the second identical call."""

    def test_dedup_cache_hit_returns_duplicate_marker(self):
        """A pre-populated tool_call_history entry must short-circuit invoke."""
        tool = MagicMock()
        tool.name = "echo_tool"
        tool.invoke.return_value = ToolMessage(
            content="should-not-execute", tool_call_id="c1", name="echo_tool"
        )

        invoker, state, _, _ = _make_invoker(tool=tool)

        # Pre-seed the cache as if a previous identical call already ran.
        cache_key = 'echo_tool:{"text": "hello"}'
        state.tool_call_history[cache_key] = "cached_world_payload"

        call = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c2"}
        result = invoker.invoke_one(call, None)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "c2"
        assert result.name == "echo_tool"
        assert result.content.startswith("[Duplicate call")
        assert "cached_world_payload" in result.content
        # The underlying tool MUST NOT have been invoked.
        assert tool.invoke.call_count == 0

    def test_duplicate_escalates_after_threshold(self):
        """#2319: repeated identical calls escalate from a soft note to a forced
        strategy change once the model is clearly looping."""
        tool = MagicMock()
        tool.name = "echo_tool"
        invoker, state, _, _ = _make_invoker(tool=tool)
        state.tool_call_history['echo_tool:{"text": "hello"}'] = "cached_payload"
        call = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c"}

        results = [invoker.invoke_one(call, None) for _ in range(3)]

        assert results[0].content.startswith("[Duplicate call")  # soft
        assert results[1].content.startswith("[Duplicate call")  # soft
        assert "stuck in a loop" in results[2].content  # escalated
        assert "write_file" in results[2].content  # offers a concrete escape
        assert state.duplicate_hit_count[0] == 3
        assert tool.invoke.call_count == 0


class TestDuplicateBanner:
    def test_banner_escalates_at_threshold(self) -> None:
        from cogtrix_core.orchestration.deduped_tool_invoker import (
            _DUPLICATE_ESCALATION_THRESHOLD,
            duplicate_call_banner,
        )

        assert duplicate_call_banner(1).startswith("[Duplicate call")
        assert duplicate_call_banner(_DUPLICATE_ESCALATION_THRESHOLD - 1).startswith(
            "[Duplicate call"
        )
        escalated = duplicate_call_banner(_DUPLICATE_ESCALATION_THRESHOLD)
        assert escalated.startswith("[You have repeated")
        assert "write_file" in escalated


class TestTOCTOUPendingEventWaitPath:
    """Unit mirror of test_parallel_duplicate_tool_call_with_slow_invoke."""

    def test_toctou_pending_event_wait_path(self):
        """Two parallel threads, slow tool: tool invoked once, both return same content."""
        invoke_lock = threading.Lock()
        invoke_calls = []

        def _slow_invoke(inp, *_a, **_kw):
            with invoke_lock:
                invoke_calls.append(inp["id"])
            time.sleep(0.05)  # widen the TOCTOU window
            return ToolMessage(content="slow_payload", tool_call_id=inp["id"], name="echo_tool")

        tool = MagicMock()
        tool.name = "echo_tool"
        tool.invoke.side_effect = _slow_invoke

        invoker, _, _, _ = _make_invoker(tool=tool)

        results: dict[str, Any] = {}
        errors: list[BaseException] = []

        def _runner(call_id: str) -> None:
            try:
                call = {"name": "echo_tool", "args": {"text": "hello"}, "id": call_id}
                results[call_id] = invoker.invoke_one(call, None)
            except BaseException as exc:  # noqa: BLE001 — test diagnostic
                errors.append(exc)

        t1 = threading.Thread(target=_runner, args=("c1",))
        t2 = threading.Thread(target=_runner, args=("c2",))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"thread raised: {errors!r}"
        assert len(results) == 2
        # Tool was only physically invoked once despite two parallel callers.
        assert tool.invoke.call_count == 1, (
            f"expected 1 underlying invocation, got {tool.invoke.call_count} "
            f"(invoke_calls={invoke_calls!r})"
        )
        # Both results carry the same cached payload (one fresh, one duplicate marker).
        assert all("slow_payload" in r.content for r in results.values())
        duplicate_count = sum(
            1 for r in results.values() if r.content.startswith("[Duplicate call")
        )
        assert duplicate_count == 1, (
            f"expected exactly one duplicate marker, got {duplicate_count} "
            f"results={ {k: v.content[:80] for k, v in results.items()} }"
        )


class TestBudgetHardCap:
    """Hard cap removes tool from active set and bumps version."""

    def test_budget_hard_removes_tool_and_bumps_version(self):
        """After N+1 calls (N = hard limit), tool is disabled and version bumped."""
        tool = MagicMock()
        tool.name = "noisy_tool"

        def _vary_invoke(inp, *_a, **_kw):
            # Each call returns a different payload so dedup never short-circuits.
            return ToolMessage(
                content=f"resp-{inp['id']}", tool_call_id=inp["id"], name="noisy_tool"
            )

        tool.invoke.side_effect = _vary_invoke

        invoker, state, _, active_tools = _make_invoker(
            tool=tool,
            tool_name="noisy_tool",
            tool_budget_hard=3,
            tool_budget_soft=2,
        )
        version_before = state.tool_version[0]

        # First N calls succeed and execute the tool.
        for i in range(3):
            call = {"name": "noisy_tool", "args": {"i": i}, "id": f"c{i}"}
            result = invoker.invoke_one(call, None)
            assert "resp-" in result.content
            assert "disabled" not in result.content

        assert tool in active_tools
        assert "noisy_tool" in state.tool_lookup
        assert state.tool_version[0] == version_before

        # The (N+1)-th call trips the hard cap.
        call_over = {"name": "noisy_tool", "args": {"i": 99}, "id": "c_over"}
        result_over = invoker.invoke_one(call_over, None)

        assert isinstance(result_over, ToolMessage)
        assert "hit its per-turn call limit (3 calls)" in result_over.content
        # #2213 per-turn stop: tool is budget-stopped (filtered from bind_tools) and
        # version bumped — but NOT removed from active_tools_list / tool_lookup and
        # NOT session-denied, so _reset_for_new_run restores it next turn.
        assert "noisy_tool" in state.budget_stopped_tools
        assert state.tool_version[0] == version_before + 1
        assert tool in active_tools  # stays — restored next turn, not session-killed
        assert "noisy_tool" in state.tool_lookup

    def test_budget_trip_does_not_session_deny(self):
        """#2213: a budget trip must NOT add to session denials (that would kill the
        tool for the whole session); it's a per-run stop only."""
        tool = MagicMock()
        tool.name = "noisy_tool"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"r-{inp['id']}", tool_call_id=inp["id"], name="noisy_tool"
        )
        ss = SessionState()
        invoker, state, _, _ = _make_invoker(
            tool=tool, tool_name="noisy_tool", session_state=ss, tool_budget_hard=2
        )
        for i in range(3):  # 3rd call trips the cap of 2
            invoker.invoke_one({"name": "noisy_tool", "args": {"i": i}, "id": f"c{i}"}, None)
        assert "noisy_tool" in state.budget_stopped_tools  # per-run stop
        assert not ss.is_denied("noisy_tool")  # NOT session-denied


class TestActionCeiling:
    """#2213 Layer 2: action tools get a recursion-aware ceiling (not the fixed
    cap, not uncapped) — high enough for a long build, still bounded."""

    def _shell(self):
        tool = MagicMock()
        tool.name = "shell"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"r-{inp['id']}", tool_call_id=inp["id"], name="shell"
        )
        return tool

    def test_action_survives_past_fixed_cap(self):
        tool = self._shell()
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell",
            tool_budget_hard=3,
            tool_budget_soft=2,
            tool_budget_action_tools={"shell"},
            tool_budget_action_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 30}  # action ceiling = max(3, 30 // 3) = 10
        for i in range(9):  # 9 calls — well past the fixed cap of 3, under ceiling 10
            r = invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content, f"action tool disabled at call {i}"
        assert tool in active
        assert "shell" in state.tool_lookup

    def test_action_still_bounded_by_ceiling(self):
        tool = self._shell()
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell",
            tool_budget_hard=3,
            tool_budget_action_tools={"shell"},
            tool_budget_action_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 12}  # action ceiling = max(3, 12 // 3) = 4
        for i in range(4):
            r = invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content
        over = invoker.invoke_one({"name": "shell", "args": {"i": 99}, "id": "over"}, cfg)
        assert "hit its per-turn call limit (4 calls)" in over.content  # bounded, not uncapped
        assert tool in active  # #2213: stays (per-turn stop, not removed)

    def test_action_gets_higher_ceiling_than_retrieval(self):
        # Same recursion budget: action divisor 3 (looser) allows more calls than
        # retrieval divisor 4 — action tools are meant to sustain longer sequences.
        tool = self._shell()
        invoker, _, _, _ = _make_invoker(
            tool=tool,
            tool_name="shell",
            tool_budget_hard=3,
            tool_budget_action_tools={"shell"},
            tool_budget_action_ceiling_divisor=3,
            tool_budget_retrieval_ceiling_divisor=4,
        )
        cfg = {"recursion_limit": 40}  # action ceiling 13 > retrieval-equiv ceiling 10
        for i in range(13):
            r = invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content, f"disabled at {i} — action ceiling too tight"


class TestExceptionCleansSentinel:
    """An exception during invoke must clear the pending event so retries proceed."""

    def test_exception_pops_sentinel_and_signals_event(self):
        """If the tool raises, pending_events must be cleared and any waiter unblocked.

        Models the realistic two-thread scenario the production code defends
        against: thread A is mid-flight in tool.invoke (sentinel registered),
        thread B arrives, sees the sentinel, parks on the Event with a 30s
        timeout.  When A's tool raises, the generic-Exception arm must pop
        the sentinel AND signal the Event so B does not block the full 30s.
        """
        ready_to_raise = threading.Event()
        release_thread_a = threading.Event()
        call = {"name": "flaky_tool", "args": {"text": "hi"}, "id": "c1"}
        call_key = 'flaky_tool:{"text": "hi"}'

        def _raises_on_release(_inp, *_a, **_kw):
            ready_to_raise.set()
            # Hold the call open so thread B reaches the wait branch before
            # A's exception arm pops the sentinel.
            release_thread_a.wait(timeout=2.0)
            raise RuntimeError("boom")

        tool = MagicMock()
        tool.name = "flaky_tool"
        tool.invoke.side_effect = _raises_on_release

        invoker, _, pending_events, _ = _make_invoker(
            tool=tool,
            tool_name="flaky_tool",
        )

        results: dict[str, Any] = {}
        errors: list[BaseException] = []

        def _run_a() -> None:
            try:
                results["a"] = invoker.invoke_one(call, None)
            except BaseException as exc:  # noqa: BLE001 — test diagnostic
                errors.append(exc)

        def _run_b() -> None:
            # Wait until A has registered its sentinel and is mid-flight.
            ready_to_raise.wait(timeout=2.0)
            # Now B enters; it should find the sentinel, park on the event,
            # and be released by A's exception arm.
            try:
                t_start = time.monotonic()
                results["b"] = invoker.invoke_one({**call, "id": "c2"}, None)
                results["b_wait_s"] = time.monotonic() - t_start  # type: ignore[assignment]
            except BaseException as exc:  # noqa: BLE001 — test diagnostic
                errors.append(exc)

        ta = threading.Thread(target=_run_a)
        tb = threading.Thread(target=_run_b)
        ta.start()
        tb.start()
        # Give B a beat to enter and park on the event.
        time.sleep(0.05)
        # Now let A raise.
        release_thread_a.set()
        ta.join(timeout=5.0)
        tb.join(timeout=5.0)

        assert not errors, f"thread raised: {errors!r}"

        # A returned the sanitised error message.
        a_result = results["a"]
        assert isinstance(a_result, ToolMessage)
        assert "Error executing flaky_tool" in a_result.content
        assert "boom" in a_result.content

        # Sentinel popped — duplicate retries won't hang.
        assert call_key not in pending_events

        # B was woken promptly (well under the 30s timeout); the value it
        # observes does not matter — what matters is it did not block 30s.
        assert results["b_wait_s"] < 2.0, (  # type: ignore[operator]
            f"thread B blocked {results['b_wait_s']:.2f}s — exception arm "
            "did not signal the pending event"
        )


class TestEarlyReturnReleasesSentinel:
    """Regression #2207: the denial / hard-budget-cap / guard-block early
    returns must release the TOCTOU sentinel reserved by the guard — exactly
    like the success and both exception arms. Leaking it strands a duplicate
    caller on the 30s Event wait, which then re-executes the tool.
    """

    def test_denied_tool_releases_sentinel(self):
        tool = MagicMock()
        tool.name = "echo_tool"
        ss = SessionState()
        ss.deny_tool("echo_tool")

        invoker, _, pending_events, _ = _make_invoker(tool=tool, session_state=ss)

        call = {"name": "echo_tool", "args": {"text": "hi"}, "id": "c1"}
        result = invoker.invoke_one(call, None)

        assert "disabled and cannot be used" in result.content
        assert tool.invoke.call_count == 0
        # The sentinel reserved by the TOCTOU guard must not be leaked.
        assert pending_events == {}

    def test_hard_budget_cap_releases_sentinel(self):
        tool = MagicMock()
        tool.name = "noisy_tool"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"resp-{inp['id']}", tool_call_id=inp["id"], name="noisy_tool"
        )

        invoker, _, pending_events, _ = _make_invoker(
            tool=tool, tool_name="noisy_tool", tool_budget_hard=2, tool_budget_soft=1
        )

        # Exhaust the hard cap (distinct args → each call_key is unique).
        for i in range(2):
            invoker.invoke_one({"name": "noisy_tool", "args": {"i": i}, "id": f"c{i}"}, None)

        # The (N+1)-th call trips the hard cap and takes the early-return arm.
        over = {"name": "noisy_tool", "args": {"i": 99}, "id": "c_over"}
        result = invoker.invoke_one(over, None)

        assert "hit its per-turn call limit (2 calls)" in result.content
        # Every key (the successful ones AND the capped one) must be released.
        assert pending_events == {}

    def test_guard_block_releases_sentinel(self):
        tool = MagicMock()
        tool.name = "echo_tool"
        blocked = MagicMock(is_safe=False, guard_name="test_guard", reason="nope")
        guard = MagicMock(return_value=blocked)

        invoker, _, pending_events, _ = _make_invoker(tool=tool, tool_call_guard=guard)

        call = {"name": "echo_tool", "args": {"text": "hi"}, "id": "c1"}
        result = invoker.invoke_one(call, None)

        assert "blocked by security policy" in result.content
        assert tool.invoke.call_count == 0
        assert pending_events == {}


class TestRetrievalRecursionCeiling:
    """#2014: retrieval/search tools were fully exempt from the hard cap, so a
    non-converging model could call them unbounded until the LangGraph recursion
    limit. They now get a recursion-aware ceiling (recursion_limit // divisor)."""

    @staticmethod
    def _search_tool() -> Any:
        tool = MagicMock()
        tool.name = "search_web"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"results-{inp['id']}", tool_call_id=inp["id"], name="search_web"
        )
        return tool

    def test_retrieval_tool_capped_at_recursion_fraction(self):
        tool = self._search_tool()
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="search_web",
            tool_budget_hard=8,
            # Mirror production: retrieval is also in the hard-exempt set, but
            # the retrieval ceiling must take precedence over full exemption.
            tool_budget_hard_exempt={"search_web"},
            tool_budget_retrieval_tools={"search_web"},
            tool_budget_retrieval_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 30}  # ceiling = max(8, 30 // 3) = 10

        for i in range(10):
            r = invoker.invoke_one({"name": "search_web", "args": {"q": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content, f"capped too early at call {i + 1}"

        # The 11th call exceeds the ceiling of 10 → hard stop (was: never capped).
        r = invoker.invoke_one({"name": "search_web", "args": {"q": 99}, "id": "c99"}, cfg)
        assert "hit its per-turn call limit (10 calls)" in r.content
        assert "search_web" in state.tool_lookup
        assert tool in active  # #2213: stays (per-turn stop, not removed)
        assert tool.invoke.call_count == 10  # the capped call did NOT execute the tool

    def test_retrieval_ceiling_scales_with_recursion_budget(self):
        # Same 11 calls, but a larger recursion budget → higher ceiling (20),
        # so the tool is still active where the tight budget (10) disabled it.
        tool = self._search_tool()
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="search_web",
            tool_budget_hard=8,
            tool_budget_hard_exempt={"search_web"},
            tool_budget_retrieval_tools={"search_web"},
            tool_budget_retrieval_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 60}  # ceiling = 60 // 3 = 20
        r = None
        for i in range(11):
            r = invoker.invoke_one({"name": "search_web", "args": {"q": i}, "id": f"c{i}"}, cfg)
        assert r is not None and "disabled" not in r.content
        assert "search_web" in state.tool_lookup

    def test_knowledge_base_gets_recursion_ceiling_not_fixed_cap(self):
        """#2213: a non-web read-only retrieval tool (query_knowledge_base, the
        #2014 runaway) must get the recursion-aware ceiling, NOT the fixed cap of
        8 — it survives past 8 progressive lookups but is still bounded (stops at
        the ceiling), rather than running unbounded to the recursion limit."""
        tool = MagicMock()
        tool.name = "query_knowledge_base"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"chunks-{inp['id']}", tool_call_id=inp["id"], name="query_knowledge_base"
        )
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="query_knowledge_base",
            tool_budget_hard=8,
            tool_budget_hard_exempt={"query_knowledge_base"},
            tool_budget_retrieval_tools={"query_knowledge_base"},
            tool_budget_retrieval_ceiling_divisor=4,
        )
        cfg = {"recursion_limit": 60}  # ceiling = max(8, 60 // 4) = 15

        # Past the fixed cap of 8 — the old classification would have stopped here.
        for i in range(15):
            r = invoker.invoke_one(
                {"name": "query_knowledge_base", "args": {"q": i}, "id": f"k{i}"}, cfg
            )
            assert "disabled" not in r.content, f"KB capped too early at call {i + 1}"

        # ...but still bounded: the 16th call exceeds the ceiling of 15 → hard stop.
        r = invoker.invoke_one(
            {"name": "query_knowledge_base", "args": {"q": 99}, "id": "k99"}, cfg
        )
        assert "hit its per-turn call limit (15 calls)" in r.content
        assert "query_knowledge_base" in state.tool_lookup
        assert tool in active  # #2213: stays (per-turn stop, not removed)


class TestDuplicateLoopTripsHardCap:
    """#2390: cache-served duplicate calls must still be charged against the
    per-tool budget, so a model spamming the EXACT same call trips the existing
    hard cap instead of looping to the wall-clock / recursion limit. Distinct
    from the reverted #2356 breaker — reuses the per-tool budget; no
    duplicate_hit_count keying, no forced thinking break, no terminal synthesis.
    """

    def test_exact_duplicate_loop_trips_hard_cap_via_cache_path(self):
        """Repeated identical calls (served from cache via the TOCTOU history
        check) eventually trip the per-tool hard cap and disable the tool."""
        tool = MagicMock()
        tool.name = "register_supplier"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content="registered", tool_call_id=inp["id"], name="register_supplier"
        )
        invoker, state, _, active = _make_invoker(
            tool=tool, tool_name="register_supplier", tool_budget_hard=3, tool_budget_soft=2
        )
        base = {"name": "register_supplier", "args": {"company": "Global Widgets"}}

        results = [invoker.invoke_one({**base, "id": f"c{i}"}, None) for i in range(4)]

        # First call executes once; calls 2-3 are cache-served (count 2,3 ≤ 3).
        assert tool.invoke.call_count == 1
        assert "disabled" not in results[1].content
        assert "disabled" not in results[2].content
        # The 4th attempt exceeds the hard cap of 3 → graceful stop, tool disabled.
        assert "hit its per-turn call limit (3 calls)" in results[3].content
        assert "synthesize" in results[3].content.lower()
        assert "register_supplier" in state.tool_lookup
        assert "register_supplier" in state.active_names
        assert tool in active  # #2213: stays (per-turn stop, not removed)

    def test_check_duplicate_pre_check_path_also_charges_budget(self):
        """In production the sequential-duplicate loop is caught by the
        check_duplicate pre-check (graph._check_duplicate), which returns BEFORE
        the TOCTOU block. That path must charge the budget too."""
        tool = MagicMock()
        tool.name = "register_supplier"

        def _always_dup(call: dict, key: str | None = None) -> ToolMessage:
            return ToolMessage(
                content="[Duplicate call] cached", tool_call_id=call["id"], name=call["name"]
            )

        invoker, state, _, _ = _make_invoker(
            tool=tool, tool_name="register_supplier", tool_budget_hard=3
        )
        invoker._check_duplicate = _always_dup  # exercise the pre-check (line-190) path

        base = {"name": "register_supplier", "args": {"x": 1}}
        results = [invoker.invoke_one({**base, "id": f"c{i}"}, None) for i in range(4)]

        # Always served as a duplicate — the tool never executes.
        assert tool.invoke.call_count == 0
        assert "hit its per-turn call limit (3 calls)" in results[3].content
        assert "register_supplier" in state.tool_lookup

    def test_hard_exempt_tool_duplicates_not_charged(self):
        """A hard-exempt tool's duplicates are not budget-charged (mirrors the
        main path's exemption) — no premature disable however many times the
        identical call repeats."""
        tool = MagicMock()
        tool.name = "shell_exec"
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content="ok", tool_call_id=inp["id"], name="shell_exec"
        )
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell_exec",
            tool_budget_hard=2,
            tool_budget_hard_exempt={"shell_exec"},
        )
        base = {"name": "shell_exec", "args": {"cmd": "ls"}}
        for i in range(6):
            r = invoker.invoke_one({**base, "id": f"c{i}"}, None)
            assert "disabled" not in r.content
        assert "shell_exec" in state.tool_lookup
        assert tool in active


@pytest.fixture(autouse=True)
def _isolate_run_state() -> None:
    """Each test gets a fresh PerRunState via _make_invoker; nothing global to reset."""
    # Placeholder for symmetry with the integration test fixtures; the class
    # is stateless across instances, so no cleanup is required.
    return None


# Defensive: re-export OrderedDict so the test stays self-contained even if a
# future refactor changes how tool_call_history is constructed (pyright would
# otherwise flag the import as unused).
_ = OrderedDict
