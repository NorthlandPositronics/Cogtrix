"""Unit tests for the extracted thinking_break_policy module (forge A3).

These tests exercise :func:`maybe_apply_thinking_break` directly. The
behaviour under test:

* When the flag is **clear**, the function is a no-op (returns None,
  no LLM call, no message append, no state mutation).
* When the flag is **set**, the function:
    - clears the flag,
    - resets the error-counter cells,
    - chooses one of five body variants based on checkpoint presence,
      tool-loop classification, search effort, and result substantiveness,
    - either fires a sub-invocation (returns a ``{"messages": [resp]}``
      dict) OR appends a STRATEGY NUDGE and returns None (low-effort
      search-loop suppression).
* On sub-invocation **timeout**, the function returns the graceful
  fallback ``{"messages": []}`` rather than re-raising.

Test-patching contract
======================
The helpers (``_compute_search_effort``, ``_has_substantive_search_results``,
``_has_arithmetic_intent``, ``_has_numeric_tool_results``,
``_MIN_SEARCH_EFFORT``) live in ``call_model.py`` and are looked up via
the module attribute. Tests patch them at
``src.orchestration.nodes.call_model.<name>`` so the patches stick when
``thinking_break_policy`` does its attribute lookup.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.orchestration.nodes.call_model import CallModelContext
from cogtrix_core.orchestration.nodes.thinking_break_policy import maybe_apply_thinking_break


class _DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[Any, ...]] = []
        self.warnings: list[tuple[Any, ...]] = []
        self.debugs: list[tuple[Any, ...]] = []

    def info(self, *args: Any) -> None:
        self.infos.append(args)

    def warning(self, *args: Any) -> None:
        self.warnings.append(args)

    def debug(self, *args: Any) -> None:
        self.debugs.append(args)


def _make_context(**overrides: Any) -> CallModelContext:
    defaults: dict[str, Any] = {
        "llm": MagicMock(),
        "tools_ready": None,
        "active_tools_list": [],
        "active_names": set(),
        "budget_stopped_tools": set(),
        "bound_cache": OrderedDict(),
        "bound_cache_lock": MagicMock(),
        "cached_fingerprint": [()],
        "compression_cache": {},
        "tool_version": [0],
        "last_tool_version": [0],
        "call_count": [1],
        "last_input_tokens": [0],
        "max_context_tokens": None,
        "context_max_messages": 0,
        "context_max_tokens": 0,
        "model_max_tokens": None,
        "compression_llm": None,
        "memory_manager": None,
        "checkpoint_store": None,
        "calls_since_last_checkpoint": [0],
        "last_checkpoint_count": [0],
        "rounds_since_checkpoint": [0],
        "force_thinking_break": [False],
        "consecutive_errors": [0],
        "last_identical_error_signature": [None],
        "consecutive_identical_error_count": [0],
        "last_reflection_at": [0],
        "tool_health_check_interval": 0,
        "last_tool_health_check_at": [0],
        "tool_quality_gate_enabled": False,
        "topic_switch_detection_enabled": False,
        "stuck_threshold": 5,
        "stuck_no_checkpoint_threshold": [20],
        "stuck_threshold_calibrated": [True],
        "checkpoint_nudge_interval": 10,
        "reflection_interval": 20,
        "max_request_tools_noops": 3,
        "sys_msg": None,
        "model_timeout": 120,
        "tool_context_limit_pct": 0.5,
        "da_enabled": False,
        "da_report_uncertainty": False,
        "da_min_confidence": 5.0,
        "apply_context_message_cap": lambda msgs, max_msgs, max_tokens: msgs,
        "maybe_compress": lambda msgs: msgs,
        "invoke_with_timeout": lambda llm, msgs, config, timeout: AIMessage(content="ok"),
        "all_tool_results_substanceless": lambda msgs: False,
    }
    defaults.update(overrides)
    return CallModelContext(**defaults)  # type: ignore[arg-type]


class _StubCheckpointStore:
    def __init__(self, count: int = 0) -> None:
        self._count = count

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._count


# ─────────────────────────────────────────────────────────────────────
# Test 7 — flag-clear path returns None
# ─────────────────────────────────────────────────────────────────────


class TestReturnsNoneWhenFlagClear:
    def test_no_invocation_no_message_no_mutation(self):
        log = _DummyLogger()
        invoke_mock = MagicMock(return_value=AIMessage(content="should not be called"))
        context = _make_context(
            force_thinking_break=[False],
            invoke_with_timeout=invoke_mock,
        )
        msgs: list[Any] = [HumanMessage(content="hello")]
        result = maybe_apply_thinking_break(context, list(msgs), list(msgs), msgs, {}, log)

        assert result is None, "flag-clear path must return None"
        invoke_mock.assert_not_called()
        # No append happened — len unchanged.
        assert len(msgs) == 1
        # Counters untouched.
        assert context.consecutive_errors[0] == 0
        assert context.force_thinking_break[0] is False


# ─────────────────────────────────────────────────────────────────────
# Test 8 — flag-set path invokes the LLM
# ─────────────────────────────────────────────────────────────────────


class TestInvokesLLMWhenFlagSet:
    def test_non_search_stuck_invokes_via_invoke_with_timeout(self):
        """Non-search tool loop (e.g. merge_pr) fires the refusal-style
        thinking break. The sub-invocation must be made via the
        ``invoke_with_timeout`` callable on the context."""
        log = _DummyLogger()
        captured: list[Any] = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured.append((llm_obj, list(msgs), timeout))
            return AIMessage(content="thinking-break response")

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        context = _make_context(
            llm=llm,
            force_thinking_break=[True],
            invoke_with_timeout=capture_invoke,
            active_tools_list=[],
        )
        # Last tool was "merge_pr" — not a search loop, so the refusal
        # body fires unconditionally.
        repaired = [
            HumanMessage(content="merge it"),
            AIMessage(content="r1", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1", name="merge_pr"),
        ]
        msgs = list(repaired)
        result = maybe_apply_thinking_break(context, list(repaired), repaired, msgs, {}, log)

        assert result is not None, "flag-set + non-search-loop must fire the sub-invocation"
        assert result == {"messages": [AIMessage(content="thinking-break response")]}
        assert len(captured) == 1, "exactly one LLM sub-invocation expected"
        # Timeout for the thinking break is hard-coded to 180 (see source).
        _, _, timeout = captured[0]
        assert timeout == 180, "thinking-break sub-invocation must use the 180s timeout"


# ─────────────────────────────────────────────────────────────────────
# Test 9 — flag cleared after invoke (regardless of body variant)
# ─────────────────────────────────────────────────────────────────────


class TestClearsFlagAfterInvoke:
    def test_flag_cleared_when_body_fires(self):
        log = _DummyLogger()
        context = _make_context(
            force_thinking_break=[True],
            consecutive_errors=[5],
            consecutive_identical_error_count=[3],
            last_identical_error_signature=[("foo", "bar")],
        )
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(content="r1", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1", name="merge_pr"),
        ]
        msgs = list(repaired)
        result = maybe_apply_thinking_break(context, list(repaired), repaired, msgs, {}, log)

        assert result is not None
        # All four state cells were reset.
        assert context.force_thinking_break[0] is False
        assert context.consecutive_errors[0] == 0
        assert context.consecutive_identical_error_count[0] == 0
        assert context.last_identical_error_signature[0] is None

    def test_flag_cleared_even_on_suppression_path(self):
        """Low-effort search loop branch: flag is set, body is
        suppressed (returns None), but flag must still be cleared and
        error counters reset (the flag was consumed)."""
        log = _DummyLogger()
        context = _make_context(
            force_thinking_break=[True],
            consecutive_errors=[5],
        )
        # search_web with empty results and low effort triggers the
        # strategy-nudge / suppression branch. Patch
        # _compute_search_effort to return (0, False) to be deterministic.
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(content="r1", tool_calls=[{"name": "search_web", "args": {}, "id": "t1"}]),
            ToolMessage(content="nothing", tool_call_id="t1", name="search_web"),
        ]
        msgs = list(repaired)
        with patch(
            "cogtrix_core.orchestration.nodes.call_model._compute_search_effort",
            return_value=(0, False),
        ):
            result = maybe_apply_thinking_break(context, list(repaired), repaired, msgs, {}, log)

        assert result is None, "low-effort suppression must return None so caller continues"
        assert context.force_thinking_break[0] is False
        assert context.consecutive_errors[0] == 0
        # STRATEGY NUDGE was appended.
        assert any(
            isinstance(m, HumanMessage) and "STRATEGY NUDGE" in m.content for m in msgs
        ), "low-effort suppression must inject the STRATEGY NUDGE before falling through"


# ─────────────────────────────────────────────────────────────────────
# Test 10 — timeout handling
# ─────────────────────────────────────────────────────────────────────


class TestHandlesTimeout:
    def test_returns_graceful_fallback_on_runtime_error(self):
        """``_invoke_with_timeout`` wraps the sub-invocation with a
        thread-pool + timeout and raises ``RuntimeError`` on timeout.
        The thinking break must catch that and return
        ``{"messages": []}`` so the graph can continue to a fresh
        round, not re-raise and crash the agent turn."""
        log = _DummyLogger()

        def raise_timeout(llm_obj, msgs, config, timeout):
            raise RuntimeError("model invocation timed out")

        context = _make_context(
            force_thinking_break=[True],
            invoke_with_timeout=raise_timeout,
        )
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(content="r1", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1", name="merge_pr"),
        ]
        msgs = list(repaired)
        result = maybe_apply_thinking_break(context, list(repaired), repaired, msgs, {}, log)

        assert result == {"messages": []}, (
            "thinking-break timeout must return the graceful "
            "empty-messages fallback, not re-raise"
        )
        # The flag must still have been cleared (consumed on entry).
        assert context.force_thinking_break[0] is False
        # Log a warning so ops can see the timeout.
        assert any("timed out" in str(w).lower() for w in log.warnings)
