"""#2269 — model-controlled tool pinning via ``request_tools(keep_loaded=[...])``.

A pinned tool is exempt from the *fixed* per-tool budget hard cap for the run;
instead it gets the recursion-aware ceiling (so a long task can use it many
times without losing it mid-run) while still bounded (a non-converging model
can't loop to the recursion limit). Pins are per-run (reset by
``_reset_for_new_run`` — a fresh ``PerRunState`` each turn), bounded to
``_MAX_PINNED_TOOLS``, and set through the ``request_tools`` control tool.

Layers under test:
  * parsing — ``_detect_tool_request`` extracts ``keep_loaded``;
  * budget — ``DedupedToolInvoker`` gives a pinned tool the ceiling, not the cap;
  * dispatch — ``process_tools`` applies/validates/bounds the pin;
  * lifecycle — a fresh ``PerRunState`` starts with no pins.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from cogtrix_core.orchestration.graph import _MAX_PINNED_TOOLS, _detect_tool_request
from cogtrix_core.orchestration.graph_runtime import PerRunState
from tests.orchestration.test_deduped_tool_invoker import _make_invoker, _make_per_run_state
from tests.orchestration.test_process_tools import _make_ai_msg, _make_node, _make_state


def _rt(args: dict) -> object:
    """An AIMessage whose only tool call is request_tools(**args)."""
    return _make_ai_msg([{"name": "request_tools", "args": args, "id": "tc1"}])


def _noisy_tool(name: str = "shell") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
        content=f"r-{inp['id']}", tool_call_id=inp["id"], name=name
    )
    return tool


class TestDetectKeepLoaded:
    def test_parses_keep_loaded_list(self) -> None:
        req = _detect_tool_request([_rt({"keep_loaded": ["shell", "http_get"]})], 0)
        assert req is not None
        assert req.keep_loaded == ["shell", "http_get"]
        # keep_loaded ALONE (no add/remove) still counts as a change to process.
        assert req.has_changes

    def test_coerces_bare_string(self) -> None:
        req = _detect_tool_request([_rt({"keep_loaded": "shell"})], 0)
        assert req is not None and req.keep_loaded == ["shell"]

    def test_dedupes_and_strips_empty(self) -> None:
        req = _detect_tool_request([_rt({"keep_loaded": ["a", "a", ""]})], 0)
        assert req is not None and req.keep_loaded == ["a"]

    def test_no_request_returns_none(self) -> None:
        assert (
            _detect_tool_request([_make_ai_msg([{"name": "t1", "args": {}, "id": "x"}])], 0) is None
        )


class TestPinnedBudget:
    """The invoker gives a pinned tool the recursion-aware ceiling, not the cap."""

    def test_pinned_survives_past_fixed_hard_cap(self) -> None:
        tool = _noisy_tool()
        state = _make_per_run_state(tool, "shell")
        state.pinned_tools.add("shell")
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell",
            per_run_state=state,
            tool_budget_hard=3,
            tool_budget_soft=2,
            tool_budget_retrieval_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 30}  # ceiling = max(3, 30 // 3) = 10
        # 9 calls — well past the fixed cap of 3, still under the ceiling of 10.
        for i in range(9):
            r = invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content, f"pinned tool disabled at call {i}"
        assert tool in active
        assert "shell" in state.tool_lookup

    def test_pinned_still_bounded_by_ceiling(self) -> None:
        tool = _noisy_tool()
        state = _make_per_run_state(tool, "shell")
        state.pinned_tools.add("shell")
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell",
            per_run_state=state,
            tool_budget_hard=3,
            tool_budget_retrieval_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 12}  # ceiling = max(3, 12 // 3) = 4
        for i in range(4):
            r = invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
            assert "disabled" not in r.content
        over = invoker.invoke_one({"name": "shell", "args": {"i": 99}, "id": "over"}, cfg)
        assert "per-turn call limit (4 calls)" in over.content  # pinned ≠ uncapped
        # #2213: budget-stop keeps the tool in active_tools_list (per-turn pause,
        # filtered from bind_tools; restored next turn) — no longer removed.
        assert tool in active
        assert "shell" in state.budget_stopped_tools

    def test_unpinned_tool_still_caps_at_fixed(self) -> None:
        tool = _noisy_tool()
        invoker, state, _, active = _make_invoker(
            tool=tool,
            tool_name="shell",
            tool_budget_hard=3,
            tool_budget_retrieval_ceiling_divisor=3,
        )
        cfg = {"recursion_limit": 30}  # would be ceiling 10 IF pinned — but it isn't
        for i in range(3):
            invoker.invoke_one({"name": "shell", "args": {"i": i}, "id": f"c{i}"}, cfg)
        over = invoker.invoke_one({"name": "shell", "args": {"i": 99}, "id": "over"}, cfg)
        assert "per-turn call limit (3 calls)" in over.content  # fixed cap, not the ceiling
        # #2213: per-turn stop keeps it in active (restored next turn), not removed.
        assert tool in active
        assert "shell" in state.budget_stopped_tools


class TestPinViaDispatcher:
    """process_tools applies keep_loaded pins with validation + the per-run limit."""

    def _node(
        self,
        pin_set: set[str],
        max_pins: int = 2,
        active: set[str] | None = None,
        session_state: object | None = None,
        budget_stopped_tools: set[str] | None = None,
        tool_version: list[int] | None = None,
    ):
        kw: dict = dict(
            _detect_tool_request=_detect_tool_request,  # the real parser
            _invoke_one=MagicMock(
                return_value=ToolMessage(content="ok", tool_call_id="tc1", name="request_tools")
            ),
            _tool_lookup={"request_tools": MagicMock()},
            _active_names=(active if active is not None else {"request_tools", "shell"}),
            _pinned_tools=pin_set,
            _max_pinned_tools=max_pins,
        )
        if session_state is not None:
            kw["session_state"] = session_state
        if budget_stopped_tools is not None:
            kw["budget_stopped_tools"] = budget_stopped_tools
        if tool_version is not None:
            kw["_tool_version"] = tool_version
        return _make_node(**kw)

    def test_keep_loaded_pins_a_known_tool(self) -> None:
        pins: set[str] = set()
        node = self._node(pins)
        node(_make_state(_rt({"keep_loaded": ["shell"]})), RunnableConfig())
        assert "shell" in pins

    def test_pin_limit_enforced(self) -> None:
        pins = {"a", "b"}  # already at the limit of 2
        node = self._node(pins, max_pins=2, active={"request_tools", "shell", "a", "b"})
        result = node(_make_state(_rt({"keep_loaded": ["shell"]})), RunnableConfig())
        assert "shell" not in pins
        assert any("Pin limit reached" in getattr(m, "content", "") for m in result["messages"])

    def test_unknown_tool_not_pinned(self) -> None:
        pins: set[str] = set()
        node = self._node(pins)
        result = node(_make_state(_rt({"keep_loaded": ["nonexistent_tool"]})), RunnableConfig())
        assert pins == set()
        assert any(
            "not a recognised tool name" in getattr(m, "content", "") for m in result["messages"]
        )

    def test_pinning_undenies_a_capped_tool(self) -> None:
        from cogtrix_core.common.types import SessionState

        ss = SessionState()
        ss.deny_tool("shell")  # simulate it having hit the cap earlier this run
        assert ss.is_denied("shell")
        pins: set[str] = set()
        node = self._node(pins, session_state=ss)
        node(_make_state(_rt({"keep_loaded": ["shell"]})), RunnableConfig())
        assert "shell" in pins
        assert not ss.is_denied("shell")  # pin rescued the capped tool

    def test_pinning_lifts_a_budget_stop(self) -> None:
        # #2213: the budget hard-cap now records a per-turn budget-stop instead of
        # a session deny. Pinning must lift THAT too, else the tool the model just
        # asked to keep stays filtered out of bind_tools for the rest of the turn.
        stopped = {"shell"}  # simulate it having hit its per-turn call limit
        tv = [7]
        pins: set[str] = set()
        node = self._node(pins, budget_stopped_tools=stopped, tool_version=tv)
        node(_make_state(_rt({"keep_loaded": ["shell"]})), RunnableConfig())
        assert "shell" in pins
        assert "shell" not in stopped  # pin lifted the per-turn budget-stop
        assert tv[0] == 8  # tool_version bumped so call_model rebinds with it back


class TestPinLifecycle:
    def test_fresh_per_run_state_has_no_pins(self) -> None:
        # A fresh PerRunState is built by _reset_for_new_run each turn, and its
        # set fields are cleared in-place — so every run starts with no pins.
        assert PerRunState().pinned_tools == set()

    def test_pins_are_per_instance(self) -> None:
        a, b = PerRunState(), PerRunState()
        a.pinned_tools.add("shell")
        assert b.pinned_tools == set()  # not shared across runs

    def test_max_pins_is_bounded(self) -> None:
        assert isinstance(_MAX_PINNED_TOOLS, int) and 1 <= _MAX_PINNED_TOOLS <= 10
