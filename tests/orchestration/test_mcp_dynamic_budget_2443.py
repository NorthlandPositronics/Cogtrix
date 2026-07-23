"""#2443 — MCP tools activated mid-turn get their declared budget ceiling.

`_resolve_budget_category_sets` freezes the retrieval/action name-sets once at
graph build. A tool activated *after* build (an MCP retrieval/action tool loaded
mid-turn via ``request_tools``) is absent from those sets, so it fell to the
STANDARD fixed cap on its first active turn. The fix resolves the category from
the LIVE tool object (via an injected ``resolve_tool_category``) in the STANDARD
branch, memoized per ``tool_version``.

These use nonsense tool names so `categorize_tool` can't classify them by name —
any recursion-aware ceiling must come from the tool's *declaration*, proving the
dynamic seam works.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage

from cogtrix_core.orchestration.graph import resolve_tool_category
from tests.orchestration.test_deduped_tool_invoker import _make_invoker


def _name(tool: Any, max_len: int = 80) -> str:
    return str(getattr(tool, "name", tool))[:max_len]


def _mcp_tool(name: str, budget_category: str | None) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.metadata = {"budget_category": budget_category} if budget_category else None
    tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
        content=f"r-{inp['id']}", tool_call_id=inp["id"], name=name
    )
    return tool


class TestMidTurnMcpBudgetCategory:
    def test_declared_retrieval_tool_absent_from_frozen_set_gets_ceiling(self) -> None:
        # "zz_widget_a" is NOT in the (empty) build-time retrieval set — the #2443
        # gap — but declares budget_category=retrieval on the live tool.
        tool = _mcp_tool("zz_widget_a", "retrieval")
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_a",
            tool_budget_hard=3,
            tool_budget_retrieval_tools=set(),  # absent at build
            tool_budget_retrieval_ceiling_divisor=3,
            resolve_tool_category=resolve_tool_category,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 30}  # retrieval ceiling = max(3, 30 // 3) = 10
        for i in range(9):  # past the fixed cap of 3, under the ceiling of 10
            r = invoker.invoke_one({"name": "zz_widget_a", "args": {"i": i}, "id": f"c{i}"}, cfg)
            # The tool actually executed (result present) — a STANDARD-capped tool
            # would be HARD-stopped at 3. Reaching call 8 (with only the advisory
            # soft nudge appended past 5) proves it got the retrieval ceiling.
            assert f"r-c{i}" in r.content, f"tool hard-stopped at call {i}: {r.content!r}"
        assert "zz_widget_a" not in state.budget_stopped_tools

    def test_declared_retrieval_tool_still_bounded_by_ceiling(self) -> None:
        # The lift is to a still-BOUNDED ceiling, never uncapped.
        tool = _mcp_tool("zz_widget_a", "retrieval")
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_a",
            tool_budget_hard=3,
            tool_budget_retrieval_ceiling_divisor=3,
            resolve_tool_category=resolve_tool_category,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 12}  # ceiling = max(3, 12 // 3) = 4
        for i in range(4):
            invoker.invoke_one({"name": "zz_widget_a", "args": {"i": i}, "id": f"c{i}"}, cfg)
        over = invoker.invoke_one({"name": "zz_widget_a", "args": {"i": 99}, "id": "over"}, cfg)
        assert "zz_widget_a" in state.budget_stopped_tools
        assert "call limit" in over.content or "disabled" in over.content

    def test_undeclared_tool_still_hits_fixed_cap(self) -> None:
        # No declaration → resolve_tool_category → STANDARD → fixed cap unchanged.
        tool = _mcp_tool("zz_widget_b", None)
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_b",
            tool_budget_hard=3,
            resolve_tool_category=resolve_tool_category,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 30}
        for i in range(3):
            invoker.invoke_one({"name": "zz_widget_b", "args": {"i": i}, "id": f"c{i}"}, cfg)
        invoker.invoke_one({"name": "zz_widget_b", "args": {"i": 99}, "id": "over"}, cfg)
        assert "zz_widget_b" in state.budget_stopped_tools

    def test_non_dict_metadata_does_not_crash_and_keeps_fixed_cap(self) -> None:
        # Gate-1 regression: a live tool whose ``.metadata`` is a NON-dict (a bare
        # MagicMock auto-attribute, as real built-ins/harness tools carry) must not
        # trigger name-based categorization or crash — it keeps the STANDARD cap.
        # (The first cut ran resolve_tool_category over every else-branch tool and
        # crashed on a MagicMock tool name in the quality harness.)
        tool = MagicMock()
        tool.name = "zz_widget_c"  # metadata left as the MagicMock auto-attr (non-dict)
        tool.invoke.side_effect = lambda inp, *a, **k: ToolMessage(
            content=f"r-{inp['id']}", tool_call_id=inp["id"], name="zz_widget_c"
        )
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_c",
            tool_budget_hard=3,
            resolve_tool_category=resolve_tool_category,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 30}
        for i in range(3):
            invoker.invoke_one({"name": "zz_widget_c", "args": {"i": i}, "id": f"c{i}"}, cfg)
        over = invoker.invoke_one({"name": "zz_widget_c", "args": {"i": 99}, "id": "over"}, cfg)
        assert "zz_widget_c" in state.budget_stopped_tools  # fixed cap, no crash
        assert "call limit" in over.content or "disabled" in over.content

    def test_no_resolver_injected_falls_back_to_fixed_cap(self) -> None:
        # Backward-compat: without a resolver, the dynamic path is inert.
        tool = _mcp_tool("zz_widget_a", "retrieval")
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_a",
            tool_budget_hard=3,
            resolve_tool_category=None,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 30}
        for i in range(3):
            invoker.invoke_one({"name": "zz_widget_a", "args": {"i": i}, "id": f"c{i}"}, cfg)
        invoker.invoke_one({"name": "zz_widget_a", "args": {"i": 99}, "id": "over"}, cfg)
        assert "zz_widget_a" in state.budget_stopped_tools

    def test_resolution_is_memoized_per_tool_version(self) -> None:
        calls: list[Any] = []

        def counting_resolver(tool: Any) -> Any:
            calls.append(tool)
            return resolve_tool_category(tool)

        tool = _mcp_tool("zz_widget_a", "retrieval")
        invoker, state, _, _ = _make_invoker(
            tool=tool,
            tool_name="zz_widget_a",
            tool_budget_hard=3,
            tool_budget_retrieval_ceiling_divisor=3,
            resolve_tool_category=counting_resolver,
            safe_tool_name=_name,
        )
        cfg = {"recursion_limit": 30}  # ceiling 10 — 3 calls stay under it
        for i in range(3):
            invoker.invoke_one({"name": "zz_widget_a", "args": {"i": i}, "id": f"c{i}"}, cfg)
        assert len(calls) == 1, "category should be resolved once and memoized"

        # A tool-set change bumps tool_version → memo invalidates → re-resolve.
        state.tool_version[0] += 1
        invoker.invoke_one({"name": "zz_widget_a", "args": {"i": 100}, "id": "c100"}, cfg)
        assert len(calls) == 2
