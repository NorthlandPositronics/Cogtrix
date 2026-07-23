"""Regression tests for the #2213 budget-stopped-tool bind_tools exclusion (#2444).

``call_model`` recomputes its ``bind_tools`` fingerprint whenever
``tool_version`` advances past ``last_tool_version`` (see
``cogtrix_core/orchestration/nodes/call_model.py`` around the ``_tool_version[0] !=
_last_tool_version[0]`` check). When it recomputes, any tool name present in
``budget_stopped_tools`` must be excluded from BOTH the fingerprint AND the
tool list actually handed to ``llm.bind_tools`` — otherwise a tool that hit
its per-turn call ceiling stays visible to the model and can be called again.

``tests/orchestration/test_call_model.py`` only ever exercises
``budget_stopped_tools=set()`` (the harness default), so this exclusion path
had no direct coverage prior to this file (#2432 follow-up / #2444).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from tests.orchestration.test_call_model import _make_node, _make_state


def _tool(name: str) -> SimpleNamespace:
    """A minimal stand-in for a bound tool — call_model only ever reads
    ``.name`` off active_tools_list entries before handing them to
    ``llm.bind_tools`` (which is mocked), so a bare namespace suffices."""
    return SimpleNamespace(name=name)


class TestBudgetStoppedToolsExcludedFromBindTools:
    """#2213 / #2432: a tool that hit its per-turn budget ceiling must be
    disarmed at the LLM boundary, not just refused inside process_tools."""

    def test_budget_stopped_tool_excluded_from_bind_tools_and_fingerprint(self) -> None:
        llm = MagicMock()
        tools = [_tool("search_web"), _tool("http_get")]
        cached_fingerprint: list[tuple[str, ...]] = [()]
        # tool_version != last_tool_version forces the fingerprint recompute
        # this turn — mirrors PerRunState's real defaults ([0], [-1]) and
        # the bump the invoker performs on a budget stop (tool_version+1,
        # last_tool_version=-1).
        tool_version = [0]
        last_tool_version = [-1]

        node = _make_node(
            llm=llm,
            active_tools_list=tools,
            budget_stopped_tools={"search_web"},
            cached_fingerprint=cached_fingerprint,
            tool_version=tool_version,
            last_tool_version=last_tool_version,
            invoke_with_timeout=lambda _llm, _msgs, _cfg, _to: AIMessage(content="ok"),
        )
        node(_make_state([HumanMessage(content="hi")]), {})

        assert llm.bind_tools.called, "bind_tools must run for a fresh fingerprint"
        bound_tool_names = {getattr(t, "name", "") for t in llm.bind_tools.call_args.args[0]}
        assert "search_web" not in bound_tool_names, (
            "A budget-stopped tool must be excluded from the tool list passed "
            f"to bind_tools; got {bound_tool_names!r}"
        )
        assert "http_get" in bound_tool_names

        assert "search_web" not in cached_fingerprint[0], (
            "The budget-stopped tool must also be excluded from the cache "
            f"fingerprint (or a future turn's cache-hit would re-expose it); "
            f"got {cached_fingerprint[0]!r}"
        )
        assert "http_get" in cached_fingerprint[0]

    def test_clearing_budget_stop_and_bumping_tool_version_restores_tool(self) -> None:
        """Next turn: the runner clears the per-turn ``budget_stopped_tools``
        set and bumps ``tool_version`` (see cogtrix_core/orchestration/graph.py's
        reset-on-stop semantics). The excluded tool must come back."""
        llm = MagicMock()
        tools = [_tool("search_web"), _tool("http_get")]
        cached_fingerprint: list[tuple[str, ...]] = [()]
        tool_version = [0]
        last_tool_version = [-1]
        budget_stopped: set[str] = {"search_web"}

        node = _make_node(
            llm=llm,
            active_tools_list=tools,
            budget_stopped_tools=budget_stopped,
            cached_fingerprint=cached_fingerprint,
            tool_version=tool_version,
            last_tool_version=last_tool_version,
            invoke_with_timeout=lambda _llm, _msgs, _cfg, _to: AIMessage(content="ok"),
        )
        node(_make_state([HumanMessage(content="hi")]), {})
        assert "search_web" not in cached_fingerprint[0]

        budget_stopped.clear()
        tool_version[0] += 1

        node(_make_state([HumanMessage(content="hi again")]), {})

        assert "search_web" in cached_fingerprint[0], (
            "Clearing budget_stopped_tools and bumping tool_version must "
            f"restore the tool to the fingerprint; got {cached_fingerprint[0]!r}"
        )
        last_call_names = {getattr(t, "name", "") for t in llm.bind_tools.call_args.args[0]}
        assert "search_web" in last_call_names, (
            "bind_tools must be re-invoked with the restored tool present; "
            f"got {last_call_names!r}"
        )
