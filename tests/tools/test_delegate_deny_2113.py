"""Regression test for #2113 — delegate sub-agents must honour the session's
runtime denials, not just the static `_DELEGATE_EXCLUDED_TOOLS` list.

Before the fix, `set_delegate_tools` filtered only by the static exclude list, so
a tool the session denied (via `api_dangerous_tools`, per-turn budget, or
`/tools disable`) that wasn't *also* hardcoded in the delegate list was still
reachable inside a delegate sub-agent — an RCE-bypass class once the two lists
drift. The fix threads the session denial snapshot (+ deny_all) into
`set_delegate_tools`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.tools.delegate import get_delegate_tools, set_delegate_tools


def _tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.delegate_exclude_override = None  # avoid MagicMock truthiness in _is_tool_excluded
    return t


def _names() -> list[str]:
    return [getattr(t, "name", "") for t in get_delegate_tools()]


def test_session_denied_tool_excluded_from_delegate() -> None:
    safe = _tool("safe_research_tool")  # not in the static exclude list, no category

    # Baseline: with no denials, the tool reaches the delegate set.
    set_delegate_tools([safe], None)
    assert "safe_research_tool" in _names()

    # With the session denying it, it must NOT reach the delegate set.
    set_delegate_tools([safe], None, denials=frozenset({"safe_research_tool"}))
    assert "safe_research_tool" not in _names()


def test_denied_tool_in_available_set_also_excluded() -> None:
    safe = _tool("avail_only_tool")
    set_delegate_tools([], {"avail_only_tool": safe}, denials=frozenset({"avail_only_tool"}))
    assert "avail_only_tool" not in _names()


def test_deny_all_yields_empty_delegate_set() -> None:
    set_delegate_tools([_tool("a"), _tool("b")], {"c": _tool("c")}, deny_all=True)
    assert get_delegate_tools() == []
