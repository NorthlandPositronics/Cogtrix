"""Regression test for #2070 — denial must be enforced at the execution chokepoint.

The #2050 deny (api_dangerous_tools) was bypassable because nothing re-checked
``is_denied`` once a tool reached the active set (e.g. via PATCH
/sessions/{id}/tools ``load``), and the safety wrapper's check is skipped for API
sessions (no_confirm=True). ``DedupedToolInvoker.invoke_one`` must block a denied
tool regardless of how it was activated.
"""

from __future__ import annotations

import threading

from cogtrix_core.orchestration.deduped_tool_invoker import DedupedToolInvoker
from cogtrix_core.orchestration.session_state import SessionState


def _make_invoker(session_state: SessionState) -> DedupedToolInvoker:
    return DedupedToolInvoker(
        per_run_state=[object()],  # untouched on the denied early-return path
        history_lock=threading.Lock(),
        tool_budget_lock=threading.Lock(),
        bound_cache_lock=threading.Lock(),
        pending_events={},
        active_tools_list=[],
        session_state=session_state,
        tool_call_guard=None,
        tool_call_key=lambda call: None,  # None -> skip the TOCTOU/dedup block
        check_duplicate=lambda call, key=None: None,
        correct_tool_args=lambda *a, **k: {},
        safe_tool_name=lambda t: str(t),
        max_tool_call_history=50,
        tool_budget_hard=8,
        tool_budget_soft=5,
        tool_budget_hard_exempt=frozenset(),
        tool_budget_soft_exempt=frozenset(),
    )


def test_denied_tool_blocked_at_execution_chokepoint() -> None:
    ss = SessionState(no_confirm=True)
    ss.deny_tool("execute_shell_command")
    result = _make_invoker(ss).invoke_one(
        {"name": "execute_shell_command", "id": "c1", "args": {"command": "id"}},
        run_config=None,
    )
    assert "disabled" in result.content.lower()
    assert result.tool_call_id == "c1"
    assert result.name == "execute_shell_command"


def test_deny_all_blocks_any_tool_at_execution() -> None:
    ss = SessionState(no_confirm=True)
    ss.set_deny_all()
    result = _make_invoker(ss).invoke_one(
        {"name": "read_file", "id": "c2", "args": {"path": "x"}},
        run_config=None,
    )
    assert "disabled" in result.content.lower()
    assert result.tool_call_id == "c2"
