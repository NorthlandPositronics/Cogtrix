"""Regression tests for #2050 — API ``api_dangerous_tools`` deny-list name mismatch.

Background
----------
``warm_session`` denied shell/python by the *module* names
(``shell`` / ``python_exec``), but ``SessionState.is_denied()`` matches on the
*agent-callable* tool name (``execute_shell_command`` / ``execute_python``).
The names never matched, so with the default ``api_dangerous_tools=False`` the
exec tools stayed loadable via ``request_tools`` — an RCE on the API host.

These tests pin the behaviour:
  - default config → exec tools are denied (``is_denied`` True) and the
    session-tools endpoint classifies them as ``disabled``;
  - ``api_dangerous_tools=True`` → exec tools are not denied (loadable);
  - the deny list itself contains the canonical tool names, so a future tool
    rename cannot silently re-open the hole.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from src.api import session_bridge  # noqa: E402
from src.api.session_bridge import _API_DENIED_DANGEROUS_TOOLS, warm_session  # noqa: E402
from src.config import Config  # noqa: E402

_EXEC_TOOL_NAMES = ("execute_shell_command", "execute_python")


def _warm(app_config: object) -> object:
    """Run ``warm_session`` with mem/LLM builders stubbed; return the ApiSession.

    ``app_config`` is placed on ``app_state.config`` — the attribute
    ``warm_session`` reads for the ``api_dangerous_tools`` gate.
    """
    record = SimpleNamespace(
        id="session-2050",
        user_id="user-2050",
        name="dangerous-tools-deny",
        config_json="{}",
        token_counts_json="{}",
        state="idle",
    )
    app_state = SimpleNamespace(
        config=app_config,
        tool_registry=SimpleNamespace(tools={"read_file": object()}),
    )
    memory_manager = SimpleNamespace(
        set_llm=lambda llm: None,
        configure_compression=lambda *a, **k: None,
    )

    async def _run() -> object:
        with (
            patch.object(session_bridge, "_build_memory_manager", return_value=memory_manager),
            patch.object(session_bridge, "_build_llm", return_value=MagicMock()),
        ):
            return await warm_session(record, app_state)

    return asyncio.run(_run())


class TestApiDangerousToolsDeny:
    def test_default_config_denies_canonical_exec_tool_names(self) -> None:
        """Default (api_dangerous_tools unset/false) → exec tools are denied by
        their agent-callable names, which is what is_denied()/process_tools check."""
        # config=None → getattr(..., "api_dangerous_tools", False) is False.
        session = _warm(None)
        ss = session.session_state
        for name in _EXEC_TOOL_NAMES:
            assert ss.is_denied(name), f"{name} must be denied on a default API session"

    def test_explicit_false_denies_exec_tools(self) -> None:
        cfg = Config()  # api_dangerous_tools defaults to False
        session = _warm(cfg)
        ss = session.session_state
        for name in _EXEC_TOOL_NAMES:
            assert ss.is_denied(name)

    def test_dangerous_tools_enabled_does_not_deny(self) -> None:
        """With api_dangerous_tools=True the exec tools stay loadable."""
        cfg = Config()
        cfg.api_dangerous_tools = True
        session = _warm(cfg)
        ss = session.session_state
        for name in _EXEC_TOOL_NAMES:
            assert not ss.is_denied(name), f"{name} must be loadable when dangerous tools enabled"

    def test_session_tools_endpoint_reports_disabled(self) -> None:
        """GET /sessions/{id}/tools must classify denied exec tools as 'disabled'
        (the symptom in the #2050 repro: they showed as 'on_demand')."""
        from src.api.routes.tools import _classify_tool_status

        session = _warm(None)
        ss = session.session_state
        for name in _EXEC_TOOL_NAMES:
            assert _classify_tool_status(name, ss) == "disabled"

    def test_deny_list_contains_canonical_names_rename_guard(self) -> None:
        """Guard: the deny list must keep the canonical tool names so a future
        rename of the module cannot silently re-open the RCE hole."""
        for name in _EXEC_TOOL_NAMES:
            assert name in _API_DENIED_DANGEROUS_TOOLS
        # module aliases retained as defence-in-depth
        for alias in ("shell", "python_exec"):
            assert alias in _API_DENIED_DANGEROUS_TOOLS
