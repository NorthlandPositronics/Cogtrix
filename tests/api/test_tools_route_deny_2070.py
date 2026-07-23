"""Regression test for #2070 — the tools route must not re-open denied exec tools.

PATCH /sessions/{id}/tools `enable` calls allow_tool() (deletes the deny) and
`load` activates a tool with no is_denied check. On an api_dangerous_tools=false
host this re-opens the #2050 RCE. The route must reject load/enable of the
denied exec tools while dangerous tools are disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

import cogtrix_core.api.routes.tools as tools_mod  # noqa: E402
from cogtrix_core.api.schemas.tool import ToolActionRequest  # noqa: E402
from cogtrix_core.orchestration.session_state import SessionState  # noqa: E402


def _request(api_dangerous_tools: bool):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(config=SimpleNamespace(api_dangerous_tools=api_dangerous_tools))
        )
    )


async def _call(body: ToolActionRequest, *, api_dangerous_tools: bool = False):
    ss = SessionState(no_confirm=True)
    ss.deny_tool("execute_shell_command")
    ss.deny_tool("execute_python")
    live = SimpleNamespace(session_state=ss)
    registry = MagicMock()
    registry.tools = {"execute_shell_command": object(), "execute_python": object()}
    sess_reg = MagicMock()
    sess_reg.get_or_warm = AsyncMock(return_value=live)
    with (
        patch.object(tools_mod, "verify_session_owner", AsyncMock()),
        patch.object(tools_mod, "_get_registry", return_value=registry),
        patch.object(tools_mod, "_get_session_registry", return_value=sess_reg),
    ):
        return await tools_mod.patch_session_tools(
            "sess1",
            body,
            _request(api_dangerous_tools),
            current_user=MagicMock(),
            db=MagicMock(),
        )


@pytest.mark.asyncio
async def test_enable_dangerous_tool_rejected_when_disabled() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(ToolActionRequest(enable=["execute_shell_command"]))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_load_dangerous_tool_rejected_when_disabled() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(ToolActionRequest(load=["execute_python"]))
    assert exc.value.status_code == 403
