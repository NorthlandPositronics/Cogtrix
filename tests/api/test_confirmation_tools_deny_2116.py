"""Regression tests for #2116 — confirmation-gated tools auto-execute on
no_confirm API sessions.

API sessions run with ``no_confirm=True``, so the safety wrapper's confirmation
block (and its ``is_denied`` re-check) is skipped — a tool whose only guard is
``requires_confirmation=True`` (e.g. ``http_post``, ``write_file``) would execute
with no gate. Agreed policy (deny by default): deny the whole confirmation class
on API sessions at warm time unless ``api_dangerous_tools`` is enabled; the
execution chokepoint then blocks them via ``is_denied``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from src.api import session_bridge  # noqa: E402
from src.api.session_bridge import confirmation_required_tool_names, warm_session  # noqa: E402
from src.config import Config  # noqa: E402


class _Registry:
    """Minimal ToolRegistry stand-in with the two methods the policy uses."""

    def __init__(self) -> None:
        self.tools = {"read_file": object(), "http_post": object(), "write_file": object()}
        self._confirm = {"http_post", "write_file"}

    def list_tools(self) -> list[str]:
        return list(self.tools)

    def requires_confirmation(self, name: str) -> bool:
        return name in self._confirm


def _warm(app_config: object, registry: object) -> object:
    record = SimpleNamespace(
        id="session-2116",
        user_id="user-2116",
        name="confirmation-deny",
        config_json="{}",
        token_counts_json="{}",
        state="idle",
    )
    app_state = SimpleNamespace(config=app_config, tool_registry=registry)
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


class TestConfirmationRequiredToolNames:
    def test_returns_only_confirmation_gated_tools(self) -> None:
        assert set(confirmation_required_tool_names(_Registry())) == {"http_post", "write_file"}

    def test_none_registry_returns_empty(self) -> None:
        assert confirmation_required_tool_names(None) == []

    def test_registry_without_methods_returns_empty(self) -> None:
        # A registry shape lacking list_tools/requires_confirmation must not raise.
        assert confirmation_required_tool_names(SimpleNamespace(tools={"x": object()})) == []


class TestWarmSessionDeniesConfirmationTools:
    def test_default_config_denies_confirmation_tools(self) -> None:
        ss = _warm(None, _Registry()).session_state
        assert ss.is_denied("http_post"), "http_post must be denied on a default API session"
        assert ss.is_denied("write_file"), "write_file must be denied on a default API session"

    def test_non_confirmation_tool_not_denied(self) -> None:
        ss = _warm(None, _Registry()).session_state
        assert not ss.is_denied("read_file"), "non-confirmation tools must stay loadable"

    def test_dangerous_tools_enabled_does_not_deny(self) -> None:
        cfg = Config()
        cfg.api_dangerous_tools = True
        ss = _warm(cfg, _Registry()).session_state
        assert not ss.is_denied("http_post"), "api_dangerous_tools=True re-enables the class"
        assert not ss.is_denied("write_file")
