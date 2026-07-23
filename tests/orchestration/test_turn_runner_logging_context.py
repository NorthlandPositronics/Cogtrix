"""Regression tests for session logging context in the API turn runner."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cogtrix_core.logging_config import get_session_id


@pytest.mark.asyncio
async def test_run_message_turn_propagates_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from cogtrix_core.api.turn_runner import run_message_turn

    session = SimpleNamespace(
        id="sess-abc",
        agent_state="idle",
        token_counts={},
        session_state=SimpleNamespace(reset_for_new_prompt=lambda: None),
        memory_manager=None,
        run_config=None,
        registry=None,
        turn_lock=asyncio.Lock(),
        cancel_event=asyncio.Event(),
        ws_queue=asyncio.Queue(),
        active_confirmation_ui=None,
        last_activity=0.0,
    )

    seen: dict[str, str] = {}

    def _fake_run_agent(*args, **kwargs) -> str:
        seen["session_id"] = get_session_id()
        return "ok"

    monkeypatch.setattr("cogtrix_core.orchestration.runner.run_agent", _fake_run_agent)

    await run_message_turn(session, "hello")

    assert seen["session_id"] == "sess-abc"
    assert get_session_id() == "-"
