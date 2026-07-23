"""Regression test for #2114 — the API turn runner must sanitize runner-level
exceptions before surfacing them to clients.

When ``run_agent`` raises, ``_run_message_turn_inner`` emits the error over the
WebSocket ``error``/``done`` frames (and the REST ``?sync=true`` path returns the
``done.error`` verbatim in its 500 body). Previously it sent raw ``str(exc)``,
leaking provider base URLs, API keys, and filesystem paths to a chat user who can
deliberately induce errors. The fix routes it through ``sanitize_error`` — the
API-side analog of the assistant fix in #2052.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_run_message_turn_sanitizes_runner_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from cogtrix_core.api.turn_runner import run_message_turn

    session = SimpleNamespace(
        id="sess-leak",
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

    secret = (
        "connect to https://api.internal-provider.example/v1 failed: "
        "key=sk-LEAKED-DEADBEEF reading /home/cogtrix/.cogtrix.yaml"
    )

    # An unrecognized exception type (as a provider/LangChain error often is) →
    # exercises the API-facing fallback added in #2114.
    class _ProviderBoom(Exception):
        pass

    def _raising_run_agent(*args: object, **kwargs: object) -> str:
        raise _ProviderBoom(secret)

    # _run_message_turn_inner imports run_agent from this module at call time.
    monkeypatch.setattr("cogtrix_core.orchestration.runner.run_agent", _raising_run_agent)

    await run_message_turn(session, "hello")

    frames = []
    while not session.ws_queue.empty():
        frames.append(session.ws_queue.get_nowait())

    error_frames = [f for f in frames if f.get("type") == "error"]
    done_frames = [f for f in frames if f.get("type") == "done"]
    assert error_frames, "expected an error frame"
    assert done_frames, "expected a done frame"

    err_msg = error_frames[0]["payload"]["message"]
    done_err = done_frames[0]["payload"].get("error", "")

    # None of the raw exception internals may reach the client (WS or REST sync).
    for leak in (
        "sk-LEAKED-DEADBEEF",
        "internal-provider.example",
        ".cogtrix.yaml",
        "RuntimeError",
    ):
        assert leak not in err_msg, f"leaked {leak!r} in WS error frame"
        assert leak not in done_err, f"leaked {leak!r} in done frame"

    # The safe fallback is what's surfaced instead.
    assert "internal error" in err_msg.lower()
    assert err_msg == done_err


@pytest.mark.asyncio
async def test_run_message_turn_agent_execution_error_is_error_frame_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2124: when run_agent raises AgentExecutionError, the API must emit a
    proper error frame (not a normal done(text=...) with the error as the reply),
    and the curated message must still be sanitized — the model id and other
    internals must not leak to API clients (preserves the #2114 guarantee)."""
    from cogtrix_core.agent.safety import AgentExecutionError
    from cogtrix_core.api.turn_runner import run_message_turn

    session = SimpleNamespace(
        id="sess-agenterr",
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

    # The curated message embeds the rejected model id — must NOT reach clients.
    curated = "**Invalid model ID:** qwen3 is not a valid model ID\n\nCheck models.<alias>.model"

    def _raising_run_agent(*args: object, **kwargs: object) -> str:
        raise AgentExecutionError(curated)

    monkeypatch.setattr("cogtrix_core.orchestration.runner.run_agent", _raising_run_agent)

    await run_message_turn(session, "hello")

    frames = []
    while not session.ws_queue.empty():
        frames.append(session.ws_queue.get_nowait())

    error_frames = [f for f in frames if f.get("type") == "error"]
    done_frames = [f for f in frames if f.get("type") == "done"]
    assert error_frames, "AgentExecutionError must surface as an error frame, not a normal reply"
    assert done_frames, "expected a done frame so the sync drain terminates"

    err_msg = error_frames[0]["payload"]["message"]
    done = done_frames[0]["payload"]
    # Structural: the turn did not return the error as normal text.
    assert done.get("text", "") == ""
    # No leak of the model id / curated internals; sanitized fallback instead.
    assert "qwen3" not in err_msg
    assert "Invalid model ID" not in err_msg
    assert "internal error" in err_msg.lower()
