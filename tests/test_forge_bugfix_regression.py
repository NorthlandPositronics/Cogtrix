"""Regression tests for forge audit fixes.

BUG-AUDIT-001 (P1) — turn_runner.py:
  asyncio.Queue.put() never raises QueueFull; only put_nowait() does.
  The dead QueueFull catch in the done-message except clause was removed.
  The three other QueueFull catches in the same function are legitimate because
  they guard put_nowait() calls, which CAN raise QueueFull.

BUG-AUDIT-002 (P2) — confirmation.py:
  _POLL_INTERVAL = 0.5 was reassigned inside the while-loop body on every
  poll iteration (up to 600 times per 5-minute confirmation).  Hoisted to
  module level so it is assigned exactly once at import time.

Additionally covers previously untested ApiConfirmationUI edge cases:
  - cancel() unblocks read_choice promptly
  - render_prompt resets _cancel_requested so a prior cancel never
    silently denies a subsequent confirmation (documented invariant)
  - displacement: a second render_prompt unblocks the first read_choice with "n"
  - unknown WebSocket action strings default to "n"
  - read_choice without a prior render_prompt returns "n"
  - no-op stub methods (show_message, pause_spinner, resume_spinner)
  - _enqueue_nowait silently drops on a full queue
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from src.api import confirmation as _confirmation_mod  # noqa: E402
from src.api.confirmation import (  # noqa: E402
    _ACTION_MAP,
    _POLL_INTERVAL,
    _TIMEOUT_SECONDS,
    ApiConfirmationUI,
)


def _make_turn_runner_session(ws_queue: asyncio.Queue) -> Any:
    """Build a minimal session object accepted by _run_message_turn_inner."""
    return SimpleNamespace(
        id="test-session-forge-regression",
        turn_lock=asyncio.Lock(),
        session_state=None,
        run_config=None,
        memory_manager=None,
        cancel_event=asyncio.Event(),
        ws_queue=ws_queue,
        active_confirmation_ui=None,
        agent_state="idle",
        token_counts={"input_tokens": 0, "output_tokens": 0},
        last_activity=0.0,
        registry=None,
    )


# ---------------------------------------------------------------------------
# BUG-AUDIT-002: _POLL_INTERVAL is a module-level constant
# ---------------------------------------------------------------------------


class TestPollIntervalModuleLevel:
    """_POLL_INTERVAL must be a module-level constant, not a loop-local."""

    def test_poll_interval_is_module_attribute(self) -> None:
        """_POLL_INTERVAL exists at module scope (importable as attribute)."""
        assert hasattr(_confirmation_mod, "_POLL_INTERVAL")

    def test_poll_interval_value(self) -> None:
        """_POLL_INTERVAL is 0.5 seconds."""
        assert _POLL_INTERVAL == 0.5

    def test_timeout_seconds_is_module_attribute(self) -> None:
        """_TIMEOUT_SECONDS exists at module scope."""
        assert hasattr(_confirmation_mod, "_TIMEOUT_SECONDS")

    def test_timeout_seconds_value(self) -> None:
        """_TIMEOUT_SECONDS is 300 (5 minutes)."""
        assert _TIMEOUT_SECONDS == 300

    @pytest.mark.asyncio
    async def test_read_choice_uses_module_poll_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavior: read_choice honors monkeypatched module poll/timeout constants.

        If read_choice reassigns _POLL_INTERVAL inside the loop, monkeypatching
        _confirmation_mod._POLL_INTERVAL would have no effect and this call would
        block for far longer than the small timeout below.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()

        monkeypatch.setattr(_confirmation_mod, "_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(_confirmation_mod, "_TIMEOUT_SECONDS", 0.06)

        started = time.monotonic()
        choice = await asyncio.to_thread(ui.read_choice)
        elapsed = time.monotonic() - started

        assert choice == "n"
        assert elapsed < 0.2, (
            "read_choice did not honor module-level _POLL_INTERVAL/_TIMEOUT_SECONDS; "
            "possible loop-local reassignment regression"
        )


# ---------------------------------------------------------------------------
# BUG-AUDIT-001: QueueFull absent from the wait_for(put()) except clause
# ---------------------------------------------------------------------------


class TestDoneMsgBlockingPut:
    """Behavioral regression tests for done-message queue handling (blocking put)."""

    @pytest.mark.asyncio
    async def test_done_message_delivered_with_blocking_put(self) -> None:
        """Done message is enqueued via blocking put and turn completes normally."""
        from src.api.turn_runner import _run_message_turn_inner

        queue = asyncio.Queue(maxsize=10)
        session = _make_turn_runner_session(queue)
        ws_callback = SimpleNamespace(input_tokens=0, output_tokens=0, tool_call_count=0)

        with patch("src.orchestration.runner.run_agent", return_value="ok"):
            with patch("src.api.callbacks.WebSocketCallbackHandler", return_value=ws_callback):
                await _run_message_turn_inner(session, "hello", "chat", None, None)

        assert session.agent_state == "idle"
        assert session.active_confirmation_ui is None
        # Verify done message was enqueued
        assert any(item.get("type") == "done" for item in queue._queue)

    @pytest.mark.asyncio
    async def test_done_message_blocks_until_space_available(self) -> None:
        """Blocking put waits until queue space is available, then delivers."""
        from src.api.turn_runner import _run_message_turn_inner

        full_queue = asyncio.Queue(maxsize=1)
        full_queue.put_nowait({"sentinel": True})
        session = _make_turn_runner_session(full_queue)
        ws_callback = SimpleNamespace(input_tokens=0, output_tokens=0, tool_call_count=0)

        async def consumer() -> None:
            """Simulate a consumer that frees up queue space."""
            await asyncio.sleep(0.05)
            full_queue.get_nowait()

        with patch("src.orchestration.runner.run_agent", return_value="ok"):
            with patch("src.api.callbacks.WebSocketCallbackHandler", return_value=ws_callback):
                await asyncio.gather(
                    consumer(),
                    _run_message_turn_inner(session, "hello", "chat", None, None),
                )

        assert session.agent_state == "idle"
        assert session.active_confirmation_ui is None
        # The done message should now be in the queue
        assert any(item.get("type") == "done" for item in full_queue._queue)


# ---------------------------------------------------------------------------
# ApiConfirmationUI: cancel() unblocks read_choice
# ---------------------------------------------------------------------------


class TestApiConfirmationUICancel:
    """cancel() must unblock an in-progress read_choice within one poll interval."""

    @pytest.mark.asyncio
    async def test_cancel_returns_n(self) -> None:
        """cancel() causes read_choice to return the default 'n'."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()  # consume the queued request

        proceed = threading.Event()

        def _cancel_after_brief_delay() -> None:
            proceed.wait(timeout=5.0)
            ui.cancel()

        t = threading.Thread(target=_cancel_after_brief_delay, daemon=True)
        t.start()

        # Give the background thread a moment to reach the wait, then signal it.
        await asyncio.sleep(0.01)
        proceed.set()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n", "cancel() should default to deny ('n')"
        t.join(timeout=5.0)

    @pytest.mark.asyncio
    async def test_cancel_sets_flag_and_event(self) -> None:
        """cancel() sets _cancel_requested and also sets the pending event."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()

        ui.cancel()

        with ui._lock:
            assert ui._cancel_requested is True
            assert ui._pending_event is not None
            assert ui._pending_event.is_set()


# ---------------------------------------------------------------------------
# ApiConfirmationUI: render_prompt resets _cancel_requested
# ---------------------------------------------------------------------------


class TestApiConfirmationUIRenderPromptReset:
    """render_prompt resets _cancel_requested so a prior cancel never bleeds
    into the next turn's confirmation flow.

    Documented invariant (CLAUDE.md):
        "render_prompt() resets _cancel_requested = False at entry so a cancel
        from a prior turn never silently denies future confirmations"
    """

    @pytest.mark.asyncio
    async def test_cancel_then_render_then_resolve_allow(self) -> None:
        """After cancel(), render_prompt for a new request clears the cancel flag.

        If _cancel_requested were not reset, the next read_choice would
        return 'n' immediately instead of waiting for resolution.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        # --- Turn 1: render, then cancel ---
        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()
        ui.cancel()

        while not queue.empty():
            await queue.get()

        # --- Turn 2: new render_prompt resets the flag; resolve with "allow" ---
        ui.render_prompt("write_file", {"path": "/tmp/x"}, frozenset(), 300)
        await asyncio.sleep(0.05)
        item = await queue.get()
        conf_id = item["payload"]["confirmation_id"]

        proceed = threading.Event()

        def _resolve() -> None:
            proceed.wait(timeout=5.0)
            ui.resolve(conf_id, "allow")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        # Let the background thread reach the wait, then signal it.
        await asyncio.sleep(0.01)
        proceed.set()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "y", (
            "render_prompt must reset _cancel_requested so the next confirmation "
            "waits for resolution instead of defaulting to 'n'"
        )
        t.join(timeout=5.0)

    @pytest.mark.asyncio
    async def test_render_prompt_resets_cancel_requested_flag_directly(self) -> None:
        """render_prompt always writes _cancel_requested = False."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        with ui._lock:
            ui._cancel_requested = True

        ui.render_prompt("tool", {}, frozenset(), 300)
        await asyncio.sleep(0.05)

        with ui._lock:
            assert (
                ui._cancel_requested is False
            ), "render_prompt must reset _cancel_requested to False"


# ---------------------------------------------------------------------------
# ApiConfirmationUI: displacement (second render_prompt displaces first)
# ---------------------------------------------------------------------------


class TestApiConfirmationUIDisplacement:
    """A second render_prompt call displaces the first pending confirmation.

    The first read_choice must return 'n' (default deny) promptly — not hang
    waiting for the full 5-minute timeout.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_second_render_displaces_first(self) -> None:
        """First read_choice returns 'n' when a second render_prompt fires."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()

        choice_holder: list[str] = []
        entered = threading.Event()
        orig_read_choice = ui.read_choice

        def _read_first() -> None:
            entered.set()
            choice_holder.append(orig_read_choice())

        ui.read_choice = _read_first
        reader = threading.Thread(target=_read_first, daemon=True)
        reader.start()

        entered.wait(timeout=2.0)
        ui.render_prompt("write_file", {"content": "x"}, frozenset(), 300)
        await asyncio.sleep(0.05)

        reader.join(timeout=3.0)
        ui.read_choice = orig_read_choice
        assert not reader.is_alive(), "read_choice hung after displacement"
        assert choice_holder == ["n"], "Displaced read_choice must return 'n'"

    @pytest.mark.asyncio
    async def test_displacement_resolves_second_correctly(self) -> None:
        """After displacing the first, the second confirmation resolves normally."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()

        ui.render_prompt("write_file", {"content": "x"}, frozenset(), 300)
        await asyncio.sleep(0.05)
        item2 = await queue.get()
        conf_id2 = item2["payload"]["confirmation_id"]

        proceed = threading.Event()

        def _resolve() -> None:
            proceed.wait(timeout=5.0)
            ui.resolve(conf_id2, "allow_all")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        await asyncio.sleep(0.01)
        proceed.set()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "a"
        t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# ApiConfirmationUI: unknown action defaults to "n"
# ---------------------------------------------------------------------------


class TestApiConfirmationUIUnknownAction:
    """resolve() with an unrecognized action string must fall back to 'n'."""

    @pytest.mark.asyncio
    async def test_unknown_action_defaults_to_n(self) -> None:
        """An action not in _ACTION_MAP maps to 'n' via dict.get default."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        item = await queue.get()
        conf_id = item["payload"]["confirmation_id"]

        proceed = threading.Event()

        def _resolve() -> None:
            proceed.wait(timeout=5.0)
            ui.resolve(conf_id, "totally_unknown_action")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        await asyncio.sleep(0.01)
        proceed.set()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n", "_ACTION_MAP.get(unknown, 'n') must default to 'n'"
        t.join(timeout=5.0)

    def test_action_map_covers_all_six_actions(self) -> None:
        """_ACTION_MAP covers the six documented WebSocket actions."""
        expected = {"allow", "deny", "allow_all", "disable", "forbid_all", "cancel"}
        assert set(_ACTION_MAP.keys()) == expected

    def test_action_map_values(self) -> None:
        """Verify the exact CLI character each action maps to."""
        assert _ACTION_MAP["allow"] == "y"
        assert _ACTION_MAP["deny"] == "n"
        assert _ACTION_MAP["allow_all"] == "a"
        assert _ACTION_MAP["disable"] == "d"
        assert _ACTION_MAP["forbid_all"] == "f"
        assert _ACTION_MAP["cancel"] == "c"


# ---------------------------------------------------------------------------
# ApiConfirmationUI: read_choice without prior render_prompt
# ---------------------------------------------------------------------------


class TestApiConfirmationUINoRender:
    """read_choice called without a preceding render_prompt returns 'n'."""

    @pytest.mark.asyncio
    async def test_read_choice_without_render_returns_n(self) -> None:
        """With no pending event, read_choice returns the default 'n' immediately."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n"

    @pytest.mark.asyncio
    async def test_resolve_without_render_returns_false(self) -> None:
        """resolve() with no pending confirmation returns False."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        result = ui.resolve("some-id", "allow")
        assert result is False


# ---------------------------------------------------------------------------
# ApiConfirmationUI: no-op stub methods
# ---------------------------------------------------------------------------


class TestApiConfirmationUINoOps:
    """show_message, pause_spinner, resume_spinner must not raise."""

    def _make_ui(self) -> ApiConfirmationUI:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.new_event_loop()
        return ApiConfirmationUI(ws_queue=queue, loop=loop)

    def test_show_message_no_raise(self) -> None:
        ui = self._make_ui()
        ui.show_message("hello", "bold")

    def test_pause_spinner_no_raise(self) -> None:
        ui = self._make_ui()
        ui.pause_spinner()

    def test_resume_spinner_no_raise(self) -> None:
        ui = self._make_ui()
        ui.resume_spinner()


# ---------------------------------------------------------------------------
# ApiConfirmationUI: _enqueue_nowait silently drops on QueueFull
# ---------------------------------------------------------------------------


class TestApiConfirmationUIEnqueueNowait:
    """_enqueue_nowait must drop the message silently when the queue is full."""

    @pytest.mark.asyncio
    async def test_full_queue_does_not_raise(self) -> None:
        """A maxsize=1 queue that is already full causes _enqueue_nowait to log
        a warning and return without propagating QueueFull."""
        full_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        full_queue.put_nowait({"sentinel": True})

        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=full_queue, loop=loop)

        # Must not raise; drops the new message silently.
        ui._try_enqueue_nowait(
            {"type": "tool_confirm_request", "payload": {"confirmation_id": "x"}}
        )

        # The original sentinel is still there (new message was dropped).
        assert full_queue.qsize() == 1
        item = full_queue.get_nowait()
        assert item == {"sentinel": True}
