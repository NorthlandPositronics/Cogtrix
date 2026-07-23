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

import ast
import asyncio
import inspect
import threading
import time

import pytest

pytest.importorskip("fastapi")

from src.api import confirmation as _confirmation_mod  # noqa: E402
from src.api.confirmation import (  # noqa: E402
    _ACTION_MAP,
    _POLL_INTERVAL,
    _TIMEOUT_SECONDS,
    ApiConfirmationUI,
)

# ---------------------------------------------------------------------------
# Shared AST helpers
# ---------------------------------------------------------------------------


def _handler_names(exc_type: ast.expr | None) -> list[str]:
    """Return the unqualified name(s) in an except-handler type node."""
    if exc_type is None:  # bare except
        return []
    if isinstance(exc_type, ast.Name):
        return [exc_type.id]
    if isinstance(exc_type, ast.Attribute):
        return [exc_type.attr]
    if isinstance(exc_type, ast.Tuple):
        names: list[str] = []
        for elt in exc_type.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(elt.attr)
        return names
    return []


def _body_has_wait_for(body: list[ast.stmt]) -> bool:
    """Return True if any node in *body* contains an asyncio.wait_for call."""
    for stmt in body:
        for child in ast.walk(stmt):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            is_wait_for = (isinstance(func, ast.Attribute) and func.attr == "wait_for") or (
                isinstance(func, ast.Name) and func.id == "wait_for"
            )
            if is_wait_for:
                return True
    return False


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

    def test_poll_interval_not_assigned_inside_function(self) -> None:
        """AST guard: _POLL_INTERVAL must not be assigned inside any function body.

        Before the fix the assignment lived inside read_choice()'s while loop,
        meaning it was re-evaluated up to 600 times per 5-minute confirmation.
        """
        source = inspect.getsource(_confirmation_mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "_POLL_INTERVAL":
                            pytest.fail(
                                f"_POLL_INTERVAL assigned inside function '{node.name}' "
                                "— must be a module-level constant (BUG-AUDIT-002)"
                            )
                if isinstance(child, ast.AnnAssign):
                    if isinstance(child.target, ast.Name) and child.target.id == "_POLL_INTERVAL":
                        pytest.fail(
                            f"_POLL_INTERVAL annotated-assigned inside function '{node.name}' "
                            "— must be a module-level constant (BUG-AUDIT-002)"
                        )


# ---------------------------------------------------------------------------
# BUG-AUDIT-001: QueueFull absent from the wait_for(put()) except clause
# ---------------------------------------------------------------------------


class TestDoneMsgNoQueueFull:
    """The done-message wait_for(put()) must NOT catch QueueFull.

    asyncio.Queue.put() blocks until space is available; it never raises
    QueueFull.  Only put_nowait() raises QueueFull.  The other three
    QueueFull catches in the same function are legitimate because they
    guard put_nowait() calls.
    """

    def test_no_queuefull_in_wait_for_except(self) -> None:
        """AST: no Try block containing wait_for() catches QueueFull."""
        from src.api import turn_runner

        source = inspect.getsource(turn_runner._run_message_turn_inner)
        tree = ast.parse(source)

        bad_handlers: list[ast.ExceptHandler] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not _body_has_wait_for(node.body):
                continue
            for handler in node.handlers:
                if "QueueFull" in _handler_names(handler.type):
                    bad_handlers.append(handler)

        assert len(bad_handlers) == 0, (
            f"Found {len(bad_handlers)} except handler(s) catching QueueFull inside a "
            "try block that contains asyncio.wait_for().  Queue.put() never raises "
            "QueueFull — this is dead code (BUG-AUDIT-001)."
        )

    def test_done_msg_uses_wait_for_with_timeout_error(self) -> None:
        """AST: the done-message put() is wrapped in wait_for and catches TimeoutError."""
        from src.api import turn_runner

        source = inspect.getsource(turn_runner._run_message_turn_inner)

        assert (
            "wait_for" in source
        ), "done message must use asyncio.wait_for() to bound the wait (BUG-209)"
        assert (
            "TimeoutError" in source
        ), "done message must catch TimeoutError from asyncio.wait_for"

    def test_put_nowait_handlers_are_legitimate(self) -> None:
        """Sanity: the remaining QueueFull catches are all around put_nowait calls.

        Verifies that each Try block catching QueueFull uses put_nowait(), not put().
        """
        from src.api import turn_runner

        source = inspect.getsource(turn_runner._run_message_turn_inner)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            has_queuefull = any("QueueFull" in _handler_names(h.type) for h in node.handlers)
            if not has_queuefull:
                continue
            # Verify the body uses put_nowait, not put().
            for stmt in node.body:
                for child in ast.walk(stmt):
                    if not isinstance(child, ast.Call):
                        continue
                    func = child.func
                    if isinstance(func, ast.Attribute) and func.attr == "put":
                        # This Try catches QueueFull AND has a plain .put() call —
                        # that is the dead-code pattern we fixed.
                        pytest.fail(
                            "Found a Try block that catches QueueFull and also "
                            "calls .put() — only put_nowait() can raise QueueFull "
                            "(BUG-AUDIT-001)."
                        )


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

        def _cancel_after_brief_delay() -> None:
            time.sleep(0.05)
            ui.cancel()

        t = threading.Thread(target=_cancel_after_brief_delay, daemon=True)
        t.start()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n", "cancel() should default to deny ('n')"
        t.join()

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

        def _resolve() -> None:
            time.sleep(0.02)
            ui.resolve(conf_id, "allow")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "y", (
            "render_prompt must reset _cancel_requested so the next confirmation "
            "waits for resolution instead of defaulting to 'n'"
        )
        t.join()

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

        def _read_first() -> None:
            choice_holder.append(ui.read_choice())

        reader = threading.Thread(target=_read_first, daemon=True)
        reader.start()

        await asyncio.sleep(0.1)
        ui.render_prompt("write_file", {"content": "x"}, frozenset(), 300)
        await asyncio.sleep(0.1)

        reader.join(timeout=3.0)
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

        def _resolve() -> None:
            time.sleep(0.02)
            ui.resolve(conf_id2, "allow_all")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "a"
        t.join()


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

        def _resolve() -> None:
            ui.resolve(conf_id, "totally_unknown_action")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n", "_ACTION_MAP.get(unknown, 'n') must default to 'n'"
        t.join()

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
        await ui._enqueue_nowait(
            {"type": "tool_confirm_request", "payload": {"confirmation_id": "x"}}
        )

        # The original sentinel is still there (new message was dropped).
        assert full_queue.qsize() == 1
        item = full_queue.get_nowait()
        assert item == {"sentinel": True}
