"""Regression tests for #2117 — a WS ``cancel`` during a REST ``?sync=true`` turn
must not corrupt the Future sentinel or drop the cancellation.

``sess.turn_task`` is a real :class:`asyncio.Task` for WS turns but a plain
:class:`asyncio.Future` *sentinel* for sync REST turns. The cancel handler used to
``.cancel()`` + ``await`` it unconditionally and clear ``cancel_event`` in its
``finally`` — for a sentinel that raced the sync handler's ``set_result`` (→
``InvalidStateError``) and cleared ``cancel_event`` before the inline turn could
observe it. ``_cancel_active_turn`` now distinguishes the two.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.routes.messages import _cancel_active_turn


def _sess(turn_task: object, confirmation_ui: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        turn_task=turn_task,
        cancel_event=asyncio.Event(),
        active_confirmation_ui=confirmation_ui,
    )


@pytest.mark.asyncio
async def test_sentinel_future_is_not_cancelled_or_awaited() -> None:
    """For a sync sentinel: cancel_event is set, but the Future is left pending so
    the sync handler can still resolve it (no InvalidStateError)."""
    sentinel: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    sess = _sess(sentinel)

    await _cancel_active_turn(sess)

    assert sess.cancel_event.is_set(), "cancel_event must be set so the inline turn winds down"
    assert not sentinel.cancelled(), "sentinel must not be cancelled by the WS cancel handler"
    assert not sentinel.done(), "sentinel must remain pending for the sync handler to resolve"

    # The sync handler's guarded finally then resolves it without error.
    if not sentinel.done():
        sentinel.set_result(None)
    assert sentinel.result() is None


@pytest.mark.asyncio
async def test_sentinel_branch_does_not_clear_cancel_event() -> None:
    """cancel_event must stay set after cancelling a sync turn — the sync handler's
    finally clears it, not the cancel handler (else the cancel is dropped)."""
    sentinel: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    sess = _sess(sentinel)

    await _cancel_active_turn(sess)

    assert sess.cancel_event.is_set()


@pytest.mark.asyncio
async def test_real_task_is_cancelled_and_event_cleared() -> None:
    """For a real turn Task: it is cancelled + awaited and cancel_event is cleared."""

    async def _long_turn() -> None:
        await asyncio.sleep(30)

    task = asyncio.create_task(_long_turn())
    await asyncio.sleep(0)  # let it start
    sess = _sess(task)

    await _cancel_active_turn(sess)

    assert task.cancelled()
    assert not sess.cancel_event.is_set(), "cancel_event cleared after the task unwinds"


@pytest.mark.asyncio
async def test_confirmation_ui_is_cancelled() -> None:
    sentinel: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    conf_ui = MagicMock()
    sess = _sess(sentinel, confirmation_ui=conf_ui)

    await _cancel_active_turn(sess)

    conf_ui.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_noop_when_no_turn_in_progress() -> None:
    sess = _sess(None)
    await _cancel_active_turn(sess)
    assert not sess.cancel_event.is_set()


@pytest.mark.asyncio
async def test_noop_when_turn_already_done() -> None:
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    done.set_result(None)
    sess = _sess(done)

    await _cancel_active_turn(sess)

    assert not sess.cancel_event.is_set()


def test_set_result_guard_tolerates_cancelled_future() -> None:
    """The sync handler's guard: a cancelled/done sentinel must not raise on the
    `if not done(): set_result()` path (InvalidStateError guard)."""
    loop = asyncio.new_event_loop()
    try:
        fut: asyncio.Future[None] = loop.create_future()
        fut.cancel()
        # Mirror the guarded finally in the sync handler.
        if not fut.done():  # cancelled() ⇒ done() is True, so this is skipped
            fut.set_result(None)
        assert fut.cancelled()
    finally:
        loop.close()
