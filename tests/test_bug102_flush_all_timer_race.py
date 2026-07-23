"""Regression tests for BUG-102: flush_all() must not cancel newly-created timers.

Before the fix, flush_all() had an inner `with self._lock:` block after each _flush()
call that cancelled any timer for the same key. This destroyed timers created by
concurrent add() calls that arrived after the outer lock was released but before
_flush() completed, silently dropping messages.

After the fix, flush_all() only cancels timers in the initial locked phase, then calls
_flush() for each key. A concurrent add() during the flush window installs a new timer
that fires normally.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from src.assistant.channel import Channel, IncomingMessage
from src.assistant.poller import MessageBuffer


def _make_msg(chat_id: str = "chat-1") -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id=chat_id,
        message_id="m1",
        sender_id="u1",
        sender_name="Alice",
        text="Hi",
        timestamp=time.time(),
    )


def _make_channel() -> Channel:
    ch = MagicMock(spec=Channel)
    ch.name = "telegram"
    return ch


def test_flush_all_does_not_cancel_new_timers_from_concurrent_add() -> None:
    """A message added during flush_all must eventually be dispatched.

    Strategy: Use a short debounce (0.05 s) and a MagicMock executor so no real
    threads are spawned. After flush_all() the new message's timer fires and
    calls executor.submit a second time.

    If BUG-102 were present the inner cancel block would destroy the new timer
    and executor.submit would only be called once.
    """
    handler = MagicMock()
    executor = MagicMock()
    buf = MessageBuffer(handler=handler, executor=executor, debounce_seconds=0.05)

    channel = _make_channel()
    msg1 = _make_msg("chat-A")
    msg2 = _make_msg("chat-A")  # arrives during flush_all window

    # Pre-load a message so flush_all has something to flush.
    buf.add(msg1, channel)

    # Patch _flush to inject a concurrent add() before the real _flush() runs.
    original_flush = buf._flush
    add_was_called = threading.Event()

    def _patched_flush(key: str) -> None:
        # Simulate a message arriving concurrently during the flush.
        buf.add(msg2, channel)
        add_was_called.set()
        original_flush(key)

    buf._flush = _patched_flush  # type: ignore[method-assign]

    buf.flush_all()

    # Wait for the new timer to fire (debounce = 0.05 s, allow generous headroom).
    time.sleep(0.5)

    assert add_was_called.is_set(), "_patched_flush was never invoked"
    # executor.submit must have been called at least twice: once for msg1 (via flush_all)
    # and once for msg2 (via the timer that fired after the debounce).
    assert executor.submit.call_count >= 2, (
        f"Expected at least 2 executor.submit calls, got {executor.submit.call_count}. "
        "BUG-102 may have regressed — flush_all() is cancelling new timers."
    )


def test_flush_all_no_double_dispatch() -> None:
    """flush_all() must not dispatch any batch twice.

    _flush() pops the buffer under the lock, so a timer firing after flush_all()
    calls _flush() for the same key finds an empty buffer and returns without
    submitting work.
    """
    handler = MagicMock()
    # Use a MagicMock executor so no real threads are spawned; timers firing
    # after the test completes won't hit a shutdown pool.
    executor = MagicMock()
    buf = MessageBuffer(handler=handler, executor=executor, debounce_seconds=0.05)
    channel = _make_channel()

    for _i in range(3):
        buf.add(_make_msg("chat-B"), channel)

    buf.flush_all()
    # Wait for any pending timers to fire.
    time.sleep(0.3)

    # Only one dispatch: the one triggered by flush_all.
    # (The timer was cancelled in the first lock block, so it cannot double-fire.)
    assert executor.submit.call_count == 1, (
        f"Expected exactly 1 executor.submit call, got {executor.submit.call_count}. "
        "flush_all() is double-dispatching."
    )


def test_flush_all_empty_buffers_is_noop() -> None:
    """flush_all() on an empty MessageBuffer must not raise and must not dispatch."""
    handler = MagicMock()
    executor = MagicMock()
    buf = MessageBuffer(handler=handler, executor=executor, debounce_seconds=3.0)
    buf.flush_all()  # must not raise
    handler.handle_batch.assert_not_called()
