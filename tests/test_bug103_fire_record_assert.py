"""Regression tests for BUG-103: DeferralManager._fire_record must use an explicit
if/raise RuntimeError instead of assert for the _reprocess_callback guard.

assert statements are silently stripped by Python in optimised mode (-O /
PYTHONOPTIMIZE=1). If _reprocess_callback is None when _fire_record runs, the
assert would be a no-op and the next line would raise:
    TypeError: 'NoneType' object is not callable
caught by the surrounding except Exception handler as a generic callback failure.

After the fix, an explicit RuntimeError is raised with a clear message that
identifies the root cause. The guard executes in both normal and optimised mode.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.assistant.channel import IncomingMessage, SendResult
from src.assistant.deferral import DeferralManager, DeferredRecord


def _make_msg(chat_id: str = "42") -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id=chat_id,
        message_id="m1",
        sender_id="u1",
        sender_name="Alice",
        text="Hello",
        timestamp=time.time(),
    )


def _make_channel(name: str = "telegram") -> MagicMock:
    ch = MagicMock()
    ch.name = name
    ch.send.return_value = SendResult(ok=True, message_id="sent-1")
    ch.is_ready.return_value = True
    return ch


def _make_manager(
    tmp_path: Path,
    callback: Any = None,
    channels: dict | None = None,
) -> DeferralManager:
    if channels is None:
        channels = {}
    return DeferralManager(
        persist_path=tmp_path / "deferrals.json",
        reprocess_callback=callback,
        channels=channels,
        check_interval=3600.0,  # never fires automatically in tests
    )


# ---------------------------------------------------------------------------
# Test 1: explicit RuntimeError is raised when _reprocess_callback is None
# ---------------------------------------------------------------------------
def test_fire_record_raises_runtime_error_when_no_callback(tmp_path: Path) -> None:
    """_fire_record must produce a RuntimeError log when callback is None.

    _fire_record catches Exception internally, logs it, and does not re-raise.
    We intercept the log.error call to confirm a RuntimeError was the cause.
    """
    channel = _make_channel()
    mgr = _make_manager(tmp_path, callback=None, channels={"telegram": channel})
    assert mgr._reprocess_callback is None

    # Build a minimal DeferredRecord.
    now = time.time()
    rec = DeferredRecord(
        id="rec-1",
        channel="telegram",
        chat_id="42",
        fire_at=now - 1.0,
        created_at=now - 60.0,
        pending_messages=[
            {
                "channel": "telegram",
                "chat_id": "42",
                "message_id": "m1",
                "sender_id": "u1",
                "sender_name": "Alice",
                "text": "Hello",
                "timestamp": now,
                "metadata": {},
                "resolved_phone": None,
            }
        ],
        deferral_depth=0,
        status="pending",
    )
    # Register the record in the manager.
    with mgr._lock:
        mgr._records["telegram::42"] = rec

    import src.assistant.deferral as deferral_mod

    captured_errors: list[str] = []
    original_error = deferral_mod.log.error

    def _capture(msg: str, *args: Any, **kwargs: Any) -> None:
        captured_errors.append(msg % args if args else msg)
        original_error(msg, *args, **kwargs)

    deferral_mod.log.error = _capture  # type: ignore[method-assign]
    try:
        mgr._fire_record("telegram::42", rec, time.monotonic())
    finally:
        deferral_mod.log.error = original_error  # type: ignore[method-assign]

    assert captured_errors, "_fire_record did not log an error when callback was None"
    combined = " ".join(captured_errors)
    # The error message must mention RuntimeError to indicate the explicit guard fired.
    assert (
        "RuntimeError" in combined or "no reprocess callback" in combined.lower()
    ), f"Expected RuntimeError or descriptive message in error log, got: {combined!r}"


# ---------------------------------------------------------------------------
# Test 2: the guard is an if/raise, not an assert — verified by source inspection
# ---------------------------------------------------------------------------
def test_fire_record_guard_is_not_assert() -> None:
    """Verify that _fire_record does not use 'assert' for the callback guard.

    We inspect the source code to confirm the guard is expressed as an explicit
    if/raise rather than an assert statement.
    """
    source = inspect.getsource(DeferralManager._fire_record)

    # There must be no bare "assert self._reprocess_callback" in the method.
    assert "assert self._reprocess_callback" not in source, (
        "_fire_record still uses 'assert self._reprocess_callback' — BUG-103 has regressed. "
        "Replace the assert with an explicit if/raise RuntimeError."
    )

    # There must be an explicit RuntimeError raise for the None case.
    assert "RuntimeError" in source, (
        "_fire_record does not contain an explicit RuntimeError raise. "
        "The guard must be an if/raise, not an assert."
    )


# ---------------------------------------------------------------------------
# Test 3: when callback IS set, _fire_record calls it normally
# ---------------------------------------------------------------------------
def test_fire_record_calls_callback_when_set(tmp_path: Path) -> None:
    """_fire_record must call the reprocess callback when it is properly configured."""
    called_with: list[tuple] = []

    def callback(messages: list, channel: Any, depth: int) -> None:
        called_with.append((messages, channel, depth))

    channel = _make_channel()
    mgr = _make_manager(tmp_path, callback=callback, channels={"telegram": channel})

    now = time.time()
    rec = DeferredRecord(
        id="rec-2",
        channel="telegram",
        chat_id="42",
        fire_at=now - 1.0,
        created_at=now - 60.0,
        pending_messages=[
            {
                "channel": "telegram",
                "chat_id": "42",
                "message_id": "m1",
                "sender_id": "u1",
                "sender_name": "Alice",
                "text": "Hello",
                "timestamp": now,
                "metadata": {},
                "resolved_phone": None,
            }
        ],
        deferral_depth=1,
        status="pending",
    )
    with mgr._lock:
        mgr._records["telegram::42"] = rec

    mgr._fire_record("telegram::42", rec, time.monotonic())

    assert len(called_with) == 1, f"Callback was not called exactly once: {called_with}"
    messages_arg, channel_arg, depth_arg = called_with[0]
    assert channel_arg is channel
    assert depth_arg == 1  # deferral_depth passed through as-is


# ---------------------------------------------------------------------------
# Test 4: start() still guards against missing callback
# ---------------------------------------------------------------------------
def test_start_raises_when_callback_not_set(tmp_path: Path) -> None:
    """start() must raise RuntimeError if set_reprocess_callback was never called."""
    mgr = _make_manager(tmp_path, callback=None)
    with pytest.raises(RuntimeError, match="set_reprocess_callback"):
        mgr.start()


# ---------------------------------------------------------------------------
# Test 5: set_reprocess_callback then start() succeeds
# ---------------------------------------------------------------------------
def test_start_succeeds_after_set_reprocess_callback(tmp_path: Path) -> None:
    """start() must succeed after set_reprocess_callback is called."""
    mgr = _make_manager(tmp_path, callback=None)

    def cb(messages: list, channel: Any, depth: int) -> None:
        pass

    mgr.set_reprocess_callback(cb)
    mgr.start()
    assert mgr._thread is not None and mgr._thread.is_alive()
    mgr.stop()
