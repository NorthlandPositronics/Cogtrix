"""#2256 — a long, actively-streaming WS agent turn must not be reaped as idle.

The WebSocket idle timeout reaps abandoned connections, but during a long agent
turn the client sends no inbound text frame (it only receives streamed tokens).
The reap decision must therefore measure *connection* idleness (an active turn or
inbound activity), not merely *inbound* idleness — otherwise a turn longer than
``COGTRIX_WS_IDLE_TIMEOUT`` (default 300s) was torn down mid-stream.

These tests pin the decision core (``_ws_idle_should_reap``) extracted from the
receive loop in ``cogtrix_core/api/routes/messages.py``.
"""

from __future__ import annotations

from cogtrix_core.api.routes.messages import _WS_IDLE_POLL_INTERVAL, _ws_idle_should_reap


class TestIdleReapDecision:
    def test_active_turn_is_never_reaped(self) -> None:
        # The #2256 core: a turn streaming far longer than the idle timeout is
        # NOT idle and must never be reaped.
        assert (
            _ws_idle_should_reap(turn_active=True, idle_elapsed=10_000.0, idle_timeout=300.0)
            is False
        )

    def test_active_turn_with_zero_timeout_still_survives(self) -> None:
        # Even a degenerate idle_timeout of 0 must not kill an active turn.
        assert _ws_idle_should_reap(turn_active=True, idle_elapsed=5.0, idle_timeout=0.0) is False

    def test_idle_connection_past_timeout_is_reaped(self) -> None:
        assert (
            _ws_idle_should_reap(turn_active=False, idle_elapsed=301.0, idle_timeout=300.0) is True
        )

    def test_idle_connection_at_exactly_timeout_is_reaped(self) -> None:
        assert (
            _ws_idle_should_reap(turn_active=False, idle_elapsed=300.0, idle_timeout=300.0) is True
        )

    def test_idle_connection_within_timeout_survives(self) -> None:
        assert (
            _ws_idle_should_reap(turn_active=False, idle_elapsed=299.9, idle_timeout=300.0) is False
        )


class TestIdlePollInterval:
    def test_poll_interval_is_positive(self) -> None:
        # The receive loop re-evaluates idleness every poll interval; it must be a
        # positive, finite cadence (it is bounded by the idle timeout at use site
        # so a small configured timeout still fires promptly). (#2256)
        assert _WS_IDLE_POLL_INTERVAL > 0
