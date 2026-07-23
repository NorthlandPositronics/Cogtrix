"""Tests for adaptive polling interval in ChannelPoller."""

from __future__ import annotations

from unittest.mock import MagicMock

from cogtrix_core.assistant.channel import IncomingMessage
from cogtrix_core.assistant.poller import ChannelPoller


def _make_poller(
    channel_name: str = "whatsapp",
    poll_results: list[list[IncomingMessage]] | None = None,
    poll_exception: Exception | None = None,
    config: dict | None = None,
    max_iterations: int = 5,
) -> tuple[ChannelPoller, MagicMock, MagicMock]:
    """Build a ChannelPoller with a mock channel and stop event.

    The stop_event.is_set side effect allows exactly *max_iterations* loop
    iterations before signalling stop.
    """
    channel = MagicMock()
    channel.name = channel_name

    if poll_exception is not None:
        channel.poll.side_effect = poll_exception
    elif poll_results is not None:
        channel.poll.side_effect = poll_results
    else:
        channel.poll.return_value = []

    handler = MagicMock()
    executor = MagicMock()
    cfg: dict = config or {}

    poller = ChannelPoller(
        channels=[channel],
        handler=handler,
        executor=executor,
        config=cfg,
        session_mgr=MagicMock(),
    )

    iteration_count = 0

    def is_set_side_effect() -> bool:
        nonlocal iteration_count
        iteration_count += 1
        # First call is at the top of the while loop, then after wait
        # Allow max_iterations passes through the loop
        return iteration_count > max_iterations

    stop_event = MagicMock()
    stop_event.is_set.side_effect = is_set_side_effect
    stop_event.wait = MagicMock()
    poller._stop_event = stop_event

    return poller, channel, stop_event


def _get_wait_timeouts(stop_event: MagicMock) -> list[float]:
    """Extract all timeout= values from stop_event.wait calls."""
    return [c.kwargs["timeout"] for c in stop_event.wait.call_args_list]


class TestAdaptivePolling:
    def test_backs_off_on_idle(self) -> None:
        """Interval increases on consecutive idle polls."""
        poller, channel, stop_event = _make_poller(max_iterations=5)
        channel.poll.return_value = []

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        assert len(timeouts) == 5
        # Each successive timeout should be >= the previous
        for i in range(1, len(timeouts)):
            assert timeouts[i] >= timeouts[i - 1]

    def test_recovers_on_activity(self) -> None:
        """Interval decreases after receiving messages."""
        msg = MagicMock(spec=IncomingMessage)
        msg.session_key = "whatsapp:123"
        # 3 idle polls to back off, then 2 active polls
        results: list[list] = [[], [], [], [msg], [msg]]
        poller, channel, stop_event = _make_poller(poll_results=results, max_iterations=5)

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        # After activity, the interval should decrease
        assert timeouts[-1] <= timeouts[2]

    def test_does_not_exceed_max(self) -> None:
        """Interval never exceeds poll_interval_max."""
        config = {"channels": {"whatsapp": {"poll_interval_max": 30.0}}}
        poller, channel, stop_event = _make_poller(config=config, max_iterations=20)
        channel.poll.return_value = []

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        for t in timeouts:
            assert t <= 30.0

    def test_does_not_go_below_min(self) -> None:
        """Interval never goes below poll_interval_min."""
        msg = MagicMock(spec=IncomingMessage)
        msg.session_key = "whatsapp:123"
        config = {"channels": {"whatsapp": {"poll_interval_min": 2.0}}}
        poller, channel, stop_event = _make_poller(config=config, max_iterations=10)
        channel.poll.return_value = [msg]

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        for t in timeouts:
            assert t >= 2.0

    def test_no_adapt_on_exception(self) -> None:
        """Interval does not change on poll exceptions."""
        config = {"channels": {"whatsapp": {"poll_interval": 5.0}}}
        poller, channel, stop_event = _make_poller(
            poll_exception=RuntimeError("network"),
            config=config,
            max_iterations=3,
        )

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        # All timeouts should be the same (base interval, no adaptation)
        assert all(t == timeouts[0] for t in timeouts)

    def test_default_config_uses_base_interval(self) -> None:
        """Without adaptive config, min_interval defaults to poll_interval."""
        poller, channel, stop_event = _make_poller(max_iterations=1)
        channel.poll.return_value = []

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        # Default WhatsApp interval is 5.0, first idle step backs off
        assert timeouts[0] == 5.0 * 1.5  # 7.5

    def test_custom_backoff_factor(self) -> None:
        """Custom backoff_factor is respected."""
        config = {
            "channels": {
                "whatsapp": {
                    "poll_interval": 10.0,
                    "poll_backoff_factor": 2.0,
                    "poll_interval_max": 100.0,
                }
            }
        }
        poller, channel, stop_event = _make_poller(config=config, max_iterations=3)
        channel.poll.return_value = []

        poller._poll_loop(channel)

        timeouts = _get_wait_timeouts(stop_event)
        assert timeouts[0] == 20.0  # 10 * 2
        assert timeouts[1] == 40.0  # 20 * 2
        assert timeouts[2] == 80.0  # 40 * 2
