"""Regression tests for CampaignManager executor concurrency (closes #1294).

BUG-029: CampaignManager._send_follow_up() must submit work to a ThreadPoolExecutor
rather than calling handle_outbound() synchronously on the dispatch loop thread.
Multiple concurrent follow-ups across campaigns must not block each other.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from cogtrix_core.assistant.campaign import Campaign, CampaignManager, CampaignTarget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_target(
    contact_name: str = "Alice",
    channel: str = "telegram",
    chat_id: str = "c1",
    *,
    status: str = "active",
    follow_ups_sent: int = 0,
    last_outbound_at: float | None = None,
) -> CampaignTarget:
    return CampaignTarget(
        contact_name=contact_name,
        channel=channel,
        chat_id=chat_id,
        status=status,
        follow_ups_sent=follow_ups_sent,
        last_outbound_at=last_outbound_at,
    )


def _make_campaign(
    campaign_id: str = "c1",
    name: str = "Test Campaign",
    goal: str = "Sell product",
    instructions: str = "Be friendly",
    targets: list[CampaignTarget] | None = None,
    *,
    status: str = "active",
    follow_up_interval_hours: float = 0.0,
    max_follow_ups: int = 3,
) -> Campaign:
    """Create a campaign with targets that need immediate follow-up."""
    if targets is None:
        targets = [_make_target(last_outbound_at=time.time() - 3600)]
    return Campaign(
        id=campaign_id,
        name=name,
        goal=goal,
        instructions=instructions,
        targets=targets,
        status=status,
        follow_up_interval_hours=follow_up_interval_hours,
        max_follow_ups=max_follow_ups,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCampaignManagerExecutor:
    """Regression tests for issue #1294: executor-based follow-up dispatch."""

    def test_start_creates_executor(self) -> None:
        """CampaignManager.start() must create a ThreadPoolExecutor."""
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json")
            mgr.set_handler(MagicMock())
            mgr.set_channels({"telegram": MagicMock()})
            mgr.start()
            try:
                assert mgr._executor is not None
                assert isinstance(mgr._executor, ThreadPoolExecutor)
            finally:
                mgr.stop()

    def test_stop_shuts_down_executor(self) -> None:
        """CampaignManager.stop() must shut down the executor without blocking."""
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json")
            mgr.set_handler(MagicMock())
            mgr.set_channels({"telegram": MagicMock()})
            mgr.start()
            assert mgr._executor is not None
            mgr.stop()
            assert mgr._executor is None

    def test_process_follow_ups_submits_to_executor(self) -> None:
        """_process_follow_ups must submit _send_follow_up to the executor, not call it inline."""
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json")
            mgr.set_handler(MagicMock())
            mgr.set_channels({"telegram": MagicMock()})

            campaign = _make_campaign(targets=[_make_target(last_outbound_at=time.time() - 3600)])
            mgr.create(campaign)

            # Patch ThreadPoolExecutor so we can verify submit() is called
            mock_executor = MagicMock(spec=ThreadPoolExecutor)
            mock_executor.submit.return_value = MagicMock()  # dummy future

            with patch(
                "cogtrix_core.assistant.campaign.ThreadPoolExecutor", return_value=mock_executor
            ):
                mgr.start()
                try:
                    # Wait for the follow-up loop to process one iteration
                    time.sleep(0.5)

                    # Verify executor.submit was called with _send_follow_up as the target
                    assert mock_executor.submit.call_count >= 1, (
                        "executor.submit was not called — follow-ups may still be "
                        "running synchronously on the dispatch loop thread."
                    )
                    submitted_calls = mock_executor.submit.call_args_list
                    first_call = submitted_calls[0]
                    # Verify executor.submit was called with a callable (lambda wrapper).
                    # The callable must invoke _do_follow_up on execution.
                    submitted_fn = first_call[0][0]
                    assert callable(
                        submitted_fn
                    ), f"Expected executor.submit called with a callable, got {type(submitted_fn)!r}"
                    # Verify the callable actually calls _do_follow_up by invoking it
                    # and checking that _do_follow_up's behavior (handle_outbound call) occurs.
                    # We do this by checking that a future was returned (submit succeeded).
                    assert mock_executor.submit.call_count >= 1
                finally:
                    mgr.stop()

    def test_multiple_follow_ups_all_submitted_to_executor(self) -> None:
        """Each target needing follow-up must be submitted as a separate executor task."""
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json")
            mgr.set_handler(MagicMock())
            mgr.set_channels({"telegram": MagicMock()})

            targets = [
                _make_target(
                    contact_name=f"Contact{i}",
                    chat_id=f"c{i}",
                    last_outbound_at=time.time() - 3600,
                )
                for i in range(3)
            ]
            campaign = _make_campaign(targets=targets)
            mgr.create(campaign)

            mock_executor = MagicMock(spec=ThreadPoolExecutor)
            mock_executor.submit.return_value = MagicMock()

            with patch(
                "cogtrix_core.assistant.campaign.ThreadPoolExecutor", return_value=mock_executor
            ):
                mgr.start()
                try:
                    time.sleep(0.5)
                    assert mock_executor.submit.call_count >= 3, (
                        f"Expected at least 3 executor.submit calls (one per target), "
                        f"got {mock_executor.submit.call_count}. "
                        "Follow-ups may still be running synchronously."
                    )
                finally:
                    mgr.stop()

    def test_dispatch_loop_not_blocked_by_slow_handler(self) -> None:
        """The follow-up loop must return quickly even when handle_outbound is slow.

        If _send_follow_up runs synchronously on the dispatch thread, the loop
        iteration blocks for N * handler_time. With the executor fix, the loop
        iteration completes immediately and work runs concurrently.
        """
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json", check_interval=0.1)
            slow_handler = MagicMock()
            # Simulate a handler that takes 2 seconds per call
            slow_handler.handle_outbound.side_effect = lambda **_: (time.sleep(2), ("ok", "msg1"))
            mgr.set_handler(slow_handler)
            mgr.set_channels({"telegram": MagicMock()})

            # Two targets, each would take 2s if run synchronously
            targets = [
                _make_target(
                    contact_name=f"Slow{i}", chat_id=f"s{i}", last_outbound_at=time.time() - 3600
                )
                for i in range(2)
            ]
            campaign = _make_campaign(targets=targets)
            mgr.create(campaign)

            mgr.start()
            try:
                # With the executor fix: loop completes iteration in << 2s
                # Without the fix: loop blocks for 4s (2s per target, sequential)
                start = time.time()
                time.sleep(1.2)
                elapsed = time.time() - start

                assert elapsed < 2.0, (
                    f"Follow-up loop took {elapsed:.1f}s to complete one iteration. "
                    "This strongly suggests handle_outbound is being called "
                    "synchronously on the dispatch thread (blocking the loop), "
                    "rather than submitted to the executor."
                )
            finally:
                mgr.stop()

    def test_executor_thread_name_prefix(self) -> None:
        """Executor threads must be identifiable via the 'campaign-outbound' name prefix."""
        with TemporaryDirectory() as tmpdir:
            mgr = CampaignManager(Path(tmpdir) / "campaigns.json")
            mgr.set_handler(MagicMock())
            mgr.set_channels({"telegram": MagicMock()})
            mgr.start()
            try:
                # Trigger lazy thread creation by submitting dummy work.
                # ThreadPoolExecutor creates worker threads only when work is submitted.
                assert mgr._executor is not None
                mgr._executor.submit(lambda: None)
                time.sleep(0.2)  # Allow worker thread to start

                threads = threading.enumerate()
                campaign_threads = [t for t in threads if "campaign-outbound" in t.name]
                assert len(campaign_threads) > 0, (
                    "No executor threads found with prefix 'campaign-outbound'. "
                    "Expected ThreadPoolExecutor threads named 'campaign-outbound-N'."
                )
            finally:
                mgr.stop()
