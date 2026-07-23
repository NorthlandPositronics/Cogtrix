"""Tests for EscapeMonitor lifecycle and spinner integration."""

import threading
from unittest.mock import MagicMock, patch


class TestEscapeMonitorUnavailable:
    """When termios is unavailable (Windows, piped stdin), everything is a no-op."""

    def test_no_op_when_unavailable(self):
        with patch("src.cli.escape_monitor._AVAILABLE", False):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            assert not mon.available
            # None of these should raise or create threads
            mon.start()
            mon.pause()
            mon.resume()
            mon.stop()
            assert mon._thread is None

    def test_double_stop_is_safe(self):
        with patch("src.cli.escape_monitor._AVAILABLE", False):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon.stop()
            mon.stop()


class TestEscapeMonitorLifecycle:
    """State transitions without actual terminal I/O."""

    def test_start_sets_running(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._fd = 0
            mon._enter_cbreak = MagicMock()
            mon._restore_terminal = MagicMock()
            mon._monitor_loop = MagicMock()

            with patch.object(threading, "Thread") as mock_thread_cls:
                mock_thread = MagicMock()
                mock_thread_cls.return_value = mock_thread

                mon.start()

                assert mon._running is True
                assert mon._paused is False
                # cbreak is now entered inside the monitor thread, not start()
                mon._enter_cbreak.assert_not_called()
                mock_thread.start.assert_called_once()

    def test_stop_clears_running(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._fd = 0
            mon._restore_terminal = MagicMock()
            mock_thread = MagicMock()
            mon._thread = mock_thread

            mon.stop()

            assert mon._running is False
            mon._restore_terminal.assert_called_once()
            mock_thread.join.assert_called_once_with(timeout=1.0)
            assert mon._thread is None

    def test_stop_when_not_running_is_noop(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = False
            mon._restore_terminal = MagicMock()

            mon.stop()

            mon._restore_terminal.assert_not_called()

    def test_idempotent_start(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._enter_cbreak = MagicMock()

            mon.start()

            mon._enter_cbreak.assert_not_called()

    def test_pause_sets_flag_and_restores_terminal(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._paused = False
            mon._fd = 0
            mon._restore_terminal = MagicMock()

            mon.pause()

            assert mon._paused is True
            mon._restore_terminal.assert_called_once()

    def test_pause_when_already_paused_is_noop(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._paused = True
            mon._restore_terminal = MagicMock()

            mon.pause()

            mon._restore_terminal.assert_not_called()

    def test_pause_when_not_running_is_noop(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = False
            mon._restore_terminal = MagicMock()

            mon.pause()

            mon._restore_terminal.assert_not_called()

    def test_resume_clears_flag_and_enters_cbreak(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._paused = True
            mon._fd = 0
            mon._enter_cbreak = MagicMock()

            mon.resume()

            assert mon._paused is False
            mon._enter_cbreak.assert_called_once()

    def test_resume_when_not_paused_is_noop(self):
        with patch("src.cli.escape_monitor._AVAILABLE", True):
            from src.cli.escape_monitor import EscapeMonitor

            mon = EscapeMonitor()
            mon._running = True
            mon._paused = False
            mon._enter_cbreak = MagicMock()

            mon.resume()

            mon._enter_cbreak.assert_not_called()


class TestSpinnerEscapeIntegration:
    """ActivityIndicator correctly delegates to escape monitor."""

    def test_spinner_delegates_start(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        mock_monitor = MagicMock()
        spinner.set_escape_monitor(mock_monitor)

        with patch.object(spinner, "_animate"):
            spinner.start()
            mock_monitor.start.assert_called_once()
            spinner._running = False  # prevent thread issues

    def test_spinner_delegates_stop(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        mock_monitor = MagicMock()
        spinner.set_escape_monitor(mock_monitor)
        spinner._running = True
        spinner._thread = MagicMock()

        spinner.stop()

        mock_monitor.stop.assert_called_once()

    def test_spinner_delegates_pause_on_first_pause(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        mock_monitor = MagicMock()
        spinner.set_escape_monitor(mock_monitor)
        spinner._running = True

        spinner.pause()
        mock_monitor.pause.assert_called_once()

        # Second pause should NOT call monitor.pause again
        mock_monitor.pause.reset_mock()
        spinner.pause()
        mock_monitor.pause.assert_not_called()

    def test_spinner_delegates_resume_on_last_resume(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        mock_monitor = MagicMock()
        spinner.set_escape_monitor(mock_monitor)
        spinner._running = True

        # Pause twice
        spinner.pause()
        spinner.pause()

        # First resume: pause_count goes from 2 to 1 — no monitor resume
        spinner.resume()
        mock_monitor.resume.assert_not_called()

        # Second resume: pause_count goes from 1 to 0 — monitor resumes
        spinner.resume()
        mock_monitor.resume.assert_called_once()

    def test_no_monitor_no_error(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        # No monitor set — none of these should raise
        spinner._running = True
        spinner.pause()
        spinner.resume()

    def test_stop_without_start_no_error(self):
        from src.ui.spinner import ActivityIndicator

        spinner = ActivityIndicator()
        mock_monitor = MagicMock()
        spinner.set_escape_monitor(mock_monitor)

        spinner.stop()
        mock_monitor.stop.assert_not_called()
