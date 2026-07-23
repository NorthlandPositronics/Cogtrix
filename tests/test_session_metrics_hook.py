"""Tests for the CLI session-metrics hook in cogtrix.py."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestMaybeWriteSessionMetrics:
    """Tests for _maybe_write_session_metrics — best-effort metrics trigger."""

    def test_none_log_file_skips(self) -> None:
        from cogtrix import _maybe_write_session_metrics

        with patch("cogtrix.write_session_metrics") as mock_write:
            _maybe_write_session_metrics(None)
            mock_write.assert_not_called()

    def test_missing_file_skips(self, tmp_path: Path) -> None:
        from cogtrix import _maybe_write_session_metrics

        with patch("cogtrix.write_session_metrics") as mock_write:
            _maybe_write_session_metrics(str(tmp_path / "nonexistent.log"))
            mock_write.assert_not_called()

    def test_existing_file_triggers_write(self, tmp_path: Path) -> None:
        from cogtrix import _maybe_write_session_metrics

        log_file = tmp_path / "session.log"
        log_file.write_text("dummy log line\n")

        with patch("cogtrix.write_session_metrics") as mock_write:
            _maybe_write_session_metrics(str(log_file))
            mock_write.assert_called_once_with(str(log_file))

    def test_empty_string_resolves_to_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cogtrix import _maybe_write_session_metrics

        monkeypatch.chdir(tmp_path)
        default_log = tmp_path / "cogtrix.log"
        default_log.write_text("dummy log line\n")

        with patch("cogtrix.write_session_metrics") as mock_write:
            _maybe_write_session_metrics("")
            mock_write.assert_called_once_with("cogtrix.log")

    def test_write_failure_is_silent(self, tmp_path: Path) -> None:
        from cogtrix import _maybe_write_session_metrics

        log_file = tmp_path / "session.log"
        log_file.write_text("dummy log line\n")

        with patch("cogtrix.write_session_metrics", side_effect=RuntimeError("boom")):
            # Must not raise — metrics are best-effort
            _maybe_write_session_metrics(str(log_file))
