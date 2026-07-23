"""Tests for ActivityIndicator pause/resume reference-counting logic."""

from cogtrix_core.ui.spinner import ActivityIndicator


class TestPauseResumeCounter:
    def setup_method(self) -> None:
        self.spinner = ActivityIndicator()

    def test_single_pause_resume(self) -> None:
        self.spinner.pause()
        assert self.spinner._pause_count == 1

        self.spinner.resume()
        assert self.spinner._pause_count == 0

    def test_nested_pause_resume(self) -> None:
        self.spinner.pause()
        self.spinner.pause()
        assert self.spinner._pause_count == 2

        self.spinner.resume()
        assert self.spinner._pause_count == 1, "still paused after first resume"

        self.spinner.resume()
        assert self.spinner._pause_count == 0, "unpaused after second resume"

    def test_resume_without_pause(self) -> None:
        assert self.spinner._pause_count == 0
        self.spinner.resume()
        assert self.spinner._pause_count == 0, "counter must not go negative"
