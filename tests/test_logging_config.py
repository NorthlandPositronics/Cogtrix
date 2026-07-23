"""Tests for src/logging_config — secret scrubbing, observability handler, stream routing."""

import logging
import sys

import pytest

from src.logging_config import (
    LLMObservabilityHandler,
    _MaxLevelFilter,
    _scrub_secrets,
    setup_logging,
)


class TestScrubSecrets:
    def test_scrubs_api_key_pattern(self) -> None:
        raw = "api_key: sk-abcdefgh12345678"
        result = _scrub_secrets(raw)
        assert "sk-abcdefgh12345678" not in result

    def test_scrubs_bearer_token(self) -> None:
        raw = "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9"
        result = _scrub_secrets(raw)
        assert "eyJhbGciOiJSUzI1NiJ9" not in result

    def test_scrubs_sk_prefix_key(self) -> None:
        raw = "key=sk-ABCDEFGHIJKLMNOP"
        result = _scrub_secrets(raw)
        assert "sk-ABCDEFGHIJKLMNOP" not in result

    def test_passthrough_plain_text(self) -> None:
        raw = "hello world"
        assert _scrub_secrets(raw) == raw


@pytest.mark.skipif(
    LLMObservabilityHandler is None,
    reason="langchain_core not installed",
)
class TestLLMObservabilityHandlerToolStart:
    def _make_handler(self) -> "LLMObservabilityHandler":
        handler = LLMObservabilityHandler(verbose=True)
        handler._logged_messages: list[tuple[str, str]] = []

        def _capture_log(level: str, message: str) -> None:
            handler._logged_messages.append((level, message))

        handler._log = _capture_log  # type: ignore[method-assign]
        return handler

    def test_on_tool_start_scrubs_api_key(self) -> None:
        handler = self._make_handler()
        secret = "sk-supersecretkey1234"
        input_str = f'{{"headers": {{"Authorization": "Bearer {secret}"}}}}'
        handler.on_tool_start({"name": "http_request"}, input_str)

        debug_messages = [msg for level, msg in handler._logged_messages if level == "debug"]
        assert any("TOOL_INPUT" in msg for msg in debug_messages), "TOOL_INPUT line missing"
        tool_input_line = next(msg for msg in debug_messages if "TOOL_INPUT" in msg)
        assert (
            secret not in tool_input_line
        ), f"Raw secret found in TOOL_INPUT log: {tool_input_line}"

    def test_on_tool_start_scrubs_password_field(self) -> None:
        handler = self._make_handler()
        input_str = "password: hunter2ABCDEFG"
        handler.on_tool_start({"name": "some_tool"}, input_str)

        debug_messages = [msg for level, msg in handler._logged_messages if level == "debug"]
        tool_input_line = next(msg for msg in debug_messages if "TOOL_INPUT" in msg)
        assert "hunter2ABCDEFG" not in tool_input_line

    def test_on_tool_start_logs_tool_name(self) -> None:
        handler = self._make_handler()
        handler.on_tool_start({"name": "web_search"}, "query=hello")

        info_messages = [msg for level, msg in handler._logged_messages if level == "info"]
        assert any("TOOL_START: web_search" in msg for msg in info_messages)

    def test_on_tool_start_safe_input_unchanged(self) -> None:
        handler = self._make_handler()
        input_str = "query=open source AI"
        handler.on_tool_start({"name": "web_search"}, input_str)

        debug_messages = [msg for level, msg in handler._logged_messages if level == "debug"]
        tool_input_line = next(msg for msg in debug_messages if "TOOL_INPUT" in msg)
        assert "open source AI" in tool_input_line


class TestMaxLevelFilter:
    """Unit tests for _MaxLevelFilter — used to split stdout/stderr streams."""

    def test_passes_records_below_max(self) -> None:
        f = _MaxLevelFilter(logging.WARNING)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True

    def test_passes_debug_below_warning(self) -> None:
        f = _MaxLevelFilter(logging.WARNING)
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        assert f.filter(record) is True

    def test_blocks_record_at_max(self) -> None:
        f = _MaxLevelFilter(logging.WARNING)
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert f.filter(record) is False

    def test_blocks_record_above_max(self) -> None:
        f = _MaxLevelFilter(logging.WARNING)
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert f.filter(record) is False


class TestSetupLoggingStreamOutput:
    """Regression tests for --debug stream routing (BUG-STREAM-001).

    When the API server is started with --debug and no --log-file:
    - DEBUG/INFO records must go to stdout only
    - WARNING/ERROR/CRITICAL records must go to stderr only
    - Providing log_file overrides stream_output entirely
    """

    def _fresh_logger(self) -> logging.Logger:
        """Return the cogtrix logger with all handlers cleared."""
        lg = logging.getLogger("cogtrix")
        lg.handlers.clear()
        return lg

    def test_stream_output_adds_stdout_and_stderr_handlers(self) -> None:
        setup_logging(log_file=None, debug=True, stream_output=True)
        lg = logging.getLogger("cogtrix")
        streams = [
            h.stream  # type: ignore[attr-defined]
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler) and hasattr(h, "stream")
        ]
        assert sys.stdout in streams, "stdout handler missing"
        assert sys.stderr in streams, "stderr handler missing"
        self._fresh_logger()

    @staticmethod
    def _would_handle(handler: logging.Handler, record: logging.LogRecord) -> bool:
        """Mirrors logging.Handler.handle(): filters AND level check must both pass."""
        return handler.filter(record) and record.levelno >= handler.level

    def test_info_record_reaches_stdout_not_stderr(self) -> None:
        setup_logging(log_file=None, debug=True, stream_output=True)
        lg = logging.getLogger("cogtrix")
        stdout_handlers = [
            h
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        ]
        stderr_handlers = [
            h
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
        ]
        record = logging.LogRecord("cogtrix", logging.INFO, "", 0, "hello", (), None)
        assert all(
            self._would_handle(h, record) for h in stdout_handlers
        ), "INFO blocked from stdout"
        assert not any(
            self._would_handle(h, record) for h in stderr_handlers
        ), "INFO leaked to stderr"
        self._fresh_logger()

    def test_warning_record_reaches_stderr_not_stdout(self) -> None:
        setup_logging(log_file=None, debug=True, stream_output=True)
        lg = logging.getLogger("cogtrix")
        stdout_handlers = [
            h
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        ]
        stderr_handlers = [
            h
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
        ]
        record = logging.LogRecord("cogtrix", logging.WARNING, "", 0, "warn", (), None)
        assert not any(
            self._would_handle(h, record) for h in stdout_handlers
        ), "WARNING leaked to stdout"
        assert all(
            self._would_handle(h, record) for h in stderr_handlers
        ), "WARNING blocked from stderr"
        self._fresh_logger()

    def test_log_file_overrides_stream_output(self, tmp_path: "pytest.TempPathFactory") -> None:
        log_path = str(tmp_path / "test.log")  # type: ignore[operator]
        setup_logging(log_file=log_path, debug=True, stream_output=True)
        lg = logging.getLogger("cogtrix")
        stream_handlers = [
            h
            for h in lg.handlers
            if isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
        ]
        assert stream_handlers == [], "stream handlers present despite log_file being set"
        self._fresh_logger()

    def test_stream_output_false_with_no_file_gives_null_handler(self) -> None:
        setup_logging(log_file=None, debug=True, stream_output=False)
        lg = logging.getLogger("cogtrix")
        assert any(isinstance(h, logging.NullHandler) for h in lg.handlers)
        self._fresh_logger()
