"""Tests for cogtrix_core/logging_config — secret scrubbing, observability handler, stream routing."""

import json
import logging
import sys
from pathlib import Path

import pytest

from cogtrix_core.logging_config import (
    LLMObservabilityHandler,
    _MaxLevelFilter,
    _scrub_secrets,
    clear_request_id,
    clear_session_id,
    get_request_id,
    get_session_id,
    log_tool_call,
    new_request_id,
    set_session_id,
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

    def test_passthrough_benign_token_word(self) -> None:
        """Standalone 'token' in prose must NOT be redacted (#994)."""
        raw = "Generating auth token for user"
        assert _scrub_secrets(raw) == raw

    def test_passthrough_benign_password_word(self) -> None:
        """Standalone 'password' in prose must NOT be redacted (#994)."""
        raw = "Password reset email sent"
        assert _scrub_secrets(raw) == raw

    def test_passthrough_benign_secret_word(self) -> None:
        """Standalone 'secret' in prose must NOT be redacted (#994)."""
        raw = "Checking secret store availability"
        assert _scrub_secrets(raw) == raw

    def test_redacts_key_name_with_colon(self) -> None:
        """Key names followed by ':' must still be redacted."""
        raw = "Authorization: Bearer xxx"
        result = _scrub_secrets(raw)
        assert "Authorization" not in result

    def test_redacts_key_name_with_equals(self) -> None:
        """Key names followed by '=' must still be redacted."""
        raw = "api_key=short"
        result = _scrub_secrets(raw)
        assert "api_key" not in result


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


class TestLogToolCallSecurityLevel:
    """Regression tests for #410 — security enforcement logs at WARNING not ERROR."""

    def _capture_records(self, tool: str, error: str) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture()
        handler.setLevel(logging.DEBUG)
        lg = logging.getLogger("cogtrix")
        lg.addHandler(handler)
        try:
            log_tool_call(tool, error=error)
        finally:
            lg.removeHandler(handler)
        return records

    def test_access_denied_logs_warning_not_error(self) -> None:
        records = self._capture_records(
            "read_text_file", error="Access denied - path outside allowed directories"
        )
        levels = {r.levelno for r in records if r.levelno >= logging.WARNING}
        assert logging.ERROR not in levels, "Access denied should not be ERROR"
        assert logging.WARNING in levels, "Access denied must be WARNING"

    def test_path_outside_allowed_logs_warning(self) -> None:
        records = self._capture_records(
            "list_directory", error="path outside allowed directories: /etc/passwd"
        )
        levels = {r.levelno for r in records if r.levelno >= logging.WARNING}
        assert logging.ERROR not in levels
        assert logging.WARNING in levels

    def test_unexpected_tool_failure_still_errors(self) -> None:
        records = self._capture_records("search_web", error="ConnectionRefusedError: [Errno 111]")
        levels = {r.levelno for r in records if r.levelno >= logging.WARNING}
        assert logging.ERROR in levels, "Unexpected failures must still be ERROR"


class TestStructuredJsonLogging:
    def _fresh_logger(self) -> logging.Logger:
        lg = logging.getLogger("cogtrix")
        lg.handlers.clear()
        return lg

    def test_json_mode_emits_context_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        log_path = tmp_path / "structured.log"
        request_id = new_request_id()
        set_session_id("sess-123")
        try:
            setup_logging(log_file=str(log_path), debug=False)
            logging.getLogger("cogtrix").info("hello world")

            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            record = next(item for item in records if item["message"] == "hello world")
            assert record["level"] == "INFO"
            assert record["logger"] == "cogtrix"
            assert record["session_id"] == "sess-123"
            assert record["request_id"] == request_id
            assert record["timestamp"].endswith("Z")
        finally:
            clear_request_id()
            clear_session_id()
            self._fresh_logger()

    def test_text_mode_remains_plain_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        log_path = tmp_path / "plain.log"
        setup_logging(log_file=str(log_path), debug=False)
        logging.getLogger("cogtrix").info("hello world")

        line = log_path.read_text().splitlines()[-1]
        assert "[INFO] hello world" in line
        assert not line.lstrip().startswith("{")
        assert '"session_id"' not in line
        self._fresh_logger()

    def test_session_context_helpers_round_trip(self) -> None:
        set_session_id("sess-999")
        assert get_session_id() == "sess-999"
        clear_session_id()
        assert get_session_id() == "-"

    def test_request_context_helpers_round_trip(self) -> None:
        req_id = new_request_id()
        assert get_request_id() == req_id
        clear_request_id()
        assert get_request_id() == "-"


@pytest.mark.skipif(
    LLMObservabilityHandler is None,
    reason="langchain_core not installed",
)
class TestLLMObservabilityOrphanDetection:
    """cogtrix47 Case D: when an LLM call exceeds its caller's
    timeout, ``_invoke_with_timeout`` calls ``Future.cancel()`` and
    moves on. Cancel is best-effort — the underlying HTTP request
    keeps running and eventually fires ``on_llm_end``. Without the
    orphan classification the log shows a normal ``LLM_COMPLETE``
    277s after start, polluting observability with what looks like
    a successful (but actually discarded) call. The handler now
    distinguishes the two cases by elapsed time.
    """

    def _make_handler(self, orphan_threshold_s: float = 180.0) -> "LLMObservabilityHandler":

        handler = LLMObservabilityHandler(verbose=True, orphan_threshold_s=orphan_threshold_s)
        handler._logged_messages: list[tuple[str, str]] = []  # type: ignore[attr-defined]

        def _capture_log(level: str, message: str) -> None:
            handler._logged_messages.append((level, message))  # type: ignore[attr-defined]

        handler._log = _capture_log  # type: ignore[method-assign]
        return handler

    def _empty_result(self):
        from langchain_core.outputs import LLMResult

        return LLMResult(generations=[])

    def test_normal_completion_logs_llm_complete(self) -> None:
        # A completion arriving inside the threshold logs the normal
        # info-level LLM_COMPLETE.
        import time

        handler = self._make_handler(orphan_threshold_s=180.0)
        handler._start_time = time.time() - 5.0  # 5s elapsed
        handler.on_llm_end(self._empty_result())

        infos = [m for lvl, m in handler._logged_messages if lvl == "info"]
        warnings = [m for lvl, m in handler._logged_messages if lvl == "warning"]
        assert any("LLM_COMPLETE" in m and "ORPHAN" not in m for m in infos)
        assert not any("ORPHAN_LLM_COMPLETE" in m for m in warnings)

    def test_orphan_completion_logs_orphan_warning(self) -> None:
        # A completion arriving past the threshold logs
        # ORPHAN_LLM_COMPLETE at WARNING level and does NOT emit the
        # normal LLM_COMPLETE.
        import time

        handler = self._make_handler(orphan_threshold_s=180.0)
        handler._start_time = time.time() - 277.0  # 277s elapsed (cogtrix47)
        handler.on_llm_end(self._empty_result())

        infos = [m for lvl, m in handler._logged_messages if lvl == "info"]
        warnings = [m for lvl, m in handler._logged_messages if lvl == "warning"]
        # The normal LLM_COMPLETE line is suppressed for orphans —
        # observability pipelines see exactly one orphan event,
        # not a stale "successful" completion line.
        assert not any("LLM_COMPLETE" in m and "ORPHAN" not in m for m in infos)
        assert any("ORPHAN_LLM_COMPLETE" in m for m in warnings)

    def test_orphan_message_carries_elapsed_and_threshold(self) -> None:
        import time

        handler = self._make_handler(orphan_threshold_s=180.0)
        handler._start_time = time.time() - 277.0
        handler.on_llm_end(self._empty_result())

        orphan_line = next(m for lvl, m in handler._logged_messages if lvl == "warning")
        # Both the elapsed and threshold values must appear so an
        # operator can read the line in isolation.
        assert "277" in orphan_line
        assert "180" in orphan_line
        # And the explanation pointing to the underlying cause.
        assert "Future.cancel" in orphan_line
        assert "discarded" in orphan_line

    def test_threshold_boundary_at_exact_threshold_is_orphan(self) -> None:
        # The ``>=`` comparison means elapsed == threshold counts as
        # orphan. Operators reading "270s ≥ 180s" expect this.
        import time

        handler = self._make_handler(orphan_threshold_s=180.0)
        handler._start_time = time.time() - 180.0
        handler.on_llm_end(self._empty_result())

        warnings = [m for lvl, m in handler._logged_messages if lvl == "warning"]
        assert any("ORPHAN_LLM_COMPLETE" in m for m in warnings)

    def test_no_start_time_no_completion_log(self) -> None:
        # When _start_time is 0 (handler never saw on_llm_start),
        # neither line fires — there's nothing meaningful to log.
        handler = self._make_handler()
        handler._start_time = 0.0
        handler.on_llm_end(self._empty_result())

        completion_lines = [
            m
            for _, m in handler._logged_messages
            if "LLM_COMPLETE" in m or "ORPHAN_LLM_COMPLETE" in m
        ]
        assert completion_lines == []

    def test_custom_threshold_honoured(self) -> None:
        # Operators can tune the threshold to match their LLM timeout
        # config. Test with a non-default value.
        import time

        handler = self._make_handler(orphan_threshold_s=60.0)
        handler._start_time = time.time() - 90.0  # 90s elapsed, below default 180
        handler.on_llm_end(self._empty_result())

        warnings = [m for lvl, m in handler._logged_messages if lvl == "warning"]
        # 90s ≥ 60s custom threshold → orphan.
        assert any("ORPHAN_LLM_COMPLETE" in m for m in warnings)
