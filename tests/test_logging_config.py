"""Tests for src/logging_config — secret scrubbing and observability handler."""

import pytest

from src.logging_config import LLMObservabilityHandler, _scrub_secrets


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
