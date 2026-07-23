"""Regression tests for #2124 — invalid model ID (provider 400) handling.

A provider 400 such as OpenRouter's ``qwen3 is not a valid model ID`` is a
user/config error, not a Cogtrix fault. It must:

* surface an *actionable* message (which model, how to fix), and
* be classified as a user-config error so the caller logs it concisely
  instead of emitting an ERROR-level stack trace.
"""

from __future__ import annotations

import logging

from src.orchestration.runner import _is_user_config_error, format_agent_error


# A stand-in for ``openai.BadRequestError`` — classification keys on the class
# *name* and the message text, so the concrete type is irrelevant.
class BadRequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


_INVALID_MODEL_400 = (
    "Error code: 400 - {'error': {'message': 'qwen3 is not a valid model ID', 'code': 400}}"
)


class TestFormatInvalidModelId:
    def test_invalid_model_id_message_is_actionable(self) -> None:
        msg = format_agent_error(BadRequestError(_INVALID_MODEL_400))
        assert "Invalid model ID" in msg
        # Names the rejected model and points at the config fix.
        assert "qwen3 is not a valid model ID" in msg
        assert "models.<alias>.model" in msg

    def test_plain_text_invalid_model_without_json_body(self) -> None:
        msg = format_agent_error(BadRequestError("qwen3 is not a valid model"))
        assert "Invalid model ID" in msg

    def test_does_not_leak_stack_trace_markers(self) -> None:
        msg = format_agent_error(BadRequestError(_INVALID_MODEL_400 + "\nTraceback: ..."))
        assert "Traceback" not in msg

    def test_generic_bad_request_still_handled(self) -> None:
        # A BadRequest that is NOT an invalid-model error keeps the generic path.
        msg = format_agent_error(BadRequestError("Error code: 400 - something else"))
        assert "Invalid model ID" not in msg
        assert "Invalid request" in msg


class TestIsUserConfigError:
    def test_bad_request_is_user_config_error(self) -> None:
        assert _is_user_config_error(BadRequestError(_INVALID_MODEL_400)) is True

    def test_authentication_is_user_config_error(self) -> None:
        assert _is_user_config_error(AuthenticationError("invalid_api_key")) is True

    def test_invalid_model_text_on_generic_type(self) -> None:
        # Even an unrecognized type counts if the message says invalid model.
        assert _is_user_config_error(RuntimeError("xyz is not a valid model")) is True

    def test_unexpected_internal_error_is_not_user_config(self) -> None:
        assert _is_user_config_error(RuntimeError("boom: null deref in graph")) is False


def test_user_config_error_logged_without_stack_trace(caplog) -> None:
    """The classifier must drive concise WARNING logging (no ERROR/exc_info).

    This is a light proxy for the runner's outer handler: it asserts the
    branch decision, which is what gates the log level + traceback.
    """
    e = BadRequestError(_INVALID_MODEL_400)
    assert _is_user_config_error(e) is True
    # And the inverse path stays ERROR-worthy.
    with caplog.at_level(logging.ERROR):
        assert _is_user_config_error(ValueError("genuine internal bug")) is False
