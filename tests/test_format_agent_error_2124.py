"""Regression tests for #2124 — invalid model ID (provider 400) handling.

A provider 400 such as OpenRouter's ``qwen3 is not a valid model ID`` is a
user/config error, not a Cogtrix fault. It must:

* surface an *actionable* message (which model, how to fix), and
* be classified as a user-config error so the caller logs it concisely
  instead of emitting an ERROR-level stack trace.
"""

from __future__ import annotations

import logging

from cogtrix_core.orchestration.runner import _is_user_config_error, format_agent_error


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


_NO_DB_400 = (
    "Error code: 400 - {'error': {'message': 'No connected db.', "
    "'type': 'no_db_connection', 'param': None, 'code': '400'}}"
)


class TestGenericBadRequestMessage:
    """#2220: the generic BadRequest branch must surface the provider's message.

    ``_sanitize_sdk_error`` alone truncates at the leading ``Error code:`` marker
    (index 0) and returns an empty string, so the operator previously saw a bare
    ``**Invalid request:** `` with no reason. The fix extracts the API ``message``
    field first (as the model-id / rate-limit branches already do).
    """

    def test_provider_message_is_surfaced(self) -> None:
        msg = format_agent_error(BadRequestError(_NO_DB_400))
        assert "Invalid request" in msg
        # The actual reason must be present — not an empty detail.
        assert "No connected db" in msg

    def test_detail_is_not_empty(self) -> None:
        # Pin the regression precisely: the text after the bold prefix is non-blank.
        msg = format_agent_error(BadRequestError(_NO_DB_400))
        detail = msg.split("**Invalid request:**", 1)[1].strip()
        assert detail, "generic BadRequest rendered an empty detail (regression #2220)"

    def test_no_message_field_falls_back_without_crashing(self) -> None:
        # A 400 with no extractable message keeps the prefix (empty detail is OK
        # when there genuinely is nothing to show) and never raises.
        msg = format_agent_error(BadRequestError("Error code: 400 - opaque"))
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


# #2358 — OpenRouter 402 (out of daily credits / max_tokens too high). The raw
# message embeds a key-management URL containing the key id, which must never
# reach a log line or a user-facing error frame.
class APIStatusError(Exception):
    pass


_402_WITH_KEY_URL = (
    "Error code: 402 - {'error': {'message': \"This request requires more credits, "
    "or fewer max_tokens. You requested up to 65536 tokens, but can only afford "
    "34908. To increase, visit https://openrouter.ai/workspaces/default/keys/"
    "4cc243185e37d3ff658e3ed51653b9fe4f0454acb802f53c3852f23e60422714 and adjust "
    "the key's daily limit\", 'code': 402}}"
)


class TestOpenRouter402:
    def test_402_is_user_config_error(self) -> None:
        assert _is_user_config_error(APIStatusError(_402_WITH_KEY_URL)) is True

    def test_402_message_is_actionable(self) -> None:
        msg = format_agent_error(APIStatusError(_402_WITH_KEY_URL)).lower()
        assert "credits" in msg or "budget" in msg
        assert "max_tokens" in msg

    def test_402_message_does_not_leak_key_url(self) -> None:
        msg = format_agent_error(APIStatusError(_402_WITH_KEY_URL))
        assert "http" not in msg
        assert "openrouter.ai" not in msg
        assert "4cc243185e37d3ff" not in msg  # the key id

    def test_sanitize_redacts_urls(self) -> None:
        from cogtrix_core.orchestration.runner import _sanitize_sdk_error

        out = _sanitize_sdk_error(_402_WITH_KEY_URL)
        assert "http" not in out
        assert "openrouter.ai" not in out
        assert "4cc243185e37d3ff" not in out
        assert out.strip()  # never empty (#2291)


class TestProviderMessageRedaction:
    """Forge audit: URL redaction must cover EVERY provider-message echo, not only
    _sanitize_sdk_error — _extract_api_message feeds the 429 / invalid-model
    branches of format_agent_error and previously leaked the raw key URL."""

    _429_WITH_KEY_URL = (
        "Error code: 429 - {'error': {'message': \"You are over your quota; top up "
        "at https://openrouter.ai/keys/4cc243185e37d3ff to continue\", 'code': 429}}"
    )

    def test_extract_api_message_redacts_url(self) -> None:
        from cogtrix_core.orchestration.runner import _extract_api_message

        out = _extract_api_message(self._429_WITH_KEY_URL)
        assert out is not None
        assert "http" not in out
        assert "openrouter.ai" not in out
        assert "4cc243185e37d3ff" not in out
        assert "over your quota" in out  # the actionable text survives

    def test_rate_limit_branch_does_not_leak_key_url(self) -> None:
        # "rate_limit"/"429" in the message routes through the rate-limit branch,
        # which echoes _extract_api_message — now redacted.
        msg = format_agent_error(Exception("rate_limit — " + self._429_WITH_KEY_URL))
        assert "4cc243185e37d3ff" not in msg
        assert "openrouter.ai" not in msg

    def test_payment_required_classified_and_actionable(self) -> None:
        e = APIStatusError("Error code: 402 - {'error': {'message': 'Payment Required'}}")
        assert _is_user_config_error(e) is True
        out = format_agent_error(e).lower()
        assert "budget" in out or "credits" in out  # hits the 402 branch, not generic
