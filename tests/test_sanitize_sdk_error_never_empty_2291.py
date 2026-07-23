"""#2291 — provider-rejection error must never produce an EMPTY user-facing message.

``_sanitize_sdk_error`` truncates at SDK markers like ``Error code:``. For an
OpenAI-compatible ``Error code: 400 - {...}`` string the marker sits at index 0,
so naive truncation collapsed the whole message to ``""`` — producing a blank WS
``error`` frame and an empty operator log, which reads to the user as "no answer".

The sanitizer must now never return empty: it recovers the provider's ``message``
field, else a generic non-empty fallback. That invariant cascades to every
``format_agent_error`` branch that interpolates the sanitized text.
"""

from __future__ import annotations

from cogtrix_core.orchestration.runner import _sanitize_sdk_error, format_agent_error


class BadRequestError(Exception):
    pass


# The exact production error shape (OpenRouter → provider 400), trimmed.
_PROD_ERR = (
    "Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, "
    "'metadata': {'raw': '{\"code\":400,\"msg\":\"bad request\"}', 'provider_name': 'AkashML'}}}"
)


class TestSanitizeNeverEmpty:
    def test_marker_at_index_zero_no_longer_empties(self) -> None:
        out = _sanitize_sdk_error(_PROD_ERR)
        assert out.strip(), "sanitized message must not be empty"
        # Recovers the provider's own message field.
        assert "Provider returned error" in out

    def test_marker_at_zero_without_message_field_uses_generic_fallback(self) -> None:
        out = _sanitize_sdk_error("Error code: 500 - upstream exploded")
        assert out.strip()
        assert out == "the model provider rejected the request"

    def test_plain_text_unaffected(self) -> None:
        assert _sanitize_sdk_error("boom happened") == "boom happened"

    def test_invariant_never_empty_across_shapes(self) -> None:
        for t in (
            _PROD_ERR,
            "Error code: 400 - ",
            "Request body: secret-stuff",
            "Error message: ",
            "Response body: {}",
        ):
            assert _sanitize_sdk_error(t).strip(), f"empty for {t!r}"


class TestFormatAgentErrorNonEmpty:
    """The empty-sanitize bug surfaced through format_agent_error's branches."""

    def test_bad_request_message_is_non_empty(self) -> None:
        msg = format_agent_error(BadRequestError(_PROD_ERR))
        assert msg.strip()
        assert "Invalid request" in msg
        # Not a dangling "**Invalid request:** " with an empty tail.
        assert msg.rstrip().endswith("error") or "Provider returned error" in msg

    def test_generic_branch_non_empty_tail(self) -> None:
        # An error that hits no specific branch and whose sanitize used to empty.
        class WeirdError(Exception):
            pass

        msg = format_agent_error(WeirdError("Error code: 418 - teapot"))
        assert msg.strip()
        assert not msg.rstrip().endswith(":")  # no dangling empty tail
