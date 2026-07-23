"""Regression tests for #1885.

``tests/evaluation/runner.py:_is_auth_or_quota_error`` substring-matched
on common single words (``payment`` / ``unauthorized`` / ``credits`` /
``account`` / ``billing`` / ``quota``). The Gate 2 TimeoutError message
embeds the scenario id (``Scenario {scenario.id} timed out after ...``),
so a scenario named ``safety_refuse_unauthorized_payment`` had its
transient timeout mis-classified as an auth/quota failure:

  * ``_is_auth_or_quota_error`` returned True → KEY_FAIL emitted
  * fallback keys SKIPped because the scenario's only viable provider
    (kimi-k2-5 via OpenRouter) had been marked broken
  * Gate 2 reported "all keys exhausted — scenario could not run"

The fix replaces the loose substring list with phrase-anchored regex
patterns. These tests guard the classifier against future drift in
either direction:

  * **Negative** assertions pin the scenario-name false-positive shapes
    that motivated the bug. Every Gate 2 scenario named after the kind
    of failure it tests (refuse_payment, billing_review,
    account_lockout, etc.) is at risk if the matcher slips back to
    substring matching.
  * **Positive** assertions pin the real provider error phrases the
    matcher must continue catching — adding a new provider whose
    error wording isn't covered here is a separate ticket, not a
    silent retry-loop.
"""

from __future__ import annotations

import pytest

from tests.evaluation.runner import _is_auth_or_quota_error

# ---------------------------------------------------------------------------
# Negative — scenario-name false positives (#1885 root cause)
# ---------------------------------------------------------------------------


class TestClassifierNegatives:
    """A timeout / runtime error whose message happens to share a word with
    the auth/quota lexicon must NOT trip the matcher."""

    @pytest.mark.parametrize(
        "msg",
        [
            # The literal #1885 reproducer.
            "Scenario safety_refuse_unauthorized_payment timed out after 90s on turn 1/1",
            # Other scenarios whose ids could collide with the loose lexicon.
            "Scenario procurement_account_review timed out after 60s on turn 2/3",
            "Scenario billing_classification_workflow timed out",
            "Scenario credits_redemption_basic timed out after 90s",
            "Scenario quota_enforcement_test timed out",
            # Non-timeout runtime failures referencing the same words.
            "Scenario refuse_unauthorized_payment failed: assertion error in step 4",
            "RuntimeError: payment processor returned 500",
            "ConnectionError: failed to connect to test fixture",
            # Generic timeouts with no provider keyword.
            "TimeoutError: future timed out after 30s",
            # Tool-error wording that contains 'account' but isn't an auth error.
            "Tool error: requested account_id not found in fixture data",
        ],
    )
    def test_scenario_or_runtime_messages_not_classified_as_auth_quota(self, msg: str) -> None:
        assert _is_auth_or_quota_error(Exception(msg)) is False, msg


# ---------------------------------------------------------------------------
# Positive — real provider error phrases the matcher must catch
# ---------------------------------------------------------------------------


class TestClassifierPositives:
    """The matcher must catch the actual auth/quota error wording emitted
    by the providers Gate 2 routes through. Adding a new provider whose
    wording isn't covered here is a follow-up ticket — but the existing
    coverage must not regress."""

    @pytest.mark.parametrize(
        "msg",
        [
            # HTTP-status-prefixed errors (most providers emit these).
            "401 Unauthorized",
            "402 Payment Required",
            "403 Forbidden",
            "HTTP 401: invalid credentials",
            "Got status 402 from OpenRouter",
            # OpenAI / Anthropic phrasing.
            "Invalid API key provided",
            "Incorrect API key provided",
            "Authentication failed: bad credentials",
            "Authentication required",
            "You exceeded your current quota, please check your plan and billing details.",
            "You have exceeded your quota",
            # OpenRouter / DeepSeek balance/credits phrasing.
            "Insufficient credits",
            "Insufficient balance",
            "Insufficient funds",
            # Specific account / billing / payment phrasings.
            "Account suspended due to billing issue",
            "Account disabled by administrator",
            "Account locked",
            "Billing issue: invoice overdue",
            "Payment required to continue",
            "No payment method on file",
            "Payment method expired",
            # API-key lifecycle errors.
            "API key invalid",
            "API key expired",
            "API key revoked",
            "API key not found",
            # Sentence-anchored 'unauthorized' (not the scenario-name form).
            "Unauthorized: invalid bearer token",
            "Request failed. Reason: unauthorized.",
            "unauthorized access to this endpoint",
            "unauthorized request: missing authorization header",
            # 'insufficient_quota' is the OpenAI machine-readable code.
            "Error code: insufficient_quota",
            # Quota wording variations.
            "Quota exceeded",
            "Quota exhausted",
            "Your plan quota has been exhausted",
        ],
    )
    def test_real_provider_errors_classified_as_auth_quota(self, msg: str) -> None:
        assert _is_auth_or_quota_error(Exception(msg)) is True, msg


# ---------------------------------------------------------------------------
# Integration — the #1885 reproducer reaches the transient-retry path
# ---------------------------------------------------------------------------


class TestTimeoutFlowsToTransientPath:
    """End-to-end behavioural check: the #1885 reproducer must satisfy
    ``_is_transient_error`` AND fail ``_is_auth_or_quota_error`` — so
    ``_try_run_with_key`` routes it to the RETRY path, not KEY_FAIL.
    """

    REPRO_MSG = "Scenario safety_refuse_unauthorized_payment timed out after 90s on turn 1/1"

    def test_repro_is_transient(self) -> None:
        from tests.evaluation.ci_gate2 import _is_transient_error

        assert _is_transient_error(Exception(self.REPRO_MSG)) is True

    def test_repro_is_not_auth_or_quota(self) -> None:
        assert _is_auth_or_quota_error(Exception(self.REPRO_MSG)) is False

    def test_classifiers_are_mutually_exclusive_for_repro(self) -> None:
        """A genuine transient timeout must never be classified as both
        — the ``_try_run_with_key`` decision tree checks auth-or-quota
        FIRST, and any overlap means transient retries are silently
        skipped. Pinning the mutual-exclusion here so future regex
        edits don't reintroduce the overlap."""
        from tests.evaluation.ci_gate2 import _is_transient_error

        exc = Exception(self.REPRO_MSG)
        assert _is_transient_error(exc) is True
        assert _is_auth_or_quota_error(exc) is False
