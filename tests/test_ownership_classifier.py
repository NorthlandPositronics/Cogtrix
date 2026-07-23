"""Tests for the task ownership classifier in src.orchestration.intent."""

from __future__ import annotations

import pytest

from src.orchestration.intent import (
    OwnershipMode,
    OwnershipResult,
    _apply_reversibility_override,
    _classify_ownership_layer1,
    _extract_action_phrase,
    classify_task_ownership,
)

# ── Shared stub ───────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Minimal LLM stub whose invoke() returns a configurable content string."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0
        self.last_prompt_text: str = ""

    def invoke(self, messages: list) -> _FakeResponse:
        self.call_count += 1
        self.last_prompt_text = messages[0].content if messages else ""
        return _FakeResponse(self._content)


class _BrokenLLM:
    def invoke(self, messages: list) -> None:
        raise RuntimeError("simulated timeout")


# ── Layer 1: happy path ───────────────────────────────────────────────────────


class TestLayer1HappyPath:
    @pytest.mark.parametrize(
        "prompt, expected_mode",
        [
            # INFORM
            ("how to install gh", OwnershipMode.INFORM),
            ("how can gh be installed", OwnershipMode.INFORM),
            ("how do I configure nginx", OwnershipMode.INFORM),
            ("what are the steps to deploy a container", OwnershipMode.INFORM),
            ("explain how to set up a virtual environment", OwnershipMode.INFORM),
            ("can I install packages without sudo", OwnershipMode.INFORM),
            ("is it possible to run this on Windows", OwnershipMode.INFORM),
            # ADVISE
            ("should I use pip or conda", OwnershipMode.ADVISE),
            ("what would you recommend for a CI tool", OwnershipMode.ADVISE),
            ("what's the best way to structure a monorepo", OwnershipMode.ADVISE),
            # EXECUTE
            ("install gh", OwnershipMode.EXECUTE),
            ("run the test suite", OwnershipMode.EXECUTE),
            ("delete the old config file", OwnershipMode.EXECUTE),
            ("create a new virtualenv", OwnershipMode.EXECUTE),
            ("can you install docker", OwnershipMode.EXECUTE),
            ("can you create a README", OwnershipMode.EXECUTE),
            ("can you run the tests", OwnershipMode.EXECUTE),
            # AMBIGUOUS
            ("check how gh can be installed", OwnershipMode.AMBIGUOUS),
            ("verify the nginx config", OwnershipMode.AMBIGUOUS),
            # "find X" is a read/search operation → falls to default_execute
            ("find the installed packages", OwnershipMode.EXECUTE),
        ],
    )
    def test_layer1_mode(self, prompt: str, expected_mode: OwnershipMode) -> None:
        result = _classify_ownership_layer1(prompt)
        assert (
            result.mode == expected_mode
        ), f"Expected {expected_mode.name} for {prompt!r}, got {result.mode.name}"


# ── Layer 1: confidence and reversibility ─────────────────────────────────────


class TestLayer1ConfidenceAndReversibility:
    def test_inform_has_high_confidence(self) -> None:
        r = _classify_ownership_layer1("how to install gh")
        assert r.confidence >= 0.7

    def test_execute_install_is_not_reversible(self) -> None:
        r = _classify_ownership_layer1("install gh")
        assert r.is_reversible is False

    def test_execute_run_is_reversible(self) -> None:
        r = _classify_ownership_layer1("run the tests")
        assert r.is_reversible is True

    def test_raw_signal_is_populated(self) -> None:
        r = _classify_ownership_layer1("how to configure nginx")
        assert r.raw_signal

    def test_inferred_action_is_populated(self) -> None:
        r = _classify_ownership_layer1("install gh on Ubuntu")
        assert r.inferred_action


# ── Layer 2: LLM fallback ─────────────────────────────────────────────────────


class TestLayer2LLMFallback:
    def test_llm_not_called_when_layer1_confident_execute(self) -> None:
        llm = _FakeLLM("AGENT")
        result = classify_task_ownership("install gh", llm=llm, llm_fallback_enabled=True)
        # Layer1 returns EXECUTE with confidence 0.80 >= threshold 0.6
        assert llm.call_count == 0
        assert result.mode == OwnershipMode.EXECUTE

    def test_llm_called_when_layer1_ambiguous(self) -> None:
        llm = _FakeLLM("AGENT")
        classify_task_ownership("check how gh can be installed", llm=llm, llm_fallback_enabled=True)
        assert llm.call_count == 1

    def test_llm_user_label_maps_to_inform(self) -> None:
        llm = _FakeLLM("USER")
        result = classify_task_ownership(
            "check how gh can be installed", llm=llm, llm_fallback_enabled=True
        )
        assert result.mode == OwnershipMode.INFORM
        assert result.raw_signal == "llm_user"

    def test_llm_agent_label_maps_to_execute(self) -> None:
        llm = _FakeLLM("AGENT")
        result = classify_task_ownership("check gh", llm=llm, llm_fallback_enabled=True)
        assert result.mode == OwnershipMode.EXECUTE

    def test_llm_ambiguous_label_maps_to_ambiguous(self) -> None:
        llm = _FakeLLM("AMBIGUOUS")
        result = classify_task_ownership("check gh", llm=llm, llm_fallback_enabled=True)
        assert result.mode == OwnershipMode.AMBIGUOUS

    def test_llm_not_called_when_fallback_disabled(self) -> None:
        llm = _FakeLLM("AGENT")
        classify_task_ownership(
            "check how gh can be installed", llm=llm, llm_fallback_enabled=False
        )
        assert llm.call_count == 0

    def test_llm_exception_falls_back_to_layer1(self) -> None:
        result = classify_task_ownership("check gh", llm=_BrokenLLM(), llm_fallback_enabled=True)
        # Must not raise; mode falls back to Layer 1 result
        assert result.mode in set(OwnershipMode)

    def test_llm_prompt_contains_sanitized_message(self) -> None:
        llm = _FakeLLM("USER")
        classify_task_ownership(
            "check <script>alert(1)</script>", llm=llm, llm_fallback_enabled=True
        )
        assert "<script>" not in llm.last_prompt_text


# ── Layer 3: reversibility override ──────────────────────────────────────────


class TestReversibilityOverride:
    def test_irreversible_execute_low_confidence_becomes_advise(self) -> None:
        r = OwnershipResult(
            mode=OwnershipMode.EXECUTE,
            confidence=0.45,
            is_reversible=False,
            raw_signal="default_execute",
        )
        out = _apply_reversibility_override(r, min_confidence=0.7)
        assert out.mode == OwnershipMode.ADVISE
        assert "reversibility_override" in out.raw_signal

    def test_irreversible_execute_high_confidence_not_downgraded(self) -> None:
        r = OwnershipResult(
            mode=OwnershipMode.EXECUTE,
            confidence=0.85,
            is_reversible=False,
            raw_signal="execute_imperative",
        )
        out = _apply_reversibility_override(r, min_confidence=0.7)
        assert out.mode == OwnershipMode.EXECUTE

    def test_reversible_execute_low_confidence_not_downgraded(self) -> None:
        r = OwnershipResult(
            mode=OwnershipMode.EXECUTE,
            confidence=0.45,
            is_reversible=True,
            raw_signal="execute_imperative",
        )
        out = _apply_reversibility_override(r, min_confidence=0.7)
        assert out.mode == OwnershipMode.EXECUTE

    def test_inform_never_downgraded(self) -> None:
        r = OwnershipResult(
            mode=OwnershipMode.INFORM,
            confidence=0.3,
            is_reversible=False,
            raw_signal="inform_pattern",
        )
        out = _apply_reversibility_override(r, min_confidence=0.9)
        assert out.mode == OwnershipMode.INFORM

    def test_override_preserves_inferred_action(self) -> None:
        r = OwnershipResult(
            mode=OwnershipMode.EXECUTE,
            confidence=0.45,
            is_reversible=False,
            raw_signal="default_execute",
            inferred_action="install gh",
        )
        out = _apply_reversibility_override(r, min_confidence=0.7)
        assert out.inferred_action == "install gh"


# ── End-to-end: motivating regression case ───────────────────────────────────


class TestMotivatingCase:
    def test_check_how_to_install_is_ambiguous(self) -> None:
        """Motivating bug: 'check how gh can be installed' must NOT return EXECUTE."""
        result = _classify_ownership_layer1("check how gh can be installed")
        assert (
            result.mode != OwnershipMode.EXECUTE
        ), "Regression: 'check how gh can be installed' must not be classified as EXECUTE"

    def test_check_how_to_install_with_llm_user_returns_inform(self) -> None:
        llm = _FakeLLM("USER")
        result = classify_task_ownership(
            "check how gh can be installed", llm=llm, llm_fallback_enabled=True
        )
        assert result.mode == OwnershipMode.INFORM

    def test_bare_install_is_execute(self) -> None:
        result = _classify_ownership_layer1("install gh")
        assert result.mode == OwnershipMode.EXECUTE


# ── Clarifying question format ────────────────────────────────────────────────


class TestClarifyingQuestion:
    def test_extract_action_phrase_basic(self) -> None:
        phrase = _extract_action_phrase("check how gh can be installed")
        assert phrase  # non-empty
        assert len(phrase) <= 60

    def test_extract_action_phrase_strips_trailing_punctuation(self) -> None:
        phrase = _extract_action_phrase("install gh?")
        assert not phrase.endswith("?")

    def test_extract_action_phrase_truncates_at_semicolon(self) -> None:
        phrase = _extract_action_phrase("install gh; then configure it")
        assert ";" not in phrase

    def test_clarifying_question_template_well_formed(self) -> None:
        action = _extract_action_phrase("check how gh can be installed")
        question = (
            f"I can either explain how to {action} or do it for you. " f"Which would you prefer?"
        )
        assert question.endswith("?")
        assert "explain how to" in question
        assert "do it for you" in question


# ── M1: _IRREVERSIBLE_TARGETS false-positive regression tests ─────────────────


class TestIrreversibleTargetsRegex:
    def _matches(self, text: str) -> bool:
        from src.orchestration.intent import _IRREVERSIBLE_TARGETS

        return bool(_IRREVERSIBLE_TARGETS.search(text))

    # Should NOT be flagged irreversible (M1 fixes)
    def test_deploy_to_staging_not_irreversible(self) -> None:
        assert not self._matches("deploy to staging")

    def test_deploy_to_dev_not_irreversible(self) -> None:
        assert not self._matches("deploy to dev")

    def test_format_output_not_irreversible(self) -> None:
        assert not self._matches("format the output")

    def test_format_json_not_irreversible(self) -> None:
        assert not self._matches("format this JSON")

    def test_destroy_variable_not_irreversible(self) -> None:
        assert not self._matches("destroy the variable")

    def test_provision_feature_not_irreversible(self) -> None:
        assert not self._matches("provision a new feature")

    def test_rm_bare_not_irreversible(self) -> None:
        assert not self._matches("rm tmp.txt")

    # Should still be flagged irreversible
    def test_deploy_to_prod_is_irreversible(self) -> None:
        assert self._matches("deploy to production")

    def test_rm_rf_is_irreversible(self) -> None:
        assert self._matches("rm -rf /var/log")

    def test_format_disk_is_irreversible(self) -> None:
        assert self._matches("format disk /dev/sda")

    def test_destroy_cluster_is_irreversible(self) -> None:
        assert self._matches("destroy cluster prod")

    def test_pip_install_is_irreversible(self) -> None:
        assert self._matches("pip install requests")

    def test_apt_install_is_irreversible(self) -> None:
        assert self._matches("apt install nginx")

    def test_drop_table_is_irreversible(self) -> None:
        assert self._matches("DROP TABLE users")

    def test_delete_all_is_irreversible(self) -> None:
        assert self._matches("delete all records")


# ── M6: _AMBIGUOUS_PATTERNS with polite prefix ────────────────────────────────


class TestAmbiguousPatternsPolitePrefix:
    @pytest.mark.parametrize(
        "prompt",
        [
            # These hit the AMBIGUOUS path (no INFORM/ADVISE signal dominates)
            "Could you verify the nginx config",
            "Would you look into the deployment",
            "Please verify the certificate",
            "check the nginx config",  # no prefix, existing case
            "verify the certificate",
        ],
    )
    def test_polite_prefix_triggers_ambiguous(self, prompt: str) -> None:
        result = _classify_ownership_layer1(prompt)
        assert (
            result.mode == OwnershipMode.AMBIGUOUS
        ), f"Expected AMBIGUOUS for {prompt!r}, got {result.mode.name}"

    @pytest.mark.parametrize(
        "prompt",
        [
            # "how to" dominates over "check" prefix → INFORM
            "Please check how to install gh",
            # "can you check" is ambiguous; it should not be forced into INFORM
            "Can you check if the service is running",
        ],
    )
    def test_inform_or_ambiguous_wins_over_polite_check_prefix(self, prompt: str) -> None:
        """How-to prompts stay INFORM; polite check prompts stay AMBIGUOUS."""
        result = _classify_ownership_layer1(prompt)
        expected_mode = (
            OwnershipMode.INFORM
            if prompt.startswith("Please check how to")
            else OwnershipMode.AMBIGUOUS
        )
        assert (
            result.mode == expected_mode
        ), f"Expected {expected_mode.name} (dominant signal) for {prompt!r}, got {result.mode.name}"


# ── M3: Layer-2 timeout ───────────────────────────────────────────────────────


class TestLayer2Timeout:
    def test_timeout_falls_back_to_layer1(self) -> None:
        """When Layer-2 LLM call times out, result falls back to Layer-1 value.

        The executor must NOT block on shutdown(wait=True) after a timeout —
        that would cause the test to hang until the background thread finishes.
        _SlowLLM sleeps 3s; the timeout fires at 1s; the whole test must
        complete well within pytest-timeout's 30s limit.
        """
        import time

        class _SlowLLM:
            def invoke(self, messages: list) -> None:
                time.sleep(3)  # longer than timeout_seconds=1, shorter than pytest limit

        result = classify_task_ownership(
            "check gh",
            llm=_SlowLLM(),
            llm_fallback_enabled=True,
            llm_timeout_seconds=1,
        )
        assert result.mode in set(OwnershipMode)

    def test_timeout_signature_accepts_parameter(self) -> None:
        result = classify_task_ownership("install gh", llm_timeout_seconds=5)
        assert result.mode == OwnershipMode.EXECUTE
