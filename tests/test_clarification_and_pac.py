"""Tests for Mechanism B (clarification policy) and Mechanism C (pre-action confirmation gate)."""

from __future__ import annotations

from src.agent.core import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from src.orchestration.reflection_delegate import (
    CLARIFICATION_POLICY_PROMPT,
    PRE_ACTION_CONFIRMATION_PROMPT,
)
from src.orchestration.run_config import AgentRunConfig

# ── Group A: build_system_prompt parameter wiring ─────────────────────────────


class TestBuildSystemPromptPAC:
    def test_pac_prompt_injected_when_provided(self) -> None:
        result = build_system_prompt(pre_action_confirmation_prompt="## PAC\nConfirm first.")
        assert "## PAC" in result
        assert "Confirm first." in result

    def test_pac_prompt_absent_when_none(self) -> None:
        result = build_system_prompt(pre_action_confirmation_prompt=None)
        assert "Shall I proceed?" not in result

    def test_pac_prompt_appended_after_accountability(self) -> None:
        result = build_system_prompt(
            decision_accountability_prompt="ACCT_MARKER",
            pre_action_confirmation_prompt="PAC_MARKER",
        )
        assert result.index("ACCT_MARKER") < result.index("PAC_MARKER")

    def test_pac_prompt_appended_after_base_prompt(self) -> None:
        result = build_system_prompt(pre_action_confirmation_prompt="PAC_MARKER")
        assert result.index("PAC_MARKER") > 0

    def test_both_prompts_present_when_both_provided(self) -> None:
        result = build_system_prompt(
            decision_accountability_prompt="ACCT",
            pre_action_confirmation_prompt="PAC",
        )
        assert "ACCT" in result
        assert "PAC" in result


# ── Group B: DEFAULT_SYSTEM_PROMPT content ────────────────────────────────────


class TestDefaultSystemPromptClarificationPolicy:
    def test_contains_clarification_policy_section(self) -> None:
        assert "Clarification Policy" in DEFAULT_SYSTEM_PROMPT

    def test_mentions_irreversible(self) -> None:
        assert "irreversible" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_requires_one_question(self) -> None:
        assert "ONE" in DEFAULT_SYSTEM_PROMPT or "one question" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_no_blanket_dont_ask_prohibition(self) -> None:
        assert "Don't stop to ask clarifying questions" not in DEFAULT_SYSTEM_PROMPT

    def test_mentions_assumption_and_proceed(self) -> None:
        lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "assumption" in lower or "assume" in lower


# ── Group C: conversation mode additions ──────────────────────────────────────


class TestConversationModeAdditions:
    def _get_additions(self) -> str:
        from unittest.mock import MagicMock

        from src.memory.modes.conversation import ConversationMemoryManager

        mgr = ConversationMemoryManager(store=MagicMock(), session_id="test", config=None)
        result = mgr.get_system_prompt_additions()
        assert result is not None
        return result

    def test_no_blanket_dont_ask_clarifying(self) -> None:
        additions = self._get_additions()
        assert "Don't stop to ask clarifying questions" not in additions

    def test_risk_weighted_language_present(self) -> None:
        additions = self._get_additions()
        lower = additions.lower()
        assert "low-risk" in lower or "reversible" in lower

    def test_still_encourages_task_completion(self) -> None:
        additions = self._get_additions()
        assert "complete requested tasks" in additions


# ── Group D: config parsing ───────────────────────────────────────────────────


class TestConfigPAC:
    def test_defaults_to_false(self) -> None:
        from src.config import Config

        assert Config().pre_action_confirmation_enabled is False

    def test_field_exists(self) -> None:
        from src.config import Config

        cfg = Config()
        assert hasattr(cfg, "pre_action_confirmation_enabled")

    def test_can_be_set_true(self) -> None:
        from src.config import Config

        cfg = Config()
        cfg.pre_action_confirmation_enabled = True
        assert cfg.pre_action_confirmation_enabled is True

    def test_yaml_parsing_enabled_true(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        import yaml

        from src.config import Config, _apply_config_file

        yaml_content = "pre_action_confirmation:\n  enabled: true\n"
        parsed = yaml.safe_load(yaml_content)
        cfg = Config()
        with patch("src.config._parse_config_file", return_value=parsed):
            _apply_config_file(cfg, Path("fake.yaml"))
        assert cfg.pre_action_confirmation_enabled is True

    def test_yaml_parsing_enabled_false(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        import yaml

        from src.config import Config, _apply_config_file

        yaml_content = "pre_action_confirmation:\n  enabled: false\n"
        parsed = yaml.safe_load(yaml_content)
        cfg = Config()
        cfg.pre_action_confirmation_enabled = True  # start True, expect False
        with patch("src.config._parse_config_file", return_value=parsed):
            _apply_config_file(cfg, Path("fake.yaml"))
        assert cfg.pre_action_confirmation_enabled is False


# ── Group E: AgentRunConfig field ─────────────────────────────────────────────


class TestAgentRunConfigPAC:
    def test_default_is_false(self) -> None:
        assert AgentRunConfig().pre_action_confirmation_enabled is False

    def test_can_be_set_true(self) -> None:
        cfg = AgentRunConfig(pre_action_confirmation_enabled=True)
        assert cfg.pre_action_confirmation_enabled is True

    def test_false_by_default_independent_of_other_flags(self) -> None:
        cfg = AgentRunConfig(decision_accountability_enabled=True)
        assert cfg.pre_action_confirmation_enabled is False


# ── Group F: prompt constants ─────────────────────────────────────────────────


class TestPromptConstants:
    def test_pac_constant_exists_and_non_empty(self) -> None:
        assert isinstance(PRE_ACTION_CONFIRMATION_PROMPT, str)
        assert len(PRE_ACTION_CONFIRMATION_PROMPT) > 0

    def test_pac_constant_mentions_irreversible(self) -> None:
        assert "irreversible" in PRE_ACTION_CONFIRMATION_PROMPT.lower()

    def test_pac_constant_contains_proceed_question(self) -> None:
        assert "Shall I proceed?" in PRE_ACTION_CONFIRMATION_PROMPT

    def test_pac_constant_has_skip_clause(self) -> None:
        lower = PRE_ACTION_CONFIRMATION_PROMPT.lower()
        assert "skip" in lower or "except" in lower

    def test_clarification_policy_constant_exists(self) -> None:
        assert isinstance(CLARIFICATION_POLICY_PROMPT, str)
        assert len(CLARIFICATION_POLICY_PROMPT) > 0

    def test_clarification_policy_mentions_irreversible(self) -> None:
        assert "irreversible" in CLARIFICATION_POLICY_PROMPT.lower()

    def test_clarification_policy_requires_one_question(self) -> None:
        assert "ONE" in CLARIFICATION_POLICY_PROMPT

    def test_clarification_policy_says_stop_after_asking(self) -> None:
        lower = CLARIFICATION_POLICY_PROMPT.lower()
        assert "stop" in lower

    def test_clarification_policy_mentions_surface_conflicts(self) -> None:
        lower = CLARIFICATION_POLICY_PROMPT.lower()
        assert "conflict" in lower or "surface" in lower
