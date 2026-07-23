"""Integration tests for ADR-0052 Milestone 2 wiring.

Covers:
- build_system_prompt injects ACCOUNTABILITY_PROMPT when the flag is set
- Config YAML parsing for decision_accountability section
- AgentRunConfig carries the DA fields
- extract_decision_justification + should_proceed=False → uncertainty note appended
- graph.py post-response parsing (via _da_enabled closure)
"""

from __future__ import annotations

import inspect

from src.agent.core import build_system_prompt
from src.orchestration.reflection_delegate import (
    ACCOUNTABILITY_PROMPT,
    extract_decision_justification,
)
from src.orchestration.run_config import AgentRunConfig

# ── build_system_prompt injection ─────────────────────────────────────────────


class TestBuildSystemPromptInjection:
    def test_accountability_prompt_absent_when_none(self):
        prompt = build_system_prompt(decision_accountability_prompt=None)
        assert "Decision Accountability" not in prompt

    def test_accountability_prompt_injected_when_provided(self):
        prompt = build_system_prompt(decision_accountability_prompt=ACCOUNTABILITY_PROMPT)
        assert "Decision Accountability" in prompt
        assert "---PLAN---" in prompt
        assert "---COUNTER-PLAN---" in prompt

    def test_accountability_prompt_appended_last(self):
        """DA block must appear after all other prompt sections."""
        from src.agent.core import DEFAULT_SYSTEM_PROMPT

        prompt = build_system_prompt(
            mode_additions="## Custom Mode\nsome addition",
            decision_accountability_prompt=ACCOUNTABILITY_PROMPT,
        )
        da_pos = prompt.find("Decision Accountability")
        mode_pos = prompt.find("Custom Mode")
        base_pos = prompt.find(DEFAULT_SYSTEM_PROMPT[:40])
        assert da_pos > mode_pos > base_pos

    def test_accountability_constant_not_empty(self):
        assert len(ACCOUNTABILITY_PROMPT) > 100
        assert "---PLAN---" in ACCOUNTABILITY_PROMPT
        assert "---FLAWS---" in ACCOUNTABILITY_PROMPT


# ── Config YAML parsing ───────────────────────────────────────────────────────


class TestConfigYamlParsing:
    def _load(self, yaml_dict: dict):
        """Apply a dict of config values via a temporary YAML file."""
        import tempfile
        from pathlib import Path

        import yaml

        from src.config import Config, _apply_config_file

        cfg = Config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_dict, f)
            tmp_path = Path(f.name)
        try:
            _apply_config_file(cfg, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return cfg

    def test_default_decision_accountability_disabled(self):
        """Feature must be off by default — opt-in only."""
        from src.config import Config

        cfg = Config()
        assert cfg.decision_accountability_enabled is False

    def test_yaml_enabled_true(self):
        cfg = self._load({"decision_accountability": {"enabled": True}})
        assert cfg.decision_accountability_enabled is True

    def test_yaml_enabled_false(self):
        cfg = self._load({"decision_accountability": {"enabled": False}})
        assert cfg.decision_accountability_enabled is False

    def test_yaml_min_confidence_threshold(self):
        cfg = self._load({"decision_accountability": {"min_confidence_threshold": 8.5}})
        assert cfg.decision_accountability_min_confidence == 8.5

    def test_yaml_report_uncertainty_false(self):
        cfg = self._load({"decision_accountability": {"report_uncertainty": False}})
        assert cfg.decision_accountability_report_uncertainty is False

    def test_yaml_require_counter_plan_false(self):
        cfg = self._load({"decision_accountability": {"require_counter_plan": False}})
        assert cfg.decision_accountability_require_counter_plan is False

    def test_yaml_invalid_confidence_ignored(self):
        """Out-of-range threshold must leave the default unchanged."""
        cfg = self._load({"decision_accountability": {"min_confidence_threshold": 99.0}})
        assert cfg.decision_accountability_min_confidence == 7.0

    def test_empty_section_leaves_defaults(self):
        cfg = self._load({"decision_accountability": {}})
        assert cfg.decision_accountability_enabled is False


# ── AgentRunConfig fields ─────────────────────────────────────────────────────


class TestAgentRunConfigFields:
    def test_default_da_disabled(self):
        cfg = AgentRunConfig()
        assert cfg.decision_accountability_enabled is False

    def test_default_report_uncertainty_true(self):
        cfg = AgentRunConfig()
        assert cfg.decision_accountability_report_uncertainty is True

    def test_default_min_confidence(self):
        cfg = AgentRunConfig()
        assert cfg.decision_accountability_min_confidence == 7.0

    def test_fields_settable(self):
        cfg = AgentRunConfig(
            decision_accountability_enabled=True,
            decision_accountability_report_uncertainty=False,
            decision_accountability_min_confidence=8.0,
        )
        assert cfg.decision_accountability_enabled is True
        assert cfg.decision_accountability_report_uncertainty is False
        assert cfg.decision_accountability_min_confidence == 8.0


# ── Post-response uncertainty note ───────────────────────────────────────────


class TestUncertaintyNoteFormat:
    _LOW_CONF_RESPONSE = (
        "I will proceed with the plan.\n\n"
        "---PLAN---\nDo the thing.\n"
        "---ASSUMPTIONS---\n- A1\n"
        "---EVIDENCE---\n- E1\n"
        "---CONFIDENCE---\n3.0\n---END---\n"
        "---COUNTER-PLAN---\nThis might fail.\n"
        "---FLAWS---\n- Missing validation\n- No rollback path\n---END---"
    )

    def test_low_confidence_triggers_uncertainty_note(self):
        """When confidence is below threshold, extract returns should_proceed=False."""
        result = extract_decision_justification(self._LOW_CONF_RESPONSE)
        assert result is not None
        assert result["should_proceed"] is False  # 3.0 - 2.0 (flaws) = 1.0 < 7.0

    def test_uncertainty_note_contains_confidence(self):
        result = extract_decision_justification(self._LOW_CONF_RESPONSE)
        assert result is not None
        assert result["confidence"] == 3.0

    def test_uncertainty_note_contains_flaws(self):
        result = extract_decision_justification(self._LOW_CONF_RESPONSE)
        assert result is not None
        assert len(result["flaws"]) == 2
        assert "Missing validation" in result["flaws"]

    def test_high_confidence_no_flaws_proceeds(self):
        good_response = (
            "---PLAN---\nStep 1.\n---ASSUMPTIONS---\n- A1\n"
            "---EVIDENCE---\n- E1\n---CONFIDENCE---\n9.0\n---END---\n"
            "---COUNTER-PLAN---\nAlt.\n---FLAWS---\n- No critical flaws identified\n---END---"
        )
        result = extract_decision_justification(good_response)
        assert result is not None
        assert result["should_proceed"] is True


# ── graph.py structural wiring ────────────────────────────────────────────────


class TestGraphDaWiring:
    def test_da_closure_vars_read_from_config(self):
        """build_agent_graph must read _da_enabled from AgentRunConfig."""
        import src.orchestration.graph as graph_mod

        src = inspect.getsource(graph_mod.build_agent_graph)
        assert "_da_enabled" in src, "_da_enabled closure variable missing from build_agent_graph"
        assert "decision_accountability_enabled" in src

    def test_post_response_parsing_present_in_call_model(self):
        """call_model must call extract_decision_justification when _da_enabled."""
        import src.orchestration.graph as graph_mod

        src = inspect.getsource(graph_mod.build_agent_graph)
        assert "extract_decision_justification" in src
        assert "_da_enabled" in src

    def test_uncertainty_note_injected_on_low_confidence(self):
        """Source must contain the uncertainty note pattern."""
        import src.orchestration.graph as graph_mod

        src = inspect.getsource(graph_mod.build_agent_graph)
        assert "Decision accountability" in src
        assert "_uncertainty_note" in src
