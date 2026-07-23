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
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cogtrix import _build_agent_graph
from src.agent.core import build_system_prompt
from src.orchestration.graph import build_agent_graph
from src.orchestration.reflection_delegate import (
    ACCOUNTABILITY_PROMPT,
    UNCERTAINTY_NOTE_PREFIX,
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

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_yaml_min_confidence_threshold_accepts_boundary_values(self, threshold):
        cfg = self._load({"decision_accountability": {"min_confidence_threshold": threshold}})
        assert cfg.decision_accountability_min_confidence == threshold

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


# ── Graph wiring ─────────────────────────────────────────────────────────────


class TestDecisionAccountabilityGraphIntegration:
    @staticmethod
    def _make_low_confidence_response(confidence: float = 0.5) -> AIMessage:
        return AIMessage(
            content=(
                "---PLAN---\n"
                "Do the thing.\n"
                "---ASSUMPTIONS---\n"
                "- A1\n"
                "---EVIDENCE---\n"
                "- E1\n"
                "---CONFIDENCE---\n"
                f"{confidence}\n"
                "---END---\n"
                "---COUNTER-PLAN---\n"
                "Alternative.\n"
                "---FLAWS---\n"
                "- No critical flaws identified\n"
                "---END---"
            ),
            id="da-low-confidence",
        )

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_uncertainty_note_appended_when_enabled(self, threshold):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = self._make_low_confidence_response()

        cfg = AgentRunConfig(
            llm=mock_llm,
            decision_accountability_enabled=True,
            decision_accountability_report_uncertainty=True,
            decision_accountability_min_confidence=threshold,
        )
        graph = build_agent_graph(config=cfg)

        result = graph.invoke({"messages": [HumanMessage(content="evaluate the plan")]})
        last = result["messages"][-1]

        assert UNCERTAINTY_NOTE_PREFIX in getattr(last, "content", "")
        assert f"threshold {threshold:.1f}." in getattr(last, "content", "")
        assert "Proceeding with caution." in getattr(last, "content", "")

    def test_uncertainty_note_suppressed_when_disabled(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        response = self._make_low_confidence_response()
        mock_llm.invoke.return_value = response

        cfg = AgentRunConfig(
            llm=mock_llm,
            decision_accountability_enabled=False,
            decision_accountability_report_uncertainty=True,
            decision_accountability_min_confidence=7.0,
        )
        graph = build_agent_graph(config=cfg)

        result = graph.invoke({"messages": [HumanMessage(content="evaluate the plan")]})
        last = result["messages"][-1]

        assert UNCERTAINTY_NOTE_PREFIX not in getattr(last, "content", "")
        assert getattr(last, "content", "") == response.content


# ── graph.py structural wiring ────────────────────────────────────────────────


class TestGraphDaWiring:
    def test_uncertainty_note_appended_by_graph_when_da_enabled(self):
        """The graph should append the uncertainty note at runtime, not just in source text."""

        response = AIMessage(
            content=(
                "I will proceed with the plan.\n\n"
                "---PLAN---\nDo the thing.\n"
                "---ASSUMPTIONS---\n- A1\n"
                "---EVIDENCE---\n- E1\n"
                "---CONFIDENCE---\n3.0\n---END---\n"
                "---COUNTER-PLAN---\nThis might fail.\n"
                "---FLAWS---\n- Missing validation\n- No rollback path\n---END---"
            ),
            id="m1",
        )
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = response

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=MagicMock(),
            approvals=set(),
            config=AgentRunConfig(
                llm=mock_llm,
                system_prompt="",
                active_tools_list=[],
                available_tools={},
                decision_accountability_enabled=True,
                decision_accountability_report_uncertainty=True,
                decision_accountability_min_confidence=7.0,
            ),
        )

        result = graph.invoke({"messages": [HumanMessage(content="check")]})
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]

        assert ai_messages, "expected the graph to return at least one AIMessage"
        assert any(
            "Decision accountability: confidence 3.0/10" in msg.content for msg in ai_messages
        )
        assert any("Missing validation" in msg.content for msg in ai_messages)

    def test_post_response_parsing_present_in_call_model(self):
        """call_model node must call extract_decision_justification when _da_enabled."""
        import src.orchestration.graph as graph_mod
        import src.orchestration.nodes.call_model as call_model_mod

        graph_src = inspect.getsource(graph_mod.build_agent_graph)
        node_src = inspect.getsource(call_model_mod.build_call_model_node)
        assert "_da_enabled" in graph_src
        assert "extract_decision_justification" in node_src
        assert "UNCERTAINTY_NOTE_PREFIX" in node_src

    def test_uncertainty_note_injected_on_low_confidence(self):
        """Source must contain the uncertainty note pattern."""
        import src.orchestration.nodes.call_model as call_model_mod

        src = inspect.getsource(call_model_mod.build_call_model_node)
        assert "_uncertainty_note" in src
        assert "Proceeding with caution" in src
