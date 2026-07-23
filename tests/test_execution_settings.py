from __future__ import annotations

import pytest

from src.config import Config
from src.orchestration.run_config import AgentRunConfig, ExecutionSettings


def test_config_to_execution_settings_maps_execution_fields() -> None:
    cfg = Config(
        context_compression=False,
        context_compression_min_age=11,
        context_compression_min_chars=3333,
        context_max_messages=144,
        tier_cache_enabled=False,
        tool_context_limit_pct=0.75,
        parallel_tool_execution=False,
        git_native=True,
        decision_accountability_enabled=True,
        decision_accountability_report_uncertainty=False,
        decision_accountability_min_confidence=8.5,
        task_ownership_classifier_enabled=False,
        task_ownership_classifier_llm_fallback=True,
        task_ownership_ambiguous_action="inform",
        pre_action_confirmation_enabled=True,
    )

    settings = cfg.to_execution_settings()

    assert settings.context_compression is False
    assert settings.compression_min_age == 11
    assert settings.compression_min_chars == 3333
    assert settings.context_max_messages == 144
    assert settings.tier_cache_enabled is False
    assert settings.tool_context_limit_pct == 0.75
    assert settings.parallel_tool_execution is False
    assert settings.git_native is True
    assert settings.decision_accountability_enabled is True
    assert settings.decision_accountability_report_uncertainty is False
    assert settings.decision_accountability_min_confidence == 8.5
    assert settings.task_ownership_classifier_enabled is False
    assert settings.task_ownership_classifier_llm_fallback is True
    assert settings.task_ownership_ambiguous_action == "inform"
    assert settings.pre_action_confirmation_enabled is True


def test_agent_run_config_compatibility_shim_proxies_to_execution_settings() -> None:
    cfg = AgentRunConfig(
        context_compression=False,
        compression_min_age=7,
        compression_min_chars=777,
        context_max_messages=99,
        tier_cache_enabled=False,
        tool_context_limit_pct=0.61,
        parallel_tool_execution=False,
        git_native=True,
        decision_accountability_enabled=True,
        decision_accountability_report_uncertainty=False,
        decision_accountability_min_confidence=9.5,
        task_ownership_classifier_enabled=False,
        task_ownership_classifier_llm_fallback=True,
        task_ownership_ambiguous_action="execute",
        pre_action_confirmation_enabled=True,
    )

    assert cfg.execution_settings.context_compression is False
    assert cfg.execution_settings.compression_min_age == 7
    assert cfg.execution_settings.compression_min_chars == 777
    assert cfg.execution_settings.context_max_messages == 99
    assert cfg.execution_settings.tier_cache_enabled is False
    assert cfg.execution_settings.tool_context_limit_pct == 0.61
    assert cfg.execution_settings.parallel_tool_execution is False
    assert cfg.execution_settings.git_native is True
    assert cfg.execution_settings.decision_accountability_enabled is True
    assert cfg.execution_settings.decision_accountability_report_uncertainty is False
    assert cfg.execution_settings.decision_accountability_min_confidence == 9.5
    assert cfg.execution_settings.task_ownership_classifier_enabled is False
    assert cfg.execution_settings.task_ownership_classifier_llm_fallback is True
    assert cfg.execution_settings.task_ownership_ambiguous_action == "execute"
    assert cfg.execution_settings.pre_action_confirmation_enabled is True

    cfg.context_compression = True
    cfg.compression_min_age = 3
    cfg.tool_context_limit_pct = 0.5

    assert cfg.execution_settings.context_compression is True
    assert cfg.execution_settings.compression_min_age == 3
    assert cfg.execution_settings.tool_context_limit_pct == 0.5

    cfg.execution_settings = ExecutionSettings(context_compression=False)
    assert cfg.context_compression is False
    assert cfg.compression_min_age is None


def test_agent_run_config_rejects_invalid_execution_settings_type() -> None:
    with pytest.raises(TypeError, match="ExecutionSettings"):
        AgentRunConfig(execution_settings="invalid")  # type: ignore[arg-type]


def test_from_app_config_uses_projected_execution_settings() -> None:
    cfg = Config(
        context_compression=False,
        context_compression_min_age=5,
        context_compression_min_chars=500,
        context_max_messages=77,
        tier_cache_enabled=False,
        tool_context_limit_pct=0.55,
        parallel_tool_execution=False,
        git_native=True,
    )

    run_cfg = AgentRunConfig.from_app_config(cfg)

    assert run_cfg.execution_settings is not None
    assert run_cfg.execution_settings.context_compression is False
    assert run_cfg.execution_settings.compression_min_age == 5
    assert run_cfg.execution_settings.compression_min_chars == 500
    assert run_cfg.execution_settings.context_max_messages == 77
    assert run_cfg.execution_settings.tier_cache_enabled is False
    assert run_cfg.execution_settings.tool_context_limit_pct == 0.55
    assert run_cfg.execution_settings.parallel_tool_execution is False
    assert run_cfg.execution_settings.git_native is True


def test_from_app_config_wires_model_timeout() -> None:
    """#2146 — from_app_config forwards the active model's timeout to
    AgentRunConfig.llm_timeout (previously the field was dead)."""

    class _FakeModel:
        timeout = 600

    class _FakeConfig:
        def resolve_llm_config(self):
            return (object(), _FakeModel())

    run_cfg = AgentRunConfig.from_app_config(_FakeConfig())
    assert run_cfg.llm_timeout == 600


def test_from_app_config_defaults_timeout_without_resolver() -> None:
    """A config lacking resolve_llm_config keeps the AgentRunConfig default."""

    class _BareConfig:
        pass

    run_cfg = AgentRunConfig.from_app_config(_BareConfig())
    assert run_cfg.llm_timeout == 180


def test_from_app_config_timeout_resolution_failure_falls_back() -> None:
    """If resolve_llm_config raises, llm_timeout falls back to the default."""

    class _BoomConfig:
        def resolve_llm_config(self):
            raise RuntimeError("no active model")

    run_cfg = AgentRunConfig.from_app_config(_BoomConfig())
    assert run_cfg.llm_timeout == 180
