"""Regression tests for #2066 — resolve_embedding_config must surface misconfig.

Two opaque-failure causes now produce a clear warning (non-fatal):
  - rag.model points to an alias that isn't defined in the models registry;
  - the resolved provider type can't produce embeddings (e.g. anthropic).
"""

from __future__ import annotations

import logging

from cogtrix_core.config import Config, ModelConfig, ProviderConfig


def _cfg(provider_type: str, rag_model: str | None) -> Config:
    cfg = Config()
    cfg.providers = {"p": ProviderConfig(name="p", type=provider_type)}
    cfg.models = {"m": ModelConfig(provider="p", model="some-model")}
    cfg.active_model_alias = "m"
    cfg.rag.model = rag_model
    return cfg


def test_missing_rag_alias_warns(caplog) -> None:
    cfg = _cfg("openai", rag_model="ghost-alias")  # not in the models registry
    with caplog.at_level(logging.WARNING, logger="cogtrix"):
        provider_type, *_ = cfg.resolve_embedding_config()
    assert provider_type == "openai"  # falls back to the active provider
    assert "not defined in the models registry" in caplog.text


def test_non_embedding_provider_warns(caplog) -> None:
    cfg = _cfg("anthropic", rag_model="m")  # alias resolves to a non-embedding provider
    with caplog.at_level(logging.WARNING, logger="cogtrix"):
        provider_type, *_ = cfg.resolve_embedding_config()
    assert provider_type == "anthropic"
    assert "does not support embeddings" in caplog.text


def test_embedding_capable_provider_no_warning(caplog) -> None:
    cfg = _cfg("openai", rag_model="m")
    with caplog.at_level(logging.WARNING, logger="cogtrix"):
        cfg.resolve_embedding_config()
    assert "does not support embeddings" not in caplog.text
