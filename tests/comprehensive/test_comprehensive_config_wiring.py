"""The dedicated comprehensive-test config + .env loader wire together correctly.

Deterministic (no live LLM): proves the config is secret-free, resolves
deterministically, and that every key is injected from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.config import load_config
from tests.comprehensive import env_loader

CONFIG_PATH = env_loader.CONFIG_PATH


def _load_cfg():
    return load_config(SimpleNamespace(config_file=str(CONFIG_PATH)))


def test_config_file_exists_and_parses():
    assert CONFIG_PATH.is_file()
    yaml.safe_load(CONFIG_PATH.read_text())  # valid YAML


def test_config_is_secret_free():
    """No api_key/token/secret literals may live in the committed config."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("api_key", "key", "token", "secret", "password") and v:
                    pytest.fail(f"secret-bearing field '{k}={v!r}' in committed config")
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(raw)


def test_active_model_and_providers():
    cfg = _load_cfg()
    assert cfg.active_model_alias == "coder"
    assert "spark" in cfg.providers
    assert "deepseek" in cfg.providers
    spark_url = cfg.providers["spark"].base_url
    assert spark_url and spark_url.startswith("http://192.168.70.254")


def test_spark_key_injected_from_env(monkeypatch):
    monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY", "sk-spark-from-env")
    cfg = _load_cfg()
    assert cfg.providers["spark"].api_key == "sk-spark-from-env"


def test_deepseek_and_service_keys_injected_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-from-env")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "ow-from-env")
    cfg = _load_cfg()
    assert cfg.providers["deepseek"].api_key == "sk-deepseek-from-env"
    assert cfg.tavily_api_key == "tvly-from-env"
    assert cfg.openweather_api_key == "ow-from-env"


def test_keys_absent_without_env(monkeypatch):
    for k in env_loader.EXPECTED_KEYS:
        monkeypatch.delenv(k, raising=False)
    cfg = _load_cfg()
    assert cfg.providers["spark"].api_key is None
    assert cfg.providers["deepseek"].api_key is None


def test_loader_sets_config_file_and_returns_path(monkeypatch):
    monkeypatch.delenv("COGTRIX_CONFIG_FILE", raising=False)
    returned = env_loader.load_comprehensive_env()
    assert returned == CONFIG_PATH
    assert Path(os.environ["COGTRIX_CONFIG_FILE"]) == CONFIG_PATH


def test_missing_keys_reports_absent(monkeypatch):
    for k in env_loader.EXPECTED_KEYS:
        monkeypatch.delenv(k, raising=False)
    assert set(env_loader.missing_keys()) == set(env_loader.EXPECTED_KEYS)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert "DEEPSEEK_API_KEY" not in env_loader.missing_keys()
