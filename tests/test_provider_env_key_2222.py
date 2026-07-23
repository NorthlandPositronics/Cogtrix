"""Generic per-provider API-key env override (#2222).

Custom / self-hosted providers (e.g. a local vLLM ``spark`` endpoint) have no
well-known ``*_API_KEY`` env name, so their key could previously only live
inline in the config file. ``COGTRIX_PROVIDER_<NAME>_API_KEY`` lets ANY
provider's key come from the environment (a ``.env``) instead — keeping secrets
out of a committable config.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogtrix_core.config import load_config

_SPARK_YAML = textwrap.dedent("""
    providers:
      spark:
        type: openai
        base_url: http://192.168.70.254:8080/v1
    models:
      coder: {provider: spark, model: qwen3-coder}
    model: coder
    """)


@pytest.fixture
def spark_config(tmp_path: Path) -> str:
    p = tmp_path / "cogtrix.comprehensive.yaml"
    p.write_text(_SPARK_YAML)
    return str(p)


def _load(config_file: str):
    return load_config(SimpleNamespace(config_file=config_file))


class TestGenericProviderEnvKey:
    def test_custom_provider_key_from_env(self, spark_config, monkeypatch):
        # The config defines `spark` with NO inline key.
        monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY", "sk-spark-secret")
        cfg = _load(spark_config)
        assert cfg.providers["spark"].api_key == "sk-spark-secret"

    def test_absent_env_leaves_key_none(self, spark_config, monkeypatch):
        monkeypatch.delenv("COGTRIX_PROVIDER_SPARK_API_KEY", raising=False)
        cfg = _load(spark_config)
        assert cfg.providers["spark"].api_key is None

    def test_env_overrides_inline_key(self, tmp_path, monkeypatch):
        # Env wins over a file-inline key, consistent with the well-known keys.
        p = tmp_path / "c.yaml"
        p.write_text(textwrap.dedent("""
                providers:
                  spark: {type: openai, base_url: http://x/v1, api_key: inline-key}
                models:
                  coder: {provider: spark, model: qwen3-coder}
                model: coder
                """))
        monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY", "env-wins")
        cfg = _load(str(p))
        assert cfg.providers["spark"].api_key == "env-wins"

    def test_name_is_lowercased(self, spark_config, monkeypatch):
        # The env NAME is conventionally upper; it maps to the lowercase provider.
        monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY", "sk-x")
        cfg = _load(spark_config)
        assert "spark" in cfg.providers
        assert cfg.providers["spark"].api_key == "sk-x"

    def test_well_known_key_still_works(self, tmp_path, monkeypatch):
        # The generic path must not regress the hardcoded well-known providers.
        p = tmp_path / "c.yaml"
        p.write_text(textwrap.dedent("""
                providers:
                  deepseek: {type: openai, base_url: https://api.deepseek.com/v1}
                models:
                  ds: {provider: deepseek, model: deepseek-v4}
                model: ds
                """))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        cfg = _load(str(p))
        assert cfg.providers["deepseek"].api_key == "sk-deepseek"

    def test_non_matching_env_var_ignored(self, spark_config, monkeypatch):
        # A var that doesn't fit the pattern must not create a bogus provider.
        monkeypatch.setenv("COGTRIX_PROVIDER_SPARK", "not-a-key")  # missing _API_KEY
        monkeypatch.setenv("SOME_OTHER_API_KEY", "unrelated")
        before = set(_load(spark_config).providers)
        assert "spark" in before
        # No provider named after the malformed vars.
        assert not any(p in before for p in ("some_other", "spark_"))
