"""Tests for _safe_int, _safe_float helpers and non-numeric config resilience."""

from pathlib import Path

from src.config import Config, _apply_config_file, _safe_float, _safe_int


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(42, "field") == 42

    def test_valid_string(self):
        assert _safe_int("42", "field") == 42

    def test_invalid_string_no_default(self):
        assert _safe_int("abc", "field") is None

    def test_invalid_string_with_default(self):
        assert _safe_int("abc", "field", default=10) == 10

    def test_none_value_no_default(self):
        assert _safe_int(None, "field") is None

    def test_none_value_with_default(self):
        assert _safe_int(None, "field", default=5) == 5

    def test_float_truncates(self):
        assert _safe_int(3.9, "field") == 3

    def test_negative_int(self):
        assert _safe_int(-1, "field") == -1


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float(0.5, "field") == 0.5

    def test_valid_string(self):
        assert _safe_float("0.5", "field") == 0.5

    def test_invalid_string_no_default(self):
        assert _safe_float("abc", "field") is None

    def test_invalid_string_with_default(self):
        assert _safe_float("abc", "field", default=1.0) == 1.0

    def test_none_value_no_default(self):
        assert _safe_float(None, "field") is None

    def test_int_coerced(self):
        assert _safe_float(2, "field") == 2.0


class TestApplyConfigFileNonNumeric:
    """_apply_config_file must not crash on non-numeric values in numeric fields."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(content)
        return cfg_file

    def test_delegate_timeout_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "delegate:\n  default_timeout: abc\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        # Should not crash; default_timeout stays unchanged
        assert config.delegate_default_timeout == 60

    def test_context_compression_min_age_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "context_compression:\n  min_age: xyz\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.context_compression_min_age == 6

    def test_context_compression_min_chars_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "context_compression:\n  min_chars: xyz\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.context_compression_min_chars == 2000

    def test_rag_chunk_size_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "rag:\n  chunk_size: notanumber\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.rag.chunk_size == 2000

    def test_rag_chunk_overlap_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "rag:\n  chunk_overlap: notanumber\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.rag.chunk_overlap == 200

    def test_research_delegate_timeout_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "research_delegate:\n  timeout: nope\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.research_delegate_timeout == 300

    def test_research_delegate_cap_ratio_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "research_delegate:\n  cap_ratio: nope\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.research_delegate_cap_ratio == 0.85

    def test_provider_num_ctx_non_numeric(self, tmp_path):
        """Non-numeric num_ctx in provider YAML auto-migrates to model; invalid value is dropped."""
        cfg_file = self._write_yaml(
            tmp_path,
            "providers:\n  my_ollama:\n    type: ollama\n    num_ctx: abc\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert "my_ollama" in config.providers
        migrated = [m for m in config.models.values() if m.provider == "my_ollama"]
        for m in migrated:
            assert m.context_window is None

    def test_provider_temperature_non_numeric(self, tmp_path):
        """Non-numeric temperature in provider YAML auto-migrates to model; invalid value is dropped."""
        cfg_file = self._write_yaml(
            tmp_path,
            "providers:\n  my_openai:\n    type: openai\n    temperature: hot\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert "my_openai" in config.providers
        migrated = [m for m in config.models.values() if m.provider == "my_openai"]
        for m in migrated:
            assert m.temperature is None

    def test_provider_max_tokens_non_numeric(self, tmp_path):
        """Non-numeric max_tokens in provider YAML auto-migrates to model; invalid value is dropped."""
        cfg_file = self._write_yaml(
            tmp_path,
            "providers:\n  my_openai:\n    type: openai\n    max_tokens: many\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert "my_openai" in config.providers
        migrated = [m for m in config.models.values() if m.provider == "my_openai"]
        for m in migrated:
            assert m.max_tokens is None


class TestShellCurlWgetAllowedDomainsConfig:
    """Tests for shell.curl_wget_allowed_domains config parsing (issue #1632)."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(content)
        return cfg_file

    def test_shell_block_parses_curl_wget_allowed_domains(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  curl_wget_allowed_domains:\n    - github.com\n    - api.stripe.com\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == ["github.com", "api.stripe.com"]

    def test_shell_block_single_string_domain(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  curl_wget_allowed_domains: github.com\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == ["github.com"]

    def test_legacy_top_level_key_still_works(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell_curl_wget_allowed_domains:\n  - example.com\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == ["example.com"]

    def test_shell_block_takes_precedence_over_legacy_top_level(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  curl_wget_allowed_domains:\n    - block.com\n"
            "shell_curl_wget_allowed_domains:\n  - legacy.com\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == ["block.com"]

    def test_invalid_type_in_shell_block_ignored(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  curl_wget_allowed_domains: 12345\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == []

    def test_no_shell_block_leaves_default_empty(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "verbosity: 0\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_curl_wget_allowed_domains == []
