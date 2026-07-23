"""Tests for _safe_int, _safe_float helpers and non-numeric config resilience."""

from pathlib import Path

from cogtrix_core.config import Config, _apply_config_file, _safe_float, _safe_int


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
        # Default lowered 2000 → 800 in #1952 Option C.
        assert config.rag.chunk_size == 800

    def test_rag_chunk_overlap_non_numeric(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "rag:\n  chunk_overlap: notanumber\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        # Default lowered 200 → 100 in #1952 Option C.
        assert config.rag.chunk_overlap == 100

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


class TestShellOperatorPolicyOverrideConfig:
    """#2392 — parsing of shell.extra_safe_commands and shell.allow_patterns."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(content)
        return cfg_file

    def test_parses_extra_safe_commands_and_allow_patterns(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n"
            "  extra_safe_commands:\n"
            "    - sync\n"
            "    - umount\n"
            "  allow_patterns:\n"
            "    - '^ssh root@192\\.168\\.70\\.20 '\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_extra_safe_commands == ["sync", "umount"]
        assert config.shell_allow_patterns == ["^ssh root@192\\.168\\.70\\.20 "]

    def test_single_string_values_coerced_to_list(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  extra_safe_commands: sync\n  allow_patterns: '^ssh '\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_extra_safe_commands == ["sync"]
        assert config.shell_allow_patterns == ["^ssh "]

    def test_defaults_empty_when_unset(self, tmp_path):
        cfg_file = self._write_yaml(tmp_path, "shell:\n  curl_wget_allowed_domains:\n    - x.com\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_extra_safe_commands == []
        assert config.shell_allow_patterns == []

    def test_invalid_type_ignored(self, tmp_path):
        cfg_file = self._write_yaml(
            tmp_path,
            "shell:\n  extra_safe_commands: 123\n  allow_patterns: 456\n",
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.shell_extra_safe_commands == []
        assert config.shell_allow_patterns == []


class TestResolveContextMaxTokens:
    """#2360: context_max_tokens auto-scales with the active model's window
    (max(40k, window//2)) unless the operator set it explicitly."""

    def _cfg_with_window(self, window, monkeypatch):
        from cogtrix_core.config import Config, ModelConfig

        c = Config()
        mc = ModelConfig(provider="p", model="m", context_window=window)
        monkeypatch.setattr(c, "resolve_llm_config", lambda: (None, mc))
        return c

    def test_big_window_scales_to_half(self, monkeypatch):
        c = self._cfg_with_window(262_144, monkeypatch)
        assert c.resolve_context_max_tokens() == 131_072  # 262144 // 2

    def test_medium_window_scales_to_half(self, monkeypatch):
        c = self._cfg_with_window(200_000, monkeypatch)
        assert c.resolve_context_max_tokens() == 100_000

    def test_small_window_keeps_40k_floor(self, monkeypatch):
        # 32k window → 16k half, but the floor keeps it at 40k (no regression).
        c = self._cfg_with_window(32_768, monkeypatch)
        assert c.resolve_context_max_tokens() == 40_000

    def test_none_window_uses_default_and_floor(self, monkeypatch):
        c = self._cfg_with_window(None, monkeypatch)
        assert c.resolve_context_max_tokens() == 40_000  # DEFAULT_CONTEXT_WINDOW 32768 → floor

    def test_explicit_value_wins_over_autoscale(self, monkeypatch):
        c = self._cfg_with_window(262_144, monkeypatch)
        c.context_max_tokens = 90_000
        c.context_max_tokens_is_default = False
        assert c.resolve_context_max_tokens() == 90_000

    def test_explicit_zero_stays_disabled(self, monkeypatch):
        c = self._cfg_with_window(262_144, monkeypatch)
        c.context_max_tokens = 0
        c.context_max_tokens_is_default = False
        assert c.resolve_context_max_tokens() == 0

    def test_resolution_failure_falls_back_to_40k(self, monkeypatch):
        from cogtrix_core.config import Config

        c = Config()

        def _boom():
            raise RuntimeError("no active model")

        monkeypatch.setattr(c, "resolve_llm_config", _boom)
        assert c.resolve_context_max_tokens() == 40_000

    def test_parser_marks_explicit_when_key_present(self, tmp_path):
        from cogtrix_core.config import Config, _apply_config_file

        cfg = tmp_path / ".cogtrix.yaml"
        cfg.write_text("context_max_tokens: 55000\n")
        config = Config()
        _apply_config_file(config, cfg)
        assert config.context_max_tokens == 55_000
        assert config.context_max_tokens_is_default is False

    def test_parser_leaves_default_flag_when_key_absent(self, tmp_path):
        from cogtrix_core.config import Config, _apply_config_file

        cfg = tmp_path / ".cogtrix.yaml"
        cfg.write_text("verbosity: 0\n")
        config = Config()
        _apply_config_file(config, cfg)
        assert config.context_max_tokens_is_default is True


class TestResolveContextMaxMessages:
    """#2397: the message cap auto-scales with the resolved token budget (so the
    token cap governs and a flat 200 doesn't evict context every turn on
    big-window models) unless the operator set it explicitly."""

    def _cfg_with_window(self, window, monkeypatch):
        from cogtrix_core.config import Config, ModelConfig

        c = Config()
        mc = ModelConfig(provider="p", model="m", context_window=window)
        monkeypatch.setattr(c, "resolve_llm_config", lambda: (None, mc))
        return c

    def test_big_window_scales_message_cap(self, monkeypatch):
        # 262k window → 131,072 token budget → 200 * (131072/40000) ≈ 655.
        c = self._cfg_with_window(262_144, monkeypatch)
        assert c.resolve_context_max_messages() == 655

    def test_medium_window_scales_message_cap(self, monkeypatch):
        # 200k window → 100,000 token budget → 200 * 2.5 = 500.
        c = self._cfg_with_window(200_000, monkeypatch)
        assert c.resolve_context_max_messages() == 500

    def test_small_window_keeps_200_floor(self, monkeypatch):
        # 32k window → 40k token floor → ratio 1.0 → 200 (no regression).
        c = self._cfg_with_window(32_768, monkeypatch)
        assert c.resolve_context_max_messages() == 200

    def test_explicit_value_wins_over_autoscale(self, monkeypatch):
        c = self._cfg_with_window(262_144, monkeypatch)
        c.context_max_messages = 50
        c.context_max_messages_is_default = False
        assert c.resolve_context_max_messages() == 50

    def test_explicit_zero_stays_disabled(self, monkeypatch):
        c = self._cfg_with_window(262_144, monkeypatch)
        c.context_max_messages = 0
        c.context_max_messages_is_default = False
        assert c.resolve_context_max_messages() == 0

    def test_default_zero_stays_disabled(self, monkeypatch):
        # 0 with the default flag still means "cap disabled" — don't scale it up.
        c = self._cfg_with_window(262_144, monkeypatch)
        c.context_max_messages = 0
        assert c.resolve_context_max_messages() == 0

    def test_resolution_failure_keeps_200(self, monkeypatch):
        from cogtrix_core.config import Config

        c = Config()

        def _boom():
            raise RuntimeError("no active model")

        monkeypatch.setattr(c, "resolve_llm_config", _boom)
        # resolve_context_max_tokens falls back to 40k → ratio 1.0 → 200.
        assert c.resolve_context_max_messages() == 200

    def test_parser_marks_explicit_when_key_present(self, tmp_path):
        from cogtrix_core.config import Config, _apply_config_file

        cfg = tmp_path / ".cogtrix.yaml"
        cfg.write_text("context_max_messages: 120\n")
        config = Config()
        _apply_config_file(config, cfg)
        assert config.context_max_messages == 120
        assert config.context_max_messages_is_default is False

    def test_parser_leaves_default_flag_when_key_absent(self, tmp_path):
        from cogtrix_core.config import Config, _apply_config_file

        cfg = tmp_path / ".cogtrix.yaml"
        cfg.write_text("verbosity: 0\n")
        config = Config()
        _apply_config_file(config, cfg)
        assert config.context_max_messages_is_default is True
