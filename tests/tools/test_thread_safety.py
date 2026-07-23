"""Thread safety tests for BUG-039 (config dict atomic swap) and BUG-048 (stderr lock)."""


class TestConfigAtomicSwap:
    """BUG-039: configure_* functions must replace the dict reference atomically."""

    def test_deep_think_config_reference_changes(self) -> None:
        import cogtrix_core.tools.deep_think as dt

        original_ref = dt._config
        dt.configure_deep_think({"default_provider": "ollama"})
        assert dt._config is not original_ref

    def test_deep_think_config_preserves_existing_keys(self) -> None:
        import cogtrix_core.tools.deep_think as dt

        dt.configure_deep_think({"key_a": "value_a"})
        dt.configure_deep_think({"key_b": "value_b"})
        assert dt._config.get("key_a") == "value_a"
        assert dt._config.get("key_b") == "value_b"

    def test_exa_config_reference_changes(self) -> None:
        import cogtrix_core.tools.exa_search as exa

        original_ref = exa._exa_config
        exa.configure_exa({"api_key": "test-key"})
        assert exa._exa_config is not original_ref

    def test_exa_config_preserves_existing_keys(self) -> None:
        import cogtrix_core.tools.exa_search as exa

        exa.configure_exa({"key_a": "value_a"})
        exa.configure_exa({"key_b": "value_b"})
        assert exa._exa_config.get("key_a") == "value_a"
        assert exa._exa_config.get("key_b") == "value_b"

    def test_brave_config_reference_changes(self) -> None:
        import cogtrix_core.tools.brave_search as brave

        original_ref = brave._brave_config
        brave.configure_brave({"api_key": "test-key"})
        assert brave._brave_config is not original_ref

    def test_brave_config_preserves_existing_keys(self) -> None:
        import cogtrix_core.tools.brave_search as brave

        brave.configure_brave({"key_a": "value_a"})
        brave.configure_brave({"key_b": "value_b"})
        assert brave._brave_config.get("key_a") == "value_a"
        assert brave._brave_config.get("key_b") == "value_b"


class TestStderrLock:
    """BUG-048: web_search must expose a module-level threading.Lock for stderr."""

    def test_stderr_lock_exists(self) -> None:
        from cogtrix_core.tools.web_search import _stderr_lock

        assert _stderr_lock is not None
