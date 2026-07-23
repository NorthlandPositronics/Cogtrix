"""Per-tool TOOL_SETUP injection tests (#2223 phase 2).

Each test:
1. Builds a minimal ``Config`` whose ``services[...]`` carries the token.
2. Calls the tool's ``TOOL_SETUP(config)`` directly (simulating ToolRegistry).
3. Asserts the tool's accessor returns the token even with the env var unset.

All tests are deterministic and make no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Minimal Config stub — avoids importing src.config (which triggers
# provider discovery and may fail in isolated environments).
# ---------------------------------------------------------------------------


@dataclass
class _FakeConfig:
    services: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def openweather_api_key(self) -> str | None:
        return self.services.get("openweather", {}).get("api_key")


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


class TestWeatherInjection:
    def setup_method(self):
        import src.tools.weather as wmod

        wmod._weather_config.clear()

    def teardown_method(self):
        # Always restore clean state so subsequent test modules are unaffected
        import src.tools.weather as wmod

        wmod._weather_config.clear()

    def test_tool_setup_captures_key(self):
        import src.tools.weather as wmod

        cfg = _FakeConfig(services={"openweather": {"api_key": "injected-ow-key"}})
        wmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert wmod._weather_config.get("api_key") == "injected-ow-key"

    def test_get_api_key_returns_injected_key_when_env_unset(self, monkeypatch):
        import src.tools.weather as wmod

        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        wmod._weather_config["api_key"] = "config-injected-key"
        assert wmod._get_api_key() == "config-injected-key"

    def test_injected_key_takes_priority_over_env(self, monkeypatch):
        import src.tools.weather as wmod

        monkeypatch.setenv("OPENWEATHER_API_KEY", "env-key")
        wmod._weather_config["api_key"] = "config-key"
        assert wmod._get_api_key() == "config-key"

    def test_env_fallback_when_no_injection(self, monkeypatch):
        import src.tools.weather as wmod

        wmod._weather_config.clear()
        monkeypatch.setenv("OPENWEATHER_API_KEY", "env-only-key")
        assert wmod._get_api_key() == "env-only-key"

    def test_tool_setup_noop_when_no_key(self):
        import src.tools.weather as wmod

        wmod._weather_config.clear()
        cfg = _FakeConfig(services={})
        wmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert "api_key" not in wmod._weather_config

    def test_tool_setup_idempotent(self):
        import src.tools.weather as wmod

        cfg = _FakeConfig(services={"openweather": {"api_key": "key-v1"}})
        wmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        cfg2 = _FakeConfig(services={"openweather": {"api_key": "key-v2"}})
        wmod.TOOL_SETUP(cfg2)  # type: ignore[arg-type]
        assert wmod._weather_config["api_key"] == "key-v2"


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------


class TestWhatsAppInjection:
    def test_tool_setup_captures_api_key(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_WHATSAPP_API_KEY", raising=False)
        monkeypatch.delenv("COGTRIX_WHATSAPP_URL", raising=False)
        monkeypatch.delenv("COGTRIX_WHATSAPP_SESSION", raising=False)

        import src.tools.whatsapp as wamod

        cfg = _FakeConfig(
            services={"whatsapp": {"api_key": "injected-wa-key", "waha_url": "http://x:3000"}}
        )
        wamod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert wamod._cfg.api_key == "injected-wa-key"
        assert wamod._cfg.waha_url == "http://x:3000"

    def test_api_key_survives_env_unset(self, monkeypatch):
        """Key captured via TOOL_SETUP remains accessible after env unset."""
        monkeypatch.delenv("COGTRIX_WHATSAPP_API_KEY", raising=False)

        import src.tools.whatsapp as wamod

        cfg = _FakeConfig(services={"whatsapp": {"api_key": "post-unset-key"}})
        wamod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        # After unset the singleton carries the injected key
        assert wamod._cfg.api_key == "post-unset-key"

    def test_tool_setup_idempotent(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_WHATSAPP_API_KEY", raising=False)

        import src.tools.whatsapp as wamod

        cfg1 = _FakeConfig(services={"whatsapp": {"api_key": "key-one"}})
        wamod.TOOL_SETUP(cfg1)  # type: ignore[arg-type]
        cfg2 = _FakeConfig(services={"whatsapp": {"api_key": "key-two"}})
        wamod.TOOL_SETUP(cfg2)  # type: ignore[arg-type]
        assert wamod._cfg.api_key == "key-two"

    def test_env_override_still_applies_before_unset(self, monkeypatch):
        """Env var in TOOL_SETUP path takes priority over Config when both present."""
        monkeypatch.setenv("COGTRIX_WHATSAPP_API_KEY", "env-override")

        import src.tools.whatsapp as wamod

        cfg = _FakeConfig(services={"whatsapp": {"api_key": "config-key"}})
        wamod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        # Env wins (it's checked after Config in the TOOL_SETUP override block)
        assert wamod._cfg.api_key == "env-override"


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


class TestTelegramInjection:
    def test_tool_setup_captures_bot_token(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_TELEGRAM_TOKEN", raising=False)

        import src.tools.telegram as tgmod

        cfg = _FakeConfig(services={"telegram": {"bot_token": "injected-tg-token"}})
        tgmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert tgmod._cfg.bot_token == "injected-tg-token"

    def test_token_survives_env_unset(self, monkeypatch):
        """Token captured via TOOL_SETUP remains accessible after env unset."""
        monkeypatch.delenv("COGTRIX_TELEGRAM_TOKEN", raising=False)

        import src.tools.telegram as tgmod

        cfg = _FakeConfig(services={"telegram": {"bot_token": "post-unset-token"}})
        tgmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert tgmod._cfg.bot_token == "post-unset-token"

    def test_tool_setup_idempotent(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_TELEGRAM_TOKEN", raising=False)

        import src.tools.telegram as tgmod

        cfg1 = _FakeConfig(services={"telegram": {"bot_token": "tok-v1"}})
        tgmod.TOOL_SETUP(cfg1)  # type: ignore[arg-type]
        cfg2 = _FakeConfig(services={"telegram": {"bot_token": "tok-v2"}})
        tgmod.TOOL_SETUP(cfg2)  # type: ignore[arg-type]
        assert tgmod._cfg.bot_token == "tok-v2"

    def test_env_override_still_applies_before_unset(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_TELEGRAM_TOKEN", "env-token")

        import src.tools.telegram as tgmod

        cfg = _FakeConfig(services={"telegram": {"bot_token": "config-token"}})
        tgmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert tgmod._cfg.bot_token == "env-token"

    def test_is_configured_with_injected_token(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_TELEGRAM_TOKEN", raising=False)

        import src.tools.telegram as tgmod

        cfg = _FakeConfig(services={"telegram": {"bot_token": "real-token", "allow_send": True}})
        tgmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        # is_configured checks _cfg.bot_token and REQUESTS_AVAILABLE
        if tgmod.REQUESTS_AVAILABLE:
            assert tgmod.is_configured() is True


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


class TestSlackInjection:
    def test_tool_setup_captures_bot_token(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

        import src.tools.slack_tools as slmod

        cfg = _FakeConfig(services={"slack": {"bot_token": "xoxb-injected-token"}})
        slmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert slmod._slack_config.get("bot_token") == "xoxb-injected-token"

    def test_token_survives_env_unset(self, monkeypatch):
        """Token captured via TOOL_SETUP remains in _slack_config after env unset."""
        monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

        import src.tools.slack_tools as slmod

        cfg = _FakeConfig(services={"slack": {"bot_token": "xoxb-post-unset"}})
        slmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert slmod._slack_config.get("bot_token") == "xoxb-post-unset"

    def test_env_override_when_env_still_set(self, monkeypatch):
        """Env var overrides config value inside configure_slack_tools."""
        monkeypatch.setenv("COGTRIX_SLACK_BOT_TOKEN", "xoxb-env-tok")

        import src.tools.slack_tools as slmod

        cfg = _FakeConfig(services={"slack": {"bot_token": "xoxb-config-tok"}})
        slmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        # configure_slack_tools applies env override on top of config dict
        assert slmod._slack_config.get("bot_token") == "xoxb-env-tok"

    def test_is_configured_with_injected_token(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

        import src.tools.slack_tools as slmod

        cfg = _FakeConfig(services={"slack": {"bot_token": "xoxb-real"}})
        slmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        # The token is always injected into the module config (the behaviour
        # under test); is_configured() additionally requires the optional
        # ``slack-sdk`` (``_HAS_SLACK``), which is NOT installed in the CI
        # unit-test extras — so only assert is_configured() when it's present
        # (mirrors the telegram REQUESTS_AVAILABLE guard above).
        assert slmod._slack_config.get("bot_token") == "xoxb-real"
        if slmod._HAS_SLACK:
            assert slmod.is_configured() is True

    def test_is_configured_false_when_no_token(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

        import src.tools.slack_tools as slmod

        cfg = _FakeConfig(services={})
        slmod.TOOL_SETUP(cfg)  # type: ignore[arg-type]
        assert slmod.is_configured() is False
