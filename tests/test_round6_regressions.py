"""Regression tests for round 6 bug fixes.

Covers:
- Scheduled message edit/cancel require admin auth (not just any user)
- Deep think _call_llm_parallel semaphore release on timeout/error
- RAG _get_uploads_dir respects COGTRIX_DATA_DIR env var
- WhatsApp message_fetch_limit is configurable and respected
"""

from __future__ import annotations

import collections
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Scheduled message endpoints require admin auth
# ---------------------------------------------------------------------------


class TestScheduledMessageAdminAuth:
    """Non-admin users must get 403 on scheduled message mutation endpoints."""

    @pytest.fixture(autouse=True)
    def _set_jwt_secret(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "a" * 64)
        from cogtrix_core.api.auth import configure_jwt_secret

        configure_jwt_secret("a" * 64)

    def _admin_headers(self) -> dict[str, str]:
        from cogtrix_core.api.auth import create_access_token

        token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        return {"Authorization": f"Bearer {token}"}

    def _user_headers(self) -> dict[str, str]:
        from cogtrix_core.api.auth import create_access_token

        token = create_access_token(user_id=str(uuid.uuid4()), role="user")
        return {"Authorization": f"Bearer {token}"}

    def test_non_admin_cannot_edit_scheduled_message(self):
        from starlette.testclient import TestClient

        from cogtrix_core.api.app import app

        with TestClient(app) as c:
            resp = c.patch(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                json={"text": "hijack"},
                headers=self._user_headers(),
            )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_non_admin_cannot_cancel_scheduled_message(self):
        from starlette.testclient import TestClient

        from cogtrix_core.api.app import app

        with TestClient(app) as c:
            resp = c.delete(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                headers=self._user_headers(),
            )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_admin_can_reach_edit_endpoint(self):
        """Admin auth passes the auth gate (409 = service not running, not 403)."""
        from starlette.testclient import TestClient

        from cogtrix_core.api.app import app

        with TestClient(app) as c:
            resp = c.patch(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                json={"text": "ok"},
                headers=self._admin_headers(),
            )
        # 409 means the request reached the handler (service not running),
        # not 403 which would mean auth failed.
        assert resp.status_code == 409

    def test_admin_can_reach_cancel_endpoint(self):
        """Admin auth passes the auth gate (409 = service not running, not 403)."""
        from starlette.testclient import TestClient

        from cogtrix_core.api.app import app

        with TestClient(app) as c:
            resp = c.delete(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                headers=self._admin_headers(),
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 2. Deep think semaphore leak prevention
# ---------------------------------------------------------------------------


class TestDeepThinkSemaphoreLeak:
    """Verify semaphore slots are released even when LLM calls fail or timeout."""

    def test_call_llm_parallel_releases_semaphore_on_success(self):
        """All semaphore slots released after successful parallel calls."""
        from cogtrix_core.tools.deep_think import _call_llm_parallel, _deep_think_sem

        initial = _deep_think_sem._value  # type: ignore[attr-defined]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="result")

        results = _call_llm_parallel(mock_llm, ["p1", "p2"], timeout=10)

        assert _deep_think_sem._value == initial  # type: ignore[attr-defined]
        assert results[0] == "result"
        assert results[1] == "result"

    def test_call_llm_parallel_releases_semaphore_on_exception(self):
        """All semaphore slots released when LLM calls raise exceptions."""
        from cogtrix_core.tools.deep_think import _call_llm_parallel, _deep_think_sem

        initial = _deep_think_sem._value  # type: ignore[attr-defined]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        _call_llm_parallel(mock_llm, ["p1", "p2", "p3"], timeout=5)

        assert _deep_think_sem._value == initial  # type: ignore[attr-defined]

    def test_call_llm_parallel_releases_semaphore_on_timeout(self):
        """All semaphore slots released when LLM calls exceed timeout."""
        from cogtrix_core.tools.deep_think import _call_llm_parallel, _deep_think_sem

        initial = _deep_think_sem._value  # type: ignore[attr-defined]

        _hang_event = threading.Event()

        def _hang(*args, **kwargs):
            _hang_event.wait()  # block indefinitely until timeout cancels us
            return MagicMock(content="late")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _hang

        # Very short timeout to trigger the timeout path quickly
        _call_llm_parallel(mock_llm, ["p1"], timeout=1)

        # Give the pool a moment to clean up
        time.sleep(0.5)
        assert _deep_think_sem._value == initial  # type: ignore[attr-defined]
        _hang_event.set()  # release any still-blocked workers

    def test_call_llm_single_releases_semaphore_on_timeout(self):
        """Single-call path releases semaphore on timeout."""
        from cogtrix_core.tools.deep_think import _call_llm, _deep_think_sem

        initial = _deep_think_sem._value  # type: ignore[attr-defined]

        _hang_event = threading.Event()

        def _hang(*args, **kwargs):
            _hang_event.wait()  # block indefinitely until timeout cancels us
            return MagicMock(content="late")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _hang

        result = _call_llm(mock_llm, "test prompt", timeout=1)

        assert _deep_think_sem._value == initial  # type: ignore[attr-defined]
        assert "Timeout" in result
        _hang_event.set()  # release any still-blocked worker

    def test_semaphore_not_over_released(self):
        """Semaphore value must not exceed initial capacity after calls."""
        from cogtrix_core.tools.deep_think import (
            _DEEP_THINK_MAX_CONCURRENT,
            _call_llm_parallel,
            _deep_think_sem,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")

        # Run several rounds
        for _ in range(3):
            _call_llm_parallel(mock_llm, ["a", "b"], timeout=5)

        assert _deep_think_sem._value == _DEEP_THINK_MAX_CONCURRENT  # type: ignore[attr-defined]

    def test_more_prompts_than_capacity_no_deadlock(self):
        """When len(prompts) > semaphore capacity, must not deadlock."""
        from cogtrix_core.tools.deep_think import (
            _DEEP_THINK_MAX_CONCURRENT,
            _call_llm_parallel,
            _deep_think_sem,
        )

        initial = _deep_think_sem._value  # type: ignore[attr-defined]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")

        # Request more prompts than semaphore capacity
        prompts = [f"p{i}" for i in range(_DEEP_THINK_MAX_CONCURRENT + 1)]
        results = _call_llm_parallel(mock_llm, prompts, timeout=10)

        assert _deep_think_sem._value == initial  # type: ignore[attr-defined]
        # All prompts should get results (not truncated)
        assert all(r == "ok" for r in results)
        assert len(results) == _DEEP_THINK_MAX_CONCURRENT + 1


# ---------------------------------------------------------------------------
# 3. RAG _get_uploads_dir respects COGTRIX_DATA_DIR
# ---------------------------------------------------------------------------


class TestRagUploadsDir:
    """_get_uploads_dir() must read COGTRIX_DATA_DIR env var."""

    def test_default_uses_data_prefix(self):
        from cogtrix_core.api.routes.rag import _get_uploads_dir

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COGTRIX_DATA_DIR", None)
            result = _get_uploads_dir()
        assert result == Path("data", "api", "uploads").resolve()

    def test_custom_data_dir_from_env(self, tmp_path):
        from cogtrix_core.api.routes.rag import _get_uploads_dir

        with patch.dict(os.environ, {"COGTRIX_DATA_DIR": str(tmp_path)}):
            result = _get_uploads_dir()
        assert result == (tmp_path / "api" / "uploads").resolve()

    def test_tasks_rag_uses_same_pattern(self, tmp_path):
        from cogtrix_core.api.tasks.rag import _get_uploads_dir

        with patch.dict(os.environ, {"COGTRIX_DATA_DIR": str(tmp_path)}):
            result = _get_uploads_dir()
        assert result == (tmp_path / "api" / "uploads").resolve()

    def test_each_call_reads_fresh_env(self, tmp_path):
        """Not cached — changing env var changes the result."""
        from cogtrix_core.api.routes.rag import _get_uploads_dir

        with patch.dict(os.environ, {"COGTRIX_DATA_DIR": str(tmp_path / "a")}):
            r1 = _get_uploads_dir()
        with patch.dict(os.environ, {"COGTRIX_DATA_DIR": str(tmp_path / "b")}):
            r2 = _get_uploads_dir()
        assert r1 != r2


# ---------------------------------------------------------------------------
# 4. WhatsApp message_fetch_limit is configurable
# ---------------------------------------------------------------------------


class TestWhatsAppMessageFetchLimit:
    """message_fetch_limit must be read from config and passed to the API call."""

    def _make_channel(self, config_overrides: dict | None = None):
        from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel

        base_config = {
            "waha_url": "http://localhost:3000",
            "session": "default",
            "filter_mode": "none",
            "contacts": [],
        }
        if config_overrides:
            base_config.update(config_overrides)

        with patch("cogtrix_core.tools._whatsapp_client.WahaClient.__init__", return_value=None):
            ch = WhatsAppChannel.__new__(WhatsAppChannel)
            ch._config = base_config
            ch._client = MagicMock()
            ch._filter_mode = "none"
            ch._contacts = []
            ch._phonebook = {}
            ch._chat_watermarks = {}
            ch._watermark_timestamps = {}
            ch._overview_snapshot = {}
            ch._snapshot_timestamps = {}
            ch._lid_cache = collections.OrderedDict()
            ch._lid_cache_lock = threading.Lock()
            ch._LID_CACHE_MAX = 1024
            ch._LID_NEGATIVE_TTL = 300.0
            ch._SNAPSHOT_TTL = 3600.0
            ch._WATERMARK_TTL = 604800.0
            ch._seen_ids = {}
            ch._SEEN_TTL = 600.0
            ch._overview_limit = 50
            ch._ignore_archived = True
            ch._ignore_older_than = None
            ch._locally_archived = set()
            ch._chat_errors = {}
            ch._FETCH_ERROR_BASE = 30.0
            ch._FETCH_ERROR_MAX = 300.0
            ch._message_fetch_limit = int(base_config.get("message_fetch_limit", 50))
            ch._session_check_interval = 60.0
            ch._last_session_check = 0.0
        return ch

    def test_default_limit_is_50(self):
        ch = self._make_channel()
        assert ch._message_fetch_limit == 50

    def test_custom_limit_from_config(self):
        ch = self._make_channel({"message_fetch_limit": 100})
        assert ch._message_fetch_limit == 100

    def test_fetch_passes_limit_to_client(self):
        from cogtrix_core.assistant.channels.whatsapp import ChatOverview

        ch = self._make_channel({"message_fetch_limit": 75})
        ch._client.get_chat_messages.return_value = []

        overview = MagicMock(spec=ChatOverview)
        overview.id = "chat@c.us"
        ch._chat_watermarks = {"chat@c.us": 1000}

        ch._fetch_new_messages(overview)

        ch._client.get_chat_messages.assert_called_once()
        call_kwargs = ch._client.get_chat_messages.call_args
        assert call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit") == 75

    def test_limit_read_from_real_init(self):
        """WhatsAppChannel.__init__ reads message_fetch_limit from config."""
        from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel

        config = {
            "waha_url": "http://localhost:3000",
            "session": "default",
            "message_fetch_limit": 200,
        }
        with patch("cogtrix_core.tools._whatsapp_client.WahaClient.__init__", return_value=None):
            ch = WhatsAppChannel(config)
        assert ch._message_fetch_limit == 200


# ---------------------------------------------------------------------------
# 5. reload_config must use load_config() not Config()
# ---------------------------------------------------------------------------


class TestReloadConfigUsesLoadConfig:
    """reload_config endpoint must produce a fully-resolved Config, not defaults."""

    @pytest.fixture(autouse=True)
    def _set_jwt_secret(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "a" * 64)
        from cogtrix_core.api.auth import configure_jwt_secret

        configure_jwt_secret("a" * 64)

    def test_reload_preserves_models_registry(self, tmp_path):
        """After reload, app.state.config.models must contain parsed aliases."""
        from starlette.testclient import TestClient

        from cogtrix_core.api.app import app

        # Create a minimal config file for load_config to find
        config_file = tmp_path / ".cogtrix.yml"
        config_file.write_text(
            "provider: openai\n"
            "model: gpt-4o\n"
            "providers:\n"
            "  openai:\n"
            "    type: openai\n"
            "    model: gpt-4o\n"
            "    api_key: sk-test\n"
            "  spark:\n"
            "    type: openai\n"
            "    model: gpt-oss\n"
            "    base_url: http://localhost:8080/v1\n"
            "    api_key: sk-test2\n"
            "models:\n"
            "  oss:\n"
            "    provider: spark\n"
            "    model: gpt-oss\n"
            "    temperature: 0.5\n"
        )

        with TestClient(app) as c:
            # Point load_config at our temp config file via env var
            with patch.dict(os.environ, {"COGTRIX_CONFIG_FILE": str(config_file)}):
                resp = c.post(
                    "/api/v1/config/reload",
                    headers=self._admin_headers(),
                )

            assert resp.status_code == 200
            cfg = c.app.state.config
            assert cfg is not None
            # Models registry must be populated — not an empty default Config()
            assert "oss" in cfg.models
            assert cfg.models["oss"].model == "gpt-oss"
            assert cfg.models["oss"].provider == "spark"

    def _admin_headers(self) -> dict[str, str]:
        from cogtrix_core.api.auth import create_access_token

        token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 6. _build_llm must not mutate shared ProviderConfig
# ---------------------------------------------------------------------------


class TestBuildLlmNoSharedMutation:
    """_build_llm must not mutate the shared ProviderConfig in the registry."""

    def test_build_llm_does_not_mutate_provider_config(self):
        """Switching session to a different model must not corrupt the provider registry."""
        from cogtrix_core.config import Config, ModelConfig, ProviderConfig

        cfg = Config(
            providers={
                "openai": ProviderConfig(name="openai", type="openai", api_key="sk-test"),
                "spark": ProviderConfig(
                    name="spark",
                    type="openai",
                    base_url="http://localhost:8080/v1",
                    api_key="sk-spark",
                ),
            },
            models={
                "default": ModelConfig(provider="openai", model="gpt-4o"),
                "oss": ModelConfig(provider="spark", model="gpt-oss", temperature=0.5),
            },
            active_model_alias="default",
        )

        original_spark_key = cfg.providers["spark"].api_key

        class FakeState:
            config = cfg

        with patch("cogtrix_core.providers.create_chat_model_from_configs") as mock_create:
            mock_create.return_value = MagicMock()
            from cogtrix_core.api.session_bridge import _build_llm

            _build_llm({"model": "oss"}, FakeState())

        assert cfg.providers["spark"].api_key == original_spark_key


# ── deep_think context pollution guard (#250) ──────────────────────────────


class TestDeepThinkContextPollutionGuard:
    """Session-history pollution detector strips cross-domain context dumps
    from deep_think calls, regardless of task domain (regression for #250)."""

    def test_clean_research_context_passes_through(self):
        """Focused web research with few unique artifacts is not stripped."""
        from cogtrix_core.tools.deep_think import configure_deep_think, deep_think

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="analysis result")
        configure_deep_think({"providers": {"mock": {}}})

        clean_context = (
            "MIITE 2026 Startup Program: Applications open Jan 15 – Mar 30. "
            "Requirements: registered UAE company, < 5 years old, pitch deck. "
            "Source: https://miite.ae/startups/apply"
        )
        result = deep_think(
            task="What are the requirements to participate as a startup in MIITE 2026?",
            context=clean_context,
            max_iterations=1,
            num_branches=2,
            llm=mock_llm,
        )
        assert result != ""

    def test_session_history_dump_stripped_for_any_task_domain(self):
        """Session history with many unique artifacts is stripped regardless of task domain."""
        from unittest.mock import patch

        from cogtrix_core.tools.deep_think import configure_deep_think, deep_think

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="analysis result")
        configure_deep_think({"providers": {"mock": {}}})

        # Many UNIQUE SHAs, PR numbers, Slack IDs — hallmark of a session history dump
        session_dump = (
            "PR #246 merged c278d37. PR #247 merged 9f14c80. "
            "PR #248 merged dceb2b2. PR #249 merged b1ef61e. "
            "PR #250 merged d5781c1. PR #251 merged ecb7a66. "
            "SHA abc1234 def5678 9ab0cd1 2ef3456 78abc90. "
            "U08K4SB05PU approved. U0B0JE40U0Z confirmed. U0B04M1SLSX merged. "
            "U0AV0EDDN3H reviewed. U0B0FG2DE8H validated. "
        )

        for task in [
            "What are the startup participation requirements for MIITE 2026?",
            "Debug the CI failure on the cross-org isolation branch",
            "Analyse Q1 procurement spend by supplier category",
        ]:
            with patch("cogtrix_core.tools.deep_think.log") as mock_log:
                deep_think(
                    task=task,
                    context=session_dump,
                    max_iterations=1,
                    num_branches=2,
                    llm=mock_llm,
                )
                warning_calls = [str(c) for c in mock_log.warning.call_args_list]
                assert any(
                    "pollution" in w.lower() for w in warning_calls
                ), f"Expected pollution warning for task: {task!r}"

    def test_small_number_of_unique_artifacts_not_stripped(self):
        """Focused context with only a few unique references is not stripped."""
        from unittest.mock import patch

        from cogtrix_core.tools.deep_think import configure_deep_think, deep_think

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="analysis result")
        configure_deep_think({"providers": {"mock": {}}})

        # Only 2 unique SHAs, 1 unique PR — well below the threshold
        focused_context = (
            "PR #246 introduced the regression. "
            "Commit abc1234 is the likely culprit; diff shows it removes the org scope. "
            "Revert candidate: def5678. PR #246 also affects the JIT path."
        )

        with patch("cogtrix_core.tools.deep_think.log") as mock_log:
            deep_think(
                task="Find the root cause of the cross-org auth regression",
                context=focused_context,
                max_iterations=1,
                num_branches=2,
                llm=mock_llm,
            )
            warning_calls = [str(c) for c in mock_log.warning.call_args_list]
            pollution_warnings = [w for w in warning_calls if "pollution" in w.lower()]
            assert (
                len(pollution_warnings) == 0
            ), "Focused context with few unique artifacts must not be stripped"
