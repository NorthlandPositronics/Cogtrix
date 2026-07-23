"""Tests for src.api.assistant_lifecycle.

Regression coverage for issue #1216 — zero test coverage for assistant startup
and shutdown helpers used by the FastAPI lifespan and ``POST /assistant/start``.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.api.assistant_lifecycle import create_and_start_assistant, shutdown_assistant_sync

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.resolve_llm_config.return_value = (MagicMock(), MagicMock())
    return cfg


def _make_tool_registry() -> MagicMock:
    reg = MagicMock()
    reg.tools = {"tool_a": MagicMock(), "tool_b": MagicMock()}
    return reg


def _make_service() -> MagicMock:
    """Return a mock AssistantService with all subsystems present."""
    svc = MagicMock()
    svc._poller = MagicMock()
    svc._scheduler = MagicMock()
    svc._deferral_mgr = MagicMock()
    svc._executor = MagicMock()
    svc._session_mgr = MagicMock()
    svc._knowledge_store = MagicMock()
    return svc


# ── create_and_start_assistant ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestCreateAndStartAssistant:
    """Happy paths and failure modes for ``create_and_start_assistant``."""

    async def test_starts_poler_and_scheduler(self):
        cfg = _make_config()
        reg = _make_tool_registry()
        mock_llm = MagicMock()
        mock_svc = _make_service()

        with (
            patch(
                "src.providers.create_chat_model_from_configs", return_value=mock_llm
            ) as mock_create_llm,
            patch("src.assistant.service.AssistantService", return_value=mock_svc) as MockSvc,
        ):
            result = await create_and_start_assistant(cfg, reg)

        assert result is mock_svc
        cfg.resolve_llm_config.assert_called_once()
        mock_create_llm.assert_called_once()
        pc, mc = cfg.resolve_llm_config.return_value
        mock_create_llm.assert_called_with(pc, mc)
        MockSvc.assert_called_once_with(
            config=cfg,
            llm=mock_llm,
            registry=reg,
            system_prompt="",
            available_tools=dict(reg.tools),
            active_tools=[],
        )
        mock_svc._poller.start.assert_called_once()
        mock_svc._scheduler.start.assert_called_once()
        assert isinstance(mock_svc._started_at, datetime)

    async def test_starts_deferral_manager_when_present(self):
        cfg = _make_config()
        reg = _make_tool_registry()
        mock_svc = _make_service()
        mock_svc._deferral_mgr = MagicMock()

        with (
            patch("src.providers.create_chat_model_from_configs", return_value=MagicMock()),
            patch("src.assistant.service.AssistantService", return_value=mock_svc),
        ):
            await create_and_start_assistant(cfg, reg)

        mock_svc._deferral_mgr.start.assert_called_once()

    async def test_skips_deferral_start_when_none(self):
        cfg = _make_config()
        reg = _make_tool_registry()
        mock_svc = _make_service()
        mock_svc._deferral_mgr = None

        with (
            patch("src.providers.create_chat_model_from_configs", return_value=MagicMock()),
            patch("src.assistant.service.AssistantService", return_value=mock_svc),
        ):
            await create_and_start_assistant(cfg, reg)

        # No AttributeError raised — test passes implicitly

    async def test_llm_creation_failure_propagates_runtime_error(self):
        cfg = _make_config()
        reg = _make_tool_registry()

        with (
            patch(
                "src.providers.create_chat_model_from_configs",
                side_effect=RuntimeError("ollama unreachable"),
            ),
            patch("src.assistant.service.AssistantService") as MockSvc,
        ):
            with pytest.raises(RuntimeError, match="ollama unreachable"):
                await create_and_start_assistant(cfg, reg)

        MockSvc.assert_not_called()

    async def test_assistant_service_init_failure_propagates(self):
        cfg = _make_config()
        reg = _make_tool_registry()

        with (
            patch("src.providers.create_chat_model_from_configs", return_value=MagicMock()),
            patch(
                "src.assistant.service.AssistantService",
                side_effect=ValueError("bad config"),
            ),
        ):
            with pytest.raises(ValueError, match="bad config"):
                await create_and_start_assistant(cfg, reg)


# ── shutdown_assistant_sync ──────────────────────────────────────────────────


class TestShutdownAssistantSync:
    """Graceful stop sequence and resilience for ``shutdown_assistant_sync``."""

    def test_stops_all_subsystems(self):
        svc = _make_service()
        shutdown_assistant_sync(svc)

        svc._poller.stop.assert_called_once()
        svc._scheduler.stop.assert_called_once()
        svc._scheduler.save.assert_called_once()
        svc._deferral_mgr.stop.assert_called_once()
        svc._deferral_mgr.save.assert_called_once()
        svc._executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)
        svc._session_mgr.save_all.assert_called_once()
        svc._knowledge_store.save.assert_called_once()

    def test_skips_missing_subsystems(self):
        svc = MagicMock()
        # None of the subsystem attributes are present
        shutdown_assistant_sync(svc)
        # Completes without error — test passes implicitly

    def test_poller_stop_failure_does_not_block_others(self):
        svc = _make_service()
        svc._poller.stop.side_effect = RuntimeError("poller boom")

        shutdown_assistant_sync(svc)

        svc._scheduler.stop.assert_called_once()
        svc._scheduler.save.assert_called_once()
        svc._executor.shutdown.assert_called_once()
        svc._session_mgr.save_all.assert_called_once()

    def test_scheduler_stop_failure_does_not_block_others(self):
        svc = _make_service()
        svc._scheduler.stop.side_effect = RuntimeError("scheduler boom")

        shutdown_assistant_sync(svc)

        svc._poller.stop.assert_called_once()
        svc._deferral_mgr.stop.assert_called_once()
        svc._executor.shutdown.assert_called_once()

    def test_deferral_mgr_stop_failure_does_not_block_others(self):
        svc = _make_service()
        svc._deferral_mgr.stop.side_effect = RuntimeError("deferral boom")

        shutdown_assistant_sync(svc)

        svc._poller.stop.assert_called_once()
        svc._scheduler.stop.assert_called_once()
        svc._executor.shutdown.assert_called_once()
        svc._session_mgr.save_all.assert_called_once()

    def test_executor_shutdown_failure_does_not_block_others(self):
        svc = _make_service()
        svc._executor.shutdown.side_effect = RuntimeError("executor boom")

        shutdown_assistant_sync(svc)

        svc._poller.stop.assert_called_once()
        svc._session_mgr.save_all.assert_called_once()

    def test_session_mgr_save_failure_does_not_block_knowledge_store(self):
        svc = _make_service()
        svc._session_mgr.save_all.side_effect = RuntimeError("session boom")

        shutdown_assistant_sync(svc)

        svc._knowledge_store.save.assert_called_once()

    def test_knowledge_store_save_failure_completes_gracefully(self):
        svc = _make_service()
        svc._knowledge_store.save.side_effect = RuntimeError("ks boom")

        shutdown_assistant_sync(svc)

        svc._poller.stop.assert_called_once()
        svc._session_mgr.save_all.assert_called_once()

    def test_multiple_failures_still_complete(self):
        svc = _make_service()
        svc._poller.stop.side_effect = RuntimeError("poller")
        svc._scheduler.stop.side_effect = RuntimeError("scheduler")
        svc._deferral_mgr.stop.side_effect = RuntimeError("deferral")
        svc._executor.shutdown.side_effect = RuntimeError("executor")
        svc._session_mgr.save_all.side_effect = RuntimeError("session")
        svc._knowledge_store.save.side_effect = RuntimeError("ks")

        shutdown_assistant_sync(svc)
        # All exceptions logged and swallowed; no propagation

    def test_logs_warnings_on_failures(self, caplog):
        svc = _make_service()
        svc._poller.stop.side_effect = RuntimeError("poller boom")

        with caplog.at_level("WARNING", logger="cogtrix.api"):
            shutdown_assistant_sync(svc)

        assert "Error stopping poller" in caplog.text
        assert "poller boom" in caplog.text
