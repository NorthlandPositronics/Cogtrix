"""Tests for AssistantService lifecycle, shutdown, and deferral coverage.

Covers bugs #904 (shutdown hang), #907 (deferral callback), #908 (executor leak).
"""

from __future__ import annotations

import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.assistant.service import AssistantService

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, **assistant_overrides: Any) -> MagicMock:
    """Return a minimal Config mock with assistant services."""
    cfg = MagicMock()
    cfg.data_dir = str(tmp_path / "data")
    cfg.services = {
        "assistant": {
            "max_concurrent": 2,
            "idle_timeout": 60.0,
            "dispatch_interval": 5.0,
            "debounce_seconds": 0.5,
            "max_sessions": 5,
            "channels": {},
            "guardrails": {},
            **assistant_overrides,
        },
    }
    cfg.parallel_tool_execution = True
    return cfg


def _make_channel(name: str = "telegram") -> MagicMock:
    ch = MagicMock()
    ch.name = name
    return ch


# ── TestBuildSystemPrompt (retained from original) ───────────────────────────


class TestBuildSystemPrompt:
    """Test the system prompt priority chain."""

    def test_cli_prompt_highest_priority(self):

        asst_cfg = {"system_prompt": "config prompt"}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", "cli prompt")
        assert result == "cli prompt"

    def test_inline_config_over_file_and_default(self, tmp_path):

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        asst_cfg = {
            "system_prompt": "inline prompt",
            "system_prompt_file": str(prompt_file),
        }
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == "inline prompt"

    def test_file_config_over_default(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt content")
        asst_cfg = {"system_prompt_file": str(prompt_file)}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", data_dir=str(tmp_path))
        assert result == "file prompt content"

    def test_file_config_missing_file_falls_to_default(self):
        from cogtrix_core.assistant.service import _ASSISTANT_SYSTEM_PROMPT

        asst_cfg = {"system_prompt_file": "/nonexistent/path/prompt.txt"}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_file_config_empty_file_falls_to_default(self, tmp_path):
        from cogtrix_core.assistant.service import _ASSISTANT_SYSTEM_PROMPT

        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   ")
        asst_cfg = {"system_prompt_file": str(prompt_file)}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", data_dir=str(tmp_path))
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_default_prompt_when_nothing_configured(self):
        from cogtrix_core.assistant.service import _ASSISTANT_SYSTEM_PROMPT

        result = AssistantService._build_system_prompt({}, "fallback")
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_default_prompt_mentions_slack_status_dedup(self):
        from cogtrix_core.assistant.service import _ASSISTANT_SYSTEM_PROMPT

        assert "Before posting a recurring status update to Slack" in _ASSISTANT_SYSTEM_PROMPT

    def test_cli_prompt_overrides_everything(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        asst_cfg = {
            "system_prompt": "inline prompt",
            "system_prompt_file": str(prompt_file),
        }
        result = AssistantService._build_system_prompt(
            asst_cfg, "fallback", "cli wins", data_dir=str(tmp_path)
        )
        assert result == "cli wins"

    def test_data_dir_none_enforces_cwd_containment(self, tmp_path):
        from cogtrix_core.assistant.service import _ASSISTANT_SYSTEM_PROMPT

        # Create a prompt file that points outside cwd when data_dir=None
        outside_file = tmp_path / "outside" / "prompt.txt"
        outside_file.parent.mkdir()
        outside_file.write_text("outside content")

        asst_cfg = {"system_prompt_file": str(outside_file)}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", data_dir=None)
        assert result == _ASSISTANT_SYSTEM_PROMPT


# ── TestLifecycle ────────────────────────────────────────────────────────────


class TestLifecycle:
    """Construction, run, and shutdown paths."""

    @pytest.fixture()
    def service(self, tmp_path: Path):
        """Yield a fully-constructed AssistantService with mocked subsystems."""
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager") as MockSessionMgr,
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
        ):
            MockSessionMgr.return_value.save_all.return_value = None
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test prompt",
                available_tools={},
                active_tools=[],
            )
            yield svc
            # Safety: ensure executor is shut down after each test
            if hasattr(svc, "_executor") and svc._executor is not None:
                svc._executor.shutdown(wait=False)

    def test_init_creates_executor(self, service: AssistantService):
        assert isinstance(service._executor, ThreadPoolExecutor)
        assert service._executor._max_workers == 2  # noqa: SLF001

    def test_init_sets_stop_event(self, service: AssistantService):
        assert isinstance(service._stop_event, threading.Event)
        assert not service._stop_event.is_set()

    def test_init_sets_shutting_down_false(self, service: AssistantService):
        assert service._shutting_down is False

    def test_run_with_no_channels_logs_error_and_returns(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with patch.object(AssistantService, "_discover_channels", return_value=[]):
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            # run() should return early when no channels are found
            with patch("cogtrix_core.assistant.service.log") as mock_log:
                svc.run()
                mock_log.error.assert_called_once()
                assert "No messaging channels are ready" in mock_log.error.call_args[0][0]
        svc._executor.shutdown(wait=False)

    def test_handle_shutdown_sets_shutting_down_flag(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        assert service._shutting_down is True

    def test_handle_shutdown_idempotent(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        # Second call should return early without raising
        service._handle_shutdown(signal.SIGINT, None)
        assert service._shutting_down is True

    def test_handle_shutdown_stops_subsystems(self, service: AssistantService):
        service._handle_shutdown(signal.SIGTERM, None)
        service._poller.stop.assert_called_once()  # type: ignore[attr-defined]
        service._scheduler.stop.assert_called_once()  # type: ignore[attr-defined]

    def test_handle_shutdown_saves_scheduler(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        service._scheduler.save.assert_called_once()  # type: ignore[attr-defined]

    def test_handle_shutdown_shuts_down_executor(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        # After shutdown, submitting new work should raise RuntimeError
        with pytest.raises(RuntimeError):
            service._executor.submit(lambda: None)

    def test_handle_shutdown_sets_stop_event(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_handle_shutdown_saves_sessions(self, service: AssistantService):
        service._handle_shutdown(signal.SIGINT, None)
        service._session_mgr.save_all.assert_called_once()  # type: ignore[attr-defined]

    def test_handle_shutdown_saves_knowledge_store_when_present(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
            patch("cogtrix_core.assistant.service.SharedKnowledgeStore") as MockKS,
        ):
            ks_instance = MockKS.return_value
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            svc._handle_shutdown(signal.SIGINT, None)
            ks_instance.save.assert_called_once()
            ks_instance.flush.assert_called_once()
        svc._executor.shutdown(wait=False)

    def test_handle_shutdown_skips_knowledge_store_when_none(self, service: AssistantService):
        service._knowledge_store = None
        service._handle_shutdown(signal.SIGINT, None)
        # Should complete without error even when knowledge_store is None
        assert service._stop_event.is_set()

    def test_handle_shutdown_saves_deferral_manager(self, service: AssistantService):
        service._deferral_mgr = MagicMock()
        service._handle_shutdown(signal.SIGINT, None)
        service._deferral_mgr.stop.assert_called_once()
        service._deferral_mgr.save.assert_called_once()

    def test_handle_shutdown_saves_campaign_manager(self, service: AssistantService):
        service._campaign_mgr = MagicMock()
        service._handle_shutdown(signal.SIGINT, None)
        service._campaign_mgr.stop.assert_called_once()
        service._campaign_mgr.save.assert_called_once()

    def test_run_installs_signal_handlers(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
        ):
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            with patch.object(svc, "_stop_event") as mock_event:
                mock_event.wait.side_effect = [None]
                with patch("signal.signal") as mock_signal:
                    svc.run()
                    assert mock_signal.call_count == 2
                    args = [call.args for call in mock_signal.call_args_list]
                    assert (signal.SIGINT, svc._handle_shutdown) in args
                    assert (signal.SIGTERM, svc._handle_shutdown) in args
        svc._executor.shutdown(wait=False)


# ── TestShutdownResilience (#904) ────────────────────────────────────────────


class TestShutdownResilience:
    """Shutdown must complete and set stop_event even when subsystems fail.

    NOTE: These tests verify the CURRENT behaviour on the ``next`` branch.
    PR #938 adds try/except wrappers so that failures are logged instead of
    propagated.  Until that PR lands, subsystem failures during shutdown will
    abort the sequence and *not* set the stop_event.
    """

    @pytest.fixture()
    def service(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager") as MockSessionMgr,
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
        ):
            MockSessionMgr.return_value.save_all.return_value = None
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            yield svc
            svc._executor.shutdown(wait=False)

    def test_poller_stop_failure_still_sets_stop_event(self, service: AssistantService):
        """FIX #904: poller.stop() failure propagates but finally ensures stop_event is set."""
        service._poller.stop.side_effect = RuntimeError("poller boom")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="poller boom"):
            service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()  # finally clause always fires

    def test_scheduler_stop_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: scheduler.stop() failure is caught; shutdown completes."""
        service._scheduler.stop.side_effect = RuntimeError("scheduler boom")  # type: ignore[attr-defined]
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_scheduler_save_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: scheduler.save() failure is caught; shutdown completes."""
        service._scheduler.save.side_effect = RuntimeError("save boom")  # type: ignore[attr-defined]
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_session_mgr_save_all_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: session_mgr.save_all() failure is caught; shutdown completes."""
        service._session_mgr.save_all.side_effect = RuntimeError("session boom")  # type: ignore[attr-defined]
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_deferral_mgr_stop_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: deferral_mgr.stop() failure is caught; shutdown completes."""
        service._deferral_mgr = MagicMock()
        service._deferral_mgr.stop.side_effect = RuntimeError("deferral stop boom")
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_campaign_mgr_stop_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: campaign_mgr.stop() failure is caught; shutdown completes."""
        service._campaign_mgr = MagicMock()
        service._campaign_mgr.stop.side_effect = RuntimeError("campaign stop boom")
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_knowledge_store_save_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: knowledge_store.save() failure is caught; shutdown completes."""
        service._knowledge_store = MagicMock()
        service._knowledge_store.save.side_effect = RuntimeError("ks save boom")
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_knowledge_store_flush_failure_does_not_block_shutdown(self, service: AssistantService):
        """FIX #904: knowledge_store.flush() failure is caught; shutdown completes."""
        service._knowledge_store = MagicMock()
        service._knowledge_store.flush.side_effect = RuntimeError("ks flush boom")
        service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()

    def test_multiple_subsystem_failures_still_complete_shutdown(self, service: AssistantService):
        """FIX #904: multiple subsystem failures are caught; stop_event is always set."""
        service._poller.stop.side_effect = RuntimeError("poller")  # type: ignore[attr-defined]
        service._scheduler.stop.side_effect = RuntimeError("scheduler")  # type: ignore[attr-defined]
        service._session_mgr.save_all.side_effect = RuntimeError("session")  # type: ignore[attr-defined]
        # Poller error still propagates; others are caught. stop_event always set via finally.
        with pytest.raises(RuntimeError, match="poller"):
            service._handle_shutdown(signal.SIGINT, None)
        assert service._stop_event.is_set()


# ── TestInitFailureCleanup (#908) ────────────────────────────────────────────


class TestInitFailureCleanup:
    """If __init__ fails after creating the executor, it must be shut down."""

    def test_executor_shutdown_when_discover_channels_raises(self, tmp_path: Path):
        """FIX #908: executor is shut down when discovery fails."""
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", side_effect=RuntimeError("discovery failed")
            ),
            patch("cogtrix_core.assistant.service.ThreadPoolExecutor") as MockTPE,
        ):
            mock_executor = MagicMock()
            MockTPE.return_value = mock_executor
            with pytest.raises(RuntimeError, match="discovery failed"):
                AssistantService(
                    config=cfg,
                    llm=MagicMock(),
                    registry=MagicMock(),
                    system_prompt="test",
                    available_tools={},
                    active_tools=[],
                )
            mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_executor_shutdown_when_scheduler_init_raises(self, tmp_path: Path):
        """FIX #908: executor is shut down when MessageScheduler init fails."""
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.MessageScheduler") as MockScheduler,
            patch("cogtrix_core.assistant.service.ThreadPoolExecutor") as MockTPE,
        ):
            mock_executor = MagicMock()
            MockTPE.return_value = mock_executor
            MockScheduler.side_effect = RuntimeError("scheduler init failed")
            with pytest.raises(RuntimeError, match="scheduler init failed"):
                AssistantService(
                    config=cfg,
                    llm=MagicMock(),
                    registry=MagicMock(),
                    system_prompt="test",
                    available_tools={},
                    active_tools=[],
                )
            mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_executor_shutdown_when_handler_init_raises(self, tmp_path: Path):
        """FIX #908: executor is shut down when MessageHandler init fails."""
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.MessageHandler") as MockHandler,
            patch("cogtrix_core.assistant.service.ThreadPoolExecutor") as MockTPE,
        ):
            mock_executor = MagicMock()
            MockTPE.return_value = mock_executor
            MockHandler.side_effect = RuntimeError("handler init failed")
            with pytest.raises(RuntimeError, match="handler init failed"):
                AssistantService(
                    config=cfg,
                    llm=MagicMock(),
                    registry=MagicMock(),
                    system_prompt="test",
                    available_tools={},
                    active_tools=[],
                )
            mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_executor_shutdown_on_value_error(self, tmp_path: Path):
        """FIX #908: executor is shut down on ValueError as well as RuntimeError."""
        cfg = _make_config(tmp_path)
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.MessageHandler") as MockHandler,
            patch("cogtrix_core.assistant.service.ThreadPoolExecutor") as MockTPE,
        ):
            mock_executor = MagicMock()
            MockTPE.return_value = mock_executor
            MockHandler.side_effect = ValueError("bad config")
            with pytest.raises(ValueError, match="bad config"):
                AssistantService(
                    config=cfg,
                    llm=MagicMock(),
                    registry=MagicMock(),
                    system_prompt="test",
                    available_tools={},
                    active_tools=[],
                )
            mock_executor.shutdown.assert_called_once_with(wait=False)


# ── TestDeferralWiring (#907) ────────────────────────────────────────────────


class TestDeferralWiring:
    """Deferral reprocess callback is wired and submits to executor."""

    @pytest.fixture()
    def service(self, tmp_path: Path):
        cfg = _make_config(tmp_path, deferral={"enabled": True})
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler") as MockHandler,
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
            patch("cogtrix_core.assistant.service.DeferralManager") as MockDeferral,
        ):
            handler_instance = MockHandler.return_value
            _ = MockDeferral.return_value
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            yield svc, handler_instance
            svc._executor.shutdown(wait=False)

    def test_deferral_manager_created_when_enabled(self, service):
        svc, _handler = service
        assert svc._deferral_mgr is not None

    def test_reprocess_callback_wired_to_deferral_manager(self, service):
        svc, _handler = service
        assert svc._deferral_mgr.set_reprocess_callback.called

    def test_reprocess_callback_submits_to_executor(self, service):
        svc, handler_instance = service
        # Extract the callback that was wired
        callback = svc._deferral_mgr.set_reprocess_callback.call_args[0][0]
        # Use an Event so we don't rely on scheduling timing.
        called = threading.Event()
        original = handler_instance.handle_batch.side_effect
        handler_instance.handle_batch.side_effect = lambda *a, **kw: called.set()
        callback(["msg"], _make_channel(), 0, "test::key")
        assert called.wait(timeout=5), "handle_batch was not called within 5 seconds"
        handler_instance.handle_batch.assert_called_once()
        handler_instance.handle_batch.side_effect = original

    def test_reprocess_callback_returns_none(self, service):
        """BUG #907: callback currently returns None (PR #974 changes to bool)."""
        svc, _handler = service
        callback = svc._deferral_mgr.set_reprocess_callback.call_args[0][0]
        result = callback(["msg"], _make_channel(), 0, "test::key")
        assert result is None

    def test_reprocess_callback_passes_is_reprocessing_flag(self, service):
        svc, handler_instance = service
        callback = svc._deferral_mgr.set_reprocess_callback.call_args[0][0]
        called = threading.Event()
        original = handler_instance.handle_batch.side_effect
        handler_instance.handle_batch.side_effect = lambda *a, **kw: called.set()
        callback(["msg"], _make_channel(), 1, "test::key")
        assert called.wait(timeout=5), "handle_batch was not called within 5 seconds"
        _, kwargs = handler_instance.handle_batch.call_args
        assert kwargs.get("is_reprocessing") is True
        assert kwargs.get("deferral_depth") == 2
        handler_instance.handle_batch.side_effect = original

    def test_no_deferral_manager_when_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path, deferral={"enabled": False})
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
        ):
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            assert svc._deferral_mgr is None
        svc._executor.shutdown(wait=False)


# ── TestCampaignWiring ───────────────────────────────────────────────────────


class TestCampaignWiring:
    """Campaign manager dependencies are wired after handler construction."""

    def test_campaign_mgr_set_handler_and_channels(self, tmp_path: Path):
        cfg = _make_config(tmp_path, campaigns={"enabled": True})
        ch = _make_channel("telegram")
        with (
            patch.object(AssistantService, "_discover_channels", return_value=[ch]),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
            patch("cogtrix_core.assistant.service.CampaignManager") as MockCampaign,
        ):
            campaign_instance = MockCampaign.return_value
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            campaign_instance.set_handler.assert_called_once()
            campaign_instance.set_channels.assert_called_once()
            # set_channels should receive a dict with the channel as positional arg
            args, _kwargs = campaign_instance.set_channels.call_args
            channels_dict = args[0] if args else _kwargs.get("channels", {})
            assert "telegram" in channels_dict
        svc._executor.shutdown(wait=False)

    def test_no_campaign_mgr_when_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path, campaigns={"enabled": False})
        with (
            patch.object(
                AssistantService, "_discover_channels", return_value=[_make_channel("telegram")]
            ),
            patch("cogtrix_core.assistant.service.ChatSessionManager"),
            patch("cogtrix_core.assistant.service.MessageScheduler"),
            patch("cogtrix_core.assistant.service.ChannelPoller"),
            patch("cogtrix_core.assistant.service.MessageHandler"),
            patch("cogtrix_core.assistant.service.GuardrailPipeline"),
            patch("cogtrix_core.assistant.workflows.WorkflowRegistry"),
        ):
            svc = AssistantService(
                config=cfg,
                llm=MagicMock(),
                registry=MagicMock(),
                system_prompt="test",
                available_tools={},
                active_tools=[],
            )
            assert svc._campaign_mgr is None
        svc._executor.shutdown(wait=False)
