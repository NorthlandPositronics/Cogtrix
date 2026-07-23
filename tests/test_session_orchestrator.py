"""Tests for SessionOrchestrator — snapshot and rollback correctness."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.orchestration.session_orchestrator import SessionOrchestrator, SessionSnapshot


def _make_orchestrator(
    active_model_alias: str = "gpt-4.1-mini",
    memory_mode: str = "conversation",
    session: str = "default",
) -> tuple[SessionOrchestrator, MagicMock, MagicMock]:
    config = MagicMock()
    config.active_model_alias = active_model_alias
    config.memory_mode = memory_mode
    config.memory_config = {}
    config.session = session

    slash_cmds = MagicMock()
    slash_cmds.system_prompt = "system"
    slash_cmds.memory_manager = MagicMock()
    slash_cmds.available_tools = {}

    orch = SessionOrchestrator(config, slash_cmds)
    return orch, config, slash_cmds


class TestSessionOrchestratorSnapshot:
    def test_snapshot_captures_config_fields(self) -> None:
        orch, config, _ = _make_orchestrator(
            active_model_alias="gpt-4.1-mini", memory_mode="conversation"
        )
        snap = orch.snapshot()
        assert snap.active_model_alias == "gpt-4.1-mini"
        assert snap.memory_mode == "conversation"

    def test_snapshot_captures_runtime_objects(self) -> None:
        orch, _, _ = _make_orchestrator()
        mm = MagicMock()
        snap = orch.snapshot(memory_manager=mm, system_prompt="custom prompt")
        assert snap.memory_manager is mm
        assert snap.system_prompt == "custom prompt"

    def test_snapshot_copies_tools(self) -> None:
        orch, _, _ = _make_orchestrator()
        tools_in = ["tool_a", "tool_b"]
        snap = orch.snapshot(tools=tools_in)
        assert snap.tools == ["tool_a", "tool_b"]
        tools_in.append("tool_c")
        assert snap.tools == ["tool_a", "tool_b"]

    def test_snapshot_copies_available_tools(self) -> None:
        orch, _, _ = _make_orchestrator()
        at = {"search": MagicMock()}
        snap = orch.snapshot(available_tools=at)
        assert "search" in snap.available_tools
        at["new"] = MagicMock()
        assert "new" not in snap.available_tools


class TestSessionOrchestratorRollback:
    def test_rollback_restores_config_fields(self) -> None:
        orch, config, slash_cmds = _make_orchestrator(active_model_alias="gpt-4.1-mini")
        snap = orch.snapshot()

        config.active_model_alias = "claude-sonnet"

        orch.rollback(snap)

        assert config.active_model_alias == "gpt-4.1-mini"

    def test_rollback_empty_tools_clears_live_list(self) -> None:
        """Rollback with snap.tools=[] correctly clears the live tools list (BUG-1845)."""
        orch, _, _ = _make_orchestrator()
        snap = SessionSnapshot(tools=[])
        live_tools: list = ["tool_a", "tool_b"]

        orch.rollback(snap, tools_list=live_tools)

        assert live_tools == []

    def test_rollback_non_empty_tools_replaces_live_list(self) -> None:
        orch, _, _ = _make_orchestrator()
        snap = SessionSnapshot(tools=["tool_x"])
        live_tools: list = ["tool_a", "tool_b"]

        orch.rollback(snap, tools_list=live_tools)

        assert live_tools == ["tool_x"]

    def test_rollback_without_tools_list_leaves_it_untouched(self) -> None:
        orch, _, _ = _make_orchestrator()
        snap = SessionSnapshot(tools=["tool_x"])
        live_tools: list = ["tool_a"]

        orch.rollback(snap)

        assert live_tools == ["tool_a"]

    def test_rollback_returns_local_var_dict(self) -> None:
        orch, _, _ = _make_orchestrator()
        mm = MagicMock()
        snap = orch.snapshot(memory_manager=mm, system_prompt="original", available_tools={})

        result = orch.rollback(snap)

        assert "memory_manager" in result
        assert "system_prompt" in result
        assert "available_tools" in result

    def test_rollback_updates_slash_cmds_available_tools(self) -> None:
        orch, _, slash_cmds = _make_orchestrator()
        tools = {"search": MagicMock()}
        snap = orch.snapshot(available_tools=tools)

        slash_cmds.available_tools = {}
        orch.rollback(snap)

        assert "search" in slash_cmds.available_tools
