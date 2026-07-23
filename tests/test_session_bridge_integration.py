"""Integration tests for the session bridge (warm_session, registry, eviction).

Covers the critical path from a DB record to a live ApiSession including
tool loading, memory initialization, concurrent warming deduplication,
and idle eviction.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.session_bridge import ApiSessionRegistry, warm_session


def _make_record(
    *,
    session_id: str = "sess-1",
    user_id: str = "user-1",
    name: str = "Test session",
    config_json: str = "{}",
    token_counts_json: str = "{}",
    state: str = "idle",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id,
        user_id=user_id,
        name=name,
        config_json=config_json,
        token_counts_json=token_counts_json,
        state=state,
    )


def _make_app_state(
    *,
    tool_registry: MagicMock | None = None,
    config: MagicMock | None = None,
    mcp_manager: MagicMock | None = None,
) -> MagicMock:
    app_state = MagicMock()
    app_state.config = config
    app_state.tool_registry = tool_registry
    app_state.mcp_manager = mcp_manager or MagicMock()
    app_state.mcp_manager.tools_ready = None
    return app_state


# ── 1. Session creation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warm_session_creates_full_api_session() -> None:
    """warm_session should build a fully populated ApiSession from a DB record."""
    record = _make_record(
        session_id="sess-create",
        config_json=json.dumps({"memory_mode": "conversation", "model": "gpt-4"}),
        token_counts_json=json.dumps({"input_tokens": 10, "output_tokens": 5}),
    )
    app_state = _make_app_state()

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("src.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("src.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        session = await warm_session(record, app_state)

    assert session.id == "sess-create"
    assert session.user_id == "user-1"
    assert session.name == "Test session"
    assert session.agent_state == "idle"
    assert session.memory_manager is mock_mm
    assert session.llm is mock_llm
    assert session.token_counts == {"input_tokens": 10, "output_tokens": 5, "context_window": 0}
    assert session.registry is None
    assert session._lock is not None
    assert session.turn_lock is not None


# ── 2. Tool loading ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warm_session_loads_tools_from_registry() -> None:
    """warm_session should populate available_tools and active_tools_list from the registry."""
    record = _make_record(session_id="sess-tools")

    mock_tool = MagicMock()
    mock_tool.name = "query_knowledge_base"
    tool_registry = MagicMock()
    tool_registry.tools = {"query_knowledge_base": mock_tool, "shell": MagicMock()}

    app_state = _make_app_state(tool_registry=tool_registry)

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("src.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("src.api.session_bridge._build_llm", return_value=mock_llm),
        patch(
            "src.agent.registry.filter_tools_for_agent",
            return_value=({"query_knowledge_base": mock_tool}, [mock_tool]),
        ),
        patch("src.tools.configure.rag_should_auto_activate", return_value=False),
        patch("src.tools.configure.build_tool_catalog", return_value={}),
        patch("src.tools.configure.create_request_tools_tool", return_value=None),
    ):
        session = await warm_session(record, app_state)

    assert "query_knowledge_base" in session.run_config.available_tools
    assert session.session_state.all_tool_originals is not None
    assert "shell" in session.session_state.all_tool_originals


@pytest.mark.asyncio
async def test_warm_session_denies_dangerous_tools_by_default() -> None:
    """API sessions should deny shell/bash/python_exec unless explicitly enabled."""
    record = _make_record(session_id="sess-deny")
    app_state = _make_app_state()

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("src.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("src.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        session = await warm_session(record, app_state)

    denied = session.session_state.get_denials_snapshot()
    assert "shell" in denied
    assert "bash" in denied
    assert "python_exec" in denied


# ── 3. Memory initialization ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warm_session_wires_memory_manager_to_run_config() -> None:
    """The memory manager should be wired with max_context_tokens and compression_llm."""
    record = _make_record(session_id="sess-mem")
    app_state = _make_app_state()

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("src.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("src.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        session = await warm_session(record, app_state)

    assert session.run_config.memory_manager is mock_mm
    assert mock_mm._max_context_tokens is not None
    assert hasattr(mock_mm, "_compression_llm")
    mock_mm.set_llm.assert_called_once_with(mock_llm)


# ── 4. Concurrent warming deduplication ────────────────────────────────────


@pytest.mark.asyncio
async def test_get_or_warm_deduplicates_concurrent_calls() -> None:
    """Concurrent requests for the same session should only warm once."""
    record = _make_record(session_id="sess-dedup")

    app_state = _make_app_state()
    registry = ApiSessionRegistry(app_state)

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    warm_count = 0

    async def _slow_warm(r, a):
        nonlocal warm_count
        warm_count += 1
        await asyncio.sleep(0.1)
        return await warm_session(r, a)

    db_session = MagicMock()

    with (
        patch("src.api.session_bridge.warm_session", side_effect=_slow_warm),
        patch("src.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("src.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        repo_mock = AsyncMock()
        repo_mock.get_by_id.return_value = record

        with patch("src.api.db.repositories.sessions.SessionRepository", return_value=repo_mock):
            results = await asyncio.gather(
                registry.get_or_warm("sess-dedup", db_session),
                registry.get_or_warm("sess-dedup", db_session),
                registry.get_or_warm("sess-dedup", db_session),
            )

    assert warm_count == 1
    assert all(r is results[0] for r in results)


# ── 5. Eviction idle ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evict_idle_saves_and_removes_stale_sessions() -> None:
    """Sessions idle longer than max_age_seconds should be saved and evicted."""
    app_state = _make_app_state()
    registry = ApiSessionRegistry(app_state)

    mock_mm = MagicMock()
    session = MagicMock()
    session.id = "sess-evict"
    session.last_activity = time.time() - 3600  # 1 hour ago
    session.turn_task = None
    session.memory_manager = mock_mm

    await registry.put(session)
    evicted = await registry.evict_idle(max_age_seconds=60)

    assert evicted == 1
    assert await registry.get_cached("sess-evict") is None
    mock_mm.save.assert_called_once()


@pytest.mark.asyncio
async def test_evict_idle_skips_active_turns() -> None:
    """Sessions with an active agent turn should not be evicted."""
    app_state = _make_app_state()
    registry = ApiSessionRegistry(app_state)

    mock_mm = MagicMock()
    session = MagicMock()
    session.id = "sess-active"
    session.last_activity = time.time() - 3600
    session.turn_task = MagicMock()
    session.turn_task.done.return_value = False
    session.memory_manager = mock_mm

    await registry.put(session)
    evicted = await registry.evict_idle(max_age_seconds=60)

    assert evicted == 0
    assert await registry.get_cached("sess-active") is session
    mock_mm.save.assert_not_called()
