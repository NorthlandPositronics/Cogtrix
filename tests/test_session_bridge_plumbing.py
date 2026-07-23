"""Targeted session bridge regression tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.api.session_bridge import warm_session
from cogtrix_core.memory.manager import BaseMemoryManager


@pytest.mark.asyncio
async def test_warm_session_threads_memory_manager_into_run_config() -> None:
    """warm_session should carry the live memory manager into AgentRunConfig."""
    record = SimpleNamespace(
        id="sess-bridge",
        user_id="user-1",
        name="Bridge test",
        config_json=json.dumps({"memory_mode": "conversation"}),
        token_counts_json=json.dumps({}),
        state="idle",
    )

    app_state = MagicMock()
    app_state.config = None
    app_state.tool_registry = None
    app_state.mcp_manager = MagicMock()
    app_state.mcp_manager.tools_ready = None

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("cogtrix_core.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("cogtrix_core.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        session = await warm_session(record, app_state)

    assert session.memory_manager is mock_mm
    assert session.run_config.memory_manager is mock_mm


@pytest.mark.asyncio
async def test_warm_session_uses_public_compression_api() -> None:
    """warm_session must call configure_compression() instead of setting private attrs."""
    record = SimpleNamespace(
        id="sess-api",
        user_id="user-1",
        name="API test",
        config_json=json.dumps({"memory_mode": "conversation"}),
        token_counts_json=json.dumps({}),
        state="idle",
    )

    app_state = MagicMock()
    app_state.config = None
    app_state.tool_registry = None
    app_state.mcp_manager = MagicMock()
    app_state.mcp_manager.tools_ready = None

    mock_mm = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("cogtrix_core.api.session_bridge._build_memory_manager", return_value=mock_mm),
        patch("cogtrix_core.api.session_bridge._build_llm", return_value=mock_llm),
    ):
        session = await warm_session(record, app_state)

    mock_mm.configure_compression.assert_called_once_with(
        max_context_tokens=session.run_config.max_context_tokens,
        compression_llm=session.run_config.compression_llm,
    )


def test_configure_compression_sets_attributes() -> None:
    """BaseMemoryManager.configure_compression should set _max_context_tokens and _compression_llm."""

    class DummyManager(BaseMemoryManager):
        @property
        def mode_name(self):
            return "dummy"

        def prepare_context(self, user_input):
            raise NotImplementedError

        def update(self, user_input, ai_response, agent_messages=None):
            raise NotImplementedError

        def get_message_count(self):
            return 0

    store = MagicMock()
    store.base_path = MagicMock()
    mgr = DummyManager(store, "sess-1")

    mock_llm = MagicMock()
    mgr.configure_compression(max_context_tokens=8192, compression_llm=mock_llm)

    assert mgr._max_context_tokens == 8192
    assert mgr._compression_llm is mock_llm
