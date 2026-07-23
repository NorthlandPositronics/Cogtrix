"""Tests for shared-dict isolation in MessageHandler.handle().

BUG-027: handle() must not mutate self._available_tools or self._active_tools
when the runner (or graph process_tools node) pops from / modifies the passed
copies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import MagicMock

from src.assistant.channel import IncomingMessage
from src.assistant.handler import MessageHandler
from src.memory.context import MemoryContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(text: str = "Hello") -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id="42",
        message_id="m1",
        sender_id="u1",
        sender_name="Alice",
        text=text,
        timestamp=time.time(),
    )


def _make_session() -> MagicMock:
    session = MagicMock()
    session.session_key = "telegram::42"
    session.lock = MagicMock()
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=None,
    )
    return session


def _make_handler(
    available_tools: dict | None = None,
    active_tools: list | None = None,
    agent_runner: Callable | None = None,
) -> tuple[MessageHandler, MagicMock]:
    """Return (handler, mock_session_mgr)."""
    session = _make_session()
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

    if agent_runner is None:
        agent_runner = MagicMock(return_value="")

    handler = MessageHandler(
        session_mgr=session_mgr,
        config={},
        llm=MagicMock(),
        system_prompt="sys",
        registry=MagicMock(),
        approvals=set(),
        available_tools=available_tools or {},
        active_tools=active_tools or [],
        agent_runner=agent_runner,
    )
    return handler, session_mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAvailableToolsIsolation:
    """handle() passes a copy of _available_tools to the runner."""

    def test_runner_pop_does_not_mutate_available_tools(self):
        """When the runner pops a key from the passed dict, the original is unchanged."""
        tool_a = MagicMock()
        tool_a.name = "tool_a"
        tool_b = MagicMock()
        tool_b.name = "tool_b"

        def _runner_that_pops(**kwargs: object) -> str:
            passed_dict = kwargs.get("available_tools", {})
            passed_dict.pop("tool_a", None)
            return "done"

        handler, _ = _make_handler(
            available_tools={"tool_a": tool_a, "tool_b": tool_b},
            agent_runner=_runner_that_pops,
        )
        original_keys = set(handler._available_tools.keys())

        handler.handle(_make_msg(), MagicMock())

        assert set(handler._available_tools.keys()) == original_keys

    def test_runner_receives_all_available_tools(self):
        """The runner receives a dict that contains all entries from _available_tools."""
        tool_x = MagicMock()
        tool_x.name = "tool_x"

        captured: list[dict] = []

        def _runner_capture(**kwargs: object) -> str:
            captured.append(dict(kwargs.get("available_tools", {})))
            return "ok"

        handler, _ = _make_handler(
            available_tools={"tool_x": tool_x},
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert "tool_x" in captured[0]

    def test_available_tools_passed_is_a_different_object(self):
        """The dict object passed to the runner is not the same as self._available_tools."""
        tool = MagicMock()
        tool.name = "some_tool"

        captured_id: list[int] = []

        def _runner_capture(**kwargs: object) -> str:
            captured_id.append(id(kwargs.get("available_tools")))
            return "ok"

        handler, _ = _make_handler(
            available_tools={"some_tool": tool},
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert captured_id[0] != id(handler._available_tools)


class TestActiveToolsIsolation:
    """handle() passes a copy of _active_tools to the runner."""

    def test_runner_pop_does_not_mutate_active_tools(self):
        """When the runner removes an item from the passed list, the original is unchanged."""
        tool_1 = MagicMock()
        tool_1.name = "tool_1"
        tool_2 = MagicMock()
        tool_2.name = "tool_2"

        def _runner_that_pops(**kwargs: object) -> str:
            passed_list = kwargs.get("active_tools_list", [])
            if passed_list:
                passed_list.pop()
            return "done"

        handler, _ = _make_handler(
            active_tools=[tool_1, tool_2],
            agent_runner=_runner_that_pops,
        )
        original_length = len(handler._active_tools)

        handler.handle(_make_msg(), MagicMock())

        assert len(handler._active_tools) == original_length

    def test_runner_receives_all_active_tools(self):
        """The runner receives a list that contains all entries from _active_tools."""
        tool_a = MagicMock()
        tool_a.name = "active_tool_a"

        captured: list[list] = []

        def _runner_capture(**kwargs: object) -> str:
            captured.append(list(kwargs.get("active_tools_list", [])))
            return "ok"

        handler, _ = _make_handler(
            active_tools=[tool_a],
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert tool_a in captured[0]

    def test_active_tools_passed_is_a_different_object(self):
        """The list object passed to the runner is not the same as self._active_tools."""
        tool = MagicMock()
        tool.name = "t"

        captured_id: list[int] = []

        def _runner_capture(**kwargs: object) -> str:
            captured_id.append(id(kwargs.get("active_tools_list")))
            return "ok"

        handler, _ = _make_handler(
            active_tools=[tool],
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert captured_id[0] != id(handler._active_tools)
