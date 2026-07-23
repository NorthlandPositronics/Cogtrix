"""Regression tests for persistent tool-message repair in the graph node.

Issue #192 showed that orphaned ToolMessages were being repaired only in the
local prompt copy. The LangGraph state itself still kept the bad messages, so
the same orphans reappeared on the next turn.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.orchestration.graph import build_agent_graph  # noqa: E402


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.requires_confirmation.return_value = False
    return registry


class TestGraphRepairPersistence:
    def test_orphaned_tool_messages_are_removed_from_state(self) -> None:
        """The repaired state should not keep the same orphaned ToolMessages."""

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Recovered prose", id="r1")

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )

        orphaned_messages = [
            HumanMessage(content="continue"),
            *[
                ToolMessage(
                    content=f"orphan-{i}",
                    tool_call_id=f"chatcmpl-tool-{i}",
                    name="get_file_contents",
                )
                for i in range(8)
            ],
        ]

        result = graph.invoke({"messages": orphaned_messages})
        messages = result["messages"]

        assert any(
            isinstance(msg, AIMessage) and msg.content == "Recovered prose" for msg in messages
        )
        assert not any(
            isinstance(msg, ToolMessage) and msg.tool_call_id.startswith("chatcmpl-tool-")
            for msg in messages
        )
