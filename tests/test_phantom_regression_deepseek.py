"""Regression tests for the DeepSeek phantom cutoff failure (Issue #153).

These tests pin the post-cutoff tool-binding refresh, context trimming, and
phantom recovery behavior that regressed when get_file_contents hit its per-tool
budget and the model started emitting raw XML instead of structured tool calls.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.orchestration.graph import (  # noqa: E402
    _apply_context_message_cap,
    _looks_like_phantom_tool_markup,
    build_agent_graph,
)


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.requires_confirmation.return_value = False
    return registry


def _tool_message(tool_call_id: str, content: str = "ok") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id, name="get_file_contents")


def _tool_call_message(call_id: str, path: str = "file.txt") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "get_file_contents", "args": {"path": path}, "id": call_id}],
        id=f"ai_{call_id}",
    )


class TestDeepSeekPhantomRegression:
    def test_budget_cutoff_rebinds_without_get_file_contents(self) -> None:
        """The next call after cutoff must rebind without the disabled tool."""

        tool = MagicMock()
        tool.name = "get_file_contents"
        tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
            content=f"payload:{payload['args']}",
            tool_call_id=payload["id"],
            name="get_file_contents",
        )

        responses = [
            _tool_call_message(f"call_{i}", path=f"/tmp/file_{i}.txt") for i in range(1, 10)
        ] + [
            AIMessage(
                content="I have enough data now and can summarize it plainly.",
                id="final",
            )
        ]
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = responses

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )

        result = graph.invoke({"messages": [HumanMessage(content="summarize the files")]})

        bind_names = [
            [getattr(t, "name", "") for t in call.args[0]]
            for call in mock_llm.bind_tools.call_args_list
        ]
        assert bind_names, "expected the LLM to be bound at least once"
        assert "get_file_contents" in bind_names[0]
        assert any(
            "get_file_contents" not in names for names in bind_names[1:]
        ), "cutoff must force a rebound that excludes the disabled tool"
        assert any(
            isinstance(msg, AIMessage) and "summarize it plainly" in str(msg.content)
            for msg in result["messages"]
        )

    def test_context_trim_no_orphaned_tool_call_ids(self) -> None:
        """Trimming must not leave ToolMessages pointing at dropped AI messages."""

        msgs = [
            HumanMessage(content="turn-0"),
            _tool_call_message("call_1"),
            _tool_message("call_1", "file-1"),
            HumanMessage(content="turn-1"),
            _tool_call_message("call_2"),
            _tool_message("call_2", "file-2"),
            HumanMessage(content="turn-2"),
        ]

        result = _apply_context_message_cap(msgs, 4)

        declared_ids = {
            tc["id"]
            for msg in result
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        }
        for idx, msg in enumerate(result):
            if not isinstance(msg, ToolMessage):
                continue
            assert msg.tool_call_id in declared_ids
            assert any(
                isinstance(prev, AIMessage)
                and any(tc["id"] == msg.tool_call_id for tc in (prev.tool_calls or []))
                for prev in result[:idx]
            )

    def test_phantom_regex_ignores_inline_documentation_snippets(self) -> None:
        """Docs that mention JSON-ish field names must not trip phantom detection."""

        msg = SimpleNamespace(
            content='The `{"tool": "hammer"}` example is documentation. Use "arguments" carefully.',
            tool_calls=None,
        )

        assert not _looks_like_phantom_tool_markup(msg)

    def test_budget_cutoff_then_phantom_recovers_to_prose(self) -> None:
        """After cutoff, a phantom XML turn must still recover to a prose answer."""

        tool = MagicMock()
        tool.name = "get_file_contents"
        tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
            content=f"payload:{payload['args']}",
            tool_call_id=payload["id"],
            name="get_file_contents",
        )

        responses = [
            _tool_call_message(f"call_{i}", path=f"/tmp/file_{i}.txt") for i in range(1, 10)
        ] + [
            AIMessage(
                content='<function_calls><invoke name="checkpoint"></invoke></function_calls>',
                id="phantom",
            ),
            AIMessage(
                content="I have enough data now and can summarize it plainly.",
                id="final",
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = responses

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )

        result = graph.invoke({"messages": [HumanMessage(content="summarize the files")]})

        assert mock_llm.invoke.call_count == 11
        assert any(
            isinstance(msg, AIMessage) and "summarize it plainly" in str(msg.content)
            for msg in result["messages"]
        )
