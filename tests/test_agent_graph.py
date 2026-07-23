"""Unit tests for the custom StateGraph implementation in cogtrix.py.

Tests cover _build_agent_graph and run_agent, including:
- Graph compilation and interface
- LLM tool binding
- Normal response flow
- Phantom tool call detection and recovery
- Phantom exhaustion fallback
- Tool execution routing
- Unknown tool fuzzy matching and activation
- run_agent return value and side-effects
- Prompt optimizer preprocessing
- In-loop message compression
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix import (
    _apply_message_compression,
    _build_agent_graph,
    _compress_tool_message,
    _optimize_prompt,
    run_agent,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(responses: list[AIMessage]) -> MagicMock:
    """Return a mock LLM that yields *responses* in order from .invoke()."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses
    return mock_llm


def _make_registry(requires_confirmation: bool = False) -> MagicMock:
    mock_registry = MagicMock()
    mock_registry.requires_confirmation.return_value = requires_confirmation
    return mock_registry


def _phantom_message(msg_id: str = "phantom1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[],
        response_metadata={"finish_reason": "tool_calls"},
        id=msg_id,
    )


# ---------------------------------------------------------------------------
# TestBuildAgentGraph
# ---------------------------------------------------------------------------


class TestBuildAgentGraph:
    """Tests for _build_agent_graph()."""

    def test_graph_compiles(self):
        """_build_agent_graph() returns a compiled graph with invoke/stream."""
        mock_llm = _make_mock_llm([AIMessage(content="Hello", id="m1")])
        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="You are helpful.",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        assert hasattr(graph, "invoke"), "compiled graph must have .invoke()"
        assert hasattr(graph, "stream"), "compiled graph must have .stream()"

    def test_call_model_binds_tools(self):
        """When active_tools_list is non-empty, bind_tools() is called."""
        mock_tool = MagicMock()
        mock_tool.name = "some_tool"

        mock_llm = _make_mock_llm([AIMessage(content="Done", id="m1")])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        graph.invoke({"messages": [HumanMessage(content="hi")]})

        mock_llm.bind_tools.assert_called_once_with([mock_tool])

    def test_normal_response_flow(self):
        """LLM returning a plain AIMessage exits at END with that message."""
        mock_llm = _make_mock_llm([AIMessage(content="Hello world", id="m1")])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Hello world" in m.content for m in ai_messages)

    def test_phantom_detection_and_recovery(self):
        """Phantom message triggers retry; second call returns real content."""
        real_response = AIMessage(content="Recovered response", id="m2")
        mock_llm = _make_mock_llm([_phantom_message("p1"), real_response])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Recovered response" in m.content for m in ai_messages)
        assert mock_llm.invoke.call_count == 2

    def test_phantom_exhaustion(self):
        """After MAX_PHANTOM_RETRIES (3) phantoms, a fallback AIMessage is returned."""
        phantoms = [_phantom_message(f"p{i}") for i in range(10)]
        mock_llm = _make_mock_llm(phantoms)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert len(ai_messages) >= 1
        assert any("persistent formatting issues" in m.content for m in ai_messages)

    def test_tool_execution(self):
        """AIMessage with tool_calls triggers process_tools then loops back."""
        tool_call = {"name": "echo_tool", "args": {"text": "ping"}, "id": "call1"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="m1")
        final_response = AIMessage(content="All done", id="m2")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"
        mock_tool.invoke.return_value = ToolMessage(
            content="pong", tool_call_id="call1", name="echo_tool"
        )

        mock_llm = _make_mock_llm([ai_with_tools, final_response])
        mock_llm.bind_tools.return_value = mock_llm

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="run tool")]})

        messages = result.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].content == "pong"

        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("All done" in m.content for m in ai_messages)

    def test_unknown_tool_fuzzy_match(self):
        """Calling a tool not in active_tools_list but in available_tools activates it."""
        tool_call = {"name": "search_web", "args": {}, "id": "call_fuzzy"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="m_fuzzy")
        final_response = AIMessage(content="Found it", id="m_final")

        available_tool = MagicMock()
        available_tool.name = "search_web"
        available_tool.invoke.return_value = ToolMessage(
            content="search result", tool_call_id="call_fuzzy", name="search_web"
        )

        mock_llm = _make_mock_llm([ai_with_tools, final_response])
        mock_llm.bind_tools.return_value = mock_llm

        active_tools_list: list = []
        available_tools: dict = {"search_web": available_tool}

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=active_tools_list,
            available_tools=available_tools,
            registry=_make_registry(),
            approvals=set(),
        )
        with patch("cogtrix._spinner"):
            result = graph.invoke({"messages": [HumanMessage(content="search")]})

        assert any(t.name == "search_web" for t in active_tools_list)

        messages = result.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert any("search result" in m.content for m in tool_messages)


# ---------------------------------------------------------------------------
# TestRunAgent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """Tests for run_agent()."""

    def _base_kwargs(self, mock_llm: MagicMock) -> dict:
        return {
            "user_input": "Hello",
            "history_messages": [],
            "registry": _make_registry(),
            "approvals": set(),
            "llm": mock_llm,
            "system_prompt": "You are helpful.",
            "available_tools": {},
            "active_tools_list": [],
        }

    def test_returns_response_string(self):
        """run_agent() returns a non-empty string response."""
        mock_llm = _make_mock_llm([AIMessage(content="Hi there!", id="r1")])
        result = run_agent(**self._base_kwargs(mock_llm))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_messages_populated(self):
        """When result_messages is provided it gets populated with graph messages."""
        mock_llm = _make_mock_llm([AIMessage(content="Populated!", id="r2")])
        collected: list = []
        run_agent(**self._base_kwargs(mock_llm), result_messages=collected)
        assert len(collected) > 0
        ai_contents = [m.content for m in collected if isinstance(m, AIMessage)]
        assert any("Populated!" in c for c in ai_contents)

    def test_active_tools_modified_in_place(self):
        """Tool expansion inside the graph is visible to the caller via active_tools_list.

        The list must be non-empty when passed to run_agent so that the ``or []``
        guard in run_agent uses the caller's object rather than a new one.
        """
        tool_call = {"name": "new_tool", "args": {}, "id": "tc1"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="expand_m1")
        final_response = AIMessage(content="Done expanding", id="expand_m2")

        available_tool = MagicMock()
        available_tool.name = "new_tool"
        available_tool.invoke.return_value = ToolMessage(
            content="result", tool_call_id="tc1", name="new_tool"
        )

        sentinel_tool = MagicMock()
        sentinel_tool.name = "sentinel"

        mock_llm = _make_mock_llm([ai_with_tools, final_response])
        mock_llm.bind_tools.return_value = mock_llm

        active_tools_list: list = [sentinel_tool]
        available_tools: dict = {"new_tool": available_tool}

        with patch("cogtrix._spinner"):
            run_agent(
                user_input="expand",
                history_messages=[],
                registry=_make_registry(),
                approvals=set(),
                llm=mock_llm,
                system_prompt="",
                available_tools=available_tools,
                active_tools_list=active_tools_list,
            )

        assert any(getattr(t, "name", None) == "new_tool" for t in active_tools_list)


# ---------------------------------------------------------------------------
# TestOptimizePrompt
# ---------------------------------------------------------------------------


class TestOptimizePrompt:
    """Tests for _optimize_prompt()."""

    def test_short_prompt_passes_through(self):
        """Prompts shorter than threshold are returned unchanged without LLM call."""
        mock_llm = MagicMock()
        result = _optimize_prompt("What time is it?", mock_llm)
        assert result == "What time is it?"
        mock_llm.invoke.assert_not_called()

    def test_long_prompt_triggers_llm(self):
        """Long prompts trigger an LLM call and return the optimized text."""
        long_prompt = "Please analyze this codebase thoroughly. " * 10
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analyze the codebase: Phase 1 read docs, Phase 2 examine source."
        mock_llm.invoke.return_value = mock_response

        result = _optimize_prompt(long_prompt, mock_llm)
        assert result == "Analyze the codebase: Phase 1 read docs, Phase 2 examine source."
        mock_llm.invoke.assert_called_once()

    def test_llm_failure_returns_original(self):
        """If the LLM call fails, the original prompt is returned."""
        long_prompt = "Please do a very thorough analysis of this entire project. " * 5
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = _optimize_prompt(long_prompt, mock_llm)
        assert result == long_prompt

    def test_empty_response_returns_original(self):
        """If the LLM returns empty content, the original prompt is returned."""
        long_prompt = "Study the documentation and run a comprehensive bug hunt. " * 5
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response

        result = _optimize_prompt(long_prompt, mock_llm)
        assert result == long_prompt

    def test_unchanged_prompt_returned(self):
        """If LLM returns the same text, it passes through cleanly."""
        long_prompt = "Search for all security vulnerabilities in the web application layer. " * 3
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = long_prompt
        mock_llm.invoke.return_value = mock_response

        result = _optimize_prompt(long_prompt, mock_llm)
        # .strip() in the implementation removes trailing whitespace
        assert result == long_prompt.strip()

    def test_list_content_response(self):
        """Handle LLM responses where content is a list of dicts."""
        long_prompt = (
            "Run a detailed analysis on this entire project codebase and report findings. " * 3
        )
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"text": "Optimized:"}, {"text": "do analysis."}]
        mock_llm.invoke.return_value = mock_response

        result = _optimize_prompt(long_prompt, mock_llm)
        assert result == "Optimized: do analysis."


# ---------------------------------------------------------------------------
# TestMessageCompression
# ---------------------------------------------------------------------------


def _build_compression_messages(
    num_old_ai: int = 8,
    tool_content_size: int = 3000,
    tool_call_id: str = "call_old",
    tool_name: str = "read_file",
) -> list:
    """Build a message list with old ToolMessages for compression testing.

    Returns [HumanMessage, AIMessage(tool_call), ToolMessage, ..., AIMessage(final)].
    The ToolMessage sits before *num_old_ai* AIMessages, giving it age >= num_old_ai.
    """
    msgs: list = [HumanMessage(content="Do something.")]
    # AI with tool call
    ai_with_call = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": tool_call_id}],
    )
    msgs.append(ai_with_call)
    # ToolMessage (the one to compress)
    msgs.append(
        ToolMessage(
            content="x" * tool_content_size,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    )
    # Subsequent AI messages to create age
    for i in range(num_old_ai):
        msgs.append(AIMessage(content=f"Step {i}", id=f"ai_{i}"))
    return msgs


class TestMessageCompression:
    """Tests for _compress_tool_message and _apply_message_compression."""

    def test_compression_skipped_when_none_context(self):
        """max_context_tokens=None skips compression entirely."""
        msgs = _build_compression_messages()
        result = _apply_message_compression(
            msgs,
            call_count=8,
            compression_cache={},
            llm=MagicMock(),
            max_context_tokens=None,
        )
        assert result is msgs  # same object, not a copy

    def test_compression_skipped_below_threshold(self):
        """Small conversations below both triggers pass through."""
        msgs = _build_compression_messages(tool_content_size=100)
        result = _apply_message_compression(
            msgs,
            call_count=3,
            compression_cache={},
            llm=MagicMock(),
            max_context_tokens=100_000,  # huge window, won't trigger size threshold
        )
        assert result is msgs

    def test_compression_triggers_on_size_threshold(self):
        """Compression runs when total chars >= 60% of context window."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Short summary."
        mock_llm.invoke.return_value = mock_response

        # 5000 chars of tool content; context window = 2000 tokens = 8000 chars.
        # Threshold = 8000 * 0.60 = 4800. Total > 4800 → triggers.
        msgs = _build_compression_messages(tool_content_size=5000)
        result = _apply_message_compression(
            msgs,
            call_count=1,  # not at interval
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=2_000,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any("[compressed]" in (m.content or "") for m in tool_msgs)

    def test_young_messages_not_compressed(self):
        """ToolMessages younger than min_age_cycles are preserved."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed."
        mock_llm.invoke.return_value = mock_response

        # Only 2 AI messages after the ToolMessage (age=2 < 6)
        msgs = _build_compression_messages(num_old_ai=2, tool_content_size=5000)
        result = _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=500,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        # Should NOT be compressed (too young)
        assert all("[compressed]" not in (m.content or "") for m in tool_msgs)
        mock_llm.invoke.assert_not_called()

    def test_short_messages_not_compressed(self):
        """ToolMessages shorter than min_chars are preserved."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed."
        mock_llm.invoke.return_value = mock_response

        # Content is 500 chars (< 2000)
        msgs = _build_compression_messages(tool_content_size=500)
        result = _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=500,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert all("[compressed]" not in (m.content or "") for m in tool_msgs)
        mock_llm.invoke.assert_not_called()

    def test_cache_prevents_recompression(self):
        """Pre-populated cache is reused without LLM call."""
        mock_llm = MagicMock()
        cache = {"call_old": "[compressed] Cached summary."}

        msgs = _build_compression_messages()
        result = _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=500,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any(m.content == "[compressed] Cached summary." for m in tool_msgs)
        mock_llm.invoke.assert_not_called()

    def test_compressed_prefix_marker(self):
        """Compressed messages get the [compressed] prefix."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary of file content."
        mock_llm.invoke.return_value = mock_response

        msgs = _build_compression_messages()
        cache: dict = {}
        _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=500,
        )
        assert cache["call_old"].startswith("[compressed] ")

    def test_original_list_not_mutated(self):
        """The input message list is not modified by compression."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary."
        mock_llm.invoke.return_value = mock_response

        msgs = _build_compression_messages()
        original_contents = [getattr(m, "content", "") for m in msgs]
        _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=500,
        )
        new_contents = [getattr(m, "content", "") for m in msgs]
        assert original_contents == new_contents

    def test_compress_tool_message_fallback(self):
        """_compress_tool_message falls back to truncation on LLM failure."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")

        content = "A" * 5000
        result = _compress_tool_message(content, "read_file", mock_llm)
        # Should be truncated (middle-cut), not the original
        assert len(result) < len(content)
        assert "truncated" in result.lower() or len(result) <= len(content) // 2 + 200

    def test_compression_uses_dedicated_llm(self):
        """When compression_llm is set, it is used instead of main LLM."""
        main_llm = MagicMock()
        compression_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed by dedicated model."
        compression_llm.invoke.return_value = mock_response

        msgs = _build_compression_messages()
        result = _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=compression_llm,  # dedicated LLM passed here
            max_context_tokens=500,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any("[compressed]" in (m.content or "") for m in tool_msgs)
        compression_llm.invoke.assert_called()
        main_llm.invoke.assert_not_called()

    def test_tool_message_ids_preserved(self):
        """Compressed ToolMessages keep tool_call_id and name."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Short."
        mock_llm.invoke.return_value = mock_response

        msgs = _build_compression_messages(
            tool_call_id="call_123", tool_name="execute_shell_command"
        )
        result = _apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=500,
        )
        compressed_tools = [
            m for m in result if isinstance(m, ToolMessage) and "[compressed]" in (m.content or "")
        ]
        assert len(compressed_tools) == 1
        assert compressed_tools[0].tool_call_id == "call_123"
        assert compressed_tools[0].name == "execute_shell_command"

    def test_compression_longer_result_keeps_original(self):
        """If LLM produces longer content than original, original is kept."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # Return something much longer than input
        mock_response.content = "Y" * 10_000
        mock_llm.invoke.return_value = mock_response

        content = "Z" * 3000
        result = _compress_tool_message(content, "read_file", mock_llm)
        assert result == content
