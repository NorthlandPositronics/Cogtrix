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

import concurrent.futures
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from cogtrix import (
    _build_agent_graph,
    run_agent,
)
from src.orchestration.compression import (
    apply_message_compression,
    compress_tool_message,
)
from src.orchestration.graph import _correct_tool_args, _tool_arg_schema_cache
from src.orchestration.intent import OwnershipMode, OwnershipResult, TaskComplexity
from src.orchestration.run_config import AgentRunConfig

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

        # The graph auto-injects the checkpoint tool, so check the original
        # tool is present alongside any auto-injected tools.
        mock_llm.bind_tools.assert_called_once()
        bound_tools = mock_llm.bind_tools.call_args[0][0]
        assert mock_tool in bound_tools

    def test_bind_tools_is_warmed_during_graph_construction(self):
        """Initial active tools should pre-warm bind_tools before the first turn."""
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

        mock_llm.bind_tools.assert_called_once()
        bound_tools = mock_llm.bind_tools.call_args[0][0]
        assert mock_tool in bound_tools

        graph.invoke({"messages": [HumanMessage(content="hi")]})
        mock_llm.bind_tools.assert_called_once()

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

    def test_tools_ready_gate_delays_call_model_until_set(self):
        """When tools_ready is unset call_model returns a transient message (#114)."""
        import threading

        from src.orchestration.run_config import AgentRunConfig

        tools_ready = threading.Event()  # not yet set — simulates post-reconnect window

        mock_llm = _make_mock_llm([AIMessage(content="should not reach here", id="m1")])

        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            tools_ready=tools_ready,
        )
        graph = _build_agent_graph(config=config, registry=_make_registry(), approvals=set())

        # Release the gate after a short delay (simulates tool re-discovery completing)
        def _release():
            time.sleep(0.05)
            tools_ready.set()

        threading.Thread(target=_release, daemon=True).start()
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        # The gate released before timeout — LLM should have been invoked normally.
        assert any(m.content for m in ai_messages)

    def test_tools_not_ready_timeout_returns_transient_message(self):
        """When tools_ready never fires within timeout, a transient retry message is returned."""
        import threading

        from src.orchestration.run_config import AgentRunConfig

        tools_ready = threading.Event()  # never set
        mock_llm = _make_mock_llm([])

        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            tools_ready=tools_ready,
        )

        # Patch the wait timeout to 0.01s so the test completes quickly.
        import unittest.mock as mock

        with mock.patch.object(tools_ready, "wait", return_value=False):
            graph = _build_agent_graph(config=config, registry=_make_registry(), approvals=set())
            result = graph.invoke({"messages": [HumanMessage(content="hi")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        # Must return the transient reconnect message, not invoke the LLM.
        assert any("reconnect" in (m.content or "").lower() for m in ai_messages)
        assert mock_llm.invoke.call_count == 0

    def test_phantom_xml_markup_triggers_retry(self):
        """Raw XML tool-call markup should follow the phantom recovery path."""
        xml_response = AIMessage(
            content='<function_calls><invoke name="list_issues"></invoke></function_calls>',
            id="p1",
        )
        real_response = AIMessage(content="Recovered response", id="m2")
        mock_llm = _make_mock_llm([xml_response, real_response])

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

    def test_markdown_phantom_report_triggers_retry(self):
        """Fabricated structured markdown report (no tool calls) triggers phantom recovery (#170)."""
        fabricated_report = AIMessage(
            content=(
                "### 1. Slack Check — #cogtrix-project-discussions\n"
                "- Retrieved last 8 messages. No new mentions.\n\n"
                "### 2. Open Issues\n"
                "| Issue | Title | Updated |\n"
                "|-------|-------|---------|\n"
                "| #42 | Fix memory leak in datastream | 10 min ago |\n"
                "| #39 | Add API usage docs | 1 hour ago |\n"
            ),
            id="p1",
        )
        real_response = AIMessage(content="Recovered after retry", id="m2")
        mock_llm = _make_mock_llm([fabricated_report, real_response])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="check for new work")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Recovered after retry" in m.content for m in ai_messages)
        assert mock_llm.invoke.call_count == 2

    def test_success_claim_after_all_tool_errors_triggers_fabrication_retry(self):
        """When all tool results are errors, fabrication-specific retry nudge is injected (#544)."""
        tool_call = {
            "name": "cron_add",
            "id": "tc1",
            "args": {"schedule": "*/15 * * * *", "command": "echo hi"},
            "type": "tool_call",
        }
        first = AIMessage(content="", tool_calls=[tool_call], id="m1")
        fabricated = AIMessage(
            content="# ✅ Cron Job Created Successfully\nCron job active every 15 minutes.",
            id="m2",
        )
        corrected = AIMessage(
            content="I could not create the cron job because the tool returned: Tool not loaded.",
            id="m3",
        )
        mock_llm = _make_mock_llm([first, fabricated, corrected])

        cron_tool = MagicMock()
        cron_tool.name = "cron_add"
        cron_tool.invoke.return_value = "Error: Tool not loaded"

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[cron_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="set a cron job")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]

        assert any("could not create the cron job" in m.content.lower() for m in ai_messages)
        assert any(
            "some of the tools you called returned errors" in (m.content or "").lower()
            for m in human_messages
        )
        assert any(
            "do not fabricate success messages" in (m.content or "").lower() for m in human_messages
        )
        assert not any("could not be parsed" in (m.content or "").lower() for m in human_messages)
        assert mock_llm.invoke.call_count == 3

    def test_plain_prose_without_tables_not_flagged(self):
        """Plain prose response without tables/numbered sections is accepted (#170 false-positive guard)."""
        normal_response = AIMessage(
            content=(
                "I checked the Slack channel and found no new messages. "
                "There are currently 3 open issues in the repository."
            ),
            id="m1",
        )
        mock_llm = _make_mock_llm([normal_response])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="check for new work")]})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any(
            "no new messages" in m.content.lower() for m in ai_messages
        ), "Normal prose must not be misclassified as phantom"
        assert mock_llm.invoke.call_count == 1

    def test_phantom_exhaustion(self):
        """After MAX_PHANTOM_RETRIES (3) phantoms, a fallback AIMessage is returned.

        After the May 2026 user-disaster fix, the give-up branch synthesizes
        from accumulated state.  With no checkpoints / tool results in this
        all-phantoms scenario, the polite "rephrase" fallback fires.
        """
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
        # The fallback message (when nothing is accumulated) is the polite
        # "rephrase the question" prompt, replacing the older hard-coded
        # "persistent formatting issues" text.
        assert any("rephrase" in m.content.lower() for m in ai_messages)

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

    def test_tool_output_is_truncated_before_history_storage(self):
        """Large ToolMessage outputs are capped before they enter history."""
        tool_call = {"name": "echo_tool", "args": {"text": "ping"}, "id": "call_cap"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="m_cap")
        final_response = AIMessage(content="Done", id="m_done")

        long_content = "Z" * 50_000
        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"
        mock_tool.invoke.return_value = ToolMessage(
            content=long_content,
            tool_call_id="call_cap",
            name="echo_tool",
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
        assert tool_messages, "expected a tool message in history"

        tool_message = tool_messages[0]
        assert len(tool_message.content) < len(long_content)
        assert len(tool_message.content) <= 30_250
        assert "truncated to fit context budget" in tool_message.content

    def test_fuzzy_matched_available_tool_returns_guidance(self):
        """Fuzzy-resolving a tool in available_tools also returns guidance, not auto-load.

        Uses 'search_web_tool' which fuzzy-matches 'search_web' via shared tokens
        and word-containment boost (score > 0.65 threshold).
        """
        # LLM calls "search_web_tool" (typo), fuzzy-match resolves to "search_web" in available
        tool_call = {"name": "search_web_tool", "args": {}, "id": "call_fuzzy2"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="m_fuzzy2")
        final_response = AIMessage(content="Done", id="m_final2")

        available_tool = MagicMock()
        available_tool.name = "search_web"
        available_tool.invoke.return_value = ToolMessage(
            content="result", tool_call_id="call_fuzzy2", name="search_web"
        )

        mock_llm = _make_mock_llm([ai_with_tools, final_response])
        mock_llm.bind_tools.return_value = mock_llm

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={"search_web": available_tool},
            registry=_make_registry(),
            approvals=set(),
        )
        with patch("cogtrix._spinner"):
            result = graph.invoke({"messages": [HumanMessage(content="search")]})

        messages = result.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        # search_web must NOT be auto-loaded; guidance message returned instead
        assert not any(
            t.content == "result" for t in tool_messages
        ), "available tool must not be invoked directly — requires request_tools() first"
        assert any(
            "request_tools" in (t.content or "") for t in tool_messages
        ), "LLM must be told to use request_tools() when calling a fuzzy-matched available tool"

    def test_denied_available_tool_returns_disabled_message(self):
        """A tool in available_tools that is denied returns 'disabled' message."""
        from src.orchestration.session_state import SessionState

        tool_call = {"name": "search_web", "args": {}, "id": "call_denied"}
        ai_with_tools = AIMessage(content="", tool_calls=[tool_call], id="m_denied")
        final_response = AIMessage(content="Done", id="m_final_denied")

        available_tool = MagicMock()
        available_tool.name = "search_web"

        session = SessionState()
        session.deny_tool("search_web")

        mock_llm = _make_mock_llm([ai_with_tools, final_response])
        mock_llm.bind_tools.return_value = mock_llm

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={"search_web": available_tool},
            registry=_make_registry(),
            approvals=set(),
            session_state=session,
        )
        with patch("cogtrix._spinner"):
            result = graph.invoke({"messages": [HumanMessage(content="search")]})

        messages = result.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert any(
            "disabled" in (t.content or "").lower() for t in tool_messages
        ), "denied tool must produce a 'disabled' message rather than guidance or invocation"

    def test_unknown_tool_returns_discovery_guidance(self):
        """Calling a tool that is in available_tools but not active now returns
        guidance to use request_tools(query=...) instead of silently auto-loading.

        This enforces tool equality: all tools — built-in and MCP — must be
        discovered through request_tools so the model picks the most specific
        one rather than falling back to training-familiar generics.
        """
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

        # Tool must NOT be auto-loaded — it stays in available_tools
        assert not any(t.name == "search_web" for t in active_tools_list)

        # The agent must receive guidance to use request_tools instead
        messages = result.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert any("request_tools" in m.content for m in tool_messages)


# ---------------------------------------------------------------------------
# TestRunAgent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """Tests for run_agent()."""

    @pytest.fixture(autouse=True)
    def _drain_compression_jobs(self):
        """Drain pending background compression jobs after each test.

        Prevents state leaks if a test fails mid-execution before its own
        drain loop completes.
        """
        yield
        from src.orchestration import runner as runner_mod

        for _ in range(100):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

    def _base_config(self, mock_llm: MagicMock) -> AgentRunConfig:
        return AgentRunConfig(
            llm=mock_llm,
            system_prompt="You are helpful.",
            available_tools={},
            active_tools_list=[],
        )

    def test_returns_response_string(self):
        """run_agent() returns a non-empty string response."""
        mock_llm = _make_mock_llm([AIMessage(content="Hi there!", id="r1")])
        result = run_agent("Hello", [], _make_registry(), set(), config=self._base_config(mock_llm))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_messages_populated(self):
        """When result_messages is provided it gets populated with graph messages."""
        mock_llm = _make_mock_llm([AIMessage(content="Populated!", id="r2")])
        collected: list = []
        run_agent(
            "Hello",
            [],
            _make_registry(),
            set(),
            config=self._base_config(mock_llm),
            result_messages=collected,
        )
        assert len(collected) > 0
        ai_contents = [m.content for m in collected if isinstance(m, AIMessage)]
        assert any("Populated!" in c for c in ai_contents)

    def test_on_demand_tool_not_auto_loaded(self):
        """Tools in available_tools are NOT auto-loaded when called directly.

        The agent must be guided to use request_tools(query=...) instead.
        This ensures all tools are discovered through the same mechanism,
        preventing training-data bias from favouring familiar tool names.
        The active_tools_list must not gain the on-demand tool.
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
                config=AgentRunConfig(
                    llm=mock_llm,
                    system_prompt="",
                    available_tools=available_tools,
                    active_tools_list=active_tools_list,
                ),
            )

        # new_tool must NOT be auto-loaded into active_tools_list
        assert not any(getattr(t, "name", None) == "new_tool" for t in active_tools_list)
        # sentinel must still be present (list not corrupted)
        assert any(getattr(t, "name", None) == "sentinel" for t in active_tools_list)

    def test_ownership_constraint_does_not_accumulate_across_runs(self):
        """run_agent() must not keep appending ownership constraints to config.system_prompt."""
        mock_llm = _make_mock_llm(
            [AIMessage(content="First turn", id="m1"), AIMessage(content="Second turn", id="m2")]
        )
        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="You are helpful.",
            available_tools={},
            active_tools_list=[],
            context_compression=False,
            max_context_tokens=4096,
        )
        ownership = OwnershipResult(
            mode=OwnershipMode.INFORM,
            confidence=0.9,
            is_reversible=True,
            raw_signal="test",
            inferred_action="explain",
        )

        with (
            patch("src.orchestration.runner.classify_task_ownership", return_value=ownership),
            patch("cogtrix._spinner"),
        ):
            first = run_agent(
                user_input="What is Docker?",
                history_messages=[],
                registry=_make_registry(),
                approvals=set(),
                config=config,
            )
            assert first == "First turn"
            assert config.system_prompt == "You are helpful."

            second = run_agent(
                user_input="What is Kubernetes?",
                history_messages=[],
                registry=_make_registry(),
                approvals=set(),
                config=config,
            )
            assert second == "Second turn"
            assert config.system_prompt == "You are helpful."

    def test_task_complexity_override_skips_reclassification(self):
        """run_agent() must reuse provided task_complexity instead of reclassifying."""
        mock_llm = _make_mock_llm([AIMessage(content="Override respected", id="m3")])
        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="You are helpful.",
            available_tools={},
            active_tools_list=[],
            context_compression=False,
            max_context_tokens=4096,
        )

        with (
            patch("src.orchestration.runner.classify_task_complexity") as classify_mock,
            patch("cogtrix._spinner"),
        ):
            result = run_agent(
                user_input="Please classify this once.",
                history_messages=[],
                registry=_make_registry(),
                approvals=set(),
                config=config,
                task_complexity=TaskComplexity.COMPLEX_RESEARCH,
            )

        assert result == "Override respected"
        classify_mock.assert_not_called()

    def test_simple_tasks_preload_common_tools(self):
        """Simple tasks should skip the request_tools bootstrap for common tools."""
        from src.orchestration.runner import _auto_load_simple_tools

        available_tools = {}
        for name in (
            "calculate",
            "search_web",
            "read_file",
            "get_current_datetime",
            "other_tool",
        ):
            available_tools[name] = SimpleNamespace(name=name)

        config = AgentRunConfig(
            llm=_make_mock_llm([AIMessage(content="Simple", id="m3a")]),
            system_prompt="You are helpful.",
            available_tools=available_tools,
            active_tools_list=[],
            context_compression=False,
            max_context_tokens=4096,
        )

        _auto_load_simple_tools(config)

        assert [tool.name for tool in config.active_tools_list] == [
            "calculate",
            "search_web",
            "read_file",
            "get_current_datetime",
        ]
        assert "other_tool" in config.available_tools
        for name in ("calculate", "search_web", "read_file", "get_current_datetime"):
            assert name not in config.available_tools

    def test_reasoning_mode_preloads_cron_tools(self):
        """Reasoning mode should auto-load the cron trio without request_tools."""
        from src.tools.configure import apply_tool_preset

        registry = SimpleNamespace(
            tools={
                name: SimpleNamespace(name=name)
                for name in (
                    "get_current_datetime",
                    "cron_add",
                    "cron_list",
                    "cron_remove",
                    "other_tool",
                )
            }
        )

        active, available = apply_tool_preset(registry, "reasoning")
        assert set(active) == {
            "get_current_datetime",
            "cron_add",
            "cron_list",
            "cron_remove",
        }
        assert "other_tool" in available
        for name in ("get_current_datetime", "cron_add", "cron_list", "cron_remove"):
            assert name not in available

        for mode in ("code", "conversation"):
            mode_active, _ = apply_tool_preset(registry, mode)
            assert "cron_add" not in mode_active
            assert "cron_list" not in mode_active
            assert "cron_remove" not in mode_active

    def test_turn_start_compression_runs_in_background(self):
        """run_agent() must not block the first token on turn-start compression."""
        from src.orchestration import runner as runner_mod

        mock_llm = _make_mock_llm([AIMessage(content="Fast response", id="m4")])
        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="You are helpful.",
            available_tools={},
            active_tools_list=[],
            context_compression=True,
            tier_cache_enabled=True,
            max_context_tokens=20_000,
        )

        started = threading.Event()
        release = threading.Event()

        def _slow_compression(*args, **kwargs):
            started.set()
            release.wait(timeout=1.0)
            return args[0]

        try:
            with (
                patch(
                    "src.orchestration.runner.apply_message_compression",
                    side_effect=_slow_compression,
                ),
                patch("cogtrix._spinner"),
            ):
                began = time.perf_counter()
                result = run_agent(
                    user_input="Compress but do not block.",
                    history_messages=[],
                    registry=_make_registry(),
                    approvals=set(),
                    config=config,
                )
                elapsed = time.perf_counter() - began
                assert result == "Fast response"
                assert elapsed < 1.0, f"run_agent() blocked turn start for {elapsed:.3f}s"
                assert started.wait(timeout=1.0), "background compression job did not start"
        finally:
            release.set()

        # Drain any completed background jobs so the queue does not leak between tests.
        for _ in range(50):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

    def test_post_turn_compression_runs_in_background(self):
        """run_agent() must not block response delivery on post-turn compression."""
        from src.orchestration import runner as runner_mod

        mock_llm = _make_mock_llm([AIMessage(content="Fast response", id="m5")])
        config = AgentRunConfig(
            llm=mock_llm,
            system_prompt="You are helpful.",
            available_tools={},
            active_tools_list=[],
            context_compression=True,
            tier_cache_enabled=False,
            max_context_tokens=20_000,
        )

        started = threading.Event()
        release = threading.Event()

        def _slow_compression(*args, **kwargs):
            if kwargs.get("call_count") == 999:
                started.set()
                release.wait(timeout=1.0)
            return args[0]

        try:
            with (
                patch(
                    "src.orchestration.runner.apply_message_compression",
                    side_effect=_slow_compression,
                ),
                patch("cogtrix._spinner"),
            ):
                began = time.perf_counter()
                result = run_agent(
                    user_input="Compress after response, not before.",
                    history_messages=[],
                    registry=_make_registry(),
                    approvals=set(),
                    config=config,
                )
                elapsed = time.perf_counter() - began
                assert result == "Fast response"
                assert elapsed < 1.0, f"run_agent() delayed response delivery for {elapsed:.3f}s"
                assert started.wait(timeout=1.0), "post-turn compression job did not start"
        finally:
            release.set()

        # Drain any completed background jobs so the queue does not leak between tests.
        for _ in range(50):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

    def test_completed_background_compression_job_merges_cache(self):
        """Completed background jobs must merge their warmed cache into the target."""
        from src.orchestration import runner as runner_mod

        target_cache: OrderedDict[str, str] = OrderedDict()
        snapshot: OrderedDict[str, str] = OrderedDict([("call_old", "Cached summary.")])
        future: concurrent.futures.Future[OrderedDict[str, str]] = concurrent.futures.Future()
        future.set_result(snapshot)

        with runner_mod._cache_lock:
            runner_mod._pending_background_compression_jobs.append((future, target_cache))

        runner_mod._drain_background_compression_jobs()

        assert target_cache["call_old"] == "Cached summary."
        with runner_mod._cache_lock:
            assert not runner_mod._pending_background_compression_jobs

    def test_drain_with_target_cache_only_drains_matching_jobs(self):
        """Per-session drain must not steal jobs from other sessions (#901)."""
        from src.orchestration import runner as runner_mod

        cache_a: OrderedDict[str, str] = OrderedDict()
        cache_b: OrderedDict[str, str] = OrderedDict()
        snapshot_a: OrderedDict[str, str] = OrderedDict([("key_a", "val_a")])
        snapshot_b: OrderedDict[str, str] = OrderedDict([("key_b", "val_b")])

        future_a: concurrent.futures.Future[OrderedDict[str, str]] = concurrent.futures.Future()
        future_a.set_result(snapshot_a)
        future_b: concurrent.futures.Future[OrderedDict[str, str]] = concurrent.futures.Future()
        future_b.set_result(snapshot_b)

        with runner_mod._cache_lock:
            runner_mod._pending_background_compression_jobs.append((future_a, cache_a))
            runner_mod._pending_background_compression_jobs.append((future_b, cache_b))

        # Drain only jobs for cache_a — cache_b's job must remain pending.
        runner_mod._drain_background_compression_jobs(cache_a)

        assert cache_a["key_a"] == "val_a"
        assert "key_b" not in cache_b
        with runner_mod._cache_lock:
            assert len(runner_mod._pending_background_compression_jobs) == 1
            assert runner_mod._pending_background_compression_jobs[0][1] is cache_b

        # Cleanup — drain the remaining job.
        runner_mod._drain_background_compression_jobs(cache_b)
        assert cache_b["key_b"] == "val_b"
        with runner_mod._cache_lock:
            assert not runner_mod._pending_background_compression_jobs

    def test_drain_runs_before_cache_snapshot_so_warm_up_visible(self):
        """Drain must happen before the local_compression_cache snapshot so that
        background warm-up results are included in the current turn's cache (#252)."""
        from src.orchestration import runner as runner_mod

        # Simulate a completed warm-up job that merged into persistent cache.
        warm_up_key = "tool_call_252"
        warm_up_value = "Pre-compressed summary from warm-up."

        # Put the finished result directly into the persistent cache (simulates
        # what _drain_background_compression_jobs does after merging).
        with runner_mod._cache_lock:
            runner_mod._persistent_compression_cache[warm_up_key] = warm_up_value

        # Snapshot the cache as run_agent would after the drain.
        runner_mod._drain_background_compression_jobs()  # no-op here; already merged
        with runner_mod._cache_lock:
            local_cache = dict(runner_mod._persistent_compression_cache)

        # The warm-up result must be in the local snapshot.
        assert warm_up_key in local_cache, (
            "Warm-up compression result was not visible in local_compression_cache — "
            "drain likely ran AFTER the snapshot instead of before it"
        )
        assert local_cache[warm_up_key] == warm_up_value

        # Cleanup
        with runner_mod._cache_lock:
            runner_mod._persistent_compression_cache.pop(warm_up_key, None)


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
    """Tests for compress_tool_message and apply_message_compression."""

    def test_compression_skipped_when_none_context(self):
        """max_context_tokens=None skips compression entirely."""
        msgs = _build_compression_messages()
        result = apply_message_compression(
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
        result = apply_message_compression(
            msgs,
            call_count=3,
            compression_cache={},
            llm=MagicMock(),
            max_context_tokens=100_000,  # huge window, won't trigger size threshold
        )
        assert result is msgs

    def test_compression_triggers_on_size_threshold(self):
        """Compression runs when total chars >= 72% of context window."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Short summary."
        mock_llm.invoke.return_value = mock_response

        # 60_000 chars of tool content; context window = 20_000 tokens = 80_000 chars.
        # Threshold = 80_000 * 0.72 = 57_600. Total > 57_600 → triggers.
        # max_context_tokens must be >= 16_384 (small-context guard).
        msgs = _build_compression_messages(tool_content_size=60_000)
        result = apply_message_compression(
            msgs,
            call_count=1,  # not at interval
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=20_000,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any(m.content != "x" * 60_000 for m in tool_msgs)

    def test_young_messages_not_compressed(self):
        """ToolMessages younger than min_age_cycles are preserved."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed."
        mock_llm.invoke.return_value = mock_response

        # Only 2 AI messages after the ToolMessage (age=2 < 6)
        msgs = _build_compression_messages(num_old_ai=2, tool_content_size=5000)
        result = apply_message_compression(
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
        result = apply_message_compression(
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
        cache = {"call_old": "Cached summary."}

        # Use large enough tool content and context window to trigger compression.
        msgs = _build_compression_messages(tool_content_size=60_000)
        result = apply_message_compression(
            msgs,
            call_count=1,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=20_000,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any(m.content == "Cached summary." for m in tool_msgs)
        mock_llm.invoke.assert_not_called()

    def test_compressed_result_stored_in_cache(self):
        """Compressed message content is stored in the cache keyed by tool_call_id."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary of file content."
        mock_llm.invoke.return_value = mock_response

        # Use large enough tool content and context window to trigger compression.
        msgs = _build_compression_messages(tool_content_size=60_000)
        cache: dict = {}
        apply_message_compression(
            msgs,
            call_count=1,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=20_000,
        )
        assert cache["call_old"] == "Summary of file content."

    def test_original_list_not_mutated(self):
        """The input message list is not modified by compression."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary."
        mock_llm.invoke.return_value = mock_response

        msgs = _build_compression_messages()
        original_contents = [getattr(m, "content", "") for m in msgs]
        apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=500,
        )
        new_contents = [getattr(m, "content", "") for m in msgs]
        assert original_contents == new_contents

    def testcompress_tool_message_fallback(self):
        """compress_tool_message falls back to truncation on LLM failure."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")

        content = "A" * 5000
        result = compress_tool_message(content, "read_file", mock_llm)
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

        # Use large enough tool content and context window to trigger compression.
        msgs = _build_compression_messages(tool_content_size=60_000)
        result = apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=compression_llm,  # dedicated LLM passed here
            max_context_tokens=20_000,
        )
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert any(m.content == "Compressed by dedicated model." for m in tool_msgs)
        compression_llm.invoke.assert_called()
        main_llm.invoke.assert_not_called()

    def test_tool_message_ids_preserved(self):
        """Compressed ToolMessages keep tool_call_id and name."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed shell output summary."
        mock_llm.invoke.return_value = mock_response

        # Use large enough tool content and context window to trigger compression.
        msgs = _build_compression_messages(
            tool_call_id="call_123", tool_name="execute_shell_command", tool_content_size=60_000
        )
        result = apply_message_compression(
            msgs,
            call_count=1,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=20_000,
        )
        compressed_tools = [
            m
            for m in result
            if isinstance(m, ToolMessage) and m.content == "Compressed shell output summary."
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
        result = compress_tool_message(content, "read_file", mock_llm)
        assert result == content


# ---------------------------------------------------------------------------
# _correct_tool_args
# ---------------------------------------------------------------------------


class _ShellSchema(BaseModel):
    cmd: str = Field(description="The command to run")
    timeout: int = Field(default=30, description="Timeout")


class _HeaderSchema(BaseModel):
    url: str = Field(description="URL")
    headers: str | None = Field(default=None, description="Headers as JSON string")


class _LongNameSchema(BaseModel):
    working_directory: str = Field(description="Working directory")
    timeout: int = Field(default=30, description="Timeout")


class TestCorrectToolArgs:
    def test_no_correction_needed(self):
        tool = MagicMock()
        tool.args_schema = _ShellSchema
        args = {"cmd": "ls -la", "timeout": 10}
        assert _correct_tool_args(tool, args) == args

    def test_substring_match_remaps(self):
        """'directory' is a substring of 'working_directory' — should be remapped."""
        tool = MagicMock()
        tool.args_schema = _LongNameSchema
        result = _correct_tool_args(tool, {"directory": "/tmp", "timeout": 10})
        assert result == {"working_directory": "/tmp", "timeout": 10}

    def test_superstring_match_remaps(self):
        """'working_directory_path' contains 'working_directory' — should be remapped."""
        tool = MagicMock()
        tool.args_schema = _LongNameSchema
        result = _correct_tool_args(tool, {"working_directory_path": "/tmp", "timeout": 10})
        assert result == {"working_directory": "/tmp", "timeout": 10}

    def test_close_fuzzy_match_remaps(self):
        """'header' vs 'headers' has ratio 0.92 — should be remapped."""
        tool = MagicMock()
        tool.args_schema = _HeaderSchema
        result = _correct_tool_args(tool, {"url": "http://x.com", "header": "{}'"})
        assert result == {"url": "http://x.com", "headers": "{}'"}

    def test_low_ratio_no_remap(self):
        """'cmd' vs 'working_directory' has very low ratio — should NOT remap."""
        tool = MagicMock()
        tool.args_schema = _LongNameSchema
        result = _correct_tool_args(tool, {"cmd": "/tmp", "timeout": 10})
        assert "cmd" in result  # not remapped

    def test_no_schema_returns_unchanged(self):
        tool = MagicMock(spec=[])  # no args_schema attribute
        args = {"cmd": "ls"}
        assert _correct_tool_args(tool, args) == args

    def test_ambiguous_match_no_remap(self):
        """If unknown key matches multiple expected fields, leave it alone."""

        class _AmbiguousSchema(BaseModel):
            command_a: str = ""
            command_b: str = ""

        tool = MagicMock()
        tool.args_schema = _AmbiguousSchema
        result = _correct_tool_args(tool, {"command": "x"})
        assert "command" in result  # not remapped

    def test_type_coercion_dict_to_str(self):
        """Schema expects str but LLM sent dict — should be JSON-encoded."""
        tool = MagicMock()
        tool.args_schema = _HeaderSchema
        result = _correct_tool_args(
            tool, {"url": "http://example.com", "headers": {"Authorization": "Bearer tok"}}
        )
        assert result["url"] == "http://example.com"
        assert isinstance(result["headers"], str)
        assert "Bearer tok" in result["headers"]

    def test_type_coercion_str_list_joined(self):
        """Schema expects str but LLM sent list of strings — should be space-joined."""
        tool = MagicMock()
        tool.args_schema = _ShellSchema
        result = _correct_tool_args(tool, {"cmd": ["ls", "-la"], "timeout": 10})
        assert result["cmd"] == "ls -la"

    def test_type_coercion_mixed_list_json(self):
        """Schema expects str but LLM sent list with non-strings — should be JSON-encoded."""
        tool = MagicMock()
        tool.args_schema = _ShellSchema
        result = _correct_tool_args(tool, {"cmd": ["echo", 42], "timeout": 10})
        assert isinstance(result["cmd"], str)
        assert "42" in result["cmd"]
        assert result["cmd"].startswith("[")  # JSON array

    def test_combined_remap_and_coerce(self):
        """Both rename and type coercion in one call."""
        tool = MagicMock()
        tool.args_schema = _LongNameSchema
        result = _correct_tool_args(tool, {"directory": ["/tmp", "/var"], "timeout": 10})
        assert "working_directory" in result
        assert "directory" not in result
        assert result["working_directory"] == "/tmp /var"

    def test_empty_args(self):
        tool = MagicMock()
        tool.args_schema = _ShellSchema
        assert _correct_tool_args(tool, {}) == {}

    def test_alias_resolution(self):
        """Pydantic alias is resolved to canonical field name before fuzzy match."""

        class _AliasSchema(BaseModel):
            command: str = Field(description="Command", alias="cmd")
            timeout: int = Field(default=30)

            model_config = {"populate_by_name": True}

        tool = MagicMock()
        tool.args_schema = _AliasSchema
        result = _correct_tool_args(tool, {"cmd": "ls -la", "timeout": 10})
        assert result == {"command": "ls -la", "timeout": 10}

    def test_alias_no_conflict(self):
        """Alias is not applied when the canonical name is already present."""

        class _AliasSchema(BaseModel):
            command: str = Field(description="Command", alias="cmd")

            model_config = {"populate_by_name": True}

        tool = MagicMock()
        tool.args_schema = _AliasSchema
        result = _correct_tool_args(tool, {"command": "ls", "cmd": "pwd"})
        # canonical already present — alias key stays unchanged
        assert result["command"] == "ls"
        assert result["cmd"] == "pwd"

    def test_validation_alias_resolution(self):
        """Pydantic validation_alias is resolved to canonical field name."""

        class _ValAliasSchema(BaseModel):
            command: str = Field(description="Command", validation_alias="cmd")

        tool = MagicMock()
        tool.args_schema = _ValAliasSchema
        result = _correct_tool_args(tool, {"cmd": "ls -la"})
        assert result == {"command": "ls -la"}

    def test_schema_cache_reused_across_recreated_mcp_schema(self):
        """Equivalent recreated schemas should reuse one cache entry (BUG-521)."""
        from pydantic import create_model

        snapshot = dict(_tool_arg_schema_cache)
        _tool_arg_schema_cache.clear()
        try:
            schema_a = create_model("ReconnectSchemaA", command=(str, ...), timeout=(int, 30))
            schema_b = create_model("ReconnectSchemaB", command=(str, ...), timeout=(int, 30))

            tool = MagicMock()
            tool.name = "mcp_shell"
            tool.args_schema = schema_a
            _correct_tool_args(tool, {"command": "ls", "timeout": 10})
            assert len(_tool_arg_schema_cache) == 1

            tool.args_schema = schema_b
            _correct_tool_args(tool, {"command": "pwd", "timeout": 10})
            assert len(_tool_arg_schema_cache) == 1
            assert ("mcp_shell", ("command", "timeout")) in _tool_arg_schema_cache
        finally:
            _tool_arg_schema_cache.clear()
            _tool_arg_schema_cache.update(snapshot)

    def test_schema_cache_key_scopes_by_tool_name(self):
        """Different tool names should keep separate entries for same field set."""
        from pydantic import create_model

        snapshot = dict(_tool_arg_schema_cache)
        _tool_arg_schema_cache.clear()
        try:
            schema = create_model("SharedSchema", command=(str, ...), timeout=(int, 30))

            tool_a = MagicMock()
            tool_a.name = "mcp_shell"
            tool_a.args_schema = schema
            _correct_tool_args(tool_a, {"command": "ls", "timeout": 10})

            tool_b = MagicMock()
            tool_b.name = "mcp_exec"
            tool_b.args_schema = schema
            _correct_tool_args(tool_b, {"command": "pwd", "timeout": 10})

            assert len(_tool_arg_schema_cache) == 2
            assert ("mcp_shell", ("command", "timeout")) in _tool_arg_schema_cache
            assert ("mcp_exec", ("command", "timeout")) in _tool_arg_schema_cache
        finally:
            _tool_arg_schema_cache.clear()
            _tool_arg_schema_cache.update(snapshot)

    def test_schema_introspection_attrerror_logs_warning(self, caplog):
        """A broken schema whose model_fields raises AttributeError should log a warning."""

        class _BrokenBool:
            def __bool__(self):
                raise AttributeError("broken")

        class _BrokenSchema:
            model_fields = _BrokenBool()

        tool = MagicMock()
        tool.name = "broken_tool"
        tool.args_schema = _BrokenSchema()
        args = {"cmd": "ls"}
        with caplog.at_level("WARNING", logger="cogtrix"):
            result = _correct_tool_args(tool, args)
        assert result == args
        assert any(
            "schema introspection failed" in r.message and "broken_tool" in r.message
            for r in caplog.records
        )

    def test_schema_introspection_typeerror_logs_warning(self, caplog):
        """A broken schema whose model_fields raises TypeError should log a warning."""

        class _BrokenBool:
            def __bool__(self):
                raise TypeError("broken")

        class _BrokenSchema:
            model_fields = _BrokenBool()

        tool = MagicMock()
        tool.name = "broken_tool"
        tool.args_schema = _BrokenSchema()
        args = {"cmd": "ls"}
        with caplog.at_level("WARNING", logger="cogtrix"):
            result = _correct_tool_args(tool, args)
        assert result == args
        assert any(
            "schema introspection failed" in r.message and "broken_tool" in r.message
            for r in caplog.records
        )

    def test_schema_introspection_unexpected_error_propagates(self):
        """An unexpected exception during schema introspection should propagate."""

        class _BrokenBool:
            def __bool__(self):
                raise RuntimeError("surprise")

        class _BrokenSchema:
            model_fields = _BrokenBool()

        tool = MagicMock()
        tool.name = "broken_tool"
        tool.args_schema = _BrokenSchema()
        with pytest.raises(RuntimeError, match="surprise"):
            _correct_tool_args(tool, {"cmd": "ls"})


# ---------------------------------------------------------------------------
# Duplicate tool call detection
# ---------------------------------------------------------------------------


class TestDuplicateToolCallDetection:
    """Tests for duplicate tool call detection in process_tools."""

    def test_duplicate_tool_call_returns_cached(self):
        """Second identical tool call should return cached result, not invoke tool again."""
        tool_call_1 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        tool_call_2 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c2"}
        ai_msg_1 = AIMessage(content="", tool_calls=[tool_call_1], id="m1")
        ai_msg_2 = AIMessage(content="", tool_calls=[tool_call_2], id="m2")
        final = AIMessage(content="done", id="m3")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"
        mock_tool.invoke.return_value = ToolMessage(
            content="world", tool_call_id="c1", name="echo_tool"
        )

        mock_llm = _make_mock_llm([ai_msg_1, ai_msg_2, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        # First call: normal result
        assert tool_msgs[0].content == "world"
        # Second call: cached with duplicate prefix
        assert "Duplicate call" in tool_msgs[1].content
        assert "world" in tool_msgs[1].content
        # Tool was only invoked once
        assert mock_tool.invoke.call_count == 1

    def test_different_args_not_duplicate(self):
        """Same tool with different args should NOT be treated as duplicate."""
        call_a = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        call_b = {"name": "echo_tool", "args": {"text": "world"}, "id": "c2"}
        ai_msg_1 = AIMessage(content="", tool_calls=[call_a], id="m1")
        ai_msg_2 = AIMessage(content="", tool_calls=[call_b], id="m2")
        final = AIMessage(content="done", id="m3")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"

        def side_effect(inp, *a, **kw):
            return ToolMessage(
                content=f"echo: {inp['args']['text']}",
                tool_call_id=inp["id"],
                name="echo_tool",
            )

        mock_tool.invoke.side_effect = side_effect

        mock_llm = _make_mock_llm([ai_msg_1, ai_msg_2, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert "Duplicate" not in tool_msgs[0].content
        assert "Duplicate" not in tool_msgs[1].content
        assert mock_tool.invoke.call_count == 2

    def test_custom_object_args_still_deduplicate(self, caplog):
        """BUG-489: structurally equal custom objects should deduplicate deterministically."""

        class Payload:
            def __init__(self, value: str) -> None:
                self.value = value

        call_a = {"name": "echo_tool", "args": {"payload": Payload("hello")}, "id": "c1"}
        call_b = {"name": "echo_tool", "args": {"payload": Payload("hello")}, "id": "c2"}
        ai_msg_1 = AIMessage(content="", tool_calls=[call_a], id="m1")
        ai_msg_2 = AIMessage(content="", tool_calls=[call_b], id="m2")
        final = AIMessage(content="done", id="m3")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"
        mock_tool.invoke.return_value = ToolMessage(
            content="world", tool_call_id="c1", name="echo_tool"
        )

        mock_llm = _make_mock_llm([ai_msg_1, ai_msg_2, final])

        with caplog.at_level("WARNING", logger="cogtrix"):
            graph = _build_agent_graph(
                llm=mock_llm,
                system_prompt="",
                active_tools_list=[mock_tool],
                available_tools={},
                registry=_make_registry(),
                approvals=set(),
            )
            result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert "Duplicate call" in tool_msgs[1].content
        assert "world" in tool_msgs[1].content
        assert mock_tool.invoke.call_count == 1
        assert not any(
            "serialization failed" in record.message.lower() for record in caplog.records
        )

    def test_request_tools_exempt_from_dedup(self):
        """request_tools calls should never be deduplicated."""
        from src.tools.configure import create_request_tools_tool

        call_1 = {"name": "request_tools", "args": {}, "id": "c1"}
        call_2 = {"name": "request_tools", "args": {}, "id": "c2"}
        ai_msg_1 = AIMessage(content="", tool_calls=[call_1], id="m1")
        ai_msg_2 = AIMessage(content="", tool_calls=[call_2], id="m2")
        final = AIMessage(content="done", id="m3")

        rt_tool = create_request_tools_tool({}, {})
        mock_llm = _make_mock_llm([ai_msg_1, ai_msg_2, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[rt_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        # Neither should be flagged as duplicate
        for msg in tool_msgs:
            assert "Duplicate" not in msg.content

    def test_parallel_duplicate_tool_call_invoked_once(self):
        """Two identical parallel tool calls must invoke the tool only once (BUG-1293)."""
        call_1 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        call_2 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c2"}
        ai_msg = AIMessage(content="", tool_calls=[call_1, call_2], id="m1")
        final = AIMessage(content="done", id="m2")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"
        mock_tool.invoke.return_value = ToolMessage(
            content="world", tool_call_id="c1", name="echo_tool"
        )

        mock_llm = _make_mock_llm([ai_msg, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert mock_tool.invoke.call_count == 1  # BUG-1293 fix validation
        # One result should be the cached duplicate
        assert any("Duplicate call" in m.content for m in tool_msgs)
        assert all("world" in m.content for m in tool_msgs)

    def test_parallel_duplicate_tool_call_with_slow_invoke(self):
        """Wide race window: slow tool still invoked only once (BUG-1293)."""
        call_1 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        call_2 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c2"}
        ai_msg = AIMessage(content="", tool_calls=[call_1, call_2], id="m1")
        final = AIMessage(content="done", id="m2")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"

        def _slow_invoke(inp, *a, **kw):
            time.sleep(0.05)  # widen the TOCTOU window
            return ToolMessage(content="world", tool_call_id=inp["id"], name="echo_tool")

        mock_tool.invoke.side_effect = _slow_invoke

        mock_llm = _make_mock_llm([ai_msg, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert mock_tool.invoke.call_count == 1  # BUG-1293 fix validation
        assert any("Duplicate call" in m.content for m in tool_msgs)
        assert all("world" in m.content for m in tool_msgs)

    def test_parallel_duplicate_tool_call_exception_cleans_pending(self):
        """If the first parallel call raises, the sentinel is removed so retries work."""
        call_1 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c1"}
        call_2 = {"name": "echo_tool", "args": {"text": "hello"}, "id": "c2"}
        ai_msg = AIMessage(content="", tool_calls=[call_1, call_2], id="m1")
        final = AIMessage(content="done", id="m2")

        mock_tool = MagicMock()
        mock_tool.name = "echo_tool"

        _call_count = 0

        def _flaky_invoke(inp, *a, **kw):
            nonlocal _call_count
            _call_count += 1
            if _call_count == 1:
                raise RuntimeError("first call fails")
            return ToolMessage(content="world", tool_call_id=inp["id"], name="echo_tool")

        mock_tool.invoke.side_effect = _flaky_invoke

        mock_llm = _make_mock_llm([ai_msg, final])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        # One call failed, the other succeeded; second should NOT be blocked by stale sentinel
        assert mock_tool.invoke.call_count == 2
        success_msgs = [m for m in tool_msgs if "world" in m.content]
        assert len(success_msgs) == 1
        error_msgs = [m for m in tool_msgs if "Error executing" in m.content]
        assert len(error_msgs) == 1


class TestIdenticalErrorStuckDetection:
    """Tests for identical-error stuck detection in process_tools."""

    def test_repeated_identical_error_tool_calls_trigger_hint_and_break(self):
        """Same failing call repeated 3x should hint on 2nd hit and force a break."""
        repeated_call_1 = {
            "name": "merge_pull_request",
            "args": {"pull_number": 149},
            "id": "c1",
        }
        repeated_call_2 = {**repeated_call_1, "id": "c2"}
        repeated_call_3 = {**repeated_call_1, "id": "c3"}
        ai_msg_1 = AIMessage(content="", tool_calls=[repeated_call_1], id="m1")
        ai_msg_2 = AIMessage(content="", tool_calls=[repeated_call_2], id="m2")
        ai_msg_3 = AIMessage(content="", tool_calls=[repeated_call_3], id="m3")
        break_response = AIMessage(content="Recovered after thinking break", id="m4")

        mock_tool = MagicMock()
        mock_tool.name = "merge_pull_request"
        mock_tool.invoke.return_value = ToolMessage(
            content="Error: Repository rule violations found.",
            tool_call_id="c1",
            name="merge_pull_request",
        )

        mock_llm = _make_mock_llm([ai_msg_1, ai_msg_2, ai_msg_3, break_response])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any("You've tried this exact action 2 times" in m.content for m in tool_msgs)
        assert any("repository rule violations" in m.content.lower() for m in tool_msgs)
        assert mock_llm.invoke.call_count == 4

        fourth_call_messages = mock_llm.invoke.call_args_list[3].args[0]
        assert any(
            "THINKING BREAK" in getattr(m, "content", "")
            for m in fourth_call_messages
            if isinstance(m, HumanMessage)
        )
        assert any(
            "Recovered after thinking break" in getattr(m, "content", "")
            for m in result["messages"]
            if isinstance(m, AIMessage)
        )


# ── extend_run wiring ─────────────────────────────────────────────────────────


class TestExtendRunWiring:
    """extend_run tool is injected into the graph when extend_run_state is provided."""

    def test_extend_run_tool_is_added_when_state_provided(self):
        from src.tools.extend_run import ExtendRunState

        state = ExtendRunState()
        active: list = []
        available: dict = {}
        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])

        _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=active,
            available_tools=available,
            registry=_make_registry(),
            approvals=set(),
            extend_run_state=state,
        )

        assert "extend_run" in available
        assert any(getattr(t, "name", "") == "extend_run" for t in active)

    def test_extend_run_tool_is_not_duplicated_when_already_present(self):
        from src.tools.extend_run import ExtendRunState

        state = ExtendRunState()
        existing_extend_tool = SimpleNamespace(name="extend_run")
        active: list = [existing_extend_tool]
        available: dict = {}
        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])

        _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=active,
            available_tools=available,
            registry=_make_registry(),
            approvals=set(),
            extend_run_state=state,
        )

        assert active.count(existing_extend_tool) == 1
        assert sum(1 for tool in active if getattr(tool, "name", "") == "extend_run") == 1
        assert "extend_run" in available

    def test_extend_run_tool_absent_without_state(self):
        active: list = []
        available: dict = {}
        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])

        _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=active,
            available_tools=available,
            registry=_make_registry(),
            approvals=set(),
        )

        assert "extend_run" not in available
        assert not any(getattr(t, "name", "") == "extend_run" for t in active)

    def test_extend_run_sets_state_requested(self):
        from src.tools.extend_run import ExtendRunState

        state = ExtendRunState()
        available: dict = {}
        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])

        _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools=available,
            registry=_make_registry(),
            approvals=set(),
            extend_run_state=state,
        )

        tool = available["extend_run"]
        tool.invoke({"mode": "continue", "reason": "need more steps"})

        assert state.requested is True
        assert state.mode == "continue"

    def test_reset_for_new_run_updates_extend_run_state(self):
        from src.tools.extend_run import ExtendRunState

        state1 = ExtendRunState()
        available: dict = {}
        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools=available,
            registry=_make_registry(),
            approvals=set(),
            extend_run_state=state1,
        )

        # Simulate a second run with a fresh state.
        state2 = ExtendRunState()
        graph._reset_for_new_run({}, {}, {}, extend_run_state=state2)

        # The tool should now write into state2, not state1.
        available["extend_run"].invoke({"mode": "continue"})
        assert state2.requested is True
        assert state1.requested is False


# ---------------------------------------------------------------------------
# TestResetForNewRun
# ---------------------------------------------------------------------------


class TestResetForNewRun:
    """Regression tests for #1292 — _reset_for_new_run must zero every PerRunState field."""

    def test_reset_for_new_run_zeros_all_per_run_state_fields(self):
        """Dirties every field, calls _reset_for_new_run, and asserts all are reset."""
        from dataclasses import fields

        from src.orchestration.graph import PerRunState

        mock_llm = _make_mock_llm([AIMessage(content="done", id="m1")])
        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )

        state = graph._per_run_state[0]

        # Capture the pristine initial values before we dirt anything.
        # tool_lookup / active_names / tool_catalog are rebuilt from
        # active_tools_list, which may contain auto-injected tools (e.g.
        # checkpoint), so we compare against the initial state rather than
        # hard-coding empty collections.
        initial_lookup = dict(state.tool_lookup)
        initial_names = set(state.active_names)
        initial_catalog = dict(state.tool_catalog)

        # Dirt every scalar counter / flag.
        state.phantom_count[0] = 99
        state.fabrication_count[0] = 99
        state.action_intent_count[0] = 99
        state.incompleteness_nudge_given[0] = 99
        state.expansion_count[0] = 99
        state.auto_expansion_count[0] = 99
        state.call_count[0] = 99
        state.last_input_tokens[0] = 99
        state.request_tools_noop_count[0] = 99
        state.tool_version[0] = 42
        state.last_tool_version[0] = 42
        state.last_reflection_at[0] = 99
        state.last_tool_health_check_at[0] = 99
        state.stuck_threshold_calibrated[0] = True
        state.stuck_no_checkpoint_threshold[0] = 99
        state.consecutive_errors[0] = 99
        state.force_thinking_break[0] = True
        state.consecutive_identical_error_count[0] = 99
        state.last_identical_error_signature[0] = ("a", "b")
        state.last_checkpoint_count[0] = 99
        state.rounds_since_checkpoint[0] = 99
        state.calls_since_last_checkpoint[0] = 99

        # Dirt every collection.
        state.tool_call_history["key"] = "value"
        state.tool_call_counts["key"] = 99
        state.same_file_writes["file"] = 99
        state.bound_cache["key"] = "value"
        state.compression_cache["key"] = "value"
        state.tool_lookup["key"] = "value"
        state.active_names.add("key")
        state.tool_catalog["key"] = "value"
        state.available_tools_ref[0] = {"dirty": True}

        graph._reset_for_new_run({}, OrderedDict(), {})

        # Verify every field against its expected post-reset value.
        for f in fields(PerRunState):
            actual = getattr(state, f.name)
            if f.name == "tool_version":
                # Should be incremented, not zeroed.
                assert actual[0] == 43, f"{f.name} should be incremented (got {actual[0]})"
            elif f.name == "tool_lookup":
                assert (
                    actual == initial_lookup
                ), f"{f.name} was not rebuilt correctly (got {actual!r})"
            elif f.name == "active_names":
                assert (
                    actual == initial_names
                ), f"{f.name} was not rebuilt correctly (got {actual!r})"
            elif f.name == "tool_catalog":
                assert (
                    actual == initial_catalog
                ), f"{f.name} was not rebuilt correctly (got {actual!r})"
            elif f.name == "available_tools_ref":
                assert actual == [{}], f"{f.name} should be [{{}}] (got {actual!r})"
            elif f.name == "bound_cache":
                assert (
                    actual == OrderedDict()
                ), f"{f.name} should be empty OrderedDict (got {actual!r})"
            elif f.name == "compression_cache":
                assert actual == {}, f"{f.name} should be empty dict (got {actual!r})"
            else:
                expected = getattr(PerRunState(), f.name)
                assert actual == expected, f"{f.name} was not reset: {actual!r} != {expected!r}"


# ---------------------------------------------------------------------------
# TestToolHealthCheck
# ---------------------------------------------------------------------------


class TestToolHealthCheck:
    """Tests for Layer-4 periodic tool-state verification (#383)."""

    def _recording_mock_llm(self, responses: list):
        """Return a mock LLM and a list that records every (messages,) call."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        recorded: list[list] = []

        def _invoke(messages, config=None):
            recorded.append(list(messages))
            return responses.pop(0)

        mock_llm.invoke.side_effect = _invoke
        return mock_llm, recorded

    def test_injects_at_configured_interval(self):
        """Tool-state verification SystemMessage is injected every N turns."""
        responses = [AIMessage(content="ok", id=f"m{i}") for i in range(22)]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_health_check_interval=5),
        )

        for i in range(22):
            graph.invoke({"messages": [HumanMessage(content=f"turn {i}", id=f"h{i}")]})

        # Interval=5, so injection happens at call_count 5, 10, 15, 20
        # (call_count>1 and call_count%5==0)
        injection_turns = []
        for idx, msgs in enumerate(recorded):
            for m in msgs:
                if getattr(m, "type", None) == "system" and "Tool-state verification" in getattr(
                    m, "content", ""
                ):
                    injection_turns.append(idx + 1)  # call_count is 1-based
                    break

        assert injection_turns == [5, 10, 15, 20]

    def test_disabled_when_interval_is_zero(self):
        """When tool_health_check_interval=0, no verification messages are injected."""
        responses = [AIMessage(content="ok", id=f"m{i}") for i in range(10)]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_health_check_interval=0),
        )

        for i in range(10):
            graph.invoke({"messages": [HumanMessage(content=f"turn {i}", id=f"h{i}")]})

        for msgs in recorded:
            for m in msgs:
                assert "Tool-state verification" not in getattr(m, "content", "")

    def test_message_lists_active_tools_from_registry(self):
        """The injected message enumerates active tool names from the registry."""
        mock_tool_a = MagicMock()
        mock_tool_a.name = "search_web"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "write_file"

        responses = [AIMessage(content="ok", id=f"m{i}") for i in range(6)]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[mock_tool_a, mock_tool_b],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_health_check_interval=5),
        )

        for i in range(6):
            graph.invoke({"messages": [HumanMessage(content=f"turn {i}", id=f"h{i}")]})

        # Turn 5 should have the injection
        msgs_turn_5 = recorded[4]
        verification_msgs = [
            m
            for m in msgs_turn_5
            if getattr(m, "type", None) == "system"
            and "Tool-state verification" in getattr(m, "content", "")
        ]
        assert len(verification_msgs) == 1
        content = verification_msgs[0].content
        assert "search_web" in content
        assert "write_file" in content
        assert "enumerated from the system registry" in content

    def test_counter_resets_on_new_run(self):
        """_reset_for_new_run clears the tool-health-check counter."""
        responses = [AIMessage(content="ok", id=f"m{i}") for i in range(12)]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_health_check_interval=5),
        )

        # 4 turns — no injection yet (need 5)
        for i in range(4):
            graph.invoke({"messages": [HumanMessage(content=f"turn {i}", id=f"h{i}")]})

        # Reset — counter should go back to 0
        graph._reset_for_new_run({}, {}, {})

        # 5 more turns — injection should fire on the 5th turn after reset
        for i in range(4, 9):
            graph.invoke({"messages": [HumanMessage(content=f"turn {i}", id=f"h{i}")]})

        injection_turns = []
        for idx, msgs in enumerate(recorded):
            for m in msgs:
                if getattr(m, "type", None) == "system" and "Tool-state verification" in getattr(
                    m, "content", ""
                ):
                    injection_turns.append(idx + 1)
                    break

        # Without reset, injection would have fired at turn 5.
        # With reset, first injection should be at turn 9 (4 before reset + 5 after).
        assert injection_turns == [9]


# ---------------------------------------------------------------------------
# TestToolQualityGate
# ---------------------------------------------------------------------------


class TestToolQualityGate:
    """Tests for Layer-3 tool output quality gate (#382)."""

    def _recording_mock_llm(self, responses: list):
        """Return a mock LLM and a list that records every (messages,) call."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        recorded: list[list] = []

        def _invoke(messages, config=None):
            recorded.append(list(messages))
            return responses.pop(0)

        mock_llm.invoke.side_effect = _invoke
        return mock_llm, recorded

    def _has_quality_gate(self, msgs: list) -> bool:
        """Return True if the message list contains the quality gate nudge."""
        for m in msgs:
            if getattr(
                m, "type", None
            ) == "system" and "All tools returned no data this turn" in getattr(m, "content", ""):
                return True
        return False

    def test_injects_when_all_tools_empty(self):
        """Quality gate nudge is injected when every ToolMessage is substanceless."""
        responses = [
            AIMessage(
                content="", tool_calls=[{"id": "tc1", "name": "search_web", "args": {}}], id="m1"
            ),
            AIMessage(content="I found nothing.", id="m2"),
        ]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_quality_gate_enabled=True),
        )

        # First invoke: model asks for tool call -> graph routes to process_tools
        # We need to simulate process_tools returning empty ToolMessages
        state = {
            "messages": [
                HumanMessage(content="find hiring managers", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "search_web", "args": {"query": "hm"}}],
                    id="m1",
                ),
                ToolMessage(content="", tool_call_id="tc1", name="search_web"),
            ]
        }
        graph.invoke(state)

        # The second call_model (after process_tools) should have the nudge
        assert len(recorded) >= 1
        assert self._has_quality_gate(recorded[0])

    def test_no_inject_when_one_tool_has_content(self):
        """Quality gate does NOT fire when at least one tool returns real content."""
        responses = [
            AIMessage(
                content="", tool_calls=[{"id": "tc1", "name": "search_web", "args": {}}], id="m1"
            ),
            AIMessage(content="Here is what I found.", id="m2"),
        ]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_quality_gate_enabled=True),
        )

        state = {
            "messages": [
                HumanMessage(content="find hiring managers", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tc1", "name": "search_web", "args": {"query": "hm"}},
                        {"id": "tc2", "name": "read_file", "args": {"path": "/tmp/x"}},
                    ],
                    id="m1",
                ),
                ToolMessage(content="Error: no results", tool_call_id="tc1", name="search_web"),
                ToolMessage(
                    content="This is a real result with substance.",
                    tool_call_id="tc2",
                    name="read_file",
                ),
            ]
        }
        graph.invoke(state)

        assert len(recorded) >= 1
        assert not self._has_quality_gate(recorded[0])

    def test_no_inject_when_no_tools_called(self):
        """Quality gate does NOT fire on a plain text turn with no ToolMessages."""
        responses = [AIMessage(content="Hello, how can I help?", id="m1")]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_quality_gate_enabled=True),
        )

        state = {"messages": [HumanMessage(content="hello", id="h1")]}
        graph.invoke(state)

        assert len(recorded) == 1
        assert not self._has_quality_gate(recorded[0])

    def test_disabled_when_flag_is_false(self):
        """When tool_quality_gate_enabled=False, no nudge is injected."""
        responses = [
            AIMessage(
                content="", tool_calls=[{"id": "tc1", "name": "search_web", "args": {}}], id="m1"
            ),
            AIMessage(content="I found nothing.", id="m2"),
        ]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_quality_gate_enabled=False),
        )

        state = {
            "messages": [
                HumanMessage(content="find hiring managers", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "search_web", "args": {"query": "hm"}}],
                    id="m1",
                ),
                ToolMessage(content="", tool_call_id="tc1", name="search_web"),
            ]
        }
        graph.invoke(state)

        assert len(recorded) >= 1
        assert not self._has_quality_gate(recorded[0])

    def test_injects_for_short_error_prefixes(self):
        """Substanceless detection catches error prefixes and short strings."""
        responses = [
            AIMessage(
                content="", tool_calls=[{"id": "tc1", "name": "search_web", "args": {}}], id="m1"
            ),
            AIMessage(content="No data.", id="m2"),
        ]
        mock_llm, recorded = self._recording_mock_llm(responses)

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(tool_quality_gate_enabled=True),
        )

        state = {
            "messages": [
                HumanMessage(content="search", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "search_web", "args": {}}],
                    id="m1",
                ),
                ToolMessage(content="Error: rate limited", tool_call_id="tc1", name="search_web"),
            ]
        }
        graph.invoke(state)

        assert len(recorded) >= 1
        assert self._has_quality_gate(recorded[0])


class TestTopicSwitchDetection:
    """Tests for automatic summary reset on short topic switches (#353)."""

    def _recording_mock_llm(self, responses: list):
        """Return a mock LLM and a list that records every (messages,) call."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        recorded: list[list] = []

        def _invoke(messages, config=None):
            recorded.append(list(messages))
            return responses.pop(0)

        mock_llm.invoke.side_effect = _invoke
        return mock_llm, recorded

    def _has_topic_switch_nudge(self, msgs: list) -> bool:
        """Return True if the hidden topic-switch nudge is present."""
        return any(
            getattr(msg, "type", None) == "system"
            and "The user has changed topic" in getattr(msg, "content", "")
            for msg in msgs
        )

    def test_short_off_topic_question_resets_summary_and_injects_nudge(self):
        """Short off-topic questions should reset rolling summary state."""
        mock_llm, recorded = self._recording_mock_llm([AIMessage(content="42", id="m1")])
        memory_manager = MagicMock()

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(
                memory_manager=memory_manager,
                topic_switch_detection_enabled=True,
            ),
        )

        state = {
            "messages": [
                HumanMessage(content="Find the hiring manager for Neologix", id="h1"),
                AIMessage(content="Working on it.", id="a1"),
                HumanMessage(content="What's the company size?", id="h2"),
            ]
        }

        graph.invoke(state)

        memory_manager.reset_summary_state.assert_called_once()
        assert len(recorded) == 1
        assert self._has_topic_switch_nudge(recorded[0])

    def test_short_same_topic_follow_up_does_not_reset_summary(self):
        """Short follow-ups about the same topic should keep the rolling summary."""
        mock_llm, recorded = self._recording_mock_llm(
            [AIMessage(content="It is 12 people", id="m1")]
        )
        memory_manager = MagicMock()

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(
                memory_manager=memory_manager,
                topic_switch_detection_enabled=True,
            ),
        )

        state = {
            "messages": [
                HumanMessage(content="Find the hiring manager for Neologix", id="h1"),
                AIMessage(content="Searching the contact list.", id="a1"),
                HumanMessage(content="What is the hiring manager's email?", id="h2"),
            ]
        }

        graph.invoke(state)

        memory_manager.reset_summary_state.assert_not_called()
        assert len(recorded) == 1
        assert not self._has_topic_switch_nudge(recorded[0])

    def test_disabled_flag_skips_detection(self):
        """When disabled, topic-switch detection must not reset memory state."""
        mock_llm, recorded = self._recording_mock_llm([AIMessage(content="12", id="m1")])
        memory_manager = MagicMock()

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
            config=AgentRunConfig(
                memory_manager=memory_manager,
                topic_switch_detection_enabled=False,
            ),
        )

        state = {
            "messages": [
                HumanMessage(content="Find the hiring manager for Neologix", id="h1"),
                AIMessage(content="Working on it.", id="a1"),
                HumanMessage(content="What's the company size?", id="h2"),
            ]
        }

        graph.invoke(state)

        memory_manager.reset_summary_state.assert_not_called()
        assert len(recorded) == 1
        assert not self._has_topic_switch_nudge(recorded[0])
