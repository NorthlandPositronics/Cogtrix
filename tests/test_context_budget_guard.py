"""Tests for the per-turn context budget guard in the agent graph."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage


def _make_ai_message(tool_calls=None, content="", input_tokens=None):
    """Create a real AIMessage with optional usage_metadata."""
    usage = (
        {"input_tokens": input_tokens, "output_tokens": 10, "total_tokens": input_tokens + 10}
        if input_tokens
        else None
    )
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage,
        id="test-id",
    )


def test_guard_not_triggered_when_under_threshold():
    """Tool calls proceed normally when context usage is below threshold."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['query']}",
        tool_call_id=payload["id"],
        name="search_web",
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [
        _make_ai_message(
            tool_calls=[
                {"name": "search_web", "args": {"query": "q1"}, "id": "c1", "type": "tool_call"}
            ],
            input_tokens=5_000,  # 50% of 10k — under 80%
        ),
        AIMessage(content="Done", id="final"),
    ]

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="search once")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 1
    assert tool_messages
    assert not any(
        "Context budget reached" in (getattr(m, "content", "") or "") for m in result["messages"]
    )


def test_guard_triggered_when_over_threshold():
    """Tool calls are stripped when context usage exceeds threshold."""
    from langchain_core.messages import HumanMessage

    mock_llm = MagicMock()
    over_budget_response = _make_ai_message(
        tool_calls=[
            {"name": "http_get", "args": {"url": "http://x.com"}, "id": "c1", "type": "tool_call"}
        ],
        input_tokens=9_000,  # 90% of 10k — over 80%
    )
    # The guard replaces this with a warning AIMessage that has no tool_calls,
    # so the graph ends without needing a second LLM call.
    # No active_tools_list → call_model uses llm.invoke directly (not bind_tools).
    mock_llm.invoke.return_value = over_budget_response
    mock_llm.bind_tools.return_value.invoke.return_value = over_budget_response

    from cogtrix_core.orchestration.graph import build_agent_graph

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
    )

    result = graph.invoke({"messages": [HumanMessage(content="search for X")]})
    final_msgs = result["messages"]
    last = final_msgs[-1]
    # The guard should have replaced the over-budget tool_call response with a warning
    assert not getattr(last, "tool_calls", []), "tool_calls should be stripped when over budget"
    assert "Context budget reached" in getattr(last, "content", "")


@pytest.mark.parametrize(
    ("input_tokens", "should_warn"),
    [
        (7_999, False),
        (8_000, False),
        (8_001, True),
    ],
)
def test_apply_context_budget_guard_boundary_conditions(input_tokens, should_warn):
    """The helper should only warn when the turn input exceeds the configured budget."""
    from cogtrix_core.orchestration.graph import _apply_context_budget_guard

    response = _make_ai_message(
        tool_calls=[{"name": "search_web", "args": {}, "id": "c1", "type": "tool_call"}],
        input_tokens=input_tokens,
    )
    guarded = _apply_context_budget_guard(
        response,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
    )

    if should_warn:
        assert guarded is not response
        assert not getattr(guarded, "tool_calls", [])
        assert "Context budget reached" in getattr(guarded, "content", "")
        assert getattr(guarded, "response_metadata", {}).get("budget_guard") is True
    else:
        assert guarded is response
        assert getattr(guarded, "tool_calls", [])


def test_guard_not_triggered_at_exact_threshold():
    """Tool calls proceed when context usage is exactly at the threshold."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['query']}",
        tool_call_id=payload["id"],
        name="search_web",
    )

    exact_threshold_response = _make_ai_message(
        tool_calls=[
            {"name": "search_web", "args": {"query": "q1"}, "id": "c1", "type": "tool_call"}
        ],
        input_tokens=8_000,  # exactly 80% of 10k — should not trigger the guard
    )
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [
        exact_threshold_response,
        AIMessage(content="Done", id="final"),
    ]

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="search once")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 1
    assert tool_messages
    assert not any(
        "Context budget reached" in (getattr(m, "content", "") or "") for m in result["messages"]
    )


def test_guard_skipped_when_no_usage_metadata():
    """Guard is a no-op when usage_metadata is absent (graceful degradation)."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['query']}",
        tool_call_id=payload["id"],
        name="search_web",
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    # Response with no usage_metadata — guard must not fire
    response_no_usage = _make_ai_message(
        tool_calls=[
            {"name": "search_web", "args": {"query": "q1"}, "id": "c1", "type": "tool_call"}
        ],
        input_tokens=None,  # usage_metadata=None
    )
    assert response_no_usage.usage_metadata is None

    mock_llm.invoke.side_effect = [
        response_no_usage,
        AIMessage(content="Done", id="final"),
    ]

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="search once")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 1
    assert tool_messages
    assert not any(
        "Context budget reached" in (getattr(m, "content", "") or "") for m in result["messages"]
    )


def test_guard_skipped_when_no_max_context_tokens():
    """Guard is a no-op when max_context_tokens is not set."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['query']}",
        tool_call_id=payload["id"],
        name="search_web",
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [
        _make_ai_message(
            tool_calls=[
                {"name": "search_web", "args": {"query": "q1"}, "id": "c1", "type": "tool_call"}
            ],
            input_tokens=900_000,  # huge — but no budget set, so guard must not fire
        ),
        AIMessage(content="Done", id="final"),
    ]

    # max_context_tokens=None → guard is completely bypassed
    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=None,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="search once")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 1
    assert tool_messages
    assert not any(
        "Context budget reached" in (getattr(m, "content", "") or "") for m in result["messages"]
    )


def test_config_default_pct():
    """Default tool_context_limit_pct is 0.80."""
    from cogtrix_core.config import Config

    c = Config()
    assert c.tool_context_limit_pct == 0.80


def test_config_custom_pct():
    """tool_context_limit_pct can be set in config."""
    from cogtrix_core.config import Config

    c = Config(tool_context_limit_pct=0.70)
    assert c.tool_context_limit_pct == 0.70


def test_agent_run_config_default_pct():
    """AgentRunConfig has tool_context_limit_pct defaulting to 0.80."""
    from cogtrix_core.orchestration.run_config import AgentRunConfig

    cfg = AgentRunConfig()
    assert cfg.tool_context_limit_pct == 0.80


def test_agent_run_config_custom_pct():
    """AgentRunConfig accepts a custom tool_context_limit_pct."""
    from cogtrix_core.orchestration.run_config import AgentRunConfig

    cfg = AgentRunConfig(tool_context_limit_pct=0.60)
    assert cfg.tool_context_limit_pct == 0.60


def test_guard_warning_message_format():
    """Warning message includes percentage and limit values."""
    from langchain_core.messages import HumanMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_llm = MagicMock()
    over_budget = _make_ai_message(
        tool_calls=[{"name": "some_tool", "args": {}, "id": "c1", "type": "tool_call"}],
        input_tokens=8_500,  # 85% of 10k
    )
    # No active_tools_list → call_model uses llm.invoke directly (not bind_tools).
    mock_llm.invoke.return_value = over_budget
    mock_llm.bind_tools.return_value.invoke.return_value = over_budget

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
    )
    result = graph.invoke({"messages": [HumanMessage(content="do stuff")]})
    last = result["messages"][-1]
    content = getattr(last, "content", "")
    assert "85%" in content
    assert "10,000" in content
    assert "80%" in content


def test_search_tools_are_not_hard_stopped_at_eight_calls():
    """Search tools should keep working past the hard cutoff for research."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['query']}",
        tool_call_id=payload["id"],
        name="search_web",
    )

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": f"q{i}"},
                    "id": f"c{i}",
                    "type": "tool_call",
                }
            ],
            id=f"m{i}",
        )
        for i in range(1, 10)
    ] + [AIMessage(content="Done", id="final")]
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="search the topic")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 9
    assert not any("budget exhausted" in (m.content or "").lower() for m in tool_messages)
    assert any("result-q9" in (m.content or "") for m in tool_messages)


def test_non_search_tools_still_hit_the_hard_cutoff():
    """The hard budget should still disable ordinary tools after eight calls."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_tool = MagicMock()
    mock_tool.name = "echo_tool"
    mock_tool.invoke.side_effect = lambda payload, _cfg=None: ToolMessage(
        content=f"result-{payload['args']['text']}",
        tool_call_id=payload["id"],
        name="echo_tool",
    )

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "echo_tool",
                    "args": {"text": f"t{i}"},
                    "id": f"e{i}",
                    "type": "tool_call",
                }
            ],
            id=f"m{i}",
        )
        for i in range(1, 10)
    ] + [AIMessage(content="Done", id="final2")]
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
        active_tools_list=[mock_tool],
        available_tools={},
        registry=MagicMock(),
        approvals=set(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="echo repeatedly")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert mock_tool.invoke.call_count == 8
    assert any("per-turn call limit (8 calls)" in (m.content or "").lower() for m in tool_messages)
    assert any("synthesize your findings" in (m.content or "").lower() for m in tool_messages)
    assert any("result-t8" in (m.content or "") for m in tool_messages)
