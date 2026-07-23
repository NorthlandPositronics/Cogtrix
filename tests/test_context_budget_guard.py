"""Tests for the per-turn context budget guard in the agent graph."""

from unittest.mock import MagicMock

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
    from src.orchestration.graph import build_agent_graph

    mock_llm = MagicMock()
    mock_response = _make_ai_message(
        tool_calls=[{"name": "search_web", "args": {}, "id": "c1", "type": "tool_call"}],
        input_tokens=5_000,  # 50% of 10k — under 80%
    )
    mock_llm.bind_tools.return_value.invoke.return_value = mock_response

    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
    )
    assert graph is not None


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

    from src.orchestration.graph import build_agent_graph

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


def test_guard_skipped_when_no_usage_metadata():
    """Guard is a no-op when usage_metadata is absent (graceful degradation)."""
    from src.orchestration.graph import build_agent_graph

    mock_llm = MagicMock()
    # Response with no usage_metadata — guard must not fire
    response_no_usage = _make_ai_message(
        tool_calls=[{"name": "search_web", "args": {}, "id": "c1", "type": "tool_call"}],
        input_tokens=None,  # usage_metadata=None
    )
    assert response_no_usage.usage_metadata is None

    mock_llm.bind_tools.return_value.invoke.return_value = response_no_usage

    # Guard is bypassed — graph builds successfully
    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=10_000,
        tool_context_limit_pct=0.80,
    )
    assert graph is not None


def test_guard_skipped_when_no_max_context_tokens():
    """Guard is a no-op when max_context_tokens is not set."""
    from src.orchestration.graph import build_agent_graph

    mock_llm = MagicMock()
    mock_response = _make_ai_message(
        tool_calls=[{"name": "search_web", "args": {}, "id": "c1", "type": "tool_call"}],
        input_tokens=900_000,  # huge — but no budget set, so guard must not fire
    )
    mock_llm.bind_tools.return_value.invoke.return_value = mock_response

    # max_context_tokens=None → guard is completely bypassed
    graph = build_agent_graph(
        llm=mock_llm,
        max_context_tokens=None,
        tool_context_limit_pct=0.80,
    )
    assert graph is not None


def test_config_default_pct():
    """Default tool_context_limit_pct is 0.80."""
    from src.config import Config

    c = Config()
    assert c.tool_context_limit_pct == 0.80


def test_config_custom_pct():
    """tool_context_limit_pct can be set in config."""
    from src.config import Config

    c = Config(tool_context_limit_pct=0.70)
    assert c.tool_context_limit_pct == 0.70


def test_agent_run_config_default_pct():
    """AgentRunConfig has tool_context_limit_pct defaulting to 0.80."""
    from src.orchestration.run_config import AgentRunConfig

    cfg = AgentRunConfig()
    assert cfg.tool_context_limit_pct == 0.80


def test_agent_run_config_custom_pct():
    """AgentRunConfig accepts a custom tool_context_limit_pct."""
    from src.orchestration.run_config import AgentRunConfig

    cfg = AgentRunConfig(tool_context_limit_pct=0.60)
    assert cfg.tool_context_limit_pct == 0.60


def test_guard_warning_message_format():
    """Warning message includes percentage and limit values."""
    from langchain_core.messages import HumanMessage

    from src.orchestration.graph import build_agent_graph

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
