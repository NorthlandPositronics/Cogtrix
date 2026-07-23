"""Regression tests for the MCP tools_ready gate in the agent graph."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage


class RecordingEvent:
    """Minimal event double that records wait calls without blocking."""

    def __init__(self, *, ready: bool, wait_result: bool) -> None:
        self._ready = ready
        self._wait_result = wait_result
        self.wait_calls: list[float | None] = []

    def is_set(self) -> bool:
        return self._ready

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_calls.append(timeout)
        self._ready = self._wait_result
        return self._wait_result


def test_call_model_waits_for_tools_ready_before_binding():
    from cogtrix_core.orchestration.graph import build_agent_graph
    from cogtrix_core.orchestration.run_config import AgentRunConfig

    event = RecordingEvent(ready=False, wait_result=True)
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="done")

    graph = build_agent_graph(
        config=AgentRunConfig(
            llm=llm,
            active_tools_list=[SimpleNamespace(name="alpha")],
            tools_ready=event,  # type: ignore[arg-type]
        )
    )

    result = graph.invoke({"messages": [HumanMessage(content="hello")]})

    assert event.wait_calls == [5.0]
    assert llm.bind_tools.called is True
    assert result["messages"][-1].content == "done"


def test_call_model_returns_retry_message_when_tools_still_not_ready():
    from cogtrix_core.orchestration.graph import build_agent_graph
    from cogtrix_core.orchestration.run_config import AgentRunConfig

    event = RecordingEvent(ready=False, wait_result=False)
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="done")

    graph = build_agent_graph(
        config=AgentRunConfig(
            llm=llm,
            active_tools_list=[SimpleNamespace(name="alpha")],
            tools_ready=event,  # type: ignore[arg-type]
        )
    )

    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    last = result["messages"][-1]

    assert event.wait_calls == [5.0]
    assert llm.bind_tools.called is False
    assert "reconnecting" in getattr(last, "content", "").lower()
