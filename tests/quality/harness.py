"""Quality harness — executes a scenario through the real Cogtrix agent graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import BaseMessage, HumanMessage

from tests.quality.mock_llm import build_mock_llm, build_tool_stubs
from tests.quality.scenario import Scenario

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class HarnessResult:
    """Output of running a scenario through the agent graph."""

    trace: list[BaseMessage]  # all messages from graph execution in order
    scenario: Scenario
    turns: int = 0  # number of call_model invocations
    prompt_tokens: int = 0  # total prompt tokens across all turns (if available)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry stub
# ---------------------------------------------------------------------------


def _make_registry(requires_confirmation: bool = False) -> MagicMock:
    """Minimal mock of the tool registry — reuses the pattern from test_agent_graph.py."""
    registry = MagicMock()
    registry.requires_confirmation.return_value = requires_confirmation
    return registry


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario) -> HarnessResult:
    """Run a scenario through the real Cogtrix agent graph with mock LLM and tools.

    Uses build_agent_graph from cogtrix_core.orchestration.graph directly (not the
    cogtrix.py wrapper) so there are no session file or config dependencies.

    Compression is disabled for determinism. Parallel tool execution is
    disabled so tool call order matches the script exactly.
    """
    from cogtrix_core.orchestration.graph import build_agent_graph

    mock_llm = build_mock_llm(scenario)
    tool_stubs = build_tool_stubs(scenario)
    active_tools = list(tool_stubs.values())

    # Count call_model invocations by wrapping invoke
    turns_counter = [0]
    prompt_tokens_counter = [0]
    original_invoke = mock_llm.invoke

    def counting_invoke(*args: Any, **kwargs: Any) -> Any:
        turns_counter[0] += 1
        return original_invoke(*args, **kwargs)

    mock_llm.invoke = counting_invoke
    mock_llm.bind_tools.return_value = mock_llm

    graph = build_agent_graph(
        llm=mock_llm,
        system_prompt=scenario.system_prompt,
        active_tools_list=active_tools,
        available_tools=tool_stubs,
        registry=_make_registry(),
        approvals=set(),
        context_max_messages=max(scenario.metrics.max_turns * 4, 50),
        context_compression=False,  # determinism
        parallel_tool_execution=False,  # ordered execution matches script
        git_native=False,
    )

    # Extract the first user message from the script
    first_user_step = next((s for s in scenario.script if s.role == "user"), None)
    if first_user_step is None:
        raise ValueError(f"Scenario '{scenario.id}' has no user step in script")

    initial_state = {"messages": [HumanMessage(content=first_user_step.content)]}
    output_state = graph.invoke(initial_state)
    trace: list[BaseMessage] = list(output_state.get("messages", []))

    return HarnessResult(
        trace=trace,
        scenario=scenario,
        turns=turns_counter[0],
        prompt_tokens=prompt_tokens_counter[0],
    )
