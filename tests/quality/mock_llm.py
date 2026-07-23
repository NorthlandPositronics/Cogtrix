"""Mock LLM and tool stubs for deterministic quality scenario execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from tests.quality.scenario import Scenario

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


def build_mock_llm(scenario: Scenario) -> MagicMock:
    """Build a mock LLM that returns AIMessages in script order.

    Each 'llm' or 'llm_phantom' step becomes one AIMessage returned by
    mock_llm.invoke() in sequence.
    """
    responses = []
    for i, step in enumerate(scenario.script):
        if step.role == "llm":
            responses.append(
                AIMessage(
                    content=step.content or "",
                    tool_calls=step.tool_calls,
                    response_metadata={
                        "finish_reason": "tool_calls" if step.tool_calls else "stop"
                    },
                    id=f"mock_ai_{i}",
                )
            )
        elif step.role == "llm_phantom":
            # Deliberate phantom: raw markup in content, no structured tool_calls
            responses.append(
                AIMessage(
                    content=step.content,
                    tool_calls=[],
                    response_metadata={"finish_reason": "stop"},
                    id=f"mock_phantom_{i}",
                )
            )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = list(responses)
    return mock_llm


# ---------------------------------------------------------------------------
# Tool stubs
# ---------------------------------------------------------------------------


def build_tool_stubs(scenario: Scenario) -> dict[str, Any]:
    """Build mock tool objects keyed by tool name.

    Each stub's invoke() returns a ToolMessage with the scripted content and
    the correct tool_call_id. Multiple calls to the same tool are served
    FIFO from the scenario script.
    """
    # Build call_id → tool_name mapping from llm steps
    call_id_to_tool: dict[str, str] = {}
    for step in scenario.script:
        if step.role == "llm":
            for tc in step.tool_calls:
                call_id_to_tool[tc["id"]] = tc["name"]

    # Build per-tool result queues: name → [(call_id, content), ...]
    queues: dict[str, list[tuple[str, str]]] = {}
    for step in scenario.script:
        if step.role == "tool_result":
            tool_name = call_id_to_tool.get(step.tool_call_id, "unknown")
            queues.setdefault(tool_name, []).append((step.tool_call_id, step.content))

    stubs: dict[str, Any] = {}
    for tool_name in scenario.tool_names:
        queue = list(queues.get(tool_name, []))

        stub = MagicMock()
        stub.name = tool_name
        stub.description = f"Mock {tool_name} tool for quality testing"
        stub.args_schema = MagicMock()

        def _make_invoke(name: str, q: list[tuple[str, str]]) -> Any:
            q_iter = iter(q)

            def invoke(*_args: Any, **_kwargs: Any) -> ToolMessage:
                try:
                    call_id, content = next(q_iter)
                except StopIteration:
                    call_id = f"call_extra_{name}"
                    content = f"[no scripted result for extra call to {name}]"
                return ToolMessage(content=content, tool_call_id=call_id, name=name)

            return invoke

        stub.invoke = _make_invoke(tool_name, queue)
        stub.run = stub.invoke
        stubs[tool_name] = stub

    return stubs
