"""Regression tests for #2124 — run_agent signals failure via a typed exception.

Previously run_agent caught every exception and *returned* ``format_agent_error``
as a string, indistinguishable from a normal answer (so the API returned a 200
with the error text as the reply). It now raises :class:`AgentExecutionError`
carrying the formatted message, so callers can render a proper error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogtrix_core.agent.safety import AgentExecutionError, UserCancelledRun
from cogtrix_core.orchestration.run_config import AgentRunConfig
from cogtrix_core.orchestration.runner import run_agent


class _ProviderBoom(Exception):
    """Stand-in for a provider SDK error (e.g. openai.BadRequestError)."""


def _raising_llm(exc: Exception) -> MagicMock:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = exc
    return llm


def _registry() -> MagicMock:
    r = MagicMock()
    r.requires_confirmation.return_value = False
    return r


def _config(llm: MagicMock) -> AgentRunConfig:
    return AgentRunConfig(
        llm=llm,
        system_prompt="You are helpful.",
        available_tools={},
        active_tools_list=[],
    )


def test_run_agent_raises_agent_execution_error_on_provider_failure() -> None:
    boom = _ProviderBoom(
        "Error code: 400 - {'error': {'message': 'qwen3 is not a valid model ID', 'code': 400}}"
    )
    with pytest.raises(AgentExecutionError) as exc_info:
        run_agent("Hello", [], _registry(), set(), config=_config(_raising_llm(boom)))

    err = exc_info.value
    # Carries a display-ready message (the invalid-model branch of
    # format_agent_error) and chains the original cause.
    assert "Invalid model ID" in err.user_message
    assert isinstance(err.__cause__, _ProviderBoom)


def test_run_agent_does_not_return_error_string() -> None:
    """The failure path must raise, never return the error as a normal answer."""
    boom = _ProviderBoom("some provider failure")
    result = None
    try:
        result = run_agent("Hi", [], _registry(), set(), config=_config(_raising_llm(boom)))
    except AgentExecutionError:
        pass
    assert result is None, "run_agent returned a value on failure instead of raising"


def test_user_cancelled_run_still_propagates_unchanged() -> None:
    """UserCancelledRun must propagate as-is (not wrapped in AgentExecutionError)."""
    with pytest.raises(UserCancelledRun):
        run_agent("Hello", [], _registry(), set(), config=_config(_raising_llm(UserCancelledRun())))
