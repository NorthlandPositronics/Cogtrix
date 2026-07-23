"""Regression test: AgentRunner Protocol must stay in sync with run_agent.

ARCH-037-08: This test catches drift between the Protocol and the concrete
implementation. When run_agent gains or loses a parameter, this test fails
and reminds the developer to update the Protocol (or remove the parameter
from run_agent if it is deprecated).
"""

from __future__ import annotations

import inspect


def test_agent_runner_protocol_core_params_present():
    """Protocol must accept the core positional parameters of run_agent."""
    from src.agent.core import AgentRunner
    from src.orchestration.runner import run_agent

    protocol_sig = inspect.signature(AgentRunner.__call__)
    run_agent_sig = inspect.signature(run_agent)

    core_params = {"user_input", "history_messages", "registry", "approvals", "config"}
    protocol_params = set(protocol_sig.parameters) - {"self"}

    missing = core_params - protocol_params
    assert not missing, (
        f"AgentRunner Protocol is missing core parameters: {missing}\n"
        f"Protocol params: {protocol_params}\n"
        f"run_agent params: {set(run_agent_sig.parameters)}"
    )


def test_run_agent_satisfies_protocol():
    """run_agent must satisfy the AgentRunner Protocol at runtime."""
    from src.agent.core import AgentRunner
    from src.orchestration.runner import run_agent

    assert isinstance(run_agent, AgentRunner), (
        "run_agent does not satisfy the AgentRunner Protocol. "
        "Check that the Protocol's __call__ signature is compatible with run_agent."
    )
