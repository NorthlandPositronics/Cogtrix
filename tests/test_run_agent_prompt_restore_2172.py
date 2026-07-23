"""Regression test for #2172 — run_agent restores the session-constant system
prompt even when it fails *before* graph execution.

``run_agent`` mutates ``config.system_prompt`` in place to apply per-run
task-ownership / clarification constraints. The restore to the base prompt used
to live only in the inner ``graph.stream`` ``finally``, so an exception raised
earlier (e.g. in ``build_agent_graph``) bypassed it and left the caller's
``AgentRunConfig`` dirty — stacking ``TASK MODE`` constraint blocks on any
cross-turn reuse. The fix moves the restore into an outer ``finally``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.safety import AgentExecutionError
from src.orchestration import runner as _runner
from src.orchestration.intent import OwnershipMode, OwnershipResult
from src.orchestration.run_config import AgentRunConfig


class _ProviderBoom(Exception):
    """Stand-in for a failure raised during graph construction."""


def _registry() -> MagicMock:
    r = MagicMock()
    r.requires_confirmation.return_value = False
    return r


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        llm=MagicMock(),
        system_prompt="You are helpful.",
        available_tools={},
        active_tools_list=[],
    )


def test_system_prompt_restored_after_early_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force an INFORM classification so a TASK MODE constraint is appended to
    # config.system_prompt at the in-place mutation point.
    monkeypatch.setattr(
        _runner,
        "classify_task_ownership",
        lambda *_a, **_k: OwnershipResult(
            mode=OwnershipMode.INFORM,
            confidence=0.95,
            is_reversible=True,
            raw_signal="forced-inform",
            inferred_action="explain something",
        ),
    )
    # Fail *before* the inner graph.stream try — this is exactly the path that
    # skipped the prompt restore prior to #2172.
    monkeypatch.setattr(
        _runner,
        "build_agent_graph",
        MagicMock(side_effect=_ProviderBoom("graph build failed")),
    )

    cfg = _config()
    original_prompt = cfg.system_prompt

    with pytest.raises(AgentExecutionError):
        _runner.run_agent("How do I deploy this?", [], _registry(), set(), config=cfg)

    assert cfg.system_prompt == original_prompt, (
        "config.system_prompt must be restored to its entry value after an early "
        "failure; the per-run TASK MODE constraint leaked into the "
        "session-constant AgentRunConfig"
    )
