"""Scenario data model and YAML loader for the quality harness."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScriptStep:
    """One step in a scripted conversation."""

    role: str  # "user" | "llm" | "llm_phantom" | "tool_result"
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""  # only for tool_result steps


@dataclass
class MetricSpec:
    """Evaluation criteria for a scenario."""

    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    task_completed: bool = True
    phantom_calls: bool = False  # False = none expected (harness asserts count == 0)
    max_turns: int = 10


@dataclass
class Scenario:
    """A single quality test scenario."""

    id: str
    complexity: str  # simple | medium | complex
    description: str
    category: str
    system_prompt: str
    tool_names: list[str]
    script: list[ScriptStep]
    metrics: MetricSpec
    tags: list[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return "critical" in self.tags

    @property
    def is_tier1(self) -> bool:
        return "tier1" in self.tags

    @property
    def is_tier2(self) -> bool:
        return "tier2" in self.tags


def _parse_script(raw: list[dict]) -> list[ScriptStep]:
    """Parse a scenario script into ScriptStep objects.

    Handles multi-tool-call LLM steps: when one LLM step declares N tool_calls,
    the N subsequent tool_result steps are matched to them in declaration order
    via a FIFO queue of pending call IDs.
    """
    steps: list[ScriptStep] = []
    pending_ids: deque[str] = deque()

    for raw_step in raw:
        if "user" in raw_step:
            steps.append(ScriptStep(role="user", content=raw_step["user"]))
            pending_ids.clear()

        elif "llm" in raw_step:
            llm_data = raw_step["llm"]
            if isinstance(llm_data, str):
                steps.append(ScriptStep(role="llm", content=llm_data))
                pending_ids.clear()
            else:
                tc_list = []
                pending_ids.clear()
                for tc in llm_data.get("tool_calls", []):
                    call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    tc_list.append(
                        {
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "id": call_id,
                            "type": "tool_call",
                        }
                    )
                    pending_ids.append(call_id)
                steps.append(
                    ScriptStep(
                        role="llm",
                        content=llm_data.get("content", ""),
                        tool_calls=tc_list,
                    )
                )

        elif "tool_result" in raw_step:
            # Pop the oldest pending call_id (FIFO — matches parallel tool calls correctly)
            call_id = pending_ids.popleft() if pending_ids else f"call_{uuid.uuid4().hex[:8]}"
            steps.append(
                ScriptStep(
                    role="tool_result",
                    content=raw_step["tool_result"],
                    tool_call_id=call_id,
                )
            )

        elif "llm_phantom" in raw_step:
            # Deliberate phantom markup — used to test phantom detection scenarios
            steps.append(ScriptStep(role="llm_phantom", content=raw_step["llm_phantom"]))

    return steps


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from a YAML file."""
    raw = yaml.safe_load(path.read_text())

    metrics_raw = raw.get("metrics", {})
    metrics = MetricSpec(
        required_tools=metrics_raw.get("required_tools", []),
        forbidden_tools=metrics_raw.get("forbidden_tools", []),
        task_completed=metrics_raw.get("task_completed", True),
        phantom_calls=metrics_raw.get("phantom_calls", False),
        max_turns=metrics_raw.get("max_turns", 10),
    )

    return Scenario(
        id=raw["id"],
        complexity=raw.get("complexity", "simple"),
        description=raw.get("description", ""),
        category=raw.get("category", "general"),
        system_prompt=raw.get("system_prompt", "You are a helpful assistant."),
        tool_names=raw.get("tools", []),
        script=_parse_script(raw.get("script", [])),
        metrics=metrics,
        tags=raw.get("tags", []),
    )


def load_all_scenarios(base_dir: Path | None = None) -> list[Scenario]:
    """Load all .yaml scenario files under tests/quality/scenarios/."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "scenarios"
    scenarios: list[Scenario] = []
    for yaml_file in sorted(base_dir.rglob("*.yaml")):
        scenarios.append(load_scenario(yaml_file))
    return scenarios
