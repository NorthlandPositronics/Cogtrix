"""Session-constant agent execution configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentRunConfig:
    """Session-constant parameters for agent execution.

    Bundled to reduce parameter counts in run_agent / build_agent_graph /
    run_execution_phase and to make omissions visible as AttributeError.
    """

    llm: Any = None
    system_prompt: str | None = None
    available_tools: dict[str, Any] | None = None
    active_tools_list: list[Any] | None = None
    max_context_tokens: int | None = None
    preset_tools: set[str] | None = None
    context_compression: bool = True
    compression_min_age: int | None = None
    compression_min_chars: int | None = None
    compression_llm: Any = None
    tool_call_guard: Any | None = None
    session_state: Any = None
    confirmation_ui: Any | None = None
    on_tool_expansion: Any | None = None
