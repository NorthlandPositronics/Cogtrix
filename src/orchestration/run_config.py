"""Session-constant agent execution configuration dataclass."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.safety import ConfirmationUI
    from src.orchestration.session_state import SessionState


@dataclass
class AgentRunConfig:
    """Session-constant parameters for agent execution.

    Bundled to reduce parameter counts in run_agent / build_agent_graph /
    run_execution_phase and to make omissions visible as AttributeError.

    ``bound_cache`` and ``compression_cache`` are per-session LRU caches.
    When non-None, ``run_agent`` uses them exclusively and skips the
    module-level globals — this prevents cross-session cache poisoning in
    API mode where multiple sessions run concurrently with different LLMs.
    CLI mode leaves both fields as ``None`` and falls back to the module globals.
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
    session_state: SessionState | None = None
    confirmation_ui: ConfirmationUI | None = None
    on_tool_expansion: Any | None = None
    parallel_tool_execution: bool = True
    git_native: bool = False
    tool_context_limit_pct: float = 0.80
    tier_cache_enabled: bool = True
    llm_timeout: int = 180  # per-call LLM request timeout (seconds)
    bound_cache: OrderedDict | None = field(default=None, compare=False, repr=False)
    compression_cache: OrderedDict | None = field(default=None, compare=False, repr=False)
    checkpoint_store: Any | None = None  # CheckpointStore for checkpoint tool
    # ADR-0052 — decision accountability feature flags (populated from app Config)
    decision_accountability_enabled: bool = False
    decision_accountability_report_uncertainty: bool = True
    decision_accountability_min_confidence: float = 7.0
    # Task ownership classifier (populated from app Config)
    task_ownership_classifier_enabled: bool = True
    task_ownership_classifier_llm_fallback: bool = False
    task_ownership_ambiguous_action: str = "ask"
    # Pre-action confirmation gate (populated from app Config)
    pre_action_confirmation_enabled: bool = False
