"""Shared types used across multiple Cogtrix layers.

AgentRunConfig and SessionState live here to break the bidirectional
dependency between src/agent/ and src/orchestration/.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.safety import ConfirmationUI


@dataclass
class SessionState:
    """All mutable state that belongs to one interactive session.

    Lifetime mapping:
    - ``denials``, ``loaded_tools``, ``approvals`` — session-scoped; cleared on session switch.
    - ``deny_all`` — per-prompt; reset at the start of each new prompt cycle.
    - ``pinned_tools`` — session-scoped; only cleared manually or on session switch.
    - ``no_confirm`` — process-scoped; set once at startup from CLI flags.
    - ``all_tool_descriptions``, ``all_tool_originals`` — process-scoped; populated once at startup.

    Tool loading tiers:
    - **Agent-loaded** — tools loaded by the LLM via ``request_tools`` during a turn.
      Tracked in ``loaded_tools``.  Cleared at the start of each new prompt cycle
      so the agent doesn't carry stale tools between turns.
    - **Pinned** — tools loaded manually by the user (``/tools load`` in CLI or
      ``PATCH /sessions/{id}/tools`` in API).  Tracked in ``pinned_tools``.
      Persist across prompt cycles until explicitly unloaded.
    """

    denials: set[str] = field(default_factory=set)
    deny_all: bool = False
    no_confirm: bool = False
    approvals: set[str] = field(default_factory=set)
    loaded_tools: set[str] = field(default_factory=set)
    pinned_tools: set[str] = field(default_factory=set)
    all_tool_descriptions: dict[str, str] = field(default_factory=dict)
    all_tool_originals: dict[str, Any] = field(default_factory=dict)
    checkpoint_store: Any | None = None  # CheckpointStore for checkpoint tool

    # Internal lock — not exposed in repr/equality; guards concurrent denial reads/writes.
    # Tool execution runs in a ThreadPoolExecutor (8 threads); API handlers mutate
    # denials/deny_all from the asyncio event loop thread via asyncio.to_thread.
    # Without this lock, budget-enforcement writes (graph.py) and API disable calls
    # (routes/tools.py) race against safety-wrapper reads (safety.py).
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def is_denied(self, tool_name: str) -> bool:
        """Atomically check deny_all and per-tool denial."""
        with self._lock:
            return self.deny_all or tool_name in self.denials

    def deny_tool(self, tool_name: str) -> None:
        """Atomically add tool_name to per-tool denials."""
        with self._lock:
            self.denials.add(tool_name)

    def allow_tool(self, tool_name: str) -> None:
        """Atomically remove tool_name from per-tool denials."""
        with self._lock:
            self.denials.discard(tool_name)

    def set_deny_all(self) -> None:
        """Atomically set deny_all = True."""
        with self._lock:
            self.deny_all = True

    def get_denials_snapshot(self) -> frozenset[str]:
        """Return an immutable snapshot of current denials for safe off-lock inspection."""
        with self._lock:
            return frozenset(self.denials)

    # ── loaded_tools helpers ─────────────────────────────────────────────────

    def add_loaded_tool(self, tool_name: str) -> None:
        with self._lock:
            self.loaded_tools.add(tool_name)

    def remove_loaded_tool(self, tool_name: str) -> None:
        with self._lock:
            self.loaded_tools.discard(tool_name)

    def get_loaded_snapshot(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self.loaded_tools)

    # ── pinned_tools helpers ─────────────────────────────────────────────────

    def add_pinned_tool(self, tool_name: str) -> None:
        with self._lock:
            self.pinned_tools.add(tool_name)
            self.loaded_tools.add(tool_name)

    def remove_pinned_tool(self, tool_name: str) -> None:
        with self._lock:
            self.pinned_tools.discard(tool_name)
            self.loaded_tools.discard(tool_name)

    def get_pinned_snapshot(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self.pinned_tools)

    # ── approvals helpers ────────────────────────────────────────────────────

    def add_approval(self, tool_name: str) -> None:
        with self._lock:
            self.approvals.add(tool_name)

    def revoke_approval(self, tool_name: str) -> None:
        with self._lock:
            self.approvals.discard(tool_name)

    def get_approvals_snapshot(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self.approvals)

    def reset_approvals(self) -> None:
        """Atomically clear all per-tool approvals."""
        with self._lock:
            self.approvals.clear()

    def reset_for_new_session(self) -> None:
        """Clear session-scoped state. Preserves no_confirm and tool catalogs.

        .. warning::
            Do NOT use ``dataclasses.replace()`` on this dataclass.  ``replace``
            copies set references by value, producing two SessionState objects
            that share the same mutable sets guarded by two unrelated locks —
            updates in one would be invisible to the other.
        """
        with self._lock:
            self.denials.clear()
            self.deny_all = False
            self.loaded_tools.clear()
            self.pinned_tools.clear()
            self.approvals.clear()

    def reset_for_new_prompt(self) -> None:
        """Reset per-prompt state.

        Clears ``deny_all`` and removes agent-loaded (non-pinned) tools from
        ``loaded_tools`` so the LLM starts each turn with a clean tool set.
        Pinned tools remain in ``loaded_tools``.
        """
        with self._lock:
            self.deny_all = False
            self.loaded_tools &= self.pinned_tools


@dataclass
class ExecutionSettings:
    """Agent-facing execution settings projected from app config."""

    context_compression: bool = True
    compression_min_age: int | None = None
    compression_min_chars: int | None = None
    context_max_messages: int = 200
    tier_cache_enabled: bool = True
    tool_context_limit_pct: float = 0.80
    parallel_tool_execution: bool = True
    git_native: bool = False
    decision_accountability_enabled: bool = False
    decision_accountability_report_uncertainty: bool = True
    decision_accountability_min_confidence: float = 7.0
    task_ownership_classifier_enabled: bool = True
    task_ownership_classifier_llm_fallback: bool = False
    task_ownership_ambiguous_action: str = "ask"
    pre_action_confirmation_enabled: bool = False


_EXECUTION_SETTING_FIELDS: tuple[str, ...] = (
    "context_compression",
    "compression_min_age",
    "compression_min_chars",
    "context_max_messages",
    "tier_cache_enabled",
    "tool_context_limit_pct",
    "parallel_tool_execution",
    "git_native",
    "decision_accountability_enabled",
    "decision_accountability_report_uncertainty",
    "decision_accountability_min_confidence",
    "task_ownership_classifier_enabled",
    "task_ownership_classifier_llm_fallback",
    "task_ownership_ambiguous_action",
    "pre_action_confirmation_enabled",
)


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
    context_max_tokens: int = 40_000
    preset_tools: set[str] | None = None
    context_compression: bool = True
    compression_min_age: int | None = None
    compression_min_chars: int | None = None
    compression_llm: Any = None
    context_max_messages: int = 200
    tool_call_guard: Any | None = None
    session_state: SessionState | None = None
    memory_manager: Any | None = None
    confirmation_ui: ConfirmationUI | None = None
    on_tool_expansion: Any | None = None
    parallel_tool_execution: bool = True
    git_native: bool = False
    tool_context_limit_pct: float = 0.80
    tier_cache_enabled: bool = True
    llm_timeout: int = 180  # per-call LLM request timeout (seconds)
    tools_ready: threading.Event | None = field(default=None, compare=False, repr=False)
    bound_cache: OrderedDict | None = field(default=None, compare=False, repr=False)
    compression_cache: OrderedDict | None = field(default=None, compare=False, repr=False)
    cache_lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
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
    # Consolidated execution settings (Issue #225)
    execution_settings: ExecutionSettings | None = field(default=None, compare=False, repr=False)
    # Layer-4: periodic tool-state verification interval (turns); 0 disables
    tool_health_check_interval: int = 20
    # Layer-3: tool output quality gate; inject honest-answer nudge when all tools return empty
    tool_quality_gate_enabled: bool = True
    # Layer-3: topic-switch detection; reset rolling summary on short off-topic questions
    topic_switch_detection_enabled: bool = True
    # Tool trust overrides: maps tool_name -> "always"|"ask"|"deny".
    # When None, create_safe_tool_wrapper defaults to "ask" for all tools.
    # Threaded through to dynamic tool loading in process_tools.py and sessions.py.
    tool_trust: dict[str, str] | None = None

    def __post_init__(self) -> None:
        settings = self.execution_settings
        if settings is None:
            object.__setattr__(self, "execution_settings", self._build_execution_settings())
            return
        if not isinstance(settings, ExecutionSettings):
            raise TypeError("execution_settings must be an ExecutionSettings instance")
        self._sync_fields_from_execution_settings(settings)

    def _build_execution_settings(self) -> ExecutionSettings:
        return ExecutionSettings(
            context_compression=self.context_compression,
            compression_min_age=self.compression_min_age,
            compression_min_chars=self.compression_min_chars,
            context_max_messages=self.context_max_messages,
            tier_cache_enabled=self.tier_cache_enabled,
            tool_context_limit_pct=self.tool_context_limit_pct,
            parallel_tool_execution=self.parallel_tool_execution,
            git_native=self.git_native,
            decision_accountability_enabled=self.decision_accountability_enabled,
            decision_accountability_report_uncertainty=self.decision_accountability_report_uncertainty,
            decision_accountability_min_confidence=self.decision_accountability_min_confidence,
            task_ownership_classifier_enabled=self.task_ownership_classifier_enabled,
            task_ownership_classifier_llm_fallback=self.task_ownership_classifier_llm_fallback,
            task_ownership_ambiguous_action=self.task_ownership_ambiguous_action,
            pre_action_confirmation_enabled=self.pre_action_confirmation_enabled,
        )

    def _sync_fields_from_execution_settings(self, settings: ExecutionSettings) -> None:
        for field_name in _EXECUTION_SETTING_FIELDS:
            object.__setattr__(self, field_name, getattr(settings, field_name))

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if name == "execution_settings":
            if value is None:
                object.__setattr__(self, "execution_settings", self._build_execution_settings())
                return
            if isinstance(value, ExecutionSettings):
                self._sync_fields_from_execution_settings(value)
            return

        if name in _EXECUTION_SETTING_FIELDS:
            settings = self.__dict__.get("execution_settings")
            if isinstance(settings, ExecutionSettings):
                setattr(settings, name, value)

    @classmethod
    def from_app_config(cls, config: Any) -> AgentRunConfig:
        """Build an AgentRunConfig from the application Config object."""
        if config is None:
            return cls()

        if hasattr(config, "to_execution_settings"):
            settings = config.to_execution_settings()
        else:
            settings = ExecutionSettings(
                context_compression=getattr(config, "context_compression", True),
                compression_min_age=getattr(config, "context_compression_min_age", None),
                compression_min_chars=getattr(config, "context_compression_min_chars", None),
                context_max_messages=getattr(config, "context_max_messages", 200),
                tier_cache_enabled=getattr(config, "tier_cache_enabled", True),
                tool_context_limit_pct=getattr(config, "tool_context_limit_pct", 0.80),
                parallel_tool_execution=getattr(config, "parallel_tool_execution", True),
                git_native=getattr(config, "git_native", False),
                decision_accountability_enabled=getattr(
                    config, "decision_accountability_enabled", False
                ),
                decision_accountability_report_uncertainty=getattr(
                    config, "decision_accountability_report_uncertainty", True
                ),
                decision_accountability_min_confidence=getattr(
                    config, "decision_accountability_min_confidence", 7.0
                ),
                task_ownership_classifier_enabled=getattr(
                    config, "task_ownership_classifier_enabled", True
                ),
                task_ownership_classifier_llm_fallback=getattr(
                    config, "task_ownership_classifier_llm_fallback", False
                ),
                task_ownership_ambiguous_action=getattr(
                    config, "task_ownership_ambiguous_action", "ask"
                ),
                pre_action_confirmation_enabled=getattr(
                    config, "pre_action_confirmation_enabled", False
                ),
            )

        return cls(
            execution_settings=settings,
            context_max_tokens=getattr(config, "context_max_tokens", 40_000),
            tool_health_check_interval=getattr(config, "tool_health_check_interval", 20),
            tool_quality_gate_enabled=getattr(config, "tool_quality_gate_enabled", True),
            topic_switch_detection_enabled=getattr(config, "topic_switch_detection_enabled", True),
        )
