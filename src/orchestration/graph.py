"""LangGraph agent graph for Cogtrix.

Builds a custom StateGraph with three nodes:
- call_model: binds active tools to LLM and invokes it
- process_tools: executes tool calls, handles fuzzy matching and expansion
- handle_phantom: recovers from phantom tool calls
"""

import json as _json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

from src.agent.core import CogtrixState
from src.logging_config import get_logger
from src.orchestration.compression import (
    _CHARS_PER_TOKEN,
    _EMERGENCY_THRESHOLD_RATIO,
    _MID_TURN_COMPRESSION_THRESHOLD,
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    _content_len,
    apply_message_compression,
)
from src.orchestration.nodes.process_tools import build_process_tools_node
from src.orchestration.nodes.recovery import (
    build_handle_action_intent_node,
    build_handle_phantom_node,
    build_handle_unsupported_quote_node,
    build_handle_unverified_claim_node,
    build_handle_unverified_entity_node,
    build_handle_version_scope_node,
)
from src.orchestration.run_config import AgentRunConfig
from src.orchestration.session_state import SessionState
from src.providers import RetryableChatModel
from src.registry import LazyToolProxy as _LazyToolProxy
from src.tools.configure import (
    TOOL_OUTPUT_CAP_MIN_CHARS,
    build_tool_catalog,
    compute_tool_output_cap,
)

DEFAULT_RECURSION_LIMIT = 90
EMPTY_RESPONSE_MSG = "**Error:** The model returned an empty response. Please try again."

# DedupedToolInvoker hosts the per-call dedup + TOCTOU-safe execution that
# was previously the ``_invoke_one`` closure inside ``build_agent_graph``.
# Extracted in /forge A4 (2026-05-24).  ``_HISTORY_TOOL_MESSAGE_CAP_CHARS``
# moved with it (sole consumer); re-exported here so any external code that
# still imports the constant from ``graph`` keeps working.
from src.orchestration.deduped_tool_invoker import (  # noqa: E402, F401
    _HISTORY_TOOL_MESSAGE_CAP_CHARS,
    DedupedToolInvoker,
)

# Executors + PerRunState moved to ``src.orchestration.graph_runtime`` in the
# /forge A1.4 extraction (2026-05-23). Re-imported here so in-file callers
# keep working without a churn-everywhere PR.
from src.orchestration.graph_runtime import (  # noqa: E402, F401
    _LLM_EXECUTOR,
    _LLM_EXECUTOR_LOCK,
    _LLM_EXECUTOR_WORKERS,
    _PARALLEL_TOOL_WORKERS,
    _TOOL_EXECUTOR,
    _TOOL_EXECUTOR_LOCK,
    PerRunState,
    _get_llm_executor,
    _get_tool_executor,
)

# Topic-switch heuristic also extracted in the same A1.4 PR. Re-imported so
# test imports + in-file callers keep working.
from src.orchestration.topic_switch import (  # noqa: E402, F401
    _TOPIC_SWITCH_MAX_WORDS,
    _TOPIC_SWITCH_MESSAGE_WINDOW,
    _TOPIC_SWITCH_MIN_SIMILARITY,
    _TOPIC_SWITCH_NUDGE,
    _TOPIC_SWITCH_STOPWORDS,
    _should_reset_summary_for_topic_switch,
    _topic_switch_tokens,
)


def _extract_llm_labels(llm: Any) -> tuple[str, str]:
    """Extract provider and model labels from a LangChain LLM instance.

    Falls back to ``"unknown"`` when the LLM object does not expose the
    expected attributes.
    """
    if llm is None:
        return "unknown", "unknown"

    # Model name — try common attribute names across LangChain providers.
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or ""
    if not model:
        _ident = getattr(llm, "_identifying_params", None) or {}
        model = _ident.get("model_name") or _ident.get("model") or "unknown"

    # Provider — normalize from _llm_type or class name.
    _llm_type = getattr(llm, "_llm_type", None)
    if _llm_type:
        provider = _llm_type.lower().replace("chat-", "").replace("-chat", "")
    else:
        cls_name = type(llm).__name__.lower()
        if "openai" in cls_name:
            provider = "openai"
        elif "anthropic" in cls_name:
            provider = "anthropic"
        elif "google" in cls_name:
            provider = "google"
        elif "ollama" in cls_name:
            provider = "ollama"
        elif "deepseek" in cls_name:
            provider = "deepseek"
        elif "xai" in cls_name:
            provider = "xai"
        else:
            provider = "unknown"

    return provider, model


# _INVALID_TOOL_RE and the message-repair helpers below moved to
# ``src.orchestration.message_repair`` in the /forge A1.1 extraction
# (2026-05-23). Re-imported here so any in-file callers in this module
# (and the test imports that depended on these symbols living in graph)
# keep working without a churn-everywhere PR.
from src.orchestration.message_repair import (  # noqa: E402, F401
    _INVALID_TOOL_RE,
    _apply_context_message_cap,
    _detect_invalid_tool_calls,
    _repair_tool_message_pairs,
    _strip_failed_tool_messages,
)

_CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded",
    "context window",
    "too long",
    "maximum context",
    "reduce the length",
    "input is too long",
    "prompt is too long",
)


def _is_context_overflow_error(exc: Exception) -> bool:
    """Return True if *exc* is a provider context-length rejection.

    Checks structured provider error fields first (OpenAI ``error.code``,
    HTTP status codes, Anthropic ``type``), then falls back to string
    matching against ``_CONTEXT_OVERFLOW_PATTERNS``.
    """
    # Structured checks — faster and language-independent
    # OpenAI / OpenAI-compatible: BadRequestError with code field
    code = getattr(exc, "code", None) or getattr(getattr(exc, "error", None), "code", None)
    if code and "context_length" in str(code).lower():
        return True
    # HTTP status 400 paired with overflow-related type/param
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 400:
        etype = getattr(exc, "type", None) or getattr(getattr(exc, "error", None), "type", None)
        if etype and any(p in str(etype).lower() for p in ("context", "length", "token")):
            return True
    # String fallback — covers providers that embed the message in the exc string
    msg = str(exc).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS)


def _stable_tool_call_value(value: Any, seen: set[int] | None = None) -> Any:
    """Return a deterministic JSON-safe representation for tool-call args."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__type__": "builtins.bytes", "__value__": value.hex()}
    if isinstance(value, bytearray):
        return {"__type__": "builtins.bytearray", "__value__": bytes(value).hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "__state__": _stable_tool_call_value(asdict(value), seen),
        }

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            pass
        else:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": _stable_tool_call_value(dumped, seen),
            }

    obj_id = id(value)
    if obj_id in seen:
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "__cycle__": True,
        }

    if isinstance(value, dict):
        seen.add(obj_id)
        try:
            return {
                str(key): _stable_tool_call_value(value[key], seen)
                for key in sorted(value, key=lambda key: str(key))
            }
        finally:
            seen.discard(obj_id)

    if isinstance(value, list):
        seen.add(obj_id)
        try:
            return [_stable_tool_call_value(item, seen) for item in value]
        finally:
            seen.discard(obj_id)

    if isinstance(value, tuple):
        seen.add(obj_id)
        try:
            return {
                "__type__": "builtins.tuple",
                "__items__": [_stable_tool_call_value(item, seen) for item in value],
            }
        finally:
            seen.discard(obj_id)

    if isinstance(value, (set, frozenset)):
        seen.add(obj_id)
        try:
            items = [_stable_tool_call_value(item, seen) for item in value]
            items.sort(key=lambda item: _json.dumps(item, sort_keys=True, separators=(",", ":")))
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__items__": items,
            }
        finally:
            seen.discard(obj_id)

    if hasattr(value, "__dict__"):
        seen.add(obj_id)
        try:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": {
                    key: _stable_tool_call_value(attr, seen)
                    for key, attr in sorted(vars(value).items())
                    if not key.startswith("__")
                },
            }
        finally:
            seen.discard(obj_id)

    slots = getattr(value, "__slots__", None)
    if slots:
        seen.add(obj_id)
        try:
            state: dict[str, Any] = {}
            slot_names = (slots,) if isinstance(slots, str) else tuple(slots)
            for slot in slot_names:
                if hasattr(value, slot):
                    state[slot] = _stable_tool_call_value(getattr(value, slot), seen)
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": state,
            }
        finally:
            seen.discard(obj_id)

    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__value__": str(value),
    }


def _apply_context_budget_guard(
    response: Any,
    *,
    max_context_tokens: int | None,
    tool_context_limit_pct: float,
) -> Any:
    """Return a warning AIMessage when tool calls exceed the turn budget."""
    if not max_context_tokens or not getattr(response, "tool_calls", None):
        return response

    um = getattr(response, "usage_metadata", None)
    if not um or not isinstance(um, dict):
        return response

    turn_input = um.get("input_tokens", 0)
    if not isinstance(turn_input, int) or turn_input <= max_context_tokens * tool_context_limit_pct:
        return response

    from langchain_core.messages import AIMessage

    pct_used = int(turn_input * 100 / max_context_tokens)
    warning = (
        f"[Context budget reached — {pct_used}% of {max_context_tokens:,} "
        f"tokens used this turn (limit: {int(tool_context_limit_pct * 100)}%). "
        "Tool execution halted. Summarising based on available information.]"
    )
    return AIMessage(
        content=warning,
        id=getattr(response, "id", None),
        response_metadata={"budget_guard": True},
    )


def _infer_llm_provider_name(llm: Any) -> str:
    """Infer a stable provider label for telemetry."""
    for attr in ("provider", "provider_name", "_provider_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value

    module = getattr(llm.__class__, "__module__", "").lower()
    if "openai" in module:
        return "openai"
    if "anthropic" in module:
        return "anthropic"
    if "ollama" in module:
        return "ollama"
    if "google" in module or "genai" in module:
        return "google"
    return llm.__class__.__name__.lower()


def _infer_llm_model_name(llm: Any) -> str:
    """Infer the model identifier for telemetry."""
    for attr in ("model", "model_name", "model_id", "model_name_or_path"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    kwargs = getattr(llm, "_default_params", None)
    if isinstance(kwargs, dict):
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    return llm.__class__.__name__


# ── Action-intent detection ───────────────────────────────────────────────────
# Catches "I'll create X" / "Let me write Y" responses that contain no tool
# calls — the model expressed intent but didn't act on it. Implementation
# moved to ``src.orchestration.response_detectors`` in the /forge A1.2
# extraction (2026-05-23). Re-imported here so in-file callers and test
# imports that depended on these symbols living in graph keep working.
from src.orchestration.response_detectors import (  # noqa: E402, F401
    _CODE_FENCE_RE,
    _FAKE_TOOL_OUTPUT_SIGNAL_RE,
    _INCOMPLETENESS_SIGNAL_RE,
    _INTENT_FALSE_POSITIVE_RE,
    _INTENT_LEAD_RE,
    _MARKDOWN_TABLE_ROW_RE,
    _NEGATED_SUCCESS_RE,
    _NUMBERED_SECTION_RE,
    _PAST_TENSE_LIST_VERB_RE,
    _PHANTOM_JSON_TOOL_RE,
    _PHANTOM_TOOL_MARKUP_RE,
    _SUCCESS_CLAIM_RE,
    _TOOL_ERROR_INDICATORS,
    _TOOL_VERB_RE,
    _has_incompleteness_signal,
    _is_action_intent,
    _is_hallucinated_completion,
    _looks_like_fabricated_success_after_tool_errors,
    _looks_like_markdown_phantom_report,
    _looks_like_phantom_tool_markup,
    _stuck_detection_headline,
    _unwrap_code_fence,
)

# The original phrase / verb / detector implementations lived here; they
# moved to ``response_detectors.py`` (see comment + re-export above).


@dataclass
class ToolManagementRequest:
    """Result of scanning agent messages for ``request_tools`` calls."""

    add: list[str]
    remove: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.add or self.remove)


def _detect_tool_request(messages: list, start_idx: int = 0) -> ToolManagementRequest | None:
    """
    Scan agent messages for a ``request_tools`` invocation.

    Supports both the new schema (``add`` / ``remove``) and the legacy
    schema (``names`` treated as additions).

    Args:
        messages: Full message list from the agent result.
        start_idx: Index to start scanning from (skip history messages).

    Returns a ``ToolManagementRequest`` or *None* if no request was made.
    """
    all_add: list[str] = []
    all_remove: list[str] = []

    for i in range(start_idx, len(messages)):
        msg = messages[i]
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("name") == "request_tools":
                args = tc.get("args", {})

                # New schema: add / remove
                add_names = args.get("add", [])
                remove_names = args.get("remove", [])

                # Legacy fallback: bare ``names`` list → treat as add
                if not add_names and not remove_names:
                    add_names = args.get("names", [])

                # Normalize bare strings to single-element lists so
                # {"add": "web_search"} works the same as {"add": ["web_search"]}
                # (BUG-204).
                if isinstance(add_names, str):
                    add_names = [add_names]
                if isinstance(remove_names, str):
                    remove_names = [remove_names]

                if isinstance(add_names, list):
                    all_add.extend(str(n) for n in add_names)
                if isinstance(remove_names, list):
                    all_remove.extend(str(n) for n in remove_names)

    if not all_add and not all_remove:
        return None
    # Deduplicate and strip empty strings so LLM-generated garbage (e.g.
    # duplicate names, empty strings from JSON coercion) produces a clear
    # no-op or guidance line rather than a silent failure.
    seen_add: set[str] = set()
    deduped_add: list[str] = []
    for n in all_add:
        if n and n not in seen_add:
            seen_add.add(n)
            deduped_add.append(n)
    seen_rem: set[str] = set()
    deduped_rem: list[str] = []
    for n in all_remove:
        if n and n not in seen_rem:
            seen_rem.add(n)
            deduped_rem.append(n)
    return ToolManagementRequest(add=deduped_add, remove=deduped_rem)


# _correct_tool_args / _safe_tool_name + the schema cache + blocklist moved
# to ``src.orchestration.tool_arg_correction`` in the /forge A1.3 extraction
# (2026-05-23). Re-imported here so in-file callers and test imports that
# depended on these symbols living in graph keep working.
from src.orchestration.tool_arg_correction import (  # noqa: E402, F401
    _FUZZY_ARG_BLOCKLIST,
    _TOOL_ARG_SCHEMA_CACHE_MAX_SIZE,
    _correct_tool_args,
    _safe_tool_name,
    _tool_arg_cache_lock,
    _tool_arg_schema_cache,
    _ToolArgSchemaCacheKey,
)


def build_agent_graph(
    llm: Any = None,
    system_prompt: str = "",
    active_tools_list: list | None = None,
    available_tools: dict | None = None,
    registry: Any = None,
    approvals: set | None = None,
    max_context_tokens: int | None = None,
    preset_tools: set[str] | None = None,
    context_compression: bool = True,
    compression_min_age: int = COMPRESSION_MIN_AGE_CYCLES,
    compression_min_chars: int = COMPRESSION_MIN_CHARS,
    compression_llm: Any = None,
    context_max_messages: int = 200,
    context_max_tokens: int = 40_000,
    tool_call_guard: Any | None = None,
    session_state: SessionState | None = None,
    confirmation_ui: Any | None = None,
    on_tool_expansion: Any | None = None,
    parallel_tool_execution: bool = True,
    git_native: bool = False,
    tool_context_limit_pct: float = 0.80,
    extend_run_state: Any = None,
    *,
    config: AgentRunConfig | None = None,
    bound_cache: OrderedDict | None = None,
    compression_cache_in: dict[str, str] | None = None,
    checkpoint_store: Any | None = None,
) -> Any:
    """Build a custom LangGraph StateGraph for the Cogtrix agent.

    The graph has three nodes:
    - call_model: binds active tools to LLM and invokes it
    - process_tools: executes tool calls, handles fuzzy matching and expansion
    - handle_phantom: recovers from phantom tool calls (malformed JSON)

    Tool management uses closured mutable references: active_tools_list and
    available_tools are modified in-place, so callers see the changes after
    graph execution.

    When *config* is provided, its fields take precedence over the individual
    keyword arguments (backward-compat layer).
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.messages.modifier import RemoveMessage
    from langgraph.graph import END, StateGraph

    _model_timeout = 180  # default LLM request timeout (seconds)

    if config is not None:
        if config.llm is not None:
            llm = config.llm
        if config.system_prompt is not None:
            system_prompt = config.system_prompt
        if config.active_tools_list is not None:
            active_tools_list = config.active_tools_list
        if config.available_tools is not None:
            available_tools = config.available_tools
        if config.max_context_tokens is not None:
            max_context_tokens = config.max_context_tokens
        if config.preset_tools is not None:
            preset_tools = config.preset_tools
        if config.tool_call_guard is not None:
            tool_call_guard = config.tool_call_guard
        if config.session_state is not None:
            session_state = config.session_state
        memory_manager = getattr(config, "memory_manager", None)
        if config.confirmation_ui is not None:
            confirmation_ui = config.confirmation_ui
        if config.on_tool_expansion is not None:
            on_tool_expansion = config.on_tool_expansion
        parallel_tool_execution = config.parallel_tool_execution
        git_native = config.git_native
        context_compression = config.context_compression
        _context_max_messages = getattr(config, "context_max_messages", context_max_messages) or 0
        _context_max_tokens = getattr(config, "context_max_tokens", context_max_tokens) or 0
        _model_timeout = getattr(config, "llm_timeout", 180)
        if config.compression_llm is not None:
            compression_llm = config.compression_llm
        if config.compression_min_age is not None:
            compression_min_age = config.compression_min_age
        if config.compression_min_chars is not None:
            compression_min_chars = config.compression_min_chars
        if hasattr(config, "tool_context_limit_pct"):
            tool_context_limit_pct = config.tool_context_limit_pct
        if hasattr(config, "checkpoint_store"):
            checkpoint_store = config.checkpoint_store
        tools_ready = getattr(config, "tools_ready", None)
        _tier_cache_enabled = getattr(config, "tier_cache_enabled", True)
        _da_enabled = getattr(config, "decision_accountability_enabled", False)
        _da_report_uncertainty = getattr(config, "decision_accountability_report_uncertainty", True)
        _da_min_confidence = getattr(config, "decision_accountability_min_confidence", 7.0)
    else:
        _context_max_messages = context_max_messages or 0
        _context_max_tokens = context_max_tokens or 0
        tools_ready = None
        memory_manager = None
        _tier_cache_enabled = False
        _da_enabled = False
        _da_report_uncertainty = True
        _da_min_confidence = 7.0

    if active_tools_list is None:
        active_tools_list = []
    if available_tools is None:
        available_tools = {}
    if approvals is None:
        approvals = set()

    if session_state is None:
        session_state = SessionState()

    # Mutable container so _reset_for_new_run can swap the state each run
    # without rebuilding the compiled graph.
    extend_run_state_ref: list[Any] = [extend_run_state]
    if extend_run_state is not None:
        try:
            from langchain_core.tools import StructuredTool as _ST

            from src.tools.extend_run import ExtendRunInput

            def _extend_run_fn(
                mode: str = "continue",
                subtasks: list[str] | None = None,
                reason: str = "",
            ) -> str:
                _state = extend_run_state_ref[0]
                if _state is None:
                    return "Error: extend_run is not available in this run context."
                if mode == "delegate" and not subtasks:
                    return (
                        "Error: mode='delegate' requires a non-empty 'subtasks' list. "
                        "Provide 2-5 independent subtask descriptions."
                    )
                _state.request_extension(mode=mode, subtasks=subtasks or [], reason=reason)
                if mode == "delegate":
                    count = len(subtasks or [])
                    return (
                        f"Extension registered: {count} subtask(s) queued for parallel "
                        "delegation. Continue sequential work; delegation runs after this run."
                    )
                return (
                    "Extension registered: the step budget will be increased when the "
                    "current limit is reached. Continue working on the task."
                )

            _extend_tool = _ST.from_function(
                func=_extend_run_fn,
                name="extend_run",
                description=(
                    "Request more execution steps or delegate work to parallel sub-agents. "
                    "Call when the task needs significantly more turns than available.\n\n"
                    "Modes:\n"
                    "- 'continue': Request more sequential steps.\n"
                    "- 'delegate': Split into parallel sub-agents (requires 'subtasks' list).\n\n"
                    "Call EARLY — don't wait until almost out of steps."
                ),
                args_schema=ExtendRunInput,
            )
            if not any(getattr(tool, "name", "") == "extend_run" for tool in active_tools_list):
                active_tools_list.append(_extend_tool)
            available_tools["extend_run"] = _extend_tool
        except ImportError:
            pass
    _MAX_PHANTOM_RETRIES = 3
    _MAX_FABRICATION_RETRIES = 3
    _MAX_ACTION_INTENT_RETRIES = 3
    # Bug L: unverified-claim guard. Lower retry budget than other
    # recovery loops — one revision attempt. If the model still
    # refuses to call the verification tool, accept and ship rather
    # than spin.
    _MAX_UNVERIFIED_CLAIM_RETRIES = 1
    # cogtrix47 Issues 5+6: unverified-entity guard. Same budget as
    # the claim guard — one revision attempt; if the model keeps
    # repeating unverified identifiers after the nudge it is
    # actively ignoring the guard, not unlucky.
    _MAX_UNVERIFIED_ENTITY_RETRIES = 1
    # #1841: output-fidelity guard. One revision attempt — a model still
    # fabricating quotes after the nudge is ignoring the guard, not unlucky.
    _MAX_UNSUPPORTED_QUOTE_RETRIES = 1
    # #1843: version-scope-collapse guard. One revision attempt — same
    # rationale as the fidelity guard it composes with.
    _MAX_VERSION_SCOPE_RETRIES = 1
    # After the 3 standard action-intent nudges are exhausted, the model
    # gets exactly one more chance if the response contains incompleteness
    # language ("first", "to start", "step 1") — a stronger nudge that
    # demands completion rather than a generic "call the appropriate tool".
    _MAX_INCOMPLETENESS_NUDGES = 1
    _MAX_TOOL_EXPANSIONS = 3
    _MAX_REQUEST_TOOLS_NOOPS = 3
    _MAX_TOOL_CALL_HISTORY = 256
    _history_lock = threading.Lock()
    # Pending events for in-flight tool calls (BUG-1293): maps call_key -> threading.Event.
    # Used to block duplicate parallel threads until the first thread stores the result.
    _pending_events: dict[str, threading.Event] = {}
    # Per-tool call counter: tracks how many times each tool is called this turn.
    # After _TOOL_BUDGET_SOFT calls, a synthesis hint is appended to the output.
    # After _TOOL_BUDGET_HARD calls, the tool returns a stop message.
    _tool_budget_lock = (
        threading.Lock()
    )  # Protects _per_run_state[0].tool_call_counts and active_tools_list
    _TOOL_BUDGET_SOFT = 5  # nudge: "please synthesize"
    _TOOL_BUDGET_HARD = 8  # stop: "budget exhausted"
    _TOOL_BUDGET_SOFT_EXEMPT = {
        "request_tools",
        "report_progress",
        "queue_reply",
        "list_scheduled_messages",
        "edit_scheduled_message",
        "cancel_scheduled_message",
        "defer_processing",
        "suppress_reply",
        # Action tools that naturally require many sequential calls for
        # complex tasks (building software, multi-file edits, etc.).
        # The budget is designed to prevent runaway *search* loops, not
        # to throttle legitimate action sequences.
        "execute_shell_command",
        "write_file",
        "append_file",
        "patch_file",
        # Progress tracking — must always be callable.
        "checkpoint",
    }
    _TOOL_BUDGET_HARD_EXEMPT = _TOOL_BUDGET_SOFT_EXEMPT | {
        # Search tools should not hard-stop at the fixed cutoff because
        # legitimate research often requires many progressive searches.
        # ADR-0056 PR-G renamed search_web → web_search; both kept so
        # any future re-introduction of search_web still exempts.
        "search_web",
        "web_search",
        "search_news",
        "google_search",
        "brave_search",
        "exa_search",
        "tavily_search",
        "serpapi_search",
        "searxng_search",
        "search_email",
        "calendar_search_events",
    }
    _DUPLICATE_EXEMPT = {
        "request_tools",
        "report_progress",
        "queue_reply",
        "list_scheduled_messages",
        "edit_scheduled_message",
        "cancel_scheduled_message",
        # These control tools must always return a fresh result; caching a
        # prior error (e.g. from a ToolCallGuard block) would cause a retry
        # to receive the stale "duplicate" error instead of being evaluated
        # on its own merits (BUG-237).
        "suppress_reply",
        "defer_processing",
    }
    protected = (preset_tools or set()) | {"request_tools"}
    _bound_cache_lock = threading.Lock()
    _REFLECTION_INTERVAL = 10  # inject reflection every N call_model cycles
    _TOOL_HEALTH_CHECK_INTERVAL = (
        getattr(config, "tool_health_check_interval", 20) if config is not None else 20
    )
    _TOOL_QUALITY_GATE_ENABLED = (
        getattr(config, "tool_quality_gate_enabled", True) if config is not None else True
    )
    _TOPIC_SWITCH_DETECTION_ENABLED = (
        getattr(config, "topic_switch_detection_enabled", True) if config is not None else True
    )

    # ── Stuck detection ───────────────────────────────────────────────
    # Tracks consecutive tool calls that produce errors.  When the count
    # reaches _STUCK_THRESHOLD, the next call_model invokes the LLM
    # WITHOUT tools (forced thinking break) so the model must produce
    # a text-only Chain-of-Thought response before it can resume tool use.
    _STUCK_THRESHOLD = 5  # consecutive error results before forcing a break
    _CHECKPOINT_NUDGE_INTERVAL = 8  # nudge after N tool calls without checkpoint
    _same_file_writes_lock = threading.Lock()
    _REWRITE_SEARCH_THRESHOLD = 2  # search reminder after N writes to same file

    _cached_fingerprint: list[tuple[str, ...]] = [()]
    output_cap = (
        compute_tool_output_cap(max_context_tokens)
        if max_context_tokens
        else TOOL_OUTPUT_CAP_MIN_CHARS
    )
    _sys_msg = SystemMessage(content=system_prompt) if system_prompt else None

    # ── Per-run mutable state (structurally reset) ────────────────────
    # All counters, collections, and lookup tables that must be zeroed
    # between agent turns live in a single dataclass.  A fresh instance
    # is created by _reset_for_new_run(), so a newly-added field is
    # automatically reset — no risk of forgetting a manual reset line.
    _tool_lookup_init: dict[str, Any] = {getattr(t, "name", ""): t for t in active_tools_list}
    _tool_lookup_init.pop("", None)
    _active_names_init: set[str] = set(_tool_lookup_init.keys())
    _per_run_state: list[PerRunState] = [
        PerRunState(
            tool_lookup=_tool_lookup_init,
            active_names=_active_names_init,
            tool_catalog=build_tool_catalog(available_tools),
            available_tools_ref=[available_tools],
            bound_cache=(bound_cache if bound_cache is not None else OrderedDict()),
            compression_cache=(compression_cache_in if compression_cache_in is not None else {}),
        )
    ]

    # ── Checkpoint store ──────────────────────────────────────────────
    from src.tools.checkpoint import CheckpointStore, create_checkpoint_tool

    if checkpoint_store is not None:
        _checkpoint_store: CheckpointStore = checkpoint_store
    else:
        _checkpoint_store = CheckpointStore()
    _checkpoint_store_lock = threading.Lock()

    _checkpoint_tool = create_checkpoint_tool(_checkpoint_store)
    if _checkpoint_tool is not None and active_tools_list is not None:
        _existing_names = {getattr(t, "name", "") for t in active_tools_list}
        if "checkpoint" not in _existing_names:
            active_tools_list.append(_checkpoint_tool)
            _per_run_state[0].active_names.add("checkpoint")
            _per_run_state[0].tool_lookup["checkpoint"] = _checkpoint_tool

    _graph_log = get_logger()

    def _warm_bound_cache() -> None:
        """Seed the bind_tools cache for the initial active tool set."""
        if llm is None or not active_tools_list:
            return
        if tools_ready is not None and not tools_ready.is_set():
            _graph_log.debug("Skipping bind_tools warm-up until MCP tools finish reconnecting")
            return
        tool_list = list(active_tools_list)
        _seen_names_rev: set[str] = set()
        deduped_rev: list[Any] = []
        for _t in reversed(tool_list):
            _tname = getattr(_t, "name", "")
            if _tname not in _seen_names_rev:
                _seen_names_rev.add(_tname)
                deduped_rev.append(_t)
        tool_list = list(reversed(deduped_rev))
        normalized_tools: list[Any] = []
        for tool_obj in tool_list:
            if isinstance(tool_obj, _LazyToolProxy):
                try:
                    tool_obj = tool_obj._resolve()
                except Exception as exc:
                    _graph_log.warning(
                        "bind_tools warm-up failed to resolve lazy tool %r: %s",
                        getattr(tool_obj, "name", ""),
                        exc,
                    )
                    continue
                if tool_obj is None:
                    continue
            normalized_tools.append(tool_obj)
        if not normalized_tools:
            return
        fingerprint = tuple(getattr(t, "name", "") for t in normalized_tools)
        if fingerprint in _per_run_state[0].bound_cache:
            return
        try:
            if len(_per_run_state[0].bound_cache) >= 8:
                _per_run_state[0].bound_cache.popitem(last=False)
            _per_run_state[0].bound_cache[fingerprint] = llm.bind_tools(normalized_tools)
            _cached_fingerprint[0] = fingerprint
            _graph_log.debug("⏱ bind_tools warm-up: %d tool(s)", len(normalized_tools))
        except Exception as exc:
            _graph_log.warning("Initial bind_tools warm-up failed: %s", exc)

    _warm_bound_cache()

    def _maybe_compress(msgs: list) -> list:
        """Pre-invoke compression check (mid-turn guard).

        Uses actual token counts from the previous model call when available,
        falling back to char-based estimates.  Fires at
        _MID_TURN_COMPRESSION_THRESHOLD (0.60) — lower than the turn-start
        token-based threshold (0.72) — so context can never grow to 100%
        during a long tool loop before compression triggers.

        When TCC is active, this guard is a safety net only — the background
        roll-forward handles compression incrementally.  The threshold is raised
        to 0.80 to avoid redundant mid-turn LLM calls.

        At 85%+ char pressure (emergency), min_age_override=0 forces all
        eligible ToolMessages to be compressed regardless of age.
        """
        _comp_llm = compression_llm or llm
        if not context_compression or _comp_llm is None:
            return msgs
        if max_context_tokens is None or max_context_tokens < 16_384:
            return msgs
        total_chars = sum(_content_len(m) for m in msgs)
        context_chars = max_context_tokens * _CHARS_PER_TOKEN
        if context_chars <= 0:
            return msgs
        ratio = total_chars / context_chars
        # Also check token-based ratio when real data is available — the
        # char estimate underestimates web/JSON content density.
        token_ratio = 0.0
        last_tokens = _per_run_state[0].last_input_tokens[0]
        if last_tokens > 0 and max_context_tokens > 0:
            token_ratio = last_tokens / max_context_tokens
        effective_ratio = max(ratio, token_ratio)
        # When TCC is active, raise the threshold to 0.80 — the mid-turn guard
        # is a safety net only; roll-forward handles most compression.
        _mid_turn_threshold = 0.80 if _tier_cache_enabled else _MID_TURN_COMPRESSION_THRESHOLD
        if effective_ratio < _mid_turn_threshold:
            return msgs
        # Emergency: min_age_override=0 compresses regardless of message age.
        # Non-emergency: min_age_override=compression_min_age bypasses the
        # internal token/char threshold check while keeping the age guard.
        min_age_ovr = 0 if effective_ratio >= _EMERGENCY_THRESHOLD_RATIO else compression_min_age
        return apply_message_compression(
            msgs,
            call_count=_per_run_state[0].call_count[0],
            compression_cache=_per_run_state[0].compression_cache,
            llm=_comp_llm,
            max_context_tokens=max_context_tokens,
            min_age_cycles=compression_min_age,
            min_chars=compression_min_chars,
            min_age_override=min_age_ovr,
            actual_input_tokens=last_tokens,
        )

    # ── LLM call with timeout ─────────────────────────────────────
    # Prevents indefinite hangs when the LLM backend disconnects.
    _LLM_RETRY_TIMEOUT = 300  # seconds — retry timeout after first attempt fails
    _LLM_MAX_RETRIES = 3  # total attempts (1 initial + 2 retries)
    _LLM_RETRY_BASE_DELAY = 2.0  # seconds — doubles on each retry (2, 4)

    def _is_retryable_error(exc: Exception) -> bool:
        """Return True for transient errors worth retrying (rate limits, 5xx)."""
        msg = str(exc).lower()
        return any(
            p in msg
            for p in (
                "rate limit",
                "rate_limit",
                "too many requests",
                "429",
                "503",
                "502",
                "500",
                "server error",
                "overloaded",
                "capacity",
                "temporarily",
            )
        )

    def _invoke_with_timeout(_model: Any, _messages: list, _cfg: Any, _timeout: int) -> Any:
        import concurrent.futures as _cf

        _executor = _get_llm_executor()
        last_exc: Exception | None = None
        for _attempt in range(_LLM_MAX_RETRIES):
            # Disable the model's inner retry loop so that retries happen
            # in this outer loop without blocking a scarce pool worker.
            if isinstance(_model, RetryableChatModel):
                _fut = _executor.submit(
                    _model.invoke, _messages, _cfg, _cogtrix_disable_retries=True
                )
            else:
                _fut = _executor.submit(_model.invoke, _messages, _cfg)
            try:
                _timeout_for_attempt = _LLM_RETRY_TIMEOUT if _attempt > 0 else _timeout
                return _fut.result(timeout=_timeout_for_attempt)
            except _cf.TimeoutError:
                # Cancel the future so the shared executor can reclaim the
                # slot.  If the underlying LLM I/O is stuck the OS thread may
                # continue running, but it is bounded by the pool's max_workers.
                _fut.cancel()
                last_exc = RuntimeError(
                    f"LLM backend not responding (timed out after {_timeout_for_attempt}s)"
                )
                _graph_log.warning(
                    "LLM call timed out after %ds (attempt %d/%d)",
                    _timeout_for_attempt,
                    _attempt + 1,
                    _LLM_MAX_RETRIES,
                )
            except Exception as _exc:
                if _is_retryable_error(_exc) and _attempt < _LLM_MAX_RETRIES - 1:
                    last_exc = _exc
                    _delay = _LLM_RETRY_BASE_DELAY * (2**_attempt)
                    _graph_log.warning(
                        "LLM call failed with retryable error (attempt %d/%d, "
                        "retrying in %.0fs): %s",
                        _attempt + 1,
                        _LLM_MAX_RETRIES,
                        _delay,
                        _exc,
                    )
                    time.sleep(_delay)
                    continue
                raise
            if _attempt < _LLM_MAX_RETRIES - 1:
                _delay = _LLM_RETRY_BASE_DELAY * (2**_attempt)
                time.sleep(_delay)
        raise last_exc or RuntimeError("LLM invocation failed after all retries")

    # ── Tool output quality gate helpers ──────────────────────────────
    _SUBSTANCELESS_PREFIXES = ("error:", "no results", "0 results")

    def _is_substanceless(content: Any) -> bool:
        """Return True if a tool result lacks actionable substance."""
        if content is None:
            return True
        if not isinstance(content, str):
            return False
        stripped = content.strip()
        if not stripped:
            return True
        # An empty JSON array/object is valid "nothing found" data, not no-data.
        # list_pull_requests returning [] means "no open PRs" — the quality gate
        # must not fire when prior turns already returned valid results.
        if stripped in ("[]", "{}", "[ ]", "{ }"):
            return False
        if len(stripped) < 20:
            return True
        lower = stripped.lower()
        if lower.startswith(_SUBSTANCELESS_PREFIXES):
            return True
        return False

    def _all_tool_results_substanceless(messages: list[Any]) -> bool:
        """Return True when the most recent contiguous ToolMessage block is non-empty
        and every message in it is substanceless.
        """
        tool_msgs: list[Any] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_msgs.append(msg)
            else:
                break
        if not tool_msgs:
            return False
        return all(_is_substanceless(getattr(m, "content", None)) for m in tool_msgs)

    from src.orchestration.nodes.call_model import CallModelContext, build_call_model_node
    from src.orchestration.search_quality import SearchQualityThresholds

    # Build search quality thresholds from Config (#1593, Option B).
    # Defaults are defined in SearchQualityThresholds; config.yaml overrides apply.
    try:
        from src.config import Config

        _cfg = Config()
        _search_sq_thresholds = SearchQualityThresholds(
            min_url_count=_cfg.search_quality_min_url_count,
            min_content_chars=_cfg.search_quality_min_chars,
        )
        # Wire curl/wget URL domain allowlisting (Caleb Varden arch review, PR #1607).
        # Without this call, _set_curl_wget_allowed_domains() is never invoked at
        # runtime and the feature is a dead letter even when users configure
        # shell_curl_wget_allowed_domains in cogtrix.yaml.
        try:
            from src.tools.shell import _set_curl_wget_allowed_domains

            _set_curl_wget_allowed_domains(_cfg.shell_curl_wget_allowed_domains)
        except (
            Exception
        ):  # noqa: BLE001 — must never crash graph build; shell domain allowlisting silently disabled
            pass
    except Exception:  # noqa: BLE001 — config loading must never crash graph build
        _search_sq_thresholds = SearchQualityThresholds()

    call_model = build_call_model_node(
        CallModelContext(
            llm=llm,
            tools_ready=tools_ready,
            active_tools_list=active_tools_list,
            active_names=_per_run_state[0].active_names,
            bound_cache=_per_run_state[0].bound_cache,
            bound_cache_lock=_bound_cache_lock,
            cached_fingerprint=_cached_fingerprint,
            compression_cache=_per_run_state[0].compression_cache,
            tool_version=_per_run_state[0].tool_version,
            last_tool_version=_per_run_state[0].last_tool_version,
            call_count=_per_run_state[0].call_count,
            last_input_tokens=_per_run_state[0].last_input_tokens,
            max_context_tokens=max_context_tokens,
            context_max_messages=_context_max_messages,
            context_max_tokens=_context_max_tokens,
            model_max_tokens=getattr(llm, "max_tokens", None),
            compression_llm=compression_llm,
            memory_manager=memory_manager,
            checkpoint_store=_checkpoint_store,
            calls_since_last_checkpoint=_per_run_state[0].calls_since_last_checkpoint,
            last_checkpoint_count=_per_run_state[0].last_checkpoint_count,
            rounds_since_checkpoint=_per_run_state[0].rounds_since_checkpoint,
            force_thinking_break=_per_run_state[0].force_thinking_break,
            consecutive_errors=_per_run_state[0].consecutive_errors,
            last_identical_error_signature=_per_run_state[0].last_identical_error_signature,
            consecutive_identical_error_count=_per_run_state[0].consecutive_identical_error_count,
            last_reflection_at=_per_run_state[0].last_reflection_at,
            tool_health_check_interval=_TOOL_HEALTH_CHECK_INTERVAL,
            last_tool_health_check_at=_per_run_state[0].last_tool_health_check_at,
            tool_quality_gate_enabled=_TOOL_QUALITY_GATE_ENABLED,
            topic_switch_detection_enabled=_TOPIC_SWITCH_DETECTION_ENABLED,
            stuck_threshold=_STUCK_THRESHOLD,
            stuck_no_checkpoint_threshold=_per_run_state[0].stuck_no_checkpoint_threshold,
            stuck_threshold_calibrated=_per_run_state[0].stuck_threshold_calibrated,
            checkpoint_nudge_interval=_CHECKPOINT_NUDGE_INTERVAL,
            reflection_interval=_REFLECTION_INTERVAL,
            max_request_tools_noops=_MAX_REQUEST_TOOLS_NOOPS,
            sys_msg=_sys_msg,
            model_timeout=_model_timeout,
            tool_context_limit_pct=tool_context_limit_pct,
            da_enabled=_da_enabled,
            da_report_uncertainty=_da_report_uncertainty,
            da_min_confidence=_da_min_confidence,
            apply_context_message_cap=_apply_context_message_cap,
            maybe_compress=_maybe_compress,
            invoke_with_timeout=_invoke_with_timeout,
            all_tool_results_substanceless=_all_tool_results_substanceless,
            search_quality_thresholds=_search_sq_thresholds,
        )
    )

    handle_phantom = build_handle_phantom_node(
        phantom_count=_per_run_state[0].phantom_count,
        max_retries=_MAX_PHANTOM_RETRIES,
    )
    handle_action_intent = build_handle_action_intent_node(
        action_intent_count=_per_run_state[0].action_intent_count,
        max_retries=_MAX_ACTION_INTENT_RETRIES,
        incompleteness_check=_has_incompleteness_signal,
    )
    handle_unverified_claim = build_handle_unverified_claim_node(
        unverified_claim_count=_per_run_state[0].unverified_claim_count,
        max_retries=_MAX_UNVERIFIED_CLAIM_RETRIES,
    )
    handle_unverified_entity = build_handle_unverified_entity_node(
        unverified_entity_count=_per_run_state[0].unverified_entity_count,
        max_retries=_MAX_UNVERIFIED_ENTITY_RETRIES,
    )
    handle_unsupported_quote = build_handle_unsupported_quote_node(
        unsupported_quote_count=_per_run_state[0].unsupported_quote_count,
        max_retries=_MAX_UNSUPPORTED_QUOTE_RETRIES,
    )
    handle_version_scope = build_handle_version_scope_node(
        version_scope_count=_per_run_state[0].version_scope_count,
        max_retries=_MAX_VERSION_SCOPE_RETRIES,
    )

    def handle_fabrication(state: CogtrixState) -> dict:
        _per_run_state[0].fabrication_count[0] += 1
        last = state["messages"][-1]
        log = get_logger()
        log.warning(
            "Fabricated success-after-error detected, attempt %d/%d. Injecting correction.",
            _per_run_state[0].fabrication_count[0],
            _MAX_FABRICATION_RETRIES,
        )
        if _per_run_state[0].fabrication_count[0] > _MAX_FABRICATION_RETRIES:
            return {
                "messages": [
                    RemoveMessage(id=last.id),
                    AIMessage(
                        content=(
                            "I reported success incorrectly after tool errors and could not "
                            "recover safely. Please retry your request."
                        )
                    ),
                ]
            }
        return {
            "messages": [
                RemoveMessage(id=last.id),
                HumanMessage(
                    content=(
                        "Some of the tools you called returned errors, but your response claims "
                        "success. Report honestly what the tools returned. Do not fabricate "
                        "success messages."
                    )
                ),
            ]
        }

    def handle_incompleteness(state: CogtrixState) -> dict:
        """Inject a strongly-worded final nudge when the model signalled
        incomplete multi-step work but exhausted standard action-intent retries.
        """
        log = get_logger()
        log.warning(
            "Incompleteness signal detected after action-intent retries exhausted. "
            "Injecting critical completion nudge."
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "CRITICAL: The task is incomplete. "
                        "You used language like 'first' or 'to start', "
                        "which implies there are more steps to complete. "
                        "Do not explain what comes next — call the remaining "
                        "tool(s) NOW."
                    )
                )
            ]
        }

    def _tool_call_key(call: dict) -> str | None:
        """Compute the deduplication key for a tool call, or None if not serializable.

        Normalizes to the canonical tool name via ``_per_run_state[0].tool_lookup`` so that an
        alias and its resolved canonical name share the same cache key.  When the
        alias is not in ``_per_run_state[0].tool_lookup`` (e.g. during the auto-expansion serial
        path) the raw call name is used instead (BUG-234).
        """
        tool_name = call["name"]
        if tool_name in _DUPLICATE_EXEMPT:
            return None
        # Prefer the canonical name stored on the live tool object.
        tool_obj = _per_run_state[0].tool_lookup.get(tool_name)
        if tool_obj is not None:
            tool_name = getattr(tool_obj, "name", tool_name) or tool_name
        args_json = _json.dumps(_stable_tool_call_value(call.get("args", {})), sort_keys=True)
        return tool_name + ":" + args_json

    def _identical_error_signature(call: dict) -> str | None:
        """Return a stable signature for repeated identical-error detection.

        Uses the tool name plus the first meaningful argument so that retry
        loops on the same action are grouped together without requiring the
        full argument payload to match byte-for-byte.
        """
        tool_name = call.get("name", "")
        if not tool_name:
            return None
        args = call.get("args", {})
        if not isinstance(args, dict):
            return None
        primary_keys = (
            "pull_number",
            "path",
            "url",
            "query",
            "command",
            "name",
            "text",
            "repo",
            "email",
            "username",
            "branch",
        )
        primary_key = next(
            (
                key
                for key in primary_keys
                if key in args and args.get(key) not in (None, "", [], {})
            ),
            None,
        )
        if primary_key is None:
            if not args:
                return None
            primary_key = next(iter(sorted(args.keys())))
        try:
            primary_value = _json.dumps(args.get(primary_key), sort_keys=True, default=str)
        except (TypeError, ValueError):
            primary_value = str(args.get(primary_key))
        return f"{tool_name}:{primary_key}={primary_value}"

    def _tool_error_class(content: str) -> str | None:
        """Normalize an error ToolMessage into a coarse error class."""
        normalized = content.strip()
        if normalized.lower().startswith("[duplicate call"):
            normalized = normalized.split("\n\n", 1)[-1]
        content_lower = _stuck_detection_headline(normalized).lower()
        if "repository rule violations" in content_lower:
            return "repository_rule_violations"
        if "permission denied" in content_lower or "forbidden" in content_lower:
            return "permission_denied"
        if "timed out" in content_lower or "timeout" in content_lower:
            return "timeout"
        if (
            "not found" in content_lower
            or "404" in content_lower
            or "no such file" in content_lower
            or "cannot open" in content_lower
        ):
            return "not_found"
        if (
            content_lower.startswith("error")
            or "error executing" in content_lower
            or "traceback" in content_lower
            or "failed" in content_lower
        ):
            return "generic_error"
        return None

    def _tool_error_guidance(error_class: str, tool_name: str) -> str:
        """Return short guidance tailored to the repeated error class."""
        if error_class == "repository_rule_violations":
            return (
                f"'{_safe_tool_name(tool_name)}' hit repository rule violations. "
                "Stop retrying and verify CI/branch protections or ask a maintainer to merge."
            )
        if error_class == "permission_denied":
            return (
                f"'{_safe_tool_name(tool_name)}' was denied. Stop retrying and verify "
                "authentication or permissions before trying again."
            )
        if error_class == "timeout":
            return (
                f"'{_safe_tool_name(tool_name)}' timed out repeatedly. Stop retrying and "
                "switch to a different approach or inspect the service health."
            )
        if error_class == "not_found":
            return (
                f"'{_safe_tool_name(tool_name)}' cannot find the target repeatedly. "
                "Verify the identifier or path before trying again."
            )
        return (
            f"'{_safe_tool_name(tool_name)}' has returned the same error repeatedly. "
            "Stop retrying and inspect the last failure before continuing."
        )

    def _check_duplicate(call: dict, key: str | None = None) -> ToolMessage | None:
        """Return a cached ToolMessage if this exact call was seen before."""
        tool_name = call["name"]
        if key is None:
            key = _tool_call_key(call)
        if key is None:
            return None
        with _history_lock:
            cached = _per_run_state[0].tool_call_history.get(key)
            if cached is not None:
                _per_run_state[0].tool_call_history.move_to_end(key)
        if cached is None:
            return None
        log = get_logger()
        log.warning("Duplicate tool call detected: %s (returning cached result)", tool_name)
        return ToolMessage(
            content=(
                "[Duplicate call — returning cached result. Do NOT repeat this call.]\n\n" + cached
            ),
            tool_call_id=call["id"],
            name=tool_name,
        )

    def _store_call_result(call: dict, result_text: str, key: str | None = None) -> None:
        """Store a tool call result for duplicate detection."""
        if key is None:
            key = _tool_call_key(call)
        if key is None:
            return
        with _history_lock:
            _per_run_state[0].tool_call_history[key] = result_text[:500]
            _per_run_state[0].tool_call_history.move_to_end(key)
            if len(_per_run_state[0].tool_call_history) > _MAX_TOOL_CALL_HISTORY:
                _per_run_state[0].tool_call_history.popitem(last=False)

    # _invoke_one + its _cap_history_tool_content helper were extracted to
    # ``src.orchestration.deduped_tool_invoker.DedupedToolInvoker`` in the
    # /forge A4 refactor (2026-05-24).  We instantiate the class here and
    # expose ``_invoker.invoke_one`` under the original symbol name so the
    # downstream contract (``build_process_tools_node(_invoke_one=…)``) is
    # byte-for-byte unchanged.  Constructor injection of locks, helpers,
    # and budget constants keeps the per-graph state model intact.
    _invoker = DedupedToolInvoker(
        per_run_state=_per_run_state,
        history_lock=_history_lock,
        tool_budget_lock=_tool_budget_lock,
        bound_cache_lock=_bound_cache_lock,
        pending_events=_pending_events,
        active_tools_list=active_tools_list,
        session_state=session_state,
        tool_call_guard=tool_call_guard,
        tool_call_key=_tool_call_key,
        check_duplicate=_check_duplicate,
        correct_tool_args=_correct_tool_args,
        safe_tool_name=_safe_tool_name,
        max_tool_call_history=_MAX_TOOL_CALL_HISTORY,
        tool_budget_hard=_TOOL_BUDGET_HARD,
        tool_budget_soft=_TOOL_BUDGET_SOFT,
        tool_budget_hard_exempt=_TOOL_BUDGET_HARD_EXEMPT,
        tool_budget_soft_exempt=_TOOL_BUDGET_SOFT_EXEMPT,
    )
    _invoke_one = _invoker.invoke_one

    process_tools = build_process_tools_node(
        _invoke_one=_invoke_one,
        _tool_lookup=_per_run_state[0].tool_lookup,
        _active_names=_per_run_state[0].active_names,
        _available_tools_ref=_per_run_state[0].available_tools_ref,
        session_state=session_state,
        parallel_tool_execution=parallel_tool_execution,
        _identical_error_signature=_identical_error_signature,
        _tool_error_class=_tool_error_class,
        _tool_error_guidance=_tool_error_guidance,
        _last_identical_error_signature=_per_run_state[0].last_identical_error_signature,
        _consecutive_identical_error_count=_per_run_state[0].consecutive_identical_error_count,
        _force_thinking_break=_per_run_state[0].force_thinking_break,
        _graph_log=_graph_log,
        protected=protected,
        tool_catalog=_per_run_state[0].tool_catalog,
        registry=registry,
        approvals=approvals,
        confirmation_ui=confirmation_ui,
        git_native=git_native,
        on_tool_expansion=on_tool_expansion,
        output_cap=output_cap,
        expansion_count=_per_run_state[0].expansion_count,
        auto_expansion_count=_per_run_state[0].auto_expansion_count,
        request_tools_noop_count=_per_run_state[0].request_tools_noop_count,
        _MAX_REQUEST_TOOLS_NOOPS=_MAX_REQUEST_TOOLS_NOOPS,
        active_tools_list=active_tools_list,
        _tool_version=_per_run_state[0].tool_version,
        _calls_since_last_checkpoint=_per_run_state[0].calls_since_last_checkpoint,
        _same_file_writes=_per_run_state[0].same_file_writes,
        _same_file_writes_lock=_same_file_writes_lock,
        _REWRITE_SEARCH_THRESHOLD=_REWRITE_SEARCH_THRESHOLD,
        _action_tier_consecutive_calls=_per_run_state[0].action_tier_consecutive_calls,
        _last_action_tier_tool=_per_run_state[0].last_action_tier_tool,
        _consecutive_errors=_per_run_state[0].consecutive_errors,
        _STUCK_THRESHOLD=_STUCK_THRESHOLD,
        _stuck_detection_headline=_stuck_detection_headline,
        _get_tool_executor=lambda: _get_tool_executor(),
        _detect_tool_request=_detect_tool_request,
        _safe_tool_name=_safe_tool_name,
        _tool_budget_lock=_tool_budget_lock,
        tool_trust=config.tool_trust if config is not None else None,
    )

    def route_after_model(state: CogtrixState) -> str:
        msgs = state["messages"]
        if not msgs:
            return END

        last = msgs[-1]
        if isinstance(last, AIMessage):
            content = getattr(last, "content", "")
            has_content = isinstance(content, str) and bool(content.strip())
            tool_calls = getattr(last, "tool_calls", None)
            meta = getattr(last, "response_metadata", None)

            if not has_content and not tool_calls:
                if meta and isinstance(meta, dict):
                    if meta.get("finish_reason") == "tool_calls":
                        return "handle_phantom"
                return END

            if meta and isinstance(meta, dict) and meta.get("budget_guard"):
                return END

            if tool_calls:
                return "process_tools"

            if _looks_like_phantom_tool_markup(last):
                return "handle_phantom"

            if _looks_like_markdown_phantom_report(last):
                return "handle_phantom"

            if _looks_like_fabricated_success_after_tool_errors(msgs, last):
                return "handle_fabrication"

            # Has content but no tool calls — check for intention-without-action.
            # Suppress the nudge when the agent is responding to an access-denied
            # tool failure: the model is offering alternatives, not planning to act.
            # Nudging it again causes a counterproductive retry of the same blocked path.
            if _is_action_intent(last):
                msgs = state.get("messages", [])
                recent_tool_errors = [
                    getattr(m, "content", "") or "" for m in msgs[-6:] if hasattr(m, "tool_call_id")
                ]
                if any(
                    "Access denied" in err or "path outside allowed" in err
                    for err in recent_tool_errors
                ):
                    pass  # skip nudge — agent handled the error gracefully
                else:
                    return "handle_action_intent"

            # Hallucinated completion: model wrote a past-tense summary
            # ("Notified the VP...") claiming it called a tool that it never
            # actually invoked. Route through the same retry/synthesis path
            # so the model gets a chance to execute the missing step.
            _available_names = [getattr(t, "name", "") for t in (active_tools_list or [])]
            if _is_hallucinated_completion(last, msgs, _available_names):
                return "handle_action_intent"

            # Bug L: unverified categorical claim about external state
            # (today's date, latest version, etc.) without calling the
            # matching verification tool. Route to a recovery node that
            # nudges the model to verify-then-revise. See
            # src/orchestration/verification.py for the rule registry.
            if has_content and isinstance(content, str):
                from src.orchestration.verification import (
                    collect_tool_message_contents,
                    collect_tool_names_this_turn,
                    detect_unsupported_quote,
                    detect_unverified_claim,
                    detect_unverified_entities,
                    detect_version_scope_mismatch,
                )

                _turn_start = 0
                for _i in range(len(msgs) - 1, -1, -1):
                    if isinstance(msgs[_i], HumanMessage):
                        _turn_start = _i
                        break
                _tools_called = collect_tool_names_this_turn(msgs, _turn_start)
                if detect_unverified_claim(content, _tools_called) is not None:
                    if _per_run_state[0].unverified_claim_count[0] <= _MAX_UNVERIFIED_CLAIM_RETRIES:
                        return "handle_unverified_claim"

                # cogtrix47 Issues 5+6: user-supplied specific
                # identifiers (SKUs, store names, multi-word product
                # names) the agent echoed without any tool result
                # confirming them. Route to the entity-recovery node
                # that nudges the model to cite evidence, hedge, or
                # substitute the verified alternative.
                _user_prompt = ""
                if _turn_start < len(msgs):
                    _up = getattr(msgs[_turn_start], "content", "") or ""
                    if isinstance(_up, str):
                        _user_prompt = _up
                _tool_contents = collect_tool_message_contents(msgs, _turn_start)
                if detect_unverified_entities(content, _user_prompt, _tool_contents):
                    if (
                        _per_run_state[0].unverified_entity_count[0]
                        <= _MAX_UNVERIFIED_ENTITY_RETRIES
                    ):
                        return "handle_unverified_entity"

                # #1841: output-fidelity guard. A verbatim quote or explicit
                # attribution in the response that appears in no tool result
                # this turn is a fabricated quote / fabricated citation.
                if detect_unsupported_quote(content, _tool_contents, _user_prompt):
                    if (
                        _per_run_state[0].unsupported_quote_count[0]
                        <= _MAX_UNSUPPORTED_QUOTE_RETRIES
                    ):
                        return "handle_unsupported_quote"

                # #1843: version-scope-collapse guard. A lifecycle status the
                # model attaches to a specific model-ID that the evidence
                # scopes only to a prefix-parent (series→version confusion).
                # Checked against the WHOLE conversation's tool output, not
                # just this turn: the misattribution often surfaces on a
                # correction turn that did no fresh research, so the only
                # ground truth is research from an earlier turn.
                _all_tool_contents = collect_tool_message_contents(msgs, 0)
                if detect_version_scope_mismatch(content, _all_tool_contents):
                    if _per_run_state[0].version_scope_count[0] <= _MAX_VERSION_SCOPE_RETRIES:
                        return "handle_version_scope"

        return END

    def route_after_phantom(state: CogtrixState) -> str:
        if _per_run_state[0].phantom_count[0] > _MAX_PHANTOM_RETRIES:
            return END
        return "call_model"

    def route_after_action_intent(state: CogtrixState) -> str:  # noqa: ARG001
        if _per_run_state[0].action_intent_count[0] > _MAX_ACTION_INTENT_RETRIES:
            # Standard retries exhausted.  Before ending, check whether
            # the model used incompleteness language ("first", "to start")
            # — a strong signal that it planned more steps but stopped.
            # Give exactly one more chance with a targeted nudge.
            if _per_run_state[0].incompleteness_nudge_given[0] < _MAX_INCOMPLETENESS_NUDGES:
                msgs = state.get("messages") or []
                last = msgs[-1] if msgs else None
                content = getattr(last, "content", "") if last is not None else ""
                if isinstance(content, str) and _has_incompleteness_signal(content):
                    _per_run_state[0].incompleteness_nudge_given[0] += 1
                    return "handle_incompleteness"
            return END
        return "call_model"

    def route_after_fabrication(state: CogtrixState) -> str:  # noqa: ARG001
        if _per_run_state[0].fabrication_count[0] > _MAX_FABRICATION_RETRIES:
            return END
        return "call_model"

    def route_after_unverified_claim(state: CogtrixState) -> str:  # noqa: ARG001
        # Recovery node has already incremented the counter and either
        # injected a nudge or short-circuited with an empty update.
        # Loop back to call_model so the agent can revise; once the
        # counter exceeds the budget we send to END so the (possibly
        # unverified) answer ships rather than spinning forever.
        if _per_run_state[0].unverified_claim_count[0] > _MAX_UNVERIFIED_CLAIM_RETRIES:
            return END
        return "call_model"

    def route_after_unverified_entity(state: CogtrixState) -> str:  # noqa: ARG001
        # Same shape as route_after_unverified_claim: one revision
        # attempt, then accept-and-ship so the model can't loop on
        # a stubborn refusal to drop the unverified identifier.
        if _per_run_state[0].unverified_entity_count[0] > _MAX_UNVERIFIED_ENTITY_RETRIES:
            return END
        return "call_model"

    def route_after_unsupported_quote(state: CogtrixState) -> str:  # noqa: ARG001
        # Same shape as the other verification guards: one revision
        # attempt, then accept-and-ship rather than loop on a model
        # that keeps re-emitting the unsupported quote.
        if _per_run_state[0].unsupported_quote_count[0] > _MAX_UNSUPPORTED_QUOTE_RETRIES:
            return END
        return "call_model"

    def route_after_version_scope(state: CogtrixState) -> str:  # noqa: ARG001
        # Same shape as the other verification guards: one revision
        # attempt, then accept-and-ship rather than loop on a model
        # that keeps collapsing the version scope.
        if _per_run_state[0].version_scope_count[0] > _MAX_VERSION_SCOPE_RETRIES:
            return END
        return "call_model"

    def _reset_for_new_run(
        new_available_tools: dict,
        new_bound_cache: "OrderedDict",
        new_compression_cache: dict,
        extend_run_state: Any = None,
    ) -> None:
        """Reset all per-run mutable state so the compiled graph can be reused.

        Called by ``run_agent()`` when the graph fingerprint matches the
        cached graph.  A fresh ``PerRunState`` instance is built and its
        values are copied in-place into the existing instance so that
        closures holding direct references to mutable fields still see
        the reset values.  Any new field added to ``PerRunState`` is
        automatically handled — no manual reset line required.
        """
        _fresh_tool_lookup = {
            getattr(t, "name", ""): t for t in active_tools_list if getattr(t, "name", "")
        }
        fresh = PerRunState(
            tool_lookup=_fresh_tool_lookup,
            active_names=set(_fresh_tool_lookup.keys()),
            tool_catalog=build_tool_catalog(new_available_tools),
            available_tools_ref=[new_available_tools],
            bound_cache=(new_bound_cache if new_bound_cache is not None else OrderedDict()),
            compression_cache=(new_compression_cache if new_compression_cache is not None else {}),
            tool_version=[_per_run_state[0].tool_version[0] + 1],
            last_tool_version=[-1],
        )

        # Copy fresh values into the existing PerRunState instance in-place.
        # This preserves object identity so closures that captured direct
        # references to mutable fields (e.g. call_count, bound_cache) still
        # see the reset values.
        for _f in fields(PerRunState):
            _current = getattr(_per_run_state[0], _f.name)
            _new = getattr(fresh, _f.name)
            if isinstance(_current, list):
                _current[:] = _new
            elif isinstance(_current, (dict, OrderedDict, set)):
                _current.clear()
                if _new:
                    _current.update(_new)
            else:
                setattr(_per_run_state[0], _f.name, _new)

        with _history_lock:
            _pending_events.clear()

        with _checkpoint_store_lock:
            _checkpoint_store.clear()

        if extend_run_state is not None:
            extend_run_state_ref[0] = extend_run_state

    graph: Any = StateGraph(CogtrixState)
    graph.add_node("call_model", call_model)
    graph.add_node("handle_phantom", handle_phantom)
    graph.add_node("handle_fabrication", handle_fabrication)
    graph.add_node("handle_action_intent", handle_action_intent)
    graph.add_node("handle_incompleteness", handle_incompleteness)
    graph.add_node("handle_unverified_claim", handle_unverified_claim)
    graph.add_node("handle_unverified_entity", handle_unverified_entity)
    graph.add_node("handle_unsupported_quote", handle_unsupported_quote)
    graph.add_node("handle_version_scope", handle_version_scope)
    graph.add_node("process_tools", process_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "process_tools": "process_tools",
            "handle_phantom": "handle_phantom",
            "handle_fabrication": "handle_fabrication",
            "handle_action_intent": "handle_action_intent",
            "handle_incompleteness": "handle_incompleteness",
            "handle_unverified_claim": "handle_unverified_claim",
            "handle_unverified_entity": "handle_unverified_entity",
            "handle_unsupported_quote": "handle_unsupported_quote",
            "handle_version_scope": "handle_version_scope",
            END: END,
        },
    )
    graph.add_edge("process_tools", "call_model")
    graph.add_conditional_edges(
        "handle_phantom",
        route_after_phantom,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_action_intent",
        route_after_action_intent,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_fabrication",
        route_after_fabrication,
        {"call_model": "call_model", END: END},
    )
    # handle_incompleteness always routes back to call_model — exactly
    # one chance to finish the task after a stronger nudge.
    graph.add_edge("handle_incompleteness", "call_model")
    graph.add_conditional_edges(
        "handle_unverified_claim",
        route_after_unverified_claim,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_unverified_entity",
        route_after_unverified_entity,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_unsupported_quote",
        route_after_unsupported_quote,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_version_scope",
        route_after_version_scope,
        {"call_model": "call_model", END: END},
    )
    compiled = graph.compile()
    compiled._reset_for_new_run = _reset_for_new_run  # type: ignore[attr-defined]
    compiled._per_run_state = _per_run_state  # type: ignore[attr-defined]
    return compiled
