"""LangGraph agent graph for Cogtrix.

Builds a custom StateGraph with three nodes:
- call_model: binds active tools to LLM and invokes it
- process_tools: executes tool calls, handles fuzzy matching and expansion
- handle_phantom: recovers from phantom tool calls
"""

import atexit
import concurrent.futures
import json as _json
import re
import threading
import time
import types
import typing
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from src.agent.core import CogtrixState
from src.agent.safety import UserCancelledRun
from src.agent.safety import create_safe_tool_wrapper as _safe_wrap
from src.logging_config import get_logger, is_trace
from src.orchestration.compression import (
    _CHARS_PER_TOKEN,
    _EMERGENCY_THRESHOLD_RATIO,
    _MID_TURN_COMPRESSION_THRESHOLD,
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    _content_len,
    apply_message_compression,
)
from src.orchestration.run_config import AgentRunConfig
from src.orchestration.session_state import SessionState
from src.tools.configure import (
    TOOL_OUTPUT_CAP_MIN_CHARS,
    apply_output_cap,
    build_tool_catalog,
    compute_tool_output_cap,
    configure_delegate_tools,
    create_request_tools_tool,
)
from src.tools.resolver import resolve_tool_name as _resolve_tool_name

DEFAULT_RECURSION_LIMIT = 90
EMPTY_RESPONSE_MSG = "**Error:** The model returned an empty response. Please try again."
_PARALLEL_TOOL_WORKERS = 8

_TOOL_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_TOOL_EXECUTOR_LOCK = threading.Lock()


def _get_tool_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level parallel tool executor, creating it on first use."""
    global _TOOL_EXECUTOR
    if _TOOL_EXECUTOR is None:
        with _TOOL_EXECUTOR_LOCK:
            if _TOOL_EXECUTOR is None:
                _TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_PARALLEL_TOOL_WORKERS,
                    thread_name_prefix="tool",
                )
                atexit.register(_TOOL_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _TOOL_EXECUTOR


_INVALID_TOOL_RE = re.compile(r"^Error:\s*(\S+)\s+is not a valid tool")

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
    """Return True if *exc* is a provider context-length rejection."""
    msg = str(exc).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS)


# ── Action-intent detection ───────────────────────────────────────────────────
# Catches "I'll create X" / "Let me write Y" responses that contain no tool
# calls — the model expressed intent but didn't act on it.

# Phrases that introduce a planned future action
_INTENT_LEAD_RE = re.compile(
    r"\b(?:"
    # First-person modal / future forms
    r"I(?:"
    r"'ll\b"  # I'll
    r"|\s+will\b"  # I will
    r"|\s+am\s+going\s+to\b"  # I am going to
    r"|'m\s+going\s+to\b"  # I'm going to
    r"|'m\s+about\s+to\b"  # I'm about to
    r"|\s+need\s+to\b"  # I need to
    r"|\s+have\s+to\b"  # I have to
    r"|\s+should\b"  # I should
    r"|\s+must\b"  # I must
    r"|\s+will\s+now\b"  # I will now
    r"|'ll\s+now\b"  # I'll now
    r"|'m\s+now\b"  # I'm now [verb-ing]
    r"|'ll\s+(?:proceed|go\s+ahead|start|begin|continue)\b"  # I'll proceed/...
    r")"
    # Imperative / collaborative
    r"|Let(?:'s|\s+(?:me|us))\b"  # Let me / Let's / Let us
    # Temporal modifiers + intent
    r"|Now\s+(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # Now I'll / Now let me
    r"|(?:First|Next|Then|Finally|Additionally|Also)(?:\s*,)?\s+"
    r"(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # First/Next/Then ... I'll
    # Implicit future
    r"|(?:Going|About)\s+to\b"  # Going to / About to
    r"|Time\s+to\b"  # Time to
    r"|I\s+can\s+now\b"  # I can now
    r")",
    re.IGNORECASE,
)

# Verb stems that indicate a tool-requiring operation.
# Long stems (5+ chars) use \w{0,8} for inflection coverage.
# Short/ambiguous stems use explicit endings to prevent overmatch.
_TOOL_VERB_RE = re.compile(
    r"\b(?:"
    # File / content operations — long stems
    r"creat\w{0,8}|generat\w{0,8}|overwrite\w{0,8}|append\w{0,8}"
    r"|delet\w{0,8}|remov\w{0,8}|replac\w{0,8}|insert\w{0,8}|rename\w{0,8}"
    r"|modif\w{0,8}|patch\w{0,8}|refactor\w{0,8}"
    # Read / fetch
    r"|fetch\w{0,8}|retriev\w{0,8}|download\w{0,8}" r"|search\w{0,8}|crawl\w{0,8}|scrape\w{0,8}"
    # Build / execution
    r"|build\w{0,8}|compil\w{0,8}|execut\w{0,8}|launch\w{0,8}"
    r"|install\w{0,8}|deploy\w{0,8}|configur\w{0,8}|initializ\w{0,8}"
    r"|bootstrap\w{0,8}|provision\w{0,8}|scaffold\w{0,8}"
    # Code / dev
    r"|implement\w{0,8}|develop\w{0,8}|debug\w{0,8}|resolv\w{0,8}"
    # Network / API
    r"|upload\w{0,8}|commit\w{0,8}|invok\w{0,8}|submit\w{0,8}|request\w{0,8}"
    r"|publish\w{0,8}|broadcast\w{0,8}"
    # Data processing
    r"|transform\w{0,8}|extract\w{0,8}|analyz\w{0,8}|analys\w{0,8}"
    r"|comput\w{0,8}|calculat\w{0,8}|process\w{0,8}|pars\w{0,8}"
    r"|export\w{0,8}|migrat\w{0,8}|convert\w{0,8}"
    # Infra
    r"|register\w{0,8}|connect\w{0,8}|verif\w{0,8}|inspect\w{0,8}|examin\w{0,8}" r"|clone\w{0,8}"
    # Short stems — explicit inflections only
    r"|writ(?:e|es|ing|ten)\b"
    r"|read(?:s|ing)?\b"
    r"|open(?:s|ing|ed)?\b"
    r"|load(?:s|ing|ed)?\b"
    r"|run(?:s|ning)?\b"
    r"|start(?:s|ing|ed)?\b"
    r"|send(?:s|ing)?\b|sent\b"
    r"|post(?:s|ing|ed)?\b"
    r"|call(?:s|ing|ed)?\b"
    r"|pull(?:s|ing|ed)?\b"
    r"|push(?:es|ing|ed)?\b"
    r"|fix(?:es|ing|ed)?\b"
    r"|test(?:s|ing|ed)?\b"
    r"|edit(?:s|ing|ed)?\b"
    r"|update(?:s|d|ing)?\b"
    r"|sav(?:e|es|ing|ed)\b"
    r"|stor(?:e|es|ing|ed)\b"
    r"|add(?:s|ing|ed)?\b"
    r"|cod(?:e|es|ing|ed)\b"
    r"|defin(?:e|es|ing|ed)\b"
    r"|list(?:s|ing|ed)?\b"
    r"|output(?:s|ting|ted)?\b"
    r"|check(?:s|ing|ed)?\b"
    # Multi-word constructions
    r"|set\s+up\b" r"|spin\s+up\b" r"|look\s+up\b" r"|wire\s+up\b" r"|stand\s+up\b" r")\b",
    re.IGNORECASE,
)


def _is_action_intent(message: Any) -> bool:
    """Return True when the model describes a planned action but emits no tool calls.

    Checks for an intent-lead phrase (``I'll``, ``Let me``, ``Going to``, etc.)
    paired with a tool-action verb (``create``, ``run``, ``fetch``, etc.).
    The two-part AND prevents false positives on pure-text responses like
    ``I'll explain...`` where the verb is not tool-requiring.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False
    return bool(_INTENT_LEAD_RE.search(text)) and bool(_TOOL_VERB_RE.search(text))


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
    return ToolManagementRequest(add=all_add, remove=all_remove)


def _detect_invalid_tool_calls(
    messages: list,
    start_idx: int = 0,
) -> list[str]:
    """
    Scan *messages* from *start_idx* for **any** "is not a valid tool"
    ToolMessage error, regardless of whether the tool is in the on-demand
    pool.

    Returns a de-duplicated, ordered list of tool names the LLM tried.
    """
    from langchain_core.messages import ToolMessage

    found: list[str] = []
    seen: set[str] = set()
    for i in range(start_idx, len(messages)):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        m = _INVALID_TOOL_RE.match(content)
        if m:
            tool_name = m.group(1)
            if tool_name not in seen:
                found.append(tool_name)
                seen.add(tool_name)
    return found


def _strip_failed_tool_messages(messages: list, tool_names: set[str]) -> list:
    """
    Return a copy of *messages* with ToolMessage errors (and their matching
    AIMessage tool_calls) removed for tools in *tool_names*.

    This cleans up the conversation history after auto-activation so the
    resumed agent doesn't see the failed "is not a valid tool" attempts.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    tool_call_ids_to_remove: set[str] = set()
    cleaned: list = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = getattr(msg, "content", "")
            if name in tool_names and isinstance(content, str) and "is not a valid tool" in content:
                tcid = getattr(msg, "tool_call_id", "")
                if tcid:
                    tool_call_ids_to_remove.add(tcid)
                continue
        cleaned.append(msg)

    if not tool_call_ids_to_remove:
        return cleaned

    final: list = []
    for msg in cleaned:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                remaining = [tc for tc in tool_calls if tc.get("id") not in tool_call_ids_to_remove]
                if len(remaining) != len(tool_calls):
                    extra = dict(getattr(msg, "additional_kwargs", {}))
                    extra.pop("tool_calls", None)
                    new_msg = AIMessage(
                        content=getattr(msg, "content", ""),
                        tool_calls=remaining,
                        additional_kwargs=extra,
                    )
                    if not remaining and not (
                        isinstance(new_msg.content, str) and new_msg.content.strip()
                    ):
                        continue
                    final.append(new_msg)
                    continue
        final.append(msg)
    return final


_FUZZY_ARG_BLOCKLIST: frozenset[str] = frozenset(
    {
        "data",
        "name",
        "port",
        "code",
        "type",
        "text",
        "path",
        "file",
        "mode",
        "size",
        "body",
        "host",
        "user",
        "role",
        "args",
        "keys",
    }
)


def _correct_tool_args(tool: Any, args: dict) -> dict:
    """Best-effort correction of misnamed tool arguments.

    Weaker LLMs sometimes send wrong parameter names (e.g. ``cmd`` instead of
    ``command``).  This function compares provided keys against the tool's
    Pydantic ``args_schema`` and applies two heuristics:

    1. **Fuzzy name match** — uses substring containment and SequenceMatcher
       to remap unknown arg names to the closest expected field.
    2. **Type coercion** — if the schema expects ``str`` and the value is a
       ``list`` or ``dict``, serialise it to a JSON string.

    Returns the (possibly corrected) args dict.  On any error, returns the
    original args unchanged.
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return args

    try:
        expected: dict[str, Any] = {}
        if hasattr(schema, "model_fields"):
            expected = schema.model_fields  # Pydantic v2
        elif hasattr(schema, "__fields__"):
            expected = schema.__fields__  # Pydantic v1
        if not expected:
            return args
    except Exception:
        return args

    expected_names = set(expected.keys())
    provided_names = set(args.keys())

    corrected = dict(args)

    # --- Name remapping ---------------------------------------------------
    unknown = provided_names - expected_names
    missing = expected_names - provided_names

    if unknown and missing:
        _REMAP_THRESHOLD = 0.75
        for unk in unknown:
            unk_lower = unk.lower()
            best: str | None = None
            best_ratio = 0.0
            tied = False
            for exp in missing:
                exp_lower = exp.lower()
                # Substring containment — only trust when the shorter
                # string is long enough to be meaningful.
                shorter_len = min(len(unk_lower), len(exp_lower))
                longer_len = max(len(unk_lower), len(exp_lower))
                if (
                    shorter_len >= 5
                    and shorter_len / longer_len >= 0.5
                    and unk_lower not in _FUZZY_ARG_BLOCKLIST
                    and (unk_lower in exp_lower or exp_lower in unk_lower)
                ):
                    ratio = 1.0
                else:
                    ratio = SequenceMatcher(None, unk_lower, exp_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = exp
                    tied = False
                elif abs(ratio - best_ratio) < 1e-9 and ratio >= _REMAP_THRESHOLD:
                    tied = True
            if best is not None and best_ratio >= _REMAP_THRESHOLD and not tied:
                corrected[best] = corrected.pop(unk)
                missing.discard(best)
                log = get_logger()
                log.info("Tool arg corrected: '%s' → '%s' (score=%.2f)", unk, best, best_ratio)

    # --- Type coercion: schema expects str but got list/dict → JSON-encode.
    for key, value in list(corrected.items()):
        if key not in expected:
            continue
        if not isinstance(value, (list, dict)):
            continue
        field_info = expected[key]
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        # Unwrap Optional[str] / str | None → str
        origin = typing.get_origin(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType):
            type_args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(type_args) == 1:
                annotation = type_args[0]
        if annotation is str:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                corrected[key] = " ".join(value)
            else:
                corrected[key] = _json.dumps(value)

    return corrected


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
    tool_call_guard: Any | None = None,
    session_state: SessionState | None = None,
    confirmation_ui: Any | None = None,
    on_tool_expansion: Any | None = None,
    parallel_tool_execution: bool = True,
    git_native: bool = False,
    tool_context_limit_pct: float = 0.80,
    *,
    config: AgentRunConfig | None = None,
    bound_cache: OrderedDict | None = None,
    compression_cache_in: dict[str, str] | None = None,
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
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, StateGraph

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
        if config.confirmation_ui is not None:
            confirmation_ui = config.confirmation_ui
        if config.on_tool_expansion is not None:
            on_tool_expansion = config.on_tool_expansion
        parallel_tool_execution = config.parallel_tool_execution
        git_native = config.git_native
        context_compression = config.context_compression
        if config.compression_llm is not None:
            compression_llm = config.compression_llm
        if config.compression_min_age is not None:
            compression_min_age = config.compression_min_age
        if config.compression_min_chars is not None:
            compression_min_chars = config.compression_min_chars
        if hasattr(config, "tool_context_limit_pct"):
            tool_context_limit_pct = config.tool_context_limit_pct
        _tier_cache_enabled = getattr(config, "tier_cache_enabled", True)
    else:
        _tier_cache_enabled = False

    if active_tools_list is None:
        active_tools_list = []
    if available_tools is None:
        available_tools = {}
    if approvals is None:
        approvals = set()

    if session_state is None:
        session_state = SessionState()

    phantom_count = [0]
    action_intent_count = [0]
    expansion_count = [0]
    auto_expansion_count = [0]
    call_count = [0]
    _last_input_tokens = [0]  # actual input tokens from the previous model call
    _MAX_PHANTOM_RETRIES = 3
    _MAX_ACTION_INTENT_RETRIES = 3
    _MAX_TOOL_EXPANSIONS = 3
    request_tools_noop_count = [0]
    _MAX_REQUEST_TOOLS_NOOPS = 3
    _tool_call_history: OrderedDict[str, str] = OrderedDict()
    _MAX_TOOL_CALL_HISTORY = 256
    _history_lock = threading.Lock()
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
    _bound_cache: OrderedDict[tuple[str, ...], Any] = (
        bound_cache if bound_cache is not None else OrderedDict()
    )
    # Lock protecting the per-graph _bound_cache; mirrors the module-level
    # _persistent_bound_cache pattern in runner.py (BUG-233).
    _bound_cache_lock = threading.Lock()
    _tool_version = [0]
    _last_tool_version = [-1]
    _cached_fingerprint: list[tuple[str, ...]] = [()]
    output_cap = (
        compute_tool_output_cap(max_context_tokens)
        if max_context_tokens
        else TOOL_OUTPUT_CAP_MIN_CHARS
    )
    _sys_msg = SystemMessage(content=system_prompt) if system_prompt else None
    _tool_lookup: dict[str, Any] = {getattr(t, "name", ""): t for t in active_tools_list}
    _tool_lookup.pop("", None)
    _active_names: set[str] = set(_tool_lookup.keys())
    tool_catalog: dict[str, str] = build_tool_catalog(available_tools)

    _compression_cache: dict[str, str] = (
        compression_cache_in if compression_cache_in is not None else {}
    )

    _graph_log = get_logger()

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
        last_tokens = _last_input_tokens[0]
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
            call_count=call_count[0],
            compression_cache=_compression_cache,
            llm=_comp_llm,
            max_context_tokens=max_context_tokens,
            min_age_cycles=compression_min_age,
            min_chars=compression_min_chars,
            min_age_override=min_age_ovr,
            actual_input_tokens=last_tokens,
        )

    def call_model(state: CogtrixState, config: RunnableConfig) -> dict:
        if llm is None:
            raise RuntimeError(
                "LLM not configured — check provider settings, API keys, and config file"
            )
        _cm_t0 = time.monotonic()
        call_count[0] += 1
        if _tool_version[0] != _last_tool_version[0]:
            _cached_fingerprint[0] = (
                tuple(getattr(t, "name", "") for t in active_tools_list)
                if active_tools_list
                else ()
            )
            _last_tool_version[0] = _tool_version[0]
        fingerprint = _cached_fingerprint[0]
        with _bound_cache_lock:
            if fingerprint in _bound_cache:
                _bound_cache.move_to_end(fingerprint)
            else:
                tool_list = list(active_tools_list) if active_tools_list else []
                if len(_bound_cache) >= 8:
                    _bound_cache.popitem(last=False)
                _bound_cache[fingerprint] = llm.bind_tools(tool_list) if tool_list else llm
            model = _bound_cache[fingerprint]
        if is_trace():
            _graph_log.debug("⏱ call_model bind_tools: %.0fms", (time.monotonic() - _cm_t0) * 1000)
        msgs = list(state["messages"])
        _comp_llm = compression_llm or llm
        # Pre-invoke compression: char-based estimate fires at 60% threshold
        # before every model.invoke() — catches mid-turn context growth that
        # the previous token-based check (using stale _last_input_tokens) would miss.
        msgs = _maybe_compress(msgs)
        full_messages = [_sys_msg, *msgs] if _sys_msg is not None else list(msgs)
        _cm_t1 = time.monotonic()
        try:
            response = model.invoke(full_messages, config)
        except Exception as _invoke_exc:
            if not _is_context_overflow_error(_invoke_exc):
                raise
            _graph_log.warning(
                "Context overflow from model (%s) — applying emergency compression and retrying",
                type(_invoke_exc).__name__,
            )
            msgs = apply_message_compression(
                msgs,
                call_count=call_count[0],
                compression_cache=_compression_cache,
                llm=_comp_llm,
                max_context_tokens=max_context_tokens,
                min_age_cycles=0,
                min_chars=0,
                emergency_threshold=0.0,
                actual_input_tokens=_last_input_tokens[0],
            )
            full_messages = [_sys_msg, *msgs] if _sys_msg is not None else list(msgs)
            try:
                response = model.invoke(full_messages, config)
            except Exception as _retry_exc:
                raise RuntimeError(
                    f"Context overflow: unable to fit conversation into model context window "
                    f"({max_context_tokens:,} tokens) even after emergency compression. "
                    "Start a new session with /session new."
                ) from _retry_exc
        if is_trace():
            _graph_log.debug(
                "⏱ call_model model.invoke: %.0fms", (time.monotonic() - _cm_t1) * 1000
            )
        # Store actual input token count for next-turn compression trigger
        _resp_um = getattr(response, "usage_metadata", None)
        if _resp_um and isinstance(_resp_um, dict):
            _resp_input = _resp_um.get("input_tokens", 0)
            if isinstance(_resp_input, int) and _resp_input > 0:
                _last_input_tokens[0] = _resp_input
        # ── Per-turn context budget guard ────────────────────────────────────
        # If the model wants more tool calls but we've already consumed a large
        # fraction of the context window this turn, abort the loop by stripping
        # tool_calls from the response and injecting a warning.
        if max_context_tokens and getattr(response, "tool_calls", None):
            um = getattr(response, "usage_metadata", None)
            if um and isinstance(um, dict):
                turn_input = um.get("input_tokens", 0)
                if turn_input > max_context_tokens * tool_context_limit_pct:
                    pct_used = int(turn_input * 100 / max_context_tokens)
                    warning = (
                        f"[Context budget reached — {pct_used}% of {max_context_tokens:,} "
                        f"tokens used this turn (limit: {int(tool_context_limit_pct * 100)}%). "
                        "Tool execution halted. Summarising based on available information.]"
                    )
                    response = AIMessage(
                        content=warning,
                        id=getattr(response, "id", None),
                    )
        return {"messages": [response]}

    def handle_phantom(state: CogtrixState) -> dict:
        phantom_count[0] += 1
        msgs = state["messages"]
        last = msgs[-1]
        log = get_logger()
        log.warning(
            "Phantom tool call detected, attempt %d/%d. Injecting hint.",
            phantom_count[0],
            _MAX_PHANTOM_RETRIES,
        )
        if phantom_count[0] > _MAX_PHANTOM_RETRIES:
            return {
                "messages": [
                    RemoveMessage(id=last.id),
                    AIMessage(
                        content=(
                            "I encountered persistent formatting issues with tool calls "
                            "and could not complete the request. Please try rephrasing "
                            "your question, or I can try to answer based on what I know."
                        )
                    ),
                ]
            }
        return {
            "messages": [
                RemoveMessage(id=last.id),
                HumanMessage(
                    content=(
                        "Your last tool call could not be parsed by the server. "
                        "The JSON was malformed. Please try your tool call again "
                        "with carefully formatted JSON arguments, or if you have "
                        "enough information, provide your answer directly."
                    )
                ),
            ]
        }

    def handle_action_intent(state: CogtrixState) -> dict:
        action_intent_count[0] += 1
        log = get_logger()
        log.warning(
            "Action-intent without tool call detected, attempt %d/%d. Injecting nudge.",
            action_intent_count[0],
            _MAX_ACTION_INTENT_RETRIES,
        )
        if action_intent_count[0] > _MAX_ACTION_INTENT_RETRIES:
            # Give up and let the model's last response stand as-is.
            return {"messages": []}
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "You described an action but did not call any tools. "
                        "Please proceed now: call the appropriate tool(s) to carry "
                        "out what you described, rather than explaining it in text."
                    )
                )
            ]
        }

    def _tool_call_key(call: dict) -> str | None:
        """Compute the deduplication key for a tool call, or None if not serializable.

        Normalizes to the canonical tool name via ``_tool_lookup`` so that an
        alias and its resolved canonical name share the same cache key.  When the
        alias is not in ``_tool_lookup`` (e.g. during the auto-expansion serial
        path) the raw call name is used instead (BUG-234).
        """
        tool_name = call["name"]
        if tool_name in _DUPLICATE_EXEMPT:
            return None
        # Prefer the canonical name stored on the live tool object.
        tool_obj = _tool_lookup.get(tool_name)
        if tool_obj is not None:
            tool_name = getattr(tool_obj, "name", tool_name) or tool_name
        try:
            return tool_name + ":" + _json.dumps(call.get("args", {}))
        except (TypeError, ValueError):
            return None

    def _check_duplicate(call: dict, key: str | None = None) -> ToolMessage | None:
        """Return a cached ToolMessage if this exact call was seen before."""
        tool_name = call["name"]
        if key is None:
            key = _tool_call_key(call)
        if key is None:
            return None
        with _history_lock:
            cached = _tool_call_history.get(key)
            if cached is not None:
                _tool_call_history.move_to_end(key)
        if cached is None:
            return None
        log = get_logger()
        log.warning("Duplicate tool call detected: %s (returning cached result)", tool_name)
        return ToolMessage(
            content=(
                "[Duplicate call — returning cached result. "
                "Do NOT repeat this call.]\n\n" + cached
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
            _tool_call_history[key] = result_text[:500]
            _tool_call_history.move_to_end(key)
            if len(_tool_call_history) > _MAX_TOOL_CALL_HISTORY:
                _tool_call_history.popitem(last=False)

    def _invoke_one(call: dict, run_config: Any) -> Any:
        """Execute a single tool call already in tool_lookup. Returns ToolMessage."""
        call_key = _tool_call_key(call)
        dup = _check_duplicate(call, key=call_key)
        if dup is not None:
            return dup

        tool_name = call["name"]
        tool_input = {**call, "type": "tool_call"}

        if tool_call_guard is not None:
            _guard_result = tool_call_guard(tool_name, call.get("args", {}))
            if hasattr(_guard_result, "is_safe") and not _guard_result.is_safe:
                log = get_logger()
                log.warning(
                    "Tool call blocked [%s]: %s — %s",
                    getattr(_guard_result, "guard_name", ""),
                    tool_name,
                    getattr(_guard_result, "reason", ""),
                )
                return ToolMessage(
                    content=(
                        f"Tool call blocked by security policy: "
                        f"{getattr(_guard_result, 'reason', 'blocked')}"
                    ),
                    tool_call_id=call["id"],
                    name=tool_name,
                )
        try:
            tool = _tool_lookup.get(tool_name)
            if tool is None:
                return ToolMessage(
                    content=f"Tool '{tool_name}' is no longer active.",
                    tool_call_id=call["id"],
                    name=tool_name,
                )
            result = tool.invoke(tool_input, run_config)
            if isinstance(result, ToolMessage):
                _store_call_result(
                    call,
                    result.content if isinstance(result.content, str) else "",
                    key=call_key,
                )
                return result
            text = str(result) if result is not None else ""
            _store_call_result(call, text, key=call_key)
            return ToolMessage(
                content=text,
                tool_call_id=call["id"],
                name=tool_name,
            )
        except UserCancelledRun:
            raise
        except Exception as exc:
            log = get_logger()
            log.warning("Tool %s raised: %s", tool_name, exc, exc_info=True)
            return ToolMessage(
                content=f"Error executing {tool_name}: {exc}",
                tool_call_id=call["id"],
                name=tool_name,
            )

    def process_tools(state: CogtrixState, config: RunnableConfig) -> dict:
        log = get_logger()
        msgs = state["messages"]
        last = msgs[-1]

        if not (isinstance(last, AIMessage) and last.tool_calls):
            return {"messages": []}

        tool_lookup_ref = _tool_lookup
        active_names_ref = _active_names

        result_msgs: list = []
        tools_activated: list[str] = []
        tools_released: list[str] = []
        guidance_lines: list[str] = []
        saw_request_tools = False

        # ── Classification pass ──────────────────────────────────
        if parallel_tool_execution and len(last.tool_calls) > 1:
            snapshot_names = set(tool_lookup_ref.keys())
            serial_first: list = []
            parallel_calls: list = []
            for call in last.tool_calls:
                name = call["name"]
                if name == "request_tools" or name not in snapshot_names:
                    serial_first.append(call)
                else:
                    parallel_calls.append(call)
        else:
            serial_first = list(last.tool_calls)
            parallel_calls = []

        # ── Serial-first execution (expansion, request_tools) ────
        cancel_requested = False
        for call in serial_first:
            tool_name = call["name"]

            if tool_name in tool_lookup_ref:
                try:
                    msg = _invoke_one(call, config)
                except UserCancelledRun:
                    cancel_requested = True
                    msg = ToolMessage(
                        content="User cancelled agent workflow",
                        tool_call_id=call["id"],
                        name=tool_name,
                    )
                result_msgs.append(msg)
                if cancel_requested:
                    break
                if tool_name == "request_tools":
                    saw_request_tools = True
            else:
                can_expand = auto_expansion_count[0] < _MAX_TOOL_EXPANSIONS

                if can_expand and available_tools:
                    match, source = _resolve_tool_name(
                        tool_name,
                        available_tools,
                        active_names_ref,
                    )
                else:
                    match, source = None, ""

                if match and source == "available":
                    if match in session_state.denials:
                        result_msgs.append(
                            ToolMessage(
                                content=f"Tool '{match}' is disabled by the user.",
                                tool_call_id=call["id"],
                                name=tool_name,
                            )
                        )
                        continue
                    tool_obj = available_tools.pop(match)
                    tool_catalog.pop(match, None)
                    apply_output_cap(tool_obj, output_cap)
                    if registry is not None and registry.requires_confirmation(match):
                        if session_state.no_confirm:
                            approvals.add(match)
                        tool_obj = _safe_wrap(
                            tool_obj,
                            match,
                            registry,
                            approvals,
                            session_state=session_state,
                            ui=confirmation_ui,
                            git_native=git_native,
                        )
                    active_tools_list.append(tool_obj)
                    active_names_ref.add(match)
                    tool_lookup_ref[match] = tool_obj
                    tools_activated.append(match)
                    session_state.loaded_tools.add(match)
                    auto_expansion_count[0] += 1

                    if match != tool_name:
                        guidance_lines.append(
                            f"'{tool_name}' resolved to '{match}' (now activated)."
                        )

                    if tool_call_guard is not None:
                        _guard_result = tool_call_guard(match, call.get("args", {}))
                        if hasattr(_guard_result, "is_safe") and not _guard_result.is_safe:
                            log.warning(
                                "Tool call blocked [%s]: %s — %s",
                                getattr(_guard_result, "guard_name", ""),
                                match,
                                getattr(_guard_result, "reason", ""),
                            )
                            result_msgs.append(
                                ToolMessage(
                                    content=(
                                        f"Tool call blocked by security policy: "
                                        f"{getattr(_guard_result, 'reason', 'blocked')}"
                                    ),
                                    tool_call_id=call["id"],
                                    name=tool_name,
                                )
                            )
                            continue
                    try:
                        corrected_args = _correct_tool_args(tool_obj, call.get("args", {}))
                        corrected_input = {
                            **call,
                            "name": match,
                            "type": "tool_call",
                            "args": corrected_args,
                        }
                        result = tool_obj.invoke(corrected_input, config)
                        # Store cache key under the resolved name so
                        # deduplication works for both alias and canonical
                        # name (BUG-198).
                        resolved_call = {**call, "name": match}
                        if isinstance(result, ToolMessage):
                            _store_call_result(
                                resolved_call,
                                result.content if isinstance(result.content, str) else "",
                            )
                            result_msgs.append(result)
                        else:
                            text = str(result) if result is not None else ""
                            _store_call_result(resolved_call, text)
                            result_msgs.append(
                                ToolMessage(
                                    content=text,
                                    tool_call_id=call["id"],
                                    name=match,
                                )
                            )
                    except UserCancelledRun:
                        cancel_requested = True
                        result_msgs.append(
                            ToolMessage(
                                content="User cancelled agent workflow",
                                tool_call_id=call["id"],
                                name=match,
                            )
                        )
                        break
                    except Exception as exc:
                        log.warning("Tool %s raised: %s", match, exc, exc_info=True)
                        result_msgs.append(
                            ToolMessage(
                                content=f"Error executing {match}: {exc}",
                                tool_call_id=call["id"],
                                name=match,
                            )
                        )

                elif match and source == "active":
                    guidance_lines.append(
                        f"'{tool_name}' is not a tool name. "
                        f"Use the already-active tool '{match}' instead."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"'{tool_name}' is not a valid tool. "
                                f"Did you mean '{match}'? It is already active."
                            ),
                            tool_call_id=call["id"],
                            name=tool_name,
                        )
                    )
                else:
                    guidance_lines.append(f"'{tool_name}' does not match any known tool.")
                    result_msgs.append(
                        ToolMessage(
                            content=f"'{tool_name}' is not a valid tool and could not be resolved.",
                            tool_call_id=call["id"],
                            name=tool_name,
                        )
                    )

        # ── Parallel execution ───────────────────────────────────
        if cancel_requested:
            for call in parallel_calls:
                result_msgs.append(
                    ToolMessage(
                        content="User cancelled agent workflow",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
        elif len(parallel_calls) == 1:
            try:
                result_msgs.append(_invoke_one(parallel_calls[0], config))
            except UserCancelledRun:
                cancel_requested = True
                result_msgs.append(
                    ToolMessage(
                        content="User cancelled agent workflow",
                        tool_call_id=parallel_calls[0]["id"],
                        name=parallel_calls[0]["name"],
                    )
                )
        elif parallel_calls:
            pool = _get_tool_executor()
            futures = [(call, pool.submit(_invoke_one, call, config)) for call in parallel_calls]
            for call, future in futures:
                try:
                    # 10-minute timeout prevents indefinite hangs from stuck
                    # tool calls (BUG-202).  On timeout, produce an error
                    # ToolMessage so LangGraph's 1:1 tool_call_id mapping
                    # is preserved.
                    result_msgs.append(future.result(timeout=600))
                except (TimeoutError, concurrent.futures.TimeoutError):
                    log.warning("Tool '%s' timed out after 600s", call["name"])
                    # Cancel the future to prevent zombie threads
                    future.cancel()
                    result_msgs.append(
                        ToolMessage(
                            content=f"Error: tool '{call['name']}' timed out after 10 minutes",
                            tool_call_id=call["id"],
                            name=call["name"],
                        )
                    )
                except UserCancelledRun:
                    cancel_requested = True
                    result_msgs.append(
                        ToolMessage(
                            content="User cancelled agent workflow",
                            tool_call_id=call["id"],
                            name=call["name"],
                        )
                    )
                    break
                except Exception as exc:
                    log.error("Unexpected future exception: %s", exc, exc_info=True)
                    result_msgs.append(
                        ToolMessage(
                            content=f"Error executing {call['name']}: {exc}",
                            tool_call_id=call["id"],
                            name=call["name"],
                        )
                    )

            if cancel_requested:
                # Note: future.cancel() only prevents not-yet-started futures.
                # In-flight futures complete naturally; the persistent pool keeps threads
                # alive for the next batch. This is intentional — abrupt thread
                # termination could leave resources in an inconsistent state.
                processed_ids = {m.tool_call_id for m in result_msgs if hasattr(m, "tool_call_id")}
                for call, future in futures:
                    if call["id"] not in processed_ids:
                        future.cancel()
                        result_msgs.append(
                            ToolMessage(
                                content="User cancelled agent workflow",
                                tool_call_id=call["id"],
                                name=call["name"],
                            )
                        )

        if cancel_requested:
            raise UserCancelledRun()

        if saw_request_tools:
            mgmt_req = _detect_tool_request(
                [last],
                start_idx=0,
            )
            if mgmt_req and mgmt_req.has_changes:
                for rname in mgmt_req.add:
                    # Fuzzy-resolve names the LLM may have abbreviated
                    if rname not in available_tools:
                        resolved, source = _resolve_tool_name(
                            rname,
                            available_tools,
                            active_names_ref,
                        )
                        if resolved and source == "available":
                            rname = resolved
                    if rname in available_tools and rname not in tools_activated:
                        tool_obj = available_tools.pop(rname)
                        tool_catalog.pop(rname, None)
                        apply_output_cap(tool_obj, output_cap)
                        if registry is not None and registry.requires_confirmation(rname):
                            if session_state.no_confirm:
                                approvals.add(rname)
                            tool_obj = _safe_wrap(
                                tool_obj,
                                rname,
                                registry,
                                approvals,
                                session_state=session_state,
                                ui=confirmation_ui,
                                git_native=git_native,
                            )
                        active_tools_list.append(tool_obj)
                        active_names_ref.add(rname)
                        tool_lookup_ref[rname] = tool_obj
                        tools_activated.append(rname)
                        session_state.loaded_tools.add(rname)

                for rname in mgmt_req.remove:
                    # Fuzzy-resolve against active pool only (not available)
                    if rname not in active_names_ref and rname not in protected:
                        resolved, source = _resolve_tool_name(
                            rname,
                            {},
                            active_names_ref,
                        )
                        if resolved and source == "active":
                            rname = resolved
                    if rname in tools_activated:
                        continue
                    if rname in protected:
                        guidance_lines.append(
                            f"'{rname}' is core to this mode and cannot be released."
                        )
                    elif rname in active_names_ref:
                        idx = next(
                            (
                                i
                                for i, t in enumerate(active_tools_list)
                                if getattr(t, "name", None) == rname
                            ),
                            None,
                        )
                        if idx is not None:
                            popped = active_tools_list.pop(idx)
                            active_names_ref.discard(rname)
                            original = session_state.all_tool_originals.get(rname, popped)
                            available_tools[rname] = original
                            tool_catalog.update(build_tool_catalog({rname: original}))
                            tools_released.append(rname)
                            session_state.loaded_tools.discard(rname)
                            if rname in tool_lookup_ref:
                                del tool_lookup_ref[rname]
                    else:
                        guidance_lines.append(f"'{rname}' is not in the active set.")

        # Circuit-breaker for repeated request_tools no-ops
        if saw_request_tools:
            if tools_activated or tools_released:
                request_tools_noop_count[0] = 0
            else:
                request_tools_noop_count[0] += 1
                if request_tools_noop_count[0] >= _MAX_REQUEST_TOOLS_NOOPS:
                    log.warning(
                        "request_tools circuit-breaker: %d consecutive no-op calls",
                        request_tools_noop_count[0],
                    )
                    guidance_lines.append(
                        "STOP: You have made multiple unsuccessful attempts to "
                        "manage tools. Work with the tools you already have, or "
                        "tell the user you cannot complete the task with the "
                        "available tools."
                    )

        if tools_activated or tools_released:
            expansion_count[0] += 1
            _tool_version[0] += 1

            active_tools_list[:] = [
                t for t in active_tools_list if getattr(t, "name", "") != "request_tools"
            ]
            _tool_lookup.pop("request_tools", None)
            releasable = active_names_ref - protected - {"request_tools"}
            if available_tools or releasable:
                rt = create_request_tools_tool(
                    available_tools,
                    tool_catalog,
                    active_names=active_names_ref,
                    protected_names=protected,
                )
                if rt:
                    active_tools_list.append(rt)
                    _tool_lookup["request_tools"] = rt

            configure_delegate_tools(active_tools_list, available_tools)

            visible_count = sum(
                1 for t in active_tools_list if getattr(t, "name", "") != "request_tools"
            )
            log.info(
                "Tool expansion round %d (auto: %d) — added: %s, released: %s (%d total)",
                expansion_count[0],
                auto_expansion_count[0],
                tools_activated,
                tools_released,
                visible_count,
            )
            if on_tool_expansion is not None:
                on_tool_expansion(tools_activated, tools_released, visible_count)

            note_parts: list[str] = []
            if tools_activated:
                note_parts.append(
                    "The following tools have been added to your toolkit: "
                    f"{', '.join(tools_activated)}. You can now use them."
                )
            if tools_released:
                note_parts.append(
                    "The following tools have been released: "
                    f"{', '.join(tools_released)}. "
                    "They are back in the catalog if you need them again."
                )
            if guidance_lines:
                note_parts.append(" ".join(guidance_lines))
            note_parts.append("Continue with your task.")
            result_msgs.append(HumanMessage(content=" ".join(note_parts)))
        elif guidance_lines:
            result_msgs.append(
                HumanMessage(content=" ".join(guidance_lines) + " Continue with your task.")
            )

        return {"messages": result_msgs}

    def route_after_model(state: CogtrixState) -> str:
        msgs = state["messages"]
        if not msgs:
            return END

        last = msgs[-1]
        if isinstance(last, AIMessage):
            content = getattr(last, "content", "")
            has_content = isinstance(content, str) and bool(content.strip())
            tool_calls = getattr(last, "tool_calls", None)

            if not has_content and not tool_calls:
                meta = getattr(last, "response_metadata", None)
                if meta and isinstance(meta, dict):
                    if meta.get("finish_reason") == "tool_calls":
                        return "handle_phantom"
                return END

            if tool_calls:
                return "process_tools"

            # Has content but no tool calls — check for intention-without-action.
            if _is_action_intent(last):
                return "handle_action_intent"

        return END

    def route_after_phantom(state: CogtrixState) -> str:
        if phantom_count[0] > _MAX_PHANTOM_RETRIES:
            return END
        return "call_model"

    def route_after_action_intent(state: CogtrixState) -> str:  # noqa: ARG001
        if action_intent_count[0] > _MAX_ACTION_INTENT_RETRIES:
            return END
        return "call_model"

    graph: Any = StateGraph(CogtrixState)
    graph.add_node("call_model", call_model)
    graph.add_node("handle_phantom", handle_phantom)
    graph.add_node("handle_action_intent", handle_action_intent)
    graph.add_node("process_tools", process_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "process_tools": "process_tools",
            "handle_phantom": "handle_phantom",
            "handle_action_intent": "handle_action_intent",
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
    return graph.compile()
