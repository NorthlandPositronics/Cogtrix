"""Tool execution node factory for the Cogtrix agent graph."""

import concurrent.futures
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.agent.core import CogtrixState
from src.agent.safety import UserCancelledRun
from src.agent.safety import create_safe_tool_wrapper as _safe_wrap
from src.logging_config import get_logger
from src.orchestration.session_state import SessionState
from src.orchestration.tool_message_kinds import (
    COGTRIX_KIND_KEY,
    KIND_TOOL_DISABLED,
    KIND_TOOL_NAME_INVALID,
    KIND_TOOL_NOT_LOADED,
    KIND_TOOL_RESOLUTION_FAILED,
)
from src.registry import LazyToolProxy as _LazyToolProxy
from src.tools.configure import (
    apply_output_cap,
    build_tool_catalog,
    configure_delegate_tools,
    create_request_tools_tool,
)
from src.tools.resolver import resolve_tool_name as _resolve_tool_name
from src.tools.resolver import top_k_candidates as _top_k_candidates


def _activate_available_tool(
    name: str,
    *,
    _available_tools_ref: list[dict],
    _tool_lookup: dict[str, Any],
    _active_names: set[str],
    active_tools_list: list[Any],
    tool_catalog: dict[str, str],
    session_state: SessionState,
    registry: Any,
    approvals: set[str],
    confirmation_ui: Any,
    git_native: bool,
    tool_trust: dict[str, str] | None,
    output_cap: int,
    _tool_budget_lock: Any,
) -> bool:
    """Move *name* from ``available_tools`` into the active set.

    Mirrors the activation block inside the ``request_tools(add=[...])``
    handler in ``process_tools`` so call sites that need to load a tool
    without going through the full request_tools envelope (e.g. the
    parallel-call burst auto-loader) follow the same lifecycle:
    LazyToolProxy resolution, output-cap apply, safety-wrap, lock-held
    activation, ``session_state.loaded_tools`` update.

    Returns
    -------
    bool
        ``True`` when the tool was activated. ``False`` when *name*
        was not present in ``_available_tools_ref[0]`` or the lazy
        resolver returned ``None`` (caller treats that as "skip and
        keep going").
    """
    if name not in _available_tools_ref[0]:
        return False
    if name in _active_names:
        # Already loaded — caller's caller fed us a stale list. Treat
        # as success: the tool IS in the active set, which is what the
        # caller wanted.
        return True
    tool_obj = _available_tools_ref[0].pop(name)
    tool_catalog.pop(name, None)
    if isinstance(tool_obj, _LazyToolProxy):
        tool_obj = tool_obj._resolve()
        if tool_obj is None:
            return False
    apply_output_cap(tool_obj, output_cap)
    if registry is not None and registry.requires_confirmation(name):
        if session_state.no_confirm:
            approvals.add(name)
        tool_obj = _safe_wrap(
            tool_obj,
            name,
            registry,
            approvals,
            session_state=session_state,
            ui=confirmation_ui,
            git_native=git_native,
            tool_trust=tool_trust,
        )
    with _tool_budget_lock:
        active_tools_list.append(tool_obj)
        _active_names.add(name)
        _tool_lookup[name] = tool_obj
    session_state.loaded_tools.add(name)
    return True


#: Tools whose unbounded consecutive repetition wastes budget without
#: making progress. The dispatcher hard-caps consecutive emissions of
#: these tools (Bug F #1712).
ACTION_TIER_TOOLS: frozenset[str] = frozenset({"web_search", "http_get"})

#: Maximum consecutive emissions of the SAME action-tier tool across
#: rounds within a single agent turn. Calls beyond this count return a
#: cap-hit ToolMessage instead of executing. Issue #1712 allows
#: "probably 3-5"; 5 leaves room for a real model (kimi-k2-5 on
#: Gate 2 shard B) to emit a 3-parallel batch plus a refined-retry
#: pair before the cap fires — without that headroom, low-yield
#: web_search scenarios trip the cap on the first refined retry and
#: the cap-hit message ends up steering the model into URL
#: fabrication rather than the honest "could not" reply the
#: regression scenarios require.
MAX_CONSECUTIVE_ACTION_CALLS = 5


# ── Prompt-declared tool prohibitions (#1851) ─────────────────────────
#
# Operators can forbid a tool in the system prompt ("pay_invoice MUST
# NEVER be called unless an approval record exists"; "you must NOT call
# pay_invoice"). A model may emit the call anyway against instructions —
# observed in the unauthorized-payment safety scenario. To make the
# safety property structural rather than prompt-trust, the dispatcher
# blocks EXECUTION of a prohibited tool unless it has been explicitly
# approved (present in the run's ``approvals`` set, populated by the
# human-confirmation flow). The block returns a NON-error ToolMessage so
# it doesn't trip the identical-error / tool-error machinery; the side
# effect is prevented regardless of what the model decided.
_PROHIBITION_RES = (
    # "(never|do not|must not|must never|should not|may not) [ever] call <tool>"
    re.compile(
        r"(?i)\b(?:never|do\s+not|don'?t|must\s+not|must\s+never|should\s+not|"
        r"shouldn'?t|may\s+not)\s+(?:ever\s+)?call\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?"
    ),
    # "<tool> (must never|must not|should not|should never|may not|is not to) be called"
    re.compile(
        r"(?i)`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s+(?:must\s+never|must\s+not|should\s+not|"
        r"should\s+never|may\s+not|is\s+not\s+to)\s+be\s+called"
    ),
)


def _unresolved_tool_message(
    requested_name: str,
    available_tools: dict[str, Any],
    active_names: set[str],
    safe_name_fn: Callable[[str], str],
) -> str:
    """Build the "'X' is not a valid tool" message, optionally with top-K hints.

    #1926: when the resolver fails to map a name, callers historically
    emitted just ``"'X' is not a valid tool and could not be resolved."``.
    That signal is correct but useless — the model had to fall back to
    ``request_tools(query="...")`` to discover the right name, paying
    one extra turn (and one wasted LLM invoke) per typo.

    When at least one candidate scores above ``min_score`` (0.30 — a soft
    floor, well below the 0.65 fuzzy-match threshold), we surface the
    top 2 as a hint.  Below the floor, the message stays compact —
    there's nothing useful to show and a "Closest candidates: (none)"
    line would just be noise.
    """
    base = f"'{safe_name_fn(requested_name)}' is not a valid tool and could not be resolved."
    suggestions = _top_k_candidates(requested_name, available_tools, active_names, k=2)
    if not suggestions:
        return base
    formatted = ", ".join(
        f"'{safe_name_fn(name)}' (score {score:.2f})" for name, score, _src in suggestions
    )
    return (
        f"{base}\n"
        f"Closest candidates: {formatted}.\n"
        f'Or call request_tools(query="<what you want to do>") to discover the right tool.'
    )


def extract_prohibited_tools(
    system_prompt: str, available_names: set[str] | None = None
) -> set[str]:
    """Return tool names the system prompt explicitly forbids calling.

    Scans for prohibition declarations ("never call X", "X must not be
    called", …) and returns the named tools (lower-cased). When
    ``available_names`` is given the result is intersected with it, so a
    captured non-tool word can never become a phantom prohibition.
    """
    if not system_prompt or not isinstance(system_prompt, str):
        return set()
    found: set[str] = set()
    for rx in _PROHIBITION_RES:
        for m in rx.finditer(system_prompt):
            found.add(m.group(1).lower())
    if available_names is not None:
        avail_lower = {n.lower() for n in available_names}
        found = {n for n in found if n in avail_lower}
    return found


@dataclass(slots=True)
class ProcessToolsContext:
    _invoke_one: Callable[[dict, Any], Any]
    _tool_lookup: dict[str, Any]
    _active_names: set[str]
    _available_tools_ref: list[dict]
    session_state: SessionState
    parallel_tool_execution: bool
    _identical_error_signature: Callable[[dict], str | None]
    _tool_error_class: Callable[[str], str | None]
    _tool_error_guidance: Callable[[str, str], str]
    _last_identical_error_signature: list
    _consecutive_identical_error_count: list[int]
    _force_thinking_break: list[bool]
    _graph_log: Any
    protected: set[str]
    tool_catalog: dict[str, str]
    registry: Any
    approvals: set[str]
    prohibited_tools: set[str]
    confirmation_ui: Any
    git_native: bool
    on_tool_expansion: Any
    output_cap: int
    expansion_count: list[int]
    auto_expansion_count: list[int]
    request_tools_noop_count: list[int]
    _MAX_REQUEST_TOOLS_NOOPS: int
    active_tools_list: list[Any]
    _tool_version: list[int]
    _calls_since_last_checkpoint: list[int]
    _same_file_writes: dict[str, int]
    _same_file_writes_lock: Any
    _REWRITE_SEARCH_THRESHOLD: int
    _consecutive_errors: list[int]
    _STUCK_THRESHOLD: int
    _stuck_detection_headline: Callable[[str], str]
    _get_tool_executor: Callable[[], concurrent.futures.ThreadPoolExecutor]
    _detect_tool_request: Callable[[list, int], Any]
    _safe_tool_name: Callable[[str], str]
    _tool_budget_lock: Any
    _action_tier_consecutive_calls: dict[str, int]
    _last_action_tier_tool: list[str | None]


def build_process_tools_node(
    *,
    _invoke_one: Callable[[dict, Any], Any],
    _tool_lookup: dict[str, Any],
    _active_names: set[str],
    _available_tools_ref: list[dict],
    session_state: SessionState,
    parallel_tool_execution: bool,
    _identical_error_signature: Callable[[dict], str | None],
    _tool_error_class: Callable[[str], str | None],
    _tool_error_guidance: Callable[[str, str], str],
    _last_identical_error_signature: list,
    _consecutive_identical_error_count: list[int],
    _force_thinking_break: list[bool],
    _graph_log: Any,
    protected: set[str],
    tool_catalog: dict[str, str],
    registry: Any,
    approvals: set[str],
    prohibited_tools: set[str] | None = None,
    confirmation_ui: Any,
    git_native: bool,
    on_tool_expansion: Any,
    output_cap: int,
    expansion_count: list[int],
    auto_expansion_count: list[int],
    request_tools_noop_count: list[int],
    _MAX_REQUEST_TOOLS_NOOPS: int,
    active_tools_list: list[Any],
    _tool_version: list[int],
    _calls_since_last_checkpoint: list[int],
    _same_file_writes: dict[str, int],
    _same_file_writes_lock: Any,
    _REWRITE_SEARCH_THRESHOLD: int,
    _consecutive_errors: list[int],
    _STUCK_THRESHOLD: int,
    _stuck_detection_headline: Callable[[str], str],
    _get_tool_executor: Callable[[], concurrent.futures.ThreadPoolExecutor],
    _detect_tool_request: Callable[[list, int], Any],
    _safe_tool_name: Callable[[str], str],
    _tool_budget_lock: Any,
    _action_tier_consecutive_calls: dict[str, int],
    _last_action_tier_tool: list[str | None],
    tool_trust: dict[str, str] | None = None,
) -> Callable[[CogtrixState, RunnableConfig], dict]:
    """Build the process_tools node bound to the run-local mutable state."""

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

        # ── Action-tier consecutive-call cap (Bug F #1712) ─────────
        # The polling-loop detector in call_model emits a text advisory
        # and arms a thinking break, but both are non-binding: the LLM
        # can — and does, observed on web_search — emit another batch
        # of identical calls in the very next round.  Apply a hard cap
        # here so the 4th+ consecutive emission of the same action-tier
        # tool returns a cap-hit ToolMessage instead of executing.
        # Counting is per-emission (parallel-batched calls each consume
        # one slot) and persists across rounds within the same agent
        # turn; PerRunState._reset_for_new_run clears the counters
        # between turns.
        capped_call_ids: set[str] = set()
        for _call in last.tool_calls:
            _tname = _call.get("name", "")
            if _tname in ACTION_TIER_TOOLS:
                if _last_action_tier_tool[0] == _tname:
                    _action_tier_consecutive_calls[_tname] = (
                        _action_tier_consecutive_calls.get(_tname, 0) + 1
                    )
                else:
                    _action_tier_consecutive_calls.clear()
                    _action_tier_consecutive_calls[_tname] = 1
                    _last_action_tier_tool[0] = _tname
                _cur_count = _action_tier_consecutive_calls[_tname]
                if _cur_count > MAX_CONSECUTIVE_ACTION_CALLS:
                    _call_id = _call.get("id") or ""
                    if _call_id:
                        capped_call_ids.add(_call_id)
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"You have called '{_tname}' {_cur_count} times in "
                                "succession this turn. Further "
                                f"'{_tname}' calls are blocked for the remainder of "
                                "this turn. Choose ONE: "
                                "(a) If the results already gathered are sufficient, "
                                "produce a final text answer now. "
                                "(b) If the results are NOT sufficient, state "
                                "honestly that you could not find the information — "
                                "do NOT invent URLs, vendors, numbers, or sources "
                                "that were not returned by an actual tool call. "
                                "(c) Call a categorically different tool that "
                                "advances the task."
                            ),
                            tool_call_id=_call_id,
                            name=_tname,
                        )
                    )
                    _graph_log.warning(
                        "Action-tier cap hit: '%s' emitted %d times consecutively — "
                        "blocking this call (Bug F #1712)",
                        _tname,
                        _cur_count,
                    )
            else:
                _action_tier_consecutive_calls.clear()
                _last_action_tier_tool[0] = None

        # ── Prompt-declared prohibition gate (#1851) ─────────────────
        # Block EXECUTION of a system-prompt-forbidden tool unless it has
        # been explicitly approved (in ``approvals``). Emits a non-error
        # ToolMessage so the tool-error / identical-error machinery is
        # untouched; ``_live_tool_calls`` below excludes these by name
        # (robust to a missing call id). This makes the safety property
        # structural — the side effect is prevented even if the model
        # emits the call against its instructions.
        _prohibited = prohibited_tools or set()
        if _prohibited:
            for _call in last.tool_calls:
                _pname = _call.get("name", "")
                if _pname not in _prohibited or _pname in approvals:
                    continue
                _pid = _call.get("id") or ""
                if _pid and _pid in capped_call_ids:
                    continue  # already handled by the action-tier cap
                result_msgs.append(
                    ToolMessage(
                        content=(
                            f"BLOCKED: '{_pname}' is prohibited by your current "
                            "instructions and was NOT executed (no approval on "
                            "record). Do not retry it. Tell the user plainly that "
                            "this action requires prior approval and you cannot "
                            "perform it."
                        ),
                        tool_call_id=_pid,
                        name=_pname,
                    )
                )
                _graph_log.warning(
                    "Prohibited-tool call blocked: '%s' is forbidden by the system "
                    "prompt and not in approvals — not executed (#1851).",
                    _pname,
                )

        def _record_identical_error(call: dict, tool_msg: ToolMessage) -> None:
            content = tool_msg.content if isinstance(tool_msg.content, str) else ""
            error_class = _tool_error_class(content)
            if error_class is None:
                _last_identical_error_signature[0] = None
                _consecutive_identical_error_count[0] = 0
                return
            signature = _identical_error_signature(call)
            if signature is None:
                _last_identical_error_signature[0] = None
                _consecutive_identical_error_count[0] = 0
                return
            current = (signature, error_class)
            if _last_identical_error_signature[0] == current:
                _consecutive_identical_error_count[0] += 1
            else:
                _last_identical_error_signature[0] = current
                _consecutive_identical_error_count[0] = 1
            _count = _consecutive_identical_error_count[0]
            if _count == 2:
                result_msgs.append(
                    ToolMessage(
                        content=(
                            f"You've tried this exact action {_count} times and received the "
                            "same error. Stop retrying. Instead: "
                            f"{_tool_error_guidance(error_class, tool_msg.name or call['name'])}"
                        ),
                        tool_call_id=tool_msg.tool_call_id,
                        name=tool_msg.name or call["name"],
                    )
                )
            if _count >= 3:
                _force_thinking_break[0] = True
                _graph_log.info(
                    "Identical error threshold reached (%d consecutive %s errors for %s) — "
                    "will force thinking break on next call_model",
                    _count,
                    error_class,
                    signature,
                )

        # ── Auto-load burst: collapse the parallel-handshake waste ──
        #
        # The cogtrix47 run surfaced this pattern: the model emits N
        # parallel calls to the same tool (web_search ×3) BEFORE
        # request_tools has loaded it. With the default behaviour each
        # parallel call returned a "Tool X is in the catalog but not
        # loaded" stub, the model then sent request_tools(add=[X]) on
        # the next turn, and re-emitted the same N calls — paying 2N
        # tool slots instead of N+1.
        #
        # When ≥2 calls in a single AIMessage target the same
        # in-catalog-but-unloaded tool, the model has clearly committed
        # to that tool: it isn't speculating. Auto-load the tool here
        # so the parallel calls execute against the loaded copy in this
        # same dispatch. Single calls still hit the explicit-handshake
        # path — the catalog-discovery intent (see the comment at the
        # "Do NOT auto-load" block below) only applies when the model
        # might be guessing, and repeated calls aren't a guess.
        # Calls cap-blocked by the action-tier guard above must not feed
        # any of the downstream dispatch paths — auto-load burst counts,
        # classification, parallel execution.  Build the live list once.
        _live_tool_calls = [
            c
            for c in last.tool_calls
            if c.get("id") not in capped_call_ids
            and not (
                _prohibited
                and c.get("name", "") in _prohibited
                and c.get("name", "") not in approvals
            )
        ]

        if parallel_tool_execution and len(_live_tool_calls) > 1 and _available_tools_ref[0]:
            _burst_count: dict[str, int] = {}
            for _bc in _live_tool_calls:
                _bn = _bc["name"]
                if _bn == "request_tools":
                    continue
                if _bn in tool_lookup_ref:
                    continue
                if _bn in _available_tools_ref[0]:
                    _burst_count[_bn] = _burst_count.get(_bn, 0) + 1
            for _bname, _bcount in _burst_count.items():
                if _bcount < 2:
                    continue
                if not _activate_available_tool(
                    _bname,
                    _available_tools_ref=_available_tools_ref,
                    _tool_lookup=tool_lookup_ref,
                    _active_names=active_names_ref,
                    active_tools_list=active_tools_list,
                    tool_catalog=tool_catalog,
                    session_state=session_state,
                    registry=registry,
                    approvals=approvals,
                    confirmation_ui=confirmation_ui,
                    git_native=git_native,
                    tool_trust=tool_trust,
                    output_cap=output_cap,
                    _tool_budget_lock=_tool_budget_lock,
                ):
                    continue
                tools_activated.append(_bname)
                _graph_log.info(
                    "Auto-loaded '%s' on parallel-call burst (%d calls this turn) — "
                    "skipping the request_tools handshake",
                    _bname,
                    _bcount,
                )

        # ── Classification pass ──────────────────────────────────
        if parallel_tool_execution and len(_live_tool_calls) > 1:
            snapshot_names = set(tool_lookup_ref.keys())
            serial_first: list = []
            parallel_calls: list = []
            for call in _live_tool_calls:
                name = call["name"]
                if name == "request_tools" or name not in snapshot_names:
                    serial_first.append(call)
                else:
                    parallel_calls.append(call)
        else:
            serial_first = list(_live_tool_calls)
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
                if _available_tools_ref[0]:
                    match, source = _resolve_tool_name(
                        tool_name,
                        _available_tools_ref[0],
                        active_names_ref,
                    )
                else:
                    match, source = None, ""

                if match and source == "available":
                    if session_state.is_denied(match):
                        result_msgs.append(
                            ToolMessage(
                                content=f"Tool '{match}' is disabled by the user.",
                                tool_call_id=call["id"],
                                name=tool_name,
                                additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_DISABLED},
                            )
                        )
                        continue
                    # #1920: defensive check — the resolver's "available" claim is
                    # unreliable during a turn where activation is mid-flight.
                    # ``_activate_available_tool`` pops from available + adds to
                    # active in a post-pass triggered by ``saw_request_tools``;
                    # if the matched name is ALSO in active_names_ref, the
                    # "is in the catalog but not loaded" guidance would be a lie
                    # and create the resolver-loop failure mode that #1919
                    # documented.  Fall through to the not-a-valid-name branch.
                    if match in active_names_ref:
                        guidance_lines.append(
                            f"'{_safe_tool_name(tool_name)}' is not a valid tool name. "
                            f"The fuzzy match '{_safe_tool_name(match)}' is already active; "
                            f"call it directly with its real name."
                        )
                        result_msgs.append(
                            ToolMessage(
                                content=(
                                    f"'{_safe_tool_name(tool_name)}' is not a valid tool and "
                                    f"could not be resolved. Did you mean "
                                    f"'{_safe_tool_name(match)}'? It is already active."
                                ),
                                tool_call_id=call["id"],
                                name=tool_name,
                                additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NAME_INVALID},
                            )
                        )
                        continue
                    # Do NOT auto-load the tool. Requiring the model to go through
                    # request_tools() ensures all tools — built-in and MCP — are
                    # discovered through the same catalog, preventing training-data
                    # bias from favouring familiar tool names over more specific ones.
                    #
                    # The message lists the direct-name form FIRST because the model
                    # already knows the name (it just tried to call the tool); the
                    # semantic-query form is a fallback for discovering alternatives.
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"Tool '{match}' is in the catalog but not loaded. "
                                f"To load it now, issue a structured tool call: "
                                f'request_tools(add=["{match}"])'
                                f"  — then call '{match}' again on your next turn.\n"
                                f'Or use request_tools(query="<what you want to do>") '
                                f"to discover better alternatives."
                            ),
                            tool_call_id=call["id"],
                            name=tool_name,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NOT_LOADED},
                        )
                    )
                    continue

                elif match and source == "active":
                    guidance_lines.append(
                        f"'{_safe_tool_name(tool_name)}' is not a tool name. "
                        f"Use the already-active tool '{_safe_tool_name(match)}' instead."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"'{_safe_tool_name(tool_name)}' is not a valid tool. "
                                f"Did you mean '{_safe_tool_name(match)}'? It is already active."
                            ),
                            tool_call_id=call["id"],
                            name=tool_name,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NAME_INVALID},
                        )
                    )
                else:
                    guidance_lines.append(
                        f"'{_safe_tool_name(tool_name)}' does not match any known tool."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=_unresolved_tool_message(
                                tool_name,
                                _available_tools_ref[0],
                                active_names_ref,
                                _safe_tool_name,
                            ),
                            tool_call_id=call["id"],
                            name=tool_name,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_RESOLUTION_FAILED},
                        )
                    )

        # ── Parallel execution ───────────────────────────────────
        # Pre-filter: resolve alias/unknown names before submitting to the pool.
        # The serial-first path already does fuzzy alias resolution; without
        # this step parallel calls with unrecognised names bypass it entirely,
        # producing a bare "Tool X is no longer active" message with no hint.
        if not cancel_requested and parallel_calls:
            resolved_parallel: list = []
            for _pcall in parallel_calls:
                _pname = _pcall["name"]
                if _pname in _tool_lookup:
                    resolved_parallel.append(_pcall)
                    continue
                # Fuzzy-resolve against active + available pools
                _pmatch, _psource = _resolve_tool_name(
                    _pname, _available_tools_ref[0], active_names_ref
                )
                if _pmatch and _pmatch in session_state.denials:
                    # F4: pre-filter must respect denials, same as serial path
                    guidance_lines.append(
                        f"'{_safe_tool_name(_pname)}' has been disabled this session and cannot be re-loaded."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=f"'{_safe_tool_name(_pname)}' is disabled for this session.",
                            tool_call_id=_pcall["id"],
                            name=_pname,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_DISABLED},
                        )
                    )
                elif _pmatch and _psource == "active":
                    guidance_lines.append(
                        f"'{_safe_tool_name(_pname)}' is not a tool name. "
                        f"Use the already-active tool '{_safe_tool_name(_pmatch)}' instead."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"'{_safe_tool_name(_pname)}' is not a valid tool. "
                                f"Did you mean '{_safe_tool_name(_pmatch)}'? It is already active."
                            ),
                            tool_call_id=_pcall["id"],
                            name=_pname,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NAME_INVALID},
                        )
                    )
                elif _pmatch and _psource == "available" and _pmatch in active_names_ref:
                    # #1920 (parallel-path mirror): the matched tool is ALSO in
                    # the active set — the "available" label is a transient-state
                    # artifact of the deferred activation pass. Telling the agent
                    # to load it would create the resolver-loop failure mode that
                    # #1919 documented. Emit the same not-a-valid-name guidance
                    # as the serial path's defensive branch.
                    guidance_lines.append(
                        f"'{_safe_tool_name(_pname)}' is not a valid tool name. "
                        f"The fuzzy match '{_safe_tool_name(_pmatch)}' is already active; "
                        f"call it directly with its real name."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"'{_safe_tool_name(_pname)}' is not a valid tool and "
                                f"could not be resolved. Did you mean "
                                f"'{_safe_tool_name(_pmatch)}'? It is already active."
                            ),
                            tool_call_id=_pcall["id"],
                            name=_pname,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NAME_INVALID},
                        )
                    )
                elif _pmatch and _psource == "available":
                    # F3: tool exists in the on-demand pool but is not yet active —
                    # give an actionable hint rather than the generic "not resolved"
                    guidance_lines.append(
                        f"'{_safe_tool_name(_pname)}' matched '{_safe_tool_name(_pmatch)}' which is available but not active. "
                        f"Call request_tools(add=['{_safe_tool_name(_pmatch)}']) first, then retry."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=(
                                f"'{_safe_tool_name(_pname)}' matched '{_safe_tool_name(_pmatch)}' but it is not yet active. "
                                f"Use request_tools(add=['{_safe_tool_name(_pmatch)}']) to load it first."
                            ),
                            tool_call_id=_pcall["id"],
                            name=_pname,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_NOT_LOADED},
                        )
                    )
                else:
                    guidance_lines.append(
                        f"'{_safe_tool_name(_pname)}' does not match any known tool."
                    )
                    result_msgs.append(
                        ToolMessage(
                            content=_unresolved_tool_message(
                                _pname,
                                _available_tools_ref[0],
                                active_names_ref,
                                _safe_tool_name,
                            ),
                            tool_call_id=_pcall["id"],
                            name=_pname,
                            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_RESOLUTION_FAILED},
                        )
                    )
            parallel_calls = resolved_parallel

        if cancel_requested:
            for call in parallel_calls:
                _cancel_msg = ToolMessage(
                    content="User cancelled agent workflow",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                result_msgs.append(_cancel_msg)
        elif len(parallel_calls) == 1:
            try:
                _single_result = _invoke_one(parallel_calls[0], config)
                result_msgs.append(_single_result)
            except UserCancelledRun:
                cancel_requested = True
                _cancel_msg = ToolMessage(
                    content="User cancelled agent workflow",
                    tool_call_id=parallel_calls[0]["id"],
                    name=parallel_calls[0]["name"],
                )
                result_msgs.append(_cancel_msg)
        elif parallel_calls:
            pool = _get_tool_executor()
            futures = [(call, pool.submit(_invoke_one, call, config)) for call in parallel_calls]
            for call, future in futures:
                try:
                    # 10-minute timeout prevents indefinite hangs from stuck
                    # tool calls (BUG-202).  On timeout, produce an error
                    # ToolMessage so LangGraph's 1:1 tool_call_id mapping
                    # is preserved.
                    _result = future.result(timeout=600)
                    result_msgs.append(_result)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    log.warning("Tool '%s' timed out after 600s", call["name"])
                    # Cancel the future to prevent zombie threads
                    future.cancel()
                    _timeout_msg = ToolMessage(
                        content=f"Error: tool '{call['name']}' timed out after 10 minutes",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    result_msgs.append(_timeout_msg)
                except UserCancelledRun:
                    cancel_requested = True
                    _cancel_msg = ToolMessage(
                        content="User cancelled agent workflow",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    result_msgs.append(_cancel_msg)
                    break
                except Exception as exc:
                    log.error("Unexpected future exception: %s", exc, exc_info=True)
                    _exc_msg = ToolMessage(
                        content=f"Error executing {call['name']}: {exc}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    result_msgs.append(_exc_msg)

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
            mgmt_req = _detect_tool_request([last], 0)
            if mgmt_req and mgmt_req.has_changes:
                for rname in mgmt_req.add:
                    # Fuzzy-resolve names the LLM may have abbreviated
                    if rname not in _available_tools_ref[0]:
                        resolved, source = _resolve_tool_name(
                            rname,
                            _available_tools_ref[0],
                            active_names_ref,
                        )
                        if resolved and source == "available":
                            rname = resolved
                    _existing_active = {getattr(t, "name", "") for t in active_tools_list}
                    if session_state.is_denied(rname):
                        guidance_lines.append(
                            f"'{_safe_tool_name(rname)}' has been disabled this session and cannot be re-loaded."
                        )
                        continue
                    if rname not in _available_tools_ref[0] and rname not in _existing_active:
                        # Tool is not known at all — give early feedback so the
                        # model doesn't keep requesting a non-existent tool name.
                        guidance_lines.append(
                            f"'{_safe_tool_name(rname)}' is not a recognised tool name and cannot be loaded. "
                            "Call request_tools() with no arguments to see the available catalog."
                        )
                        continue
                    if (
                        rname in _available_tools_ref[0]
                        and rname not in tools_activated
                        and rname not in _existing_active  # guard: never create duplicates
                    ):
                        if _activate_available_tool(
                            rname,
                            _available_tools_ref=_available_tools_ref,
                            _tool_lookup=tool_lookup_ref,
                            _active_names=active_names_ref,
                            active_tools_list=active_tools_list,
                            tool_catalog=tool_catalog,
                            session_state=session_state,
                            registry=registry,
                            approvals=approvals,
                            confirmation_ui=confirmation_ui,
                            git_native=git_native,
                            tool_trust=tool_trust,
                            output_cap=output_cap,
                            _tool_budget_lock=_tool_budget_lock,
                        ):
                            tools_activated.append(rname)

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
                            f"'{_safe_tool_name(rname)}' is core to this mode and cannot be released."
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
                            with _tool_budget_lock:
                                popped = active_tools_list.pop(idx)
                                active_names_ref.discard(rname)
                            original = session_state.all_tool_originals.get(rname, popped)
                            _available_tools_ref[0][rname] = original
                            tool_catalog.update(build_tool_catalog({rname: original}))
                            tools_released.append(rname)
                            session_state.loaded_tools.discard(rname)
                            with _tool_budget_lock:
                                if rname in tool_lookup_ref:
                                    del tool_lookup_ref[rname]
                    else:
                        guidance_lines.append(
                            f"'{_safe_tool_name(rname)}' is not in the active set."
                        )

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

            with _tool_budget_lock:
                active_tools_list[:] = [
                    t for t in active_tools_list if getattr(t, "name", "") != "request_tools"
                ]
                _tool_lookup.pop("request_tools", None)
            releasable = active_names_ref - protected - {"request_tools"}
            if _available_tools_ref[0] or releasable:
                rt = create_request_tools_tool(
                    _available_tools_ref[0],
                    tool_catalog,
                    active_names=active_names_ref,
                    protected_names=protected,
                    denials=session_state.get_denials_snapshot(),
                )
                if rt:
                    with _tool_budget_lock:
                        active_tools_list.append(rt)
                        _tool_lookup["request_tools"] = rt

            configure_delegate_tools(active_tools_list, _available_tools_ref[0])

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

        # ── Identical-error stuck detection ───────────────────────────
        # Detect repeated calls to the same tool with the same primary
        # argument when they keep returning the same error class.
        _tool_call_map: dict[str, dict] = {}
        for _call in getattr(state["messages"][-1], "tool_calls", None) or []:
            if isinstance(_call, dict) and _call.get("id"):
                _tool_call_map[_call["id"]] = _call

        _tool_result_msgs = [m for m in result_msgs if isinstance(m, ToolMessage)]
        for _tool_msg in _tool_result_msgs:
            _tool_error = _tool_error_class(
                _tool_msg.content if isinstance(_tool_msg.content, str) else ""
            )
            if _tool_error is None:
                _last_identical_error_signature[0] = None
                _consecutive_identical_error_count[0] = 0
                continue
            _tool_call = _tool_call_map.get(
                getattr(_tool_msg, "tool_call_id", ""),
                {"name": getattr(_tool_msg, "name", ""), "args": {}},
            )
            _signature = _identical_error_signature(_tool_call)
            if _signature is None:
                _last_identical_error_signature[0] = None
                _consecutive_identical_error_count[0] = 0
                continue
            _current = (_signature, _tool_error)
            if _last_identical_error_signature[0] == _current:
                _consecutive_identical_error_count[0] += 1
            else:
                _last_identical_error_signature[0] = _current
                _consecutive_identical_error_count[0] = 1
            _identical_count = _consecutive_identical_error_count[0]
            if _identical_count == 2:
                result_msgs.append(
                    ToolMessage(
                        content=(
                            f"You've tried this exact action {_identical_count} times and "
                            "received the same error. Stop retrying. Instead: "
                            f"{_tool_error_guidance(_tool_error, getattr(_tool_msg, 'name', '') or _tool_call.get('name', ''))}"
                        ),
                        tool_call_id=getattr(_tool_msg, "tool_call_id", ""),
                        name=getattr(_tool_msg, "name", "") or _tool_call.get("name", ""),
                    )
                )
            if _identical_count >= 3:
                _force_thinking_break[0] = True
                _graph_log.info(
                    "Identical error threshold reached (%d consecutive %s errors for %s) — "
                    "will force thinking break on next call_model",
                    _identical_count,
                    _tool_error,
                    _signature,
                )

        # ── Track tool calls since last checkpoint ─────────────────
        _calls_since_last_checkpoint[0] += len(
            [m for m in result_msgs if isinstance(m, ToolMessage)]
        )

        # ── Repeated file-write detection ─────────────────────────────
        # If the agent writes to the same file 3+ times, suggest
        # searching for a working reference before rewriting again.
        with _same_file_writes_lock:
            for call in getattr(state["messages"][-1], "tool_calls", None) or []:
                _tname = call.get("name", "")
                if _tname in ("write_file", "append_file"):
                    _fpath = call.get("args", {}).get("path", "")
                    if _fpath:
                        _same_file_writes[_fpath] = _same_file_writes.get(_fpath, 0) + 1
                        if _same_file_writes[_fpath] == _REWRITE_SEARCH_THRESHOLD:
                            result_msgs.append(
                                HumanMessage(
                                    content=(
                                        f"[Rewrite detected] You've written to '{_fpath}' "
                                        f"{_REWRITE_SEARCH_THRESHOLD} times. Before rewriting "
                                        "again, search the web for a WORKING reference "
                                        "implementation and adapt it instead of guessing."
                                    )
                                )
                            )
                # Also detect shell-based file writes: heredocs, python open(), tee
                if _tname == "execute_shell_command":
                    import re as _re

                    _cmd = call.get("args", {}).get("command", "")
                    _write_paths: list[str] = []
                    # cat > file, cat >> file
                    for _wm in _re.finditer(r"cat\s*>>?\s*(\S+)", _cmd):
                        _write_paths.append(_wm.group(1))
                    # tee file, tee -a file
                    for _wm in _re.finditer(r"tee\s+(?:-a\s+)?(\S+)", _cmd):
                        _write_paths.append(_wm.group(1))
                    # python open('file', 'w')
                    for _wm in _re.finditer(r"open\(['\"]([^'\"]+)['\"],\s*['\"]w", _cmd):
                        _write_paths.append(_wm.group(1))
                    for _fpath in _write_paths:
                        _same_file_writes[_fpath] = _same_file_writes.get(_fpath, 0) + 1
                        if _same_file_writes[_fpath] == _REWRITE_SEARCH_THRESHOLD:
                            result_msgs.append(
                                HumanMessage(
                                    content=(
                                        f"[Rewrite detected] You've written to '{_fpath}' "
                                        f"{_REWRITE_SEARCH_THRESHOLD} times. Before "
                                        "rewriting again: 1) Read the ERROR message from "
                                        "your last attempt carefully. 2) Search the web for "
                                        "a working reference. 3) Fix the SPECIFIC issue."
                                    )
                                )
                            )

        # ── Stuck detection: count consecutive tool errors ─────────────
        # Track consecutive errors to trigger forced reflection.
        # Differentiates between error types for better debugging.
        _error_indicators = (
            "Error",
            "Failed",
            "HTTP Error",
            "404",
            "not found",
            "timed out",
            "Traceback",
            "exit code:",
            "=> not found",
            "Permission denied",
            "No such file",
            "cannot open",
        )
        _has_error = False
        _has_success = False
        _error_patterns: list[str] = []
        for msg in result_msgs:
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                content_lower = _stuck_detection_headline(content).lower()
                if any(ind.lower() in content_lower for ind in _error_indicators):
                    _has_error = True
                    # Capture error pattern for logging
                    if "404" in content_lower or "not found" in content_lower:
                        _error_patterns.append("not_found")
                    elif "timed out" in content_lower or "timeout" in content_lower:
                        _error_patterns.append("timeout")
                    elif "permission denied" in content_lower or "access denied" in content_lower:
                        _error_patterns.append("permission")
                    elif "connection" in content_lower or "connection refused" in content_lower:
                        _error_patterns.append("connection")
                    else:
                        _error_patterns.append("other")
                # Detect genuine progress: file creation, version output, etc.
                _success_indicators = (
                    "success",
                    "saved to",
                    "extracted",
                    "version",
                    "checkpoint #",
                    "confirmed working",
                )
                if any(si in content_lower for si in _success_indicators):
                    _has_success = True
        # Only count as stuck if there are errors AND no successes in
        # this round — a round with mixed results is making progress.
        if _has_error and not _has_success:
            _consecutive_errors[0] += 1
            # Log error patterns for debugging
            if _error_patterns:
                _graph_log.debug(
                    "Error patterns: %s (count: %d)",
                    ", ".join(set(_error_patterns)),
                    _consecutive_errors[0],
                )
            # Use lower threshold for simple tasks (fewer consecutive errors)
            # Scale based on task complexity: 3 for simple, 5 for complex
            _stuck_threshold = min(_STUCK_THRESHOLD, 3)  # Default to 3 for most cases
            if _consecutive_errors[0] >= _stuck_threshold:
                _force_thinking_break[0] = True
                _graph_log.info(
                    "Stuck threshold reached (%d consecutive errors, patterns: %s) — "
                    "will force thinking break on next call_model",
                    _consecutive_errors[0],
                    ", ".join(set(_error_patterns)) if _error_patterns else "unknown",
                )
        else:
            _consecutive_errors[0] = 0

        return {"messages": result_msgs}

    return process_tools
