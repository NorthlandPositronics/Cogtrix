"""LangGraph agent graph for Cogtrix.

Builds a custom StateGraph with three nodes:
- call_model: binds active tools to LLM and invokes it
- process_tools: executes tool calls, handles fuzzy matching and expansion
- handle_phantom: recovers from phantom tool calls
"""

import concurrent.futures
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from src.agent.core import CogtrixState
from src.logging_config import get_logger
from src.orchestration.compression import (
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
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

DEFAULT_RECURSION_LIMIT = 150
EMPTY_RESPONSE_MSG = "**Error:** The model returned an empty response. Please try again."
_PARALLEL_TOOL_WORKERS = 8

_INVALID_TOOL_RE = re.compile(r"^Error:\s*(\S+)\s+is not a valid tool")


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
    found: list[str] = []
    seen: set[str] = set()
    for i in range(start_idx, len(messages)):
        msg = messages[i]
        if type(msg).__name__ != "ToolMessage":
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
    tool_call_ids_to_remove: set[str] = set()
    cleaned: list = []

    for msg in messages:
        if type(msg).__name__ == "ToolMessage":
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
        if type(msg).__name__ == "AIMessage":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                remaining = [tc for tc in tool_calls if tc.get("id") not in tool_call_ids_to_remove]
                if len(remaining) != len(tool_calls):
                    try:
                        from langchain_core.messages import AIMessage as AI

                        extra = dict(getattr(msg, "additional_kwargs", {}))
                        extra.pop("tool_calls", None)
                        new_msg = AI(
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
                    except ImportError:
                        pass
        final.append(msg)
    return final


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
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
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
        context_compression = config.context_compression
        if config.compression_llm is not None:
            compression_llm = config.compression_llm
        if config.tool_call_guard is not None:
            tool_call_guard = config.tool_call_guard
        if config.session_state is not None:
            session_state = config.session_state
        if config.confirmation_ui is not None:
            confirmation_ui = config.confirmation_ui
        if config.on_tool_expansion is not None:
            on_tool_expansion = config.on_tool_expansion
        parallel_tool_execution = config.parallel_tool_execution

    if active_tools_list is None:
        active_tools_list = []
    if available_tools is None:
        available_tools = {}
    if approvals is None:
        approvals = set()

    if session_state is None:
        session_state = SessionState()

    phantom_count = [0]
    expansion_count = [0]
    auto_expansion_count = [0]
    call_count = [0]
    compression_cache: dict[str, str] = (
        compression_cache_in if compression_cache_in is not None else {}
    )
    _MAX_PHANTOM_RETRIES = 3
    _MAX_TOOL_EXPANSIONS = 3
    request_tools_noop_count = [0]
    _MAX_REQUEST_TOOLS_NOOPS = 3
    protected = (preset_tools or set()) | {"request_tools"}
    _bound_cache: OrderedDict[tuple[str, ...], Any] = (
        bound_cache if bound_cache is not None else OrderedDict()
    )
    _tool_version = [0]
    _last_tool_version = [-1]
    _cached_fingerprint: list[tuple[str, ...]] = [()]
    output_cap = (
        compute_tool_output_cap(max_context_tokens)
        if max_context_tokens
        else TOOL_OUTPUT_CAP_MIN_CHARS
    )
    _sys_msg = SystemMessage(content=system_prompt)
    _tool_lookup: dict[str, Any] = {getattr(t, "name", ""): t for t in active_tools_list}
    _tool_lookup.pop("", None)
    _active_names: set[str] = set(_tool_lookup.keys())

    def call_model(state: CogtrixState, config: RunnableConfig) -> dict:
        call_count[0] += 1
        if _tool_version[0] != _last_tool_version[0]:
            _cached_fingerprint[0] = (
                tuple(getattr(t, "name", "") for t in active_tools_list)
                if active_tools_list
                else ()
            )
            _last_tool_version[0] = _tool_version[0]
        fingerprint = _cached_fingerprint[0]
        if fingerprint in _bound_cache:
            _bound_cache.move_to_end(fingerprint)
        else:
            tool_list = list(active_tools_list) if active_tools_list else []
            if len(_bound_cache) > 8:
                _bound_cache.popitem(last=False)
            _bound_cache[fingerprint] = llm.bind_tools(tool_list) if tool_list else llm
        model = _bound_cache[fingerprint]
        if context_compression:
            full_messages = [_sys_msg] + list(state["messages"])
            full_messages = apply_message_compression(
                full_messages,
                call_count=call_count[0],
                compression_cache=compression_cache,
                llm=compression_llm or llm,
                max_context_tokens=max_context_tokens,
                min_age_cycles=compression_min_age,
                min_chars=compression_min_chars,
            )
        else:
            full_messages = [_sys_msg, *state["messages"]]
        response = model.invoke(full_messages, config)
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
                SystemMessage(
                    content=(
                        "Your last tool call could not be parsed by the server. "
                        "The JSON was malformed. Please try your tool call again "
                        "with carefully formatted JSON arguments, or if you have "
                        "enough information, provide your answer directly."
                    )
                ),
            ]
        }

    def _invoke_one(call: dict, run_config: Any) -> Any:
        """Execute a single tool call already in tool_lookup. Returns ToolMessage."""
        from src.agent.safety import UserCancelledRun

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
                return result
            return ToolMessage(
                content=str(result) if result is not None else "",
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
        from src.agent.safety import UserCancelledRun
        from src.agent.safety import create_safe_tool_wrapper as _safe_wrap

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
                    apply_output_cap(tool_obj, output_cap)
                    if registry.requires_confirmation(match):
                        if session_state.no_confirm:
                            approvals.add(match)
                        tool_obj = _safe_wrap(
                            tool_obj,
                            match,
                            registry,
                            approvals,
                            session_state=session_state,
                            ui=confirmation_ui,
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
                        corrected_input = {**call, "name": match, "type": "tool_call"}
                        result = tool_obj.invoke(corrected_input, config)
                        if isinstance(result, ToolMessage):
                            result_msgs.append(result)
                        else:
                            result_msgs.append(
                                ToolMessage(
                                    content=str(result) if result is not None else "",
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
            if parallel_calls[0]["name"] == "request_tools":
                saw_request_tools = True
        elif parallel_calls:
            workers = min(len(parallel_calls), _PARALLEL_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="tool"
            ) as pool:
                futures = [
                    (call, pool.submit(_invoke_one, call, config)) for call in parallel_calls
                ]
                for call, future in futures:
                    try:
                        result_msgs.append(future.result())
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
                    if call["name"] == "request_tools":
                        saw_request_tools = True

                if cancel_requested:
                    processed_ids = {
                        m.tool_call_id for m in result_msgs if hasattr(m, "tool_call_id")
                    }
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
                        apply_output_cap(tool_obj, output_cap)
                        if registry.requires_confirmation(rname):
                            if session_state.no_confirm:
                                approvals.add(rname)
                            tool_obj = _safe_wrap(
                                tool_obj,
                                rname,
                                registry,
                                approvals,
                                session_state=session_state,
                                ui=confirmation_ui,
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
                    build_tool_catalog(available_tools),
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
            result_msgs.append(SystemMessage(content=" ".join(note_parts)))
        elif guidance_lines:
            result_msgs.append(
                SystemMessage(content=" ".join(guidance_lines) + " Continue with your task.")
            )

        return {"messages": result_msgs}

    def route_after_model(state: CogtrixState) -> str:
        msgs = state["messages"]
        if not msgs:
            return END

        last = msgs[-1]
        if type(last).__name__ == "AIMessage":
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

        return END

    def route_after_phantom(state: CogtrixState) -> str:
        if phantom_count[0] > _MAX_PHANTOM_RETRIES:
            return END
        return "call_model"

    graph: Any = StateGraph(CogtrixState)
    graph.add_node("call_model", call_model)
    graph.add_node("handle_phantom", handle_phantom)
    graph.add_node("process_tools", process_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {"process_tools": "process_tools", "handle_phantom": "handle_phantom", END: END},
    )
    graph.add_edge("process_tools", "call_model")
    graph.add_conditional_edges(
        "handle_phantom",
        route_after_phantom,
        {"call_model": "call_model", END: END},
    )
    return graph.compile()
