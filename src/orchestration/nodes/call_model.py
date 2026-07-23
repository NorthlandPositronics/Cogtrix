"""Extracted call_model graph node."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry.trace import Status, StatusCode

from src.agent.core import CogtrixState
from src.api.telemetry import start_span
from src.logging_config import get_logger, is_trace
from src.orchestration.compression import apply_message_compression
from src.orchestration.graph import (
    _TOPIC_SWITCH_NUDGE,
    _apply_context_budget_guard,
    _infer_llm_model_name,
    _infer_llm_provider_name,
    _is_context_overflow_error,
    _repair_tool_message_pairs,
    _should_reset_summary_for_topic_switch,
)
from src.orchestration.reflection_delegate import (
    UNCERTAINTY_NOTE_PREFIX,
    extract_decision_justification,
)


@dataclass(slots=True)
class CallModelContext:
    llm: Any
    tools_ready: Any | None
    active_tools_list: list[Any]
    active_names: set[str]
    bound_cache: OrderedDict[tuple[str, ...], Any]
    bound_cache_lock: Any
    cached_fingerprint: list[tuple[str, ...]]
    compression_cache: dict[str, str]
    tool_version: list[int]
    last_tool_version: list[int]
    call_count: list[int]
    last_input_tokens: list[int]
    max_context_tokens: int | None
    context_max_messages: int
    context_max_tokens: int
    model_max_tokens: int | None
    compression_llm: Any
    memory_manager: Any | None
    checkpoint_store: Any | None
    calls_since_last_checkpoint: list[int]
    last_checkpoint_count: list[int]
    rounds_since_checkpoint: list[int]
    force_thinking_break: list[bool]
    consecutive_errors: list[int]
    last_identical_error_signature: list[tuple[str, str] | None]
    consecutive_identical_error_count: list[int]
    last_reflection_at: list[int]
    tool_health_check_interval: int
    last_tool_health_check_at: list[int]
    tool_quality_gate_enabled: bool
    topic_switch_detection_enabled: bool
    stuck_threshold: int
    stuck_no_checkpoint_threshold: list[int]
    stuck_threshold_calibrated: list[bool]
    checkpoint_nudge_interval: int
    reflection_interval: int
    max_request_tools_noops: int
    sys_msg: SystemMessage | None
    model_timeout: int
    tool_context_limit_pct: float
    da_enabled: bool
    da_report_uncertainty: bool
    da_min_confidence: float
    apply_context_message_cap: Callable[[list[Any], int, int], list[Any]]
    maybe_compress: Callable[[list[Any]], list[Any]]
    invoke_with_timeout: Callable[[Any, list[Any], Any, int], Any]
    all_tool_results_substanceless: Callable[[list[Any]], bool]


# finish_reason values that definitively indicate the LLM was cut off by
# the token limit.  Providers differ: OpenAI uses "length", Anthropic uses
# "max_tokens", Azure OpenAI follows OpenAI.
_TRUNCATED_FINISH_REASONS: frozenset[str] = frozenset({"length", "max_tokens"})

# Minimum aggregate tool-call argument size (chars) to apply the heuristic
# truncation check.  Below this, even exact-boundary completion_tokens is
# unlikely to represent a truncated large write.
_TRUNCATION_HEURISTIC_MIN_ARG_CHARS = 1_000

# Well-known provider token-cap boundaries.  If completion_tokens lands
# exactly on one of these AND the total tool-call args are large, we treat
# the response as heuristically truncated.  Covers providers such as
# qwen3-coder via OpenRouter/spark that always report finish_reason=
# "tool_calls" regardless of whether the output was cut at the limit.
_COMMON_TOKEN_CAPS: frozenset[int] = frozenset({512, 1024, 2048, 4096, 8192, 16384, 32768})


def _guard_truncated_tool_calls(
    response: Any,
    model_max_tokens: int | None,
    log: Any,
) -> Any:
    """Replace truncated tool-call responses with an instructive error message.

    When the LLM hits its output-token limit mid-generation the tool-call
    arguments can be silently truncated.  A ``write_file`` call with a
    truncated ``content`` argument will overwrite a good file with an
    incomplete fragment — exactly the failure mode seen in the May 2026
    stock-data session (see investigation in bugfix/101-failed-logic).

    Two detection tiers:

    1. **Definitive** — ``finish_reason`` is an explicit truncation signal
       (``"length"`` for OpenAI/Azure, ``"max_tokens"`` for Anthropic).
       Always intercept.

    2. **Heuristic** — ``finish_reason`` is ``"tool_calls"`` but
       ``completion_tokens`` lands exactly on a known provider cap *and*
       the combined tool-call argument payload is large (≥ 1 KB).  Covers
       providers such as qwen3-coder via spark/OpenRouter that always report
       ``finish_reason="tool_calls"`` even when output was cut at the limit.

    In both cases the tool calls are stripped from the response and replaced
    with a plain-text message instructing the agent to retry with smaller
    content.
    """
    try:
        from langchain_core.messages import AIMessage as _AIMessage

        if not isinstance(response, _AIMessage):
            return response

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response

        meta = getattr(response, "response_metadata", None) or {}
        finish_reason = (
            meta.get("finish_reason")
            or meta.get("stop_reason")
            or meta.get("finish_reason_details", {}).get("reason", "")
            or ""
        )

        # --- tier 1: definitive ---
        truncated = finish_reason.lower() in _TRUNCATED_FINISH_REASONS

        # --- tier 2: heuristic ---
        if not truncated:
            usage = meta.get("token_usage") or {}
            completion_tokens = usage.get("completion_tokens", 0) or 0
            if not completion_tokens:
                # Some providers expose via usage_metadata on the message
                um = getattr(response, "usage_metadata", None) or {}
                completion_tokens = um.get("output_tokens", 0) or 0

            # When ``model_max_tokens`` is configured, treat it as the
            # authoritative truncation signal: the power-of-2 heuristic
            # is meant for providers that don't expose their cap, so
            # second-guessing a sub-cap response when we already know
            # the real cap produces false positives (e.g. a legitimate
            # 4096-token complete response under an 8192-token cap).
            # When ``model_max_tokens`` is unset, fall back to the
            # heuristic to cover providers like qwen3-coder via
            # spark/OpenRouter that always report ``finish_reason=
            # "tool_calls"`` regardless of truncation.
            if model_max_tokens:
                if completion_tokens >= model_max_tokens:
                    truncated = True
            elif completion_tokens in _COMMON_TOKEN_CAPS:
                # Only flag heuristically when the tool-call payload is large
                # enough that a truncation would be consequential.
                total_arg_chars = sum(
                    len(str(v)) for tc in tool_calls for v in (tc.get("args") or {}).values()
                )
                if total_arg_chars >= _TRUNCATION_HEURISTIC_MIN_ARG_CHARS:
                    truncated = True

        if not truncated:
            return response

        tool_names = ", ".join(tc.get("name", "?") for tc in tool_calls)
        log.warning(
            "Tool call arguments truncated at token limit "
            "(finish_reason=%r, tool_calls=[%s]). "
            "Suppressing execution to prevent partial/corrupt data.",
            finish_reason,
            tool_names,
        )

        return _AIMessage(
            content=(
                "⚠️ Token limit reached mid-generation — the tool call arguments "
                f"for [{tool_names}] may be incomplete and were NOT executed.\n\n"
                "To continue:\n"
                "• Break large file writes into multiple smaller append_file calls "
                "(one function or section at a time).\n"
                "• For write_file, keep each call under ~300 lines of content.\n"
                "• Retry the intended operation with reduced content per call."
            ),
            id=getattr(response, "id", None),
            response_metadata=meta,
        )
    except Exception:  # noqa: BLE001 — guard must never crash the agent turn
        return response


def build_call_model_node(
    context: CallModelContext,
) -> Callable[[CogtrixState, RunnableConfig], dict]:
    """Build the extracted call_model node."""

    def call_model(state: CogtrixState, config: RunnableConfig) -> dict:
        llm = context.llm
        tools_ready = context.tools_ready
        active_tools_list = context.active_tools_list
        _active_names = context.active_names
        _bound_cache = context.bound_cache
        _bound_cache_lock = context.bound_cache_lock
        _cached_fingerprint = context.cached_fingerprint
        _compression_cache = context.compression_cache
        _tool_version = context.tool_version
        _last_tool_version = context.last_tool_version
        call_count = context.call_count
        _last_input_tokens = context.last_input_tokens
        _max_context_tokens = context.max_context_tokens
        _context_max_messages = context.context_max_messages
        _context_max_tokens = context.context_max_tokens
        _model_max_tokens = context.model_max_tokens
        compression_llm = context.compression_llm
        memory_manager = context.memory_manager
        _checkpoint_store = context.checkpoint_store
        _calls_since_last_checkpoint = context.calls_since_last_checkpoint
        _last_checkpoint_count = context.last_checkpoint_count
        _rounds_since_checkpoint = context.rounds_since_checkpoint
        _force_thinking_break = context.force_thinking_break
        _consecutive_errors = context.consecutive_errors
        _last_identical_error_signature = context.last_identical_error_signature
        _consecutive_identical_error_count = context.consecutive_identical_error_count
        _last_reflection_at = context.last_reflection_at
        _TOOL_HEALTH_CHECK_INTERVAL = context.tool_health_check_interval
        _last_tool_health_check_at = context.last_tool_health_check_at
        _TOOL_QUALITY_GATE_ENABLED = context.tool_quality_gate_enabled
        _TOPIC_SWITCH_DETECTION_ENABLED = context.topic_switch_detection_enabled
        _STUCK_THRESHOLD = context.stuck_threshold
        _STUCK_NO_CHECKPOINT_THRESHOLD = context.stuck_no_checkpoint_threshold
        _STUCK_THRESHOLD_CALIBRATED = context.stuck_threshold_calibrated
        _CHECKPOINT_NUDGE_INTERVAL = context.checkpoint_nudge_interval
        _REFLECTION_INTERVAL = context.reflection_interval
        _MAX_REQUEST_TOOLS_NOOPS = context.max_request_tools_noops
        _sys_msg = context.sys_msg
        _model_timeout = context.model_timeout
        _tool_context_limit_pct = context.tool_context_limit_pct
        _da_enabled = context.da_enabled
        _da_report_uncertainty = context.da_report_uncertainty
        _da_min_confidence = context.da_min_confidence
        _apply_context_message_cap = context.apply_context_message_cap
        _maybe_compress = context.maybe_compress
        _invoke_with_timeout = context.invoke_with_timeout
        _all_tool_results_substanceless = context.all_tool_results_substanceless
        _graph_log = get_logger()

        if llm is None:
            raise RuntimeError(
                "LLM not configured — check provider settings, API keys, and config file"
            )
        if tools_ready is not None and not tools_ready.is_set():
            _graph_log.warning(
                "MCP tools are still reconnecting; waiting briefly before binding the model"
            )
            if not tools_ready.wait(timeout=5.0):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "MCP tools are reconnecting and are not ready yet. "
                                "Please retry this turn in a moment."
                            ),
                            response_metadata={"transient": True},
                        )
                    ]
                }
        _cm_t0 = time.monotonic()
        _llm_provider = _infer_llm_provider_name(llm)
        _llm_model = _infer_llm_model_name(llm)
        call_count[0] += 1
        state_messages = list(state["messages"])
        repaired_state_messages = _repair_tool_message_pairs(state_messages)
        repaired_state_ids = {id(msg) for msg in repaired_state_messages}
        repair_removals = []
        for msg in state_messages:
            msg_id = getattr(msg, "id", None)
            if id(msg) not in repaired_state_ids and isinstance(msg_id, str) and msg_id:
                repair_removals.append(RemoveMessage(id=msg_id))
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
                _seen_names_rev: set[str] = set()
                deduped_rev: list = []
                for _t in reversed(tool_list):
                    _tname = getattr(_t, "name", "")
                    if _tname not in _seen_names_rev:
                        _seen_names_rev.add(_tname)
                        deduped_rev.append(_t)
                    else:
                        _graph_log.warning(
                            "Duplicate tool name %r in active_tools_list — "
                            "dropping extra instance to avoid API 400. "
                            "Check budget-enforcement and request_tools paths.",
                            _tname,
                        )
                        try:
                            active_tools_list.remove(_t)
                        except ValueError:
                            pass
                tool_list = list(reversed(deduped_rev))
                clean_fingerprint = tuple(getattr(t, "name", "") for t in tool_list)
                if clean_fingerprint != fingerprint:
                    _cached_fingerprint[0] = clean_fingerprint
                    fingerprint = clean_fingerprint
                if len(_bound_cache) >= 8:
                    _bound_cache.popitem(last=False)
                _bound_cache[fingerprint] = llm.bind_tools(tool_list) if tool_list else llm
            model = _bound_cache[fingerprint]
        if is_trace():
            _graph_log.debug("⏱ call_model bind_tools: %.0fms", (time.monotonic() - _cm_t0) * 1000)
        msgs = [
            m
            for m in repaired_state_messages
            if not (
                hasattr(m, "response_metadata")
                and isinstance(m.response_metadata, dict)
                and m.response_metadata.get("transient")
            )
        ]
        _comp_llm = compression_llm or llm
        if _context_max_messages > 0 or _context_max_tokens > 0:
            msgs = _apply_context_message_cap(
                msgs,
                _context_max_messages,
                _context_max_tokens,
            )
        msgs = _maybe_compress(msgs)

        if (
            _TOPIC_SWITCH_DETECTION_ENABLED
            and memory_manager is not None
            and _should_reset_summary_for_topic_switch(msgs)
        ):
            _reset_summary_state = getattr(memory_manager, "reset_summary_state", None)
            if callable(_reset_summary_state):
                _reset_summary_state()
            else:
                _legacy_reset_summary = getattr(memory_manager, "_reset_summary_state", None)
                if callable(_legacy_reset_summary):
                    _legacy_reset_summary()
            msgs.append(SystemMessage(content=_TOPIC_SWITCH_NUDGE))
            _graph_log.info("Topic switch detected — resetting summary state and nudging model")

        if not _STUCK_THRESHOLD_CALIBRATED[0] and call_count[0] == 1:
            _STUCK_THRESHOLD_CALIBRATED[0] = True
            from src.orchestration.intent import (
                TaskComplexity as _TC,
            )
            from src.orchestration.intent import (
                classify_task_complexity as _classify_tc,
            )

            _user_text = ""
            for _m in msgs:
                if hasattr(_m, "type") and _m.type == "human":
                    _user_text = getattr(_m, "content", "")
                    break
            _tc = _classify_tc(_user_text)
            if _tc == _TC.COMPLEX_ACTION:
                _STUCK_NO_CHECKPOINT_THRESHOLD[0] = 35
            elif _tc == _TC.COMPLEX_RESEARCH:
                _STUCK_NO_CHECKPOINT_THRESHOLD[0] = 20
            else:
                _STUCK_NO_CHECKPOINT_THRESHOLD[0] = 20
            _graph_log.debug(
                "Stuck threshold calibrated to %d (complexity=%s)",
                _STUCK_NO_CHECKPOINT_THRESHOLD[0],
                _tc.name,
            )

        if _calls_since_last_checkpoint[0] >= _CHECKPOINT_NUDGE_INTERVAL and call_count[0] > 3:
            _graph_log.info(
                "Checkpoint nudge fired (calls_since=%d, round=%d)",
                _calls_since_last_checkpoint[0],
                call_count[0],
            )
            msgs.append(
                HumanMessage(
                    content=(
                        "[Checkpoint reminder] You've made several actions without "
                        "recording a checkpoint. Use the checkpoint tool now to record "
                        "what you've accomplished or learned since your last checkpoint."
                    )
                )
            )
            _calls_since_last_checkpoint[0] = 0

        if _checkpoint_store is not None and len(_checkpoint_store) > 0:
            _ckpt_summary = _checkpoint_store.summary()
            if _ckpt_summary:
                msgs.append(HumanMessage(content=_ckpt_summary))

        if _checkpoint_store is not None:
            current_ckpt_count = len(_checkpoint_store)
            if current_ckpt_count > _last_checkpoint_count[0]:
                _last_checkpoint_count[0] = current_ckpt_count
                _rounds_since_checkpoint[0] = 0
                _calls_since_last_checkpoint[0] = 0
            else:
                _rounds_since_checkpoint[0] += 1
                _threshold = _STUCK_NO_CHECKPOINT_THRESHOLD[0]
                if _rounds_since_checkpoint[0] >= _threshold and call_count[0] > _threshold:
                    _force_thinking_break[0] = True
                    _rounds_since_checkpoint[0] = 0
                    _graph_log.info(
                        "No new checkpoints in %d rounds — forcing thinking break",
                        _threshold,
                    )

        if _force_thinking_break[0]:
            _force_thinking_break[0] = False
            _consecutive_errors[0] = 0
            _consecutive_identical_error_count[0] = 0
            _last_identical_error_signature[0] = None
            _graph_log.info(
                "Stuck detected — forcing thinking break (only request_tools available)"
            )
            msgs.append(
                HumanMessage(
                    content=(
                        "[THINKING BREAK — most tools temporarily disabled]\n"
                        "You have been repeating similar actions that keep failing. "
                        "STOP and think carefully:\n\n"
                        "1. Review your checkpoints — what has actually WORKED so far?\n"
                        "2. What approaches have FAILED and WHY? List each failed category.\n"
                        "3. Have you SEARCHED THE WEB? If not, after this analysis call "
                        'request_tools(add=["search_web"]) then search.\n'
                        "4. List exactly THREE categorically different strategies you "
                        "haven't tried. 'Different category' means a completely different "
                        "method — not the same method with a different URL or version.\n\n"
                        "Write out your analysis and pick ONE of your three strategies. "
                        "Do NOT guess URLs — search for real ones. "
                        "request_tools is still available so you can load whatever you need; "
                        "all other tools are restored on the next round.\n\n"
                        "If you intend to call a tool, emit a structured tool_call — "
                        "do NOT write the call as XML in your text response."
                    )
                )
            )
            # Keep request_tools bound so the model can fix the underlying
            # 'tool not loaded' problem during the thinking break itself.
            # Stripping every tool forces the model into a text-only mode where
            # qwen3-coder and similar models emit XML tool calls in content.
            _request_tools_only = [
                t for t in (active_tools_list or []) if getattr(t, "name", "") == "request_tools"
            ]
            if _request_tools_only:
                think_model = llm.bind_tools(_request_tools_only)
            else:
                think_model = llm
            think_messages = [_sys_msg, *msgs] if _sys_msg is not None else list(msgs)
            _cm_t1 = time.monotonic()
            with start_span(
                "src.orchestration.graph",
                "llm.invoke",
                attributes={
                    "llm.provider": _llm_provider,
                    "llm.model": _llm_model,
                },
            ) as _llm_span:
                try:
                    response = _invoke_with_timeout(think_model, think_messages, config, 180)
                except RuntimeError as exc:
                    _llm_span.record_exception(exc)
                    _llm_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    _graph_log.warning("LLM timed out during thinking break")
                    return {"messages": []}
                from src.orchestration.phases import normalize_native_tool_calls

                response = normalize_native_tool_calls(response)
                if is_trace():
                    _graph_log.debug(
                        "⏱ call_model thinking_break: %.0fms",
                        (time.monotonic() - _cm_t1) * 1000,
                    )
                _llm_span.set_attribute("llm.tokens_input", 0)
                _llm_span.set_attribute("llm.tokens_output", 0)
                _llm_span.set_attribute("llm.duration_ms", int((time.monotonic() - _cm_t1) * 1000))
                _llm_span.set_attribute("llm.status", "success")
                _llm_span.set_status(Status(StatusCode.OK))
            return {"messages": [response]}

        if (
            _TOOL_HEALTH_CHECK_INTERVAL > 0
            and call_count[0] > 1
            and call_count[0] % _TOOL_HEALTH_CHECK_INTERVAL == 0
            and call_count[0] != _last_tool_health_check_at[0]
        ):
            _last_tool_health_check_at[0] = call_count[0]
            _active_tool_names = sorted(getattr(context, "active_names", set()))
            if _active_tool_names:
                _tool_verification_msg = (
                    "[Tool-state verification] Confirm your current tool inventory. "
                    "You currently have access to the following tools (enumerated from the "
                    "system registry — do not rely on memory):\n"
                    + "\n".join(f"  • {name}" for name in _active_tool_names)
                    + "\n\nIf a task requires a tool not listed above, use request_tools() to load it."
                )
            else:
                _tool_verification_msg = (
                    "[Tool-state verification] You currently have NO tools loaded. "
                    "Use request_tools() to load tools before attempting actions."
                )
            msgs.append(SystemMessage(content=_tool_verification_msg))
            _graph_log.info(
                "Tool-state verification injected at turn %d (interval=%d)",
                call_count[0],
                _TOOL_HEALTH_CHECK_INTERVAL,
            )

        if (
            call_count[0] > 1
            and call_count[0] % _REFLECTION_INTERVAL == 0
            and call_count[0] != _last_reflection_at[0]
        ):
            _last_reflection_at[0] = call_count[0]
            if _consecutive_errors[0] >= 2:
                msgs.append(
                    HumanMessage(
                        content=(
                            "[Debug cycle check] You've had recent errors. Before continuing:\n"
                            "1. Read the EXACT error message from your last failed attempt.\n"
                            "2. What SPECIFIC line or issue does it point to?\n"
                            "3. Have you searched the web for that specific error or for a "
                            "working reference implementation?\n"
                            "4. Run ONLY the failing test case in isolation, not the full suite.\n"
                            "5. Fix the ONE thing the error message identifies. Don't rewrite "
                            "the whole file."
                        )
                    )
                )
            else:
                msgs.append(
                    HumanMessage(
                        content=(
                            "[Work cycle check] Before continuing:\n"
                            "1. EVALUATE: What did your last actions achieve? "
                            "Checkpoint any new findings.\n"
                            "2. PLAN: What specific information do you still need? "
                            "Write it out clearly.\n"
                            "3. RESEARCH: Search for that specific information. After getting "
                            "results, ask: do I have a SPECIFIC URL/command/answer, or just "
                            "general info? If general → refine query and search again.\n"
                            "4. ACT only when you have actionable specifics from research.\n"
                            "Do NOT guess URLs or fill in details from memory — "
                            "search until you have concrete answers."
                        )
                    )
                )

        _MAX_CONSECUTIVE_SAME_TOOL = 3
        _recent_tool_names = [
            getattr(m, "name", None)
            for m in repaired_state_messages[-(_MAX_CONSECUTIVE_SAME_TOOL * 2) :]
            if hasattr(m, "tool_call_id")
        ]
        if (
            len(_recent_tool_names) >= _MAX_CONSECUTIVE_SAME_TOOL
            and len(set(_recent_tool_names[-_MAX_CONSECUTIVE_SAME_TOOL:])) == 1
        ):
            _stuck_tool = _recent_tool_names[-1]
            msgs.append(
                SystemMessage(
                    content=(
                        f"You have called '{_stuck_tool}' {_MAX_CONSECUTIVE_SAME_TOOL} "
                        f"times in a row without making progress. Stop calling "
                        f"'{_stuck_tool}'. Choose ONE of:\n"
                        f"  (a) Produce a final text response now, summarising what you "
                        f"have already accomplished. Do not call any tools.\n"
                        f"  (b) Call a categorically different tool that advances the "
                        f"task — not the same tool with different arguments.\n"
                        f"  (c) If you genuinely need to wait for a future event, call "
                        f"cron_add to schedule it, then produce a final text response.\n"
                        f"Do NOT call '{_stuck_tool}' again."
                    )
                )
            )
            # Escalate: arm a thinking break for the next round. If the model
            # heeds the advisory above and produces text, the graph terminates
            # before the flag is checked. If it ignores the advisory and calls
            # any tool again, _force_thinking_break fires on the next call_model
            # invocation, stripping tools and forcing a text-only response.
            # Without this escalation, the duplicate-call cache returns
            # success-shaped ToolMessages, _consecutive_errors never advances,
            # and the loop runs to recursion_limit. Observed for Llama 3.3 70B
            # on the Gate 2 finance_invoice_approval_workflow scenario.
            _force_thinking_break[0] = True
            _graph_log.warning(
                "Temporal polling loop detected — '%s' called %d+ consecutive times; "
                "injecting advisory + arming thinking break for next round",
                _stuck_tool,
                _MAX_CONSECUTIVE_SAME_TOOL,
            )

        if _TOOL_QUALITY_GATE_ENABLED and _all_tool_results_substanceless(repaired_state_messages):
            msgs.append(
                SystemMessage(
                    content=(
                        "All tools returned no data this turn. Do not synthesise an answer "
                        "from prior context or memory. Report honestly that the tools "
                        "returned nothing and ask the user how to proceed."
                    )
                )
            )
            _graph_log.info("Tool output quality gate injected — all tools returned empty")

        with start_span(
            "src.orchestration.graph",
            "llm.invoke",
            attributes={
                "llm.provider": _llm_provider,
                "llm.model": _llm_model,
            },
        ) as _llm_span:
            msgs = _repair_tool_message_pairs(msgs)
            full_messages = [_sys_msg, *msgs] if _sys_msg is not None else list(msgs)
            _cm_t1 = time.monotonic()
            _LLM_TIMEOUT = _model_timeout if call_count[0] > 1 else max(_model_timeout, 300)

            try:
                response = _invoke_with_timeout(model, full_messages, config, _LLM_TIMEOUT)
            except Exception as _invoke_exc:
                if not _is_context_overflow_error(_invoke_exc):
                    _llm_span.record_exception(_invoke_exc)
                    _llm_span.set_status(Status(StatusCode.ERROR, str(_invoke_exc)))
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
                    max_context_tokens=_context_max_tokens,
                    min_age_cycles=0,
                    min_chars=0,
                    emergency_threshold=0.0,
                    actual_input_tokens=_last_input_tokens[0],
                )
                msgs = _repair_tool_message_pairs(msgs)
                full_messages = [_sys_msg, *msgs] if _sys_msg is not None else list(msgs)
                try:
                    response = _invoke_with_timeout(model, full_messages, config, 300)
                except Exception as _retry_exc:
                    _llm_span.record_exception(_retry_exc)
                    _llm_span.set_status(Status(StatusCode.ERROR, str(_retry_exc)))
                    raise RuntimeError(
                        f"Context overflow: unable to fit conversation into model context window "
                        f"({_context_max_tokens:,} tokens) even after emergency compression. "
                        "Start a new session with /session new."
                    ) from _retry_exc

            # Some models (DeepSeek-V3 via OpenRouter, qwen3-coder) emit
            # tool calls in their native format inside the assistant
            # content rather than via structured tool_calls.  Pull those
            # out before downstream routing reads the message shape.
            # No-op when the response uses standard structured calls.
            from src.orchestration.phases import normalize_native_tool_calls

            response = normalize_native_tool_calls(response)
            response = _guard_truncated_tool_calls(response, _model_max_tokens, _graph_log)

            if is_trace():
                _graph_log.debug(
                    "⏱ call_model model.invoke: %.0fms", (time.monotonic() - _cm_t1) * 1000
                )
            _span_input_tokens = 0
            _resp_um = getattr(response, "usage_metadata", None)
            if _resp_um and isinstance(_resp_um, dict):
                _resp_input = _resp_um.get("input_tokens", 0)
                if isinstance(_resp_input, int) and _resp_input > 0:
                    _last_input_tokens[0] = _resp_input
                    _span_input_tokens = _resp_input
                _total_tokens = _resp_um.get("total_tokens", 0)
                if not _total_tokens:
                    _total_tokens = _resp_um.get("input_tokens", 0) + _resp_um.get(
                        "output_tokens", 0
                    )
                if isinstance(_total_tokens, int) and _total_tokens > 0:
                    _provider, _model = _llm_provider, _llm_model
                    try:
                        from src.api.routes.metrics import LLM_TOKENS_TOTAL
                    except ModuleNotFoundError:
                        LLM_TOKENS_TOTAL = None

                    if LLM_TOKENS_TOTAL is not None:
                        LLM_TOKENS_TOTAL.labels(provider=_provider, model=_model).inc(_total_tokens)
            _resp_output = 0
            if _resp_um and isinstance(_resp_um, dict):
                _resp_output = _resp_um.get("output_tokens", 0) or 0
            _llm_span.set_attribute("llm.tokens_input", _span_input_tokens)
            _llm_span.set_attribute("llm.tokens_output", _resp_output)
            _llm_span.set_attribute("llm.duration_ms", int((time.monotonic() - _cm_t1) * 1000))
            _llm_span.set_attribute("llm.status", "success")
            _llm_span.set_status(Status(StatusCode.OK))

        response = _apply_context_budget_guard(
            response,
            max_context_tokens=_max_context_tokens,
            tool_context_limit_pct=_tool_context_limit_pct,
        )

        if _da_enabled:
            try:
                _raw_content = getattr(response, "content", "") or ""
                if isinstance(_raw_content, list):
                    _da_content = " ".join(
                        str(c.get("text", c) if isinstance(c, dict) else c) for c in _raw_content
                    )
                else:
                    _da_content = str(_raw_content)

                _da_result = extract_decision_justification(_da_content)
                if _da_result is not None:
                    _has_tool_calls = bool(getattr(response, "tool_calls", None))
                    _graph_log.info(
                        "decision_accountability: confidence=%.1f adjustment=%.1f "
                        "flaws=%d should_proceed=%s%s",
                        _da_result["confidence"],
                        _da_result["confidence_adjustment"],
                        len(_da_result["flaws"]),
                        _da_result["should_proceed"],
                        " (note suppressed: tool-calls present)" if _has_tool_calls else "",
                    )
                    if (
                        not _da_result["should_proceed"]
                        and _da_report_uncertainty
                        and not _has_tool_calls
                    ):
                        _flaw_suffix = (
                            f" with {len(_da_result['flaws'])} critical flaw(s): "
                            + "; ".join(_da_result["flaws"][:2])
                            if _da_result["flaws"]
                            else ""
                        )
                        _adj_conf = _da_result["confidence"] + _da_result["confidence_adjustment"]
                        _uncertainty_note = (
                            f"\n\n{UNCERTAINTY_NOTE_PREFIX} confidence "
                            f"{_da_result['confidence']:.1f}/10{_flaw_suffix}. "
                            f"Adjusted confidence {_adj_conf:.1f}/10 is below "
                            f"threshold {_da_min_confidence:.1f}. Proceeding with caution."
                        )
                        response = AIMessage(
                            content=_da_content + _uncertainty_note,
                            id=getattr(response, "id", None),
                        )
            except Exception as _da_exc:  # noqa: BLE001 — DA is non-critical; never crash a turn
                _graph_log.warning("decision_accountability parsing failed: %s", _da_exc)

        return {"messages": [*repair_removals, response]}

    return call_model
