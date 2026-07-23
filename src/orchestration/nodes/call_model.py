"""Extracted call_model graph node."""

from __future__ import annotations

import re
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

# ``_TOPIC_SWITCH_NUDGE`` and ``_should_reset_summary_for_topic_switch``
# are re-exported here purely for the test-patching contract: tests do
# ``patch("src.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
# ...)`` and the extracted ``pre_invoke_directives`` module resolves
# these via attribute lookup on this module so the patches stick. Do not
# remove without updating both the tests and the extracted module.
from src.orchestration.graph import (
    _TOPIC_SWITCH_NUDGE,  # noqa: F401 — re-exported for test patching
    _apply_context_budget_guard,
    _infer_llm_model_name,
    _infer_llm_provider_name,
    _is_context_overflow_error,
    _repair_tool_message_pairs,
    _should_reset_summary_for_topic_switch,  # noqa: F401 — re-exported for test patching
)
from src.orchestration.nodes.pre_invoke_directives import (
    apply_late_directives,
    apply_pre_invoke_directives,
)
from src.orchestration.nodes.thinking_break_policy import maybe_apply_thinking_break
from src.orchestration.reflection_delegate import (
    UNCERTAINTY_NOTE_PREFIX,
    extract_decision_justification,
)
from src.orchestration.response_detectors import _SYCOPHANTIC_PREFIX_RE
from src.orchestration.search_quality import (
    SearchQualityThresholds,
    has_substantive_search_results,
)

# Bug K #1720 — CJK detection. Multilingual models (qwen3-coder
# observed in cogtrix57.log lines 27652, 30376) occasionally sample a
# CJK token where an English equivalent was expected ("立场" / "表态"
# in place of "stance" / "statement"). The character ranges cover:
#   U+4E00..U+9FFF — CJK Unified Ideographs (Chinese, Japanese kanji)
#   U+3040..U+309F — Hiragana
#   U+30A0..U+30FF — Katakana
#   U+3400..U+4DBF — CJK Extension A
# This is detection-only — we log a warning and do not strip / rewrite
# content. Some legitimate responses contain CJK (the user explicitly
# asks about Chinese terms, or a tool returned CJK content the agent
# is quoting verbatim) and silent stripping would corrupt those.
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ㐀-䶿]")

# Bug G #1713 follow-up — orchestrator-side detection of sycophantic
# validation prefixes. The system-prompt rule in
# ``src/agent/core.py:build_system_prompt`` is the primary defense, but
# RLHF-tuned chat models (qwen3-coder observed in 2026-05-21 corpus
# replay on E03; deepseek-v4-flash in next69) still emit "You're right —
# let me ..." prefixes even when the rule explicitly forbids them. The
# regex itself lives in ``response_detectors`` alongside the other
# router-level predicates (imported at module top); this module retains
# the strip helper below for the LOGGING-ONLY observation path. The
# router-level recovery node (``handle_sycophancy`` in graph.py) does
# the actual remove-and-regenerate using the same regex.


def _strip_sycophantic_prefix(text: str) -> tuple[str, str | None]:
    """Return ``(stripped_text, matched_prefix_or_None)``.

    When the response starts with a sycophantic validation phrase, the
    matched prefix is removed and the next character is capitalised so
    the remaining text reads as a complete sentence. When no prefix
    matches, returns the input unchanged with ``None``.
    """
    m = _SYCOPHANTIC_PREFIX_RE.match(text)
    if not m:
        return text, None
    matched = m.group(0).strip()
    remainder = text[m.end() :].lstrip()
    if remainder and remainder[0].islower():
        remainder = remainder[0].upper() + remainder[1:]
    return remainder, matched


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
    search_quality_thresholds: Any | None = None


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

# Minimum effort threshold before the no-checkpoint thinking break may
# instruct the model to refuse.  Prevents "lazy refusal" where the agent
# gives up after only shallow searches (#1520).
_MIN_SEARCH_EFFORT = 3

# Stopwords dropped when normalising search queries for distinctness.
_SEARCH_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
    }
)


def _normalise_query(query: str) -> frozenset[str]:
    """Lower-case, tokenise, and drop stopwords for distinctness comparison."""
    words = re.findall(r"\b\w+\b", query.lower())
    return frozenset(w for w in words if w not in _SEARCH_STOPWORDS)


# ── Arithmetic-intent detection (cogtrix47 Issue 4) ──────────────────
#
# When the user asks "how many X for $Y" or "convert N units of X",
# and the message history contains numeric tool results (prices,
# rates, quantities), the agent must attempt the calculation before
# emitting a flat refusal. cogtrix47 had NZD→EUR rate + EUR prices
# in hand but answered "I could not retrieve current data" instead
# of bounding "approximately N units" with explicit caveats — that
# is laziness dressed up as honesty.

# Phrase signals that mark the user's prompt as an arithmetic-style
# question (count / quantity / conversion / total). Each must be
# tight enough that ordinary research prompts ("how does X work")
# don't trip — anchor with word-boundary + the quantifying noun.
_ARITHMETIC_INTENT_RE = re.compile(
    r"(?i)"
    r"\bhow\s+many\b"
    r"|\bhow\s+much\b"
    r"|\bhow\s+long\b"
    r"|\b(?:can|could)\s+i\s+(?:afford|buy|get)\b"
    r"|\bconvert(?:ed|ing)?\b"
    r"|\bcalculate\b"
    r"|\bwhat(?:'?s|\s+is)\s+the\s+(?:total|sum|cost|price|exchange|conversion)\b"
    r"|\bin\s+(?:USD|EUR|GBP|NZD|AUD|CAD|JPY|CNY|INR|CHF)\b"
)

# Numeric / currency tokens that mark a ToolMessage as carrying
# data worth computing on. The currency-prefix forms ($100, €50)
# and bare ISO codes (USD, EUR, ...) cover the FX and pricing
# shapes most likely to appear in search-fetched extracts.
_NUMERIC_RESULT_RE = re.compile(
    r"(?:"
    r"[$€£¥₹]\s?\d"  # $100, € 50, £1.99
    r"|\b\d+(?:[.,]\d+)?\s*(?:USD|EUR|GBP|NZD|AUD|CAD|JPY|CNY|INR|CHF)\b"
    r"|\b(?:USD|EUR|GBP|NZD|AUD|CAD|JPY|CNY|INR|CHF)\s*\d"
    # Percentage: no trailing \b — ``%`` is non-word and a following
    # punctuation char (``.``, ``,``) is also non-word, so \b fails.
    r"|\b\d+(?:[.,]\d+)?\s*(?:%|\bpercent\b)"
    r")"
)


def _has_arithmetic_intent(messages: list[Any]) -> bool:
    """Return True when the most recent user prompt looks like an
    arithmetic / quantity / conversion question.

    Scans only the last HumanMessage so a single arithmetic question
    earlier in the session doesn't flag every subsequent prompt.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            return bool(_ARITHMETIC_INTENT_RE.search(content))
    return False


def _has_numeric_tool_results(messages: list[Any]) -> bool:
    """Return True when any tool message in the *current turn* carries
    a currency / percentage / numeric data token.

    Pairs with ``_has_arithmetic_intent`` to determine whether the
    agent has the raw material to attempt a calculation. Current-turn
    scoping mirrors ``_compute_search_effort`` (#1532).
    """
    last_human_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    scope = messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages
    for msg in scope:
        if not hasattr(msg, "tool_call_id"):
            continue
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            continue
        if _NUMERIC_RESULT_RE.search(content):
            return True
    return False


def _compute_search_effort(messages: list[Any]) -> tuple[int, bool]:
    """Count *distinct* search_web calls and detect http_get attempts.

    Returns ``(distinct_search_count, http_get_attempted)``.  Used by the
    thinking-break effort gate.  Near-duplicate queries (e.g. reordered
    words or added stopwords) are collapsed to a single count so the gate
    measures *lateral* effort, not mere repetition (#1520).

    Effort is scoped to the *current user turn* only — not session-cumulative.
    This prevents long sessions with prior search activity from short-circuiting
    fresh questions into refusal without the strategy nudge firing (#1532).
    """
    # Find the last HumanMessage to scope effort to the current turn.
    last_human_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    scope = messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages

    search_count = 0
    http_get_attempted = False
    seen_queries: set[frozenset[str]] = set()

    for i, msg in enumerate(scope):
        if not hasattr(msg, "tool_call_id"):
            continue
        tool_name = getattr(msg, "name", None)
        if tool_name == "http_get":
            http_get_attempted = True
            continue
        # Accept both the legacy ``search_web`` tool name and the
        # modern ``web_search`` tool that superseded it (PR-G /
        # ADR-0056). Without ``web_search`` in this set, the
        # cogtrix47 run's 7 web_search calls counted as zero effort
        # and the stuck-detection branched into the non-search-loop
        # refusal body.
        if tool_name not in ("search_web", "web_search"):
            continue

        # Skip error / stub results that indicate the search never ran.
        content = getattr(msg, "content", "") or ""
        if content.startswith("Error searching") or "not loaded" in content.lower():
            continue

        # Walk backward to find the AIMessage.tool_calls that triggered this
        # search so we can extract the query text for distinctness checking.
        tool_call_id = getattr(msg, "tool_call_id", None)
        query = ""
        for prev_msg in reversed(scope[:i]):
            if not hasattr(prev_msg, "tool_calls"):
                continue
            for tc in getattr(prev_msg, "tool_calls", []):
                if tc.get("id") == tool_call_id:
                    args = tc.get("args", {})
                    query = args.get("query", "") if isinstance(args, dict) else str(args)
                    break
            if query:
                break

        normalised = _normalise_query(query)
        if normalised and normalised not in seen_queries:
            seen_queries.add(normalised)
            search_count += 1

    return search_count, http_get_attempted


def _has_substantive_search_results(
    messages: list[Any],
    thresholds: SearchQualityThresholds | None = None,
) -> bool:
    """Thin wrapper around ``has_substantive_search_results`` (#1593, Option B).

    Delegates to the dedicated ``search_quality`` module, which fixes the
    dead ``startswith("Error searching")`` check (actual error format is
    ``"Tool failed: search_web - Error searching..."``) and adds
    observability logging for false-negative detection.

    The ``thresholds`` parameter is pulled from ``CallModelContext`` at
    call time so the heuristic remains configurable via ``cogtrix.yaml``.
    """
    return has_substantive_search_results(messages, thresholds)


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

        # ── Phase 1 (P1 + P0) — pre-invoke directives ──────────────
        # Builds ``msgs`` from ``repaired_state_messages`` via:
        # transient filter → context cap → compress → topic-switch
        # nudge → stuck-conclusion nudge → stuck-threshold calibration
        # → checkpoint nudge → checkpoint summary → rounds-since-
        # checkpoint accounting (sets / clears
        # ``_force_thinking_break[0]`` for THIS round).
        msgs = repaired_state_messages
        msgs = apply_pre_invoke_directives(
            context,
            state_messages,
            repaired_state_messages,
            msgs,
            _graph_log,
        )
        _comp_llm = compression_llm or llm

        # ── Phase 2 — thinking-break sub-invocation ────────────────
        # Consumes ``_force_thinking_break[0]`` if set. Returns an
        # early-exit dict when the sub-invocation fired (graph treats
        # the round as terminal-ish); returns ``None`` when the flag
        # was clear or when the low-effort search-loop suppression
        # branch fell through (a STRATEGY NUDGE was appended to
        # ``msgs`` and we continue to normal processing).
        _tb_result = maybe_apply_thinking_break(
            context,
            state_messages,
            repaired_state_messages,
            msgs,
            config,
            _graph_log,
        )
        if _tb_result is not None:
            return _tb_result

        # ── Phase 3 (P2) — late directives ─────────────────────────
        # Tool-state verification → reflection → polling-loop advisory
        # → tool-output quality gate. The polling-loop branch may arm
        # ``_force_thinking_break[0]`` for the NEXT call_model round
        # (the current round's consumer above has already run).
        msgs = apply_late_directives(
            context,
            state_messages,
            repaired_state_messages,
            msgs,
            _graph_log,
        )

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

        # Bug K #1720 — CJK leakage detection. Multilingual LLMs
        # (qwen3-coder in cogtrix57) occasionally sample a CJK token
        # where an English one was expected. Detection-only: log a
        # WARNING so ops can see the rate without altering the model
        # output (legitimate quoted CJK content must survive).
        _resp_content = getattr(response, "content", "")
        _resp_text = ""
        if isinstance(_resp_content, str):
            _resp_text = _resp_content
        elif isinstance(_resp_content, list):
            _resp_text = " ".join(
                str(c.get("text", c) if isinstance(c, dict) else c) for c in _resp_content
            )
        if _resp_text:
            _cjk_matches = _CJK_RE.findall(_resp_text)
            if _cjk_matches:
                _graph_log.warning(
                    "Non-Latin (CJK) characters in assistant response (%d found); "
                    "possible multilingual-model leakage (Bug K #1720). Sample: %r",
                    len(_cjk_matches),
                    _cjk_matches[:5],
                )

        # Bug G #1713 follow-up — detect sycophantic validation prefix.
        # The system-prompt rule forbids "You're absolutely right" /
        # "I apologize" prefixes on unchanged answers, but the 2026-05-21
        # corpus replay (E03 captured intra-turn LLM_GENERATION) showed
        # qwen3-coder still emits "You're right - let me ..." after an
        # orchestrator nudge. Detection-only here: log a WARNING so ops
        # can see the bypass rate without modifying the response.
        #
        # We deliberately do NOT strip the prefix at this layer. An
        # earlier prototype (PR #1731 first/second iterations) stripped
        # the matched span and rebuilt the response via
        # ``response.model_copy``. That broke Gate 2 shard D × kimi-k2-5
        # and shard B × kimi-k2-5 because the strip altered intra-turn
        # responses kimi later referenced (the post-strip remainder no
        # longer carried context that downstream scoring depended on).
        # The prompt rule remains the primary defense; detection-only
        # surfaces the rate so the impact can be quantified.
        if isinstance(_resp_content, str) and _resp_content:
            _, _matched_prefix = _strip_sycophantic_prefix(_resp_content)
            if _matched_prefix is not None:
                _graph_log.warning(
                    "Sycophantic prefix detected in response: %r (Bug G #1713). "
                    "Prompt rule was bypassed by the model; logging only — no "
                    "content modification.",
                    _matched_prefix,
                )

        return {"messages": [*repair_removals, response]}

    return call_model
