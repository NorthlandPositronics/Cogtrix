"""Thinking-break sub-invocation extracted from the call_model node
(forge A3).

The thinking break is a short, *tool-restricted* sub-invocation that
fires when the agent has gotten stuck — either because it has run too
many rounds without recording a checkpoint (set in
:func:`pre_invoke_directives._phase_p0_calibration_and_checkpoint`) or
because a polling-loop advisory armed the flag in the previous round
(set in :func:`pre_invoke_directives._phase_p2_late_directives`).

When the flag is consumed:

* All tools are stripped except ``request_tools`` (kept so qwen3-coder
  and similar models don't fall back to emitting XML tool calls inside
  the assistant content when forced into text-only mode).
* The system prompt + current ``msgs`` are sent through a one-shot LLM
  invocation with a 180-second timeout.
* The response replaces the normal call_model output — the function
  returns ``{"messages": [response]}`` so the graph treats this round
  as terminal-ish (downstream routing decides whether tools were called
  and where to go next).

Three thinking-break body variants:

1. **Has checkpoints** — synthesise from recorded checkpoints.
2. **Search loop with effort + results** — synthesise from search
   message history (cogtrix47 Issue 4 arithmetic clause included when
   numeric data is present).
3. **Search loop with effort, no results** — honest refusal (don't
   fabricate).
4. **Search loop with low effort** — STRATEGY NUDGE injected; thinking
   break is **suppressed** and the function returns ``None`` so the
   caller continues to normal tool-enabled processing (#1520).
5. **Non-search stuck** — fall back to the honest-refusal variant.

The flag is *always* cleared at function entry, so a follow-up round
will not loop unless re-armed by P0 or P2.

Test-patching contract
======================
Tests patch helpers via the ``src.orchestration.nodes.call_model.<name>``
attribute. To preserve that, this module imports the ``call_model``
module at call time and resolves ``_compute_search_effort``,
``_has_substantive_search_results``, ``_has_arithmetic_intent``,
``_has_numeric_tool_results``, ``_MIN_SEARCH_EFFORT`` via attribute
lookup on it. The helpers stay in ``call_model.py``.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry.trace import Status, StatusCode

from cogtrix_core.api.telemetry import start_span
from cogtrix_core.logging_config import is_trace
from cogtrix_core.orchestration.graph import (
    _infer_llm_model_name,
    _infer_llm_provider_name,
)


def maybe_apply_thinking_break(
    context: Any,
    state_messages: list[Any],
    repaired_state_messages: list[Any],
    msgs: list[Any],
    config: RunnableConfig,
    log: Any,
) -> dict | None:
    """Run the thinking-break sub-invocation if
    ``context.force_thinking_break[0]`` is armed.

    Returns:
        * ``{"messages": [response]}`` if the thinking break fired and
          its sub-invocation completed (or timed out — a graceful
          ``{"messages": []}`` is returned on timeout).
        * ``None`` if the flag was clear OR if the flag was set but the
          low-effort suppression branch fired (a STRATEGY NUDGE was
          appended to ``msgs`` and the caller should continue with
          normal tool-enabled processing).

    Side effects:
        * Clears ``context.force_thinking_break[0]`` when the flag was
          set (regardless of which body variant fired).
        * Resets ``consecutive_errors[0]``,
          ``consecutive_identical_error_count[0]`` and
          ``last_identical_error_signature[0]`` when the flag was set.
        * May append a ``HumanMessage`` (thinking-break body) or a
          ``HumanMessage`` (STRATEGY NUDGE) to ``msgs``.
    """
    if not context.force_thinking_break[0]:
        return None

    # Defer to module-attribute lookup so test patches resolve correctly.
    from cogtrix_core.orchestration.nodes import call_model as _cm

    context.force_thinking_break[0] = False
    context.consecutive_errors[0] = 0
    context.consecutive_identical_error_count[0] = 0
    context.last_identical_error_signature[0] = None
    log.info("Stuck detected — forcing thinking break (only request_tools available)")
    _has_checkpoints = context.checkpoint_store is not None and len(context.checkpoint_store) > 0

    # Determine whether the stuck state is a search loop.  The effort gate
    # only applies when the agent is looping on search_web; for non-search
    # stuck tools (e.g. merge_pull_request, write_file) the original
    # THINKING BREAK behaviour is preserved (#1520).
    _recent_tool_names = [
        getattr(m, "name", None) for m in repaired_state_messages[-6:] if hasattr(m, "tool_call_id")
    ]
    _stuck_tool_name = _recent_tool_names[-1] if _recent_tool_names else None
    # PR-G / ADR-0056: ``web_search`` replaced the legacy
    # ``search_web``. Accept both so the search-loop branch
    # — which knows how to grade effort and steer toward
    # synthesis vs. refusal — still fires for the modern
    # tool (cogtrix47 regression).
    _is_search_loop = _stuck_tool_name in ("search_web", "web_search")

    # Defaults so the final dispatch condition has every name bound.
    _effort_met = False
    _tb_body: str | None = None

    if _has_checkpoints:
        _tb_body = (
            "[THINKING BREAK — tools disabled this round]\n"
            "Recent attempts are not producing new information. "
            "You have recorded checkpoints during this session. "
            "Synthesize a final answer from those checkpoints.\n\n"
            "Write the answer directly — do not narrate your approach, do "
            "not enumerate what failed, do not list alternative methods. "
            "Just answer the question.\n\n"
            "Tools are restored on the next round if you still need them."
        )
    elif _is_search_loop:
        _search_count, _http_get_attempted = _cm._compute_search_effort(msgs)
        _effort_met = _search_count >= _cm._MIN_SEARCH_EFFORT or _http_get_attempted
        _has_results = _cm._has_substantive_search_results(
            msgs, thresholds=getattr(context, "search_quality_thresholds", None)
        )
        if _effort_met and _has_results:
            # Effort spent AND results came back rich — synthesise,
            # don't refuse.  Closes #1585: a cold-start session
            # with successful searches was emitting one-line
            # "I could not retrieve current data" refusals
            # despite having a message history full of real
            # product names and URLs.  The agent had material
            # but the prior thinking-break wording framed
            # refusal as the preferred output, and at high
            # temperature the model took the easy path.
            #
            # cogtrix47 follow-up (Issue 4): when the user
            # asked an arithmetic question ("how many for
            # $100") and the tool results contain numeric
            # data (prices, FX rates), the synthesise nudge
            # must explicitly require an attempted
            # calculation — not just a "structured listing"
            # — so the agent doesn't refuse with the math
            # half-done.
            _has_arith = _cm._has_arithmetic_intent(msgs)
            _has_nums = _cm._has_numeric_tool_results(msgs)
            log.info(
                "Thinking break — effort met (%d distinct searches) and "
                "substantive results present; instructing model to synthesise"
                " (arithmetic_intent=%s, numeric_data=%s)",
                _search_count,
                _has_arith,
                _has_nums,
            )
            _arith_clause = ""
            if _has_arith and _has_nums:
                _arith_clause = (
                    "\n\nARITHMETIC INTENT DETECTED — the user asked a "
                    "quantity / conversion / total question, and your "
                    "tool results contain numeric values (prices, "
                    "exchange rates, percentages). You MUST attempt "
                    "the calculation before any refusal. Even a "
                    "BOUNDED estimate with explicit caveats "
                    "('using the cheapest comparable SKU at €X and "
                    "today's NZD→EUR rate of ~Y, $100 NZD buys "
                    "approximately N units, plus or minus M due to "
                    "Z') is more useful than 'I could not retrieve'."
                    " Show your working in one or two sentences so "
                    "the user can sanity-check the assumptions."
                )
            _tb_body = (
                "[THINKING BREAK — tools disabled this round]\n"
                "Your prior search results in this turn contain substantive "
                "content — real product names, URLs, and descriptions in "
                "the message history.  Stop searching and synthesise.\n\n"
                "Review the search results that have already been returned. "
                "Produce a structured answer that lists what was found and "
                "how each result relates to the user's question.  If the "
                "user asked for a specific named product that didn't appear "
                "in the results, name the products that DID appear and "
                "explain whether each one solves their problem.\n\n"
                "Do NOT respond with 'I could not retrieve current data' "
                "or any equivalent one-line refusal — your message history "
                "shows real search results, so that response would be "
                "lazy rather than honest.  An honest refusal applies only "
                "when results were genuinely empty.\n\n"
                "Do NOT fabricate URLs, facts, or product details beyond "
                "what appeared verbatim in the actual search results — only "
                "synthesise content that is already in your message history."
                f"{_arith_clause}"
                "\n\n"
                "Tools are restored on the next round if you still need them."
            )
        elif _effort_met:
            # Effort spent but results were empty / errors / blocked
            # pages — honest refusal is appropriate (#1520).
            _tb_body = (
                "[THINKING BREAK — tools disabled this round]\n"
                "Recent attempts have not produced useful data. "
                "No checkpoint information has been accumulated in this session.\n\n"
                "If the topic is one where you cannot reach a definitive answer "
                "without live data (current prices, stock levels, recent events, "
                "specific SKUs, FX rates), state plainly that you could not "
                "retrieve the data and suggest the user contact the source "
                "directly. Do NOT fabricate specific numbers, percentages, "
                "citations, URLs, or links from your training data — that is "
                "a worse failure than not answering.  If you mention a website "
                "or repository it must come from an actual tool result, not "
                "from what you would expect to see online.  A pointer to a "
                "non-existent URL is worse than no pointer at all.\n\n"
                "A short honest 'I could not retrieve current data on this topic' "
                "is far better than a confident fabrication.\n\n"
                "Tools are restored on the next round if you still need them."
            )
        else:
            # Low effort — the agent has not earned the right to refuse.
            # Inject a strategy nudge and continue with tools enabled (#1520).
            log.info(
                "Thinking break suppressed — low effort (%d distinct searches, "
                "http_get=%s); injecting strategy nudge instead",
                _search_count,
                _http_get_attempted,
            )
            msgs.append(
                HumanMessage(
                    content=(
                        "[STRATEGY NUDGE] Your searches have not yet surfaced a "
                        "definitive answer. Before giving up, try harder:\n"
                        "  1. Change the search language if the topic is region-specific.\n"
                        "  2. Drop overly-specific identifiers and search for broader categories.\n"
                        "  3. Follow a promising URL from your search results with http_get.\n"
                        "  4. Search for distributors or official contact pages instead of retailers.\n"
                        "  5. Use the checkpoint tool to record any partial findings.\n\n"
                        "Only refuse after you have tried at least one of these angles."
                    )
                )
            )
            # Skip the forced text-only break so the model can act on the nudge.
            # Fall through to normal tool-enabled processing below.
    else:
        # Not a search loop — fire the normal refusal thinking break.
        _tb_body = (
            "[THINKING BREAK — tools disabled this round]\n"
            "Recent attempts have not produced useful data. "
            "No checkpoint information has been accumulated in this session.\n\n"
            "If the topic is one where you cannot reach a definitive answer "
            "without live data (current prices, stock levels, recent events, "
            "specific SKUs, FX rates), state plainly that you could not "
            "retrieve the data and suggest the user contact the source "
            "directly. Do NOT fabricate specific numbers, percentages, "
            "citations, URLs, or links from your training data — that is a "
            "worse failure than not answering.  If you mention a website or "
            "repository it must come from an actual tool result, not from "
            "what you would expect to see online.  A pointer to a non-existent "
            "URL is worse than no pointer at all.\n\n"
            "A short honest 'I could not retrieve current data on this topic' "
            "is far better than a confident fabrication.\n\n"
            "Tools are restored on the next round if you still need them."
        )

    # Dispatch: fire the sub-invocation only when one of the body
    # variants was selected. The low-effort search-loop branch
    # intentionally falls through (returns None) so the caller can
    # continue to normal tool-enabled processing.
    if _has_checkpoints or (_is_search_loop and _effort_met) or not _is_search_loop:
        msgs.append(HumanMessage(content=_tb_body))
        # Keep request_tools bound so the model can fix the underlying
        # 'tool not loaded' problem during the thinking break itself.
        # Stripping every tool forces the model into a text-only mode where
        # qwen3-coder and similar models emit XML tool calls in content.
        _request_tools_only = [
            t
            for t in (context.active_tools_list or [])
            if getattr(t, "name", "") == "request_tools"
        ]
        if _request_tools_only:
            think_model = context.llm.bind_tools(_request_tools_only)
        else:
            think_model = context.llm
        think_messages = [context.sys_msg, *msgs] if context.sys_msg is not None else list(msgs)
        _llm_provider = _infer_llm_provider_name(context.llm)
        _llm_model = _infer_llm_model_name(context.llm)
        _cm_t1 = time.monotonic()
        with start_span(
            "cogtrix_core.orchestration.graph",
            "llm.invoke",
            attributes={
                "llm.provider": _llm_provider,
                "llm.model": _llm_model,
            },
        ) as _llm_span:
            try:
                response = context.invoke_with_timeout(think_model, think_messages, config, 180)
            except RuntimeError as exc:
                _llm_span.record_exception(exc)
                _llm_span.set_status(Status(StatusCode.ERROR, str(exc)))
                log.warning("LLM timed out during thinking break")
                return {"messages": []}
            from cogtrix_core.orchestration.phases import normalize_native_tool_calls

            response = normalize_native_tool_calls(response)
            if is_trace():
                log.debug(
                    "⏱ call_model thinking_break: %.0fms",
                    (time.monotonic() - _cm_t1) * 1000,
                )
            _llm_span.set_attribute("llm.tokens_input", 0)
            _llm_span.set_attribute("llm.tokens_output", 0)
            _llm_span.set_attribute("llm.duration_ms", int((time.monotonic() - _cm_t1) * 1000))
            _llm_span.set_attribute("llm.status", "success")
            _llm_span.set_status(Status(StatusCode.OK))
        return {"messages": [response]}

    # Low-effort search-loop suppression path: no thinking break fired,
    # let the caller continue with normal tool-enabled processing.
    return None
