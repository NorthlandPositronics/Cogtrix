"""Pre- and late-invoke directive injectors for the call_model node.

Extracted from ``cogtrix_core/orchestration/nodes/call_model.py`` (forge A2). This
module owns the two directive phases that bracket the optional
thinking-break sub-invocation:

* :func:`apply_pre_invoke_directives` — runs immediately after the bound
  model is selected. It performs the post-bind message preparation
  (P1: transient filtering, context cap, compression, topic-switch
  nudge, stuck-conclusion nudge) and the calibration / checkpoint
  bookkeeping (P0: stuck-threshold calibration, checkpoint nudge,
  checkpoint summary injection, rounds-since-checkpoint accounting).
  P0 also **arms or clears** ``context.force_thinking_break[0]`` for the
  *current* round — the thinking-break sub-invocation that runs next
  consumes the flag.

* :func:`apply_late_directives` — runs after the thinking-break
  sub-invocation has either fired and returned (early exit, never
  reaches this function) or been skipped. It injects the late-round
  directives (P2: tool-state verification, reflection cycle, polling
  loop advisory, tool-output quality gate). The polling-loop branch
  here **arms** ``context.force_thinking_break[0]`` for the *next*
  call_model round — the current round's consumer has already run by
  the time this executes.

Behaviour preservation
======================
The function bodies are a 1:1 lift from the original ``call_model``
closure (lines 622-782 for P1+P0, lines 1011-1158 for P2). The
ordering of message appends and side effects is identical to the
original:

P1 + P0 order:

1. transient-filter (drops messages flagged ``response_metadata
   ["transient"]=True``)
2. compress (``maybe_compress``) — reduces size by summarising old
   ToolMessages while preserving the content (lossy but recoverable).
   Runs FIRST so the cap below has less to drop.
3. context-cap (``apply_context_message_cap``) — last-resort eviction
   when compression alone didn't fit the budget.  Drops oldest message
   chunks and emits a SystemMessage marker noting that data was lost
   (#1943 PR #1).
4. topic-switch nudge (``_TOPIC_SWITCH_NUDGE`` appended +
   ``memory_manager.reset_summary_state`` invoked)
5. stuck-conclusion nudge (only at ``call_count == 1`` when prior 2
   final assistant responses are >=90% similar)
6. stuck-threshold calibration (only on ``call_count == 1``)
7. checkpoint nudge (when ``calls_since_last_checkpoint >=
   CHECKPOINT_NUDGE_INTERVAL`` and ``call_count > 3``)
8. checkpoint summary (appended verbatim from
   ``checkpoint_store.summary()``)
9. rounds-since-checkpoint accounting (sets / clears
   ``force_thinking_break[0]`` for the current round)

P2 order:

11. tool-verification (at every ``TOOL_HEALTH_CHECK_INTERVAL`` rounds)
12. reflection cycle (every ``REFLECTION_INTERVAL`` rounds; debug-cycle
    variant when ``consecutive_errors[0] >= 2``)
13. polling-loop advisory (3 consecutive identical tool calls; arms
    ``force_thinking_break[0]`` for the *next* round unless all stub
    results)
14. tool-quality gate (when all tool results are substanceless)

Test-patching contract
======================
The original ``call_model.py`` exports ``_should_reset_summary_for_topic_switch``
as a re-exported name; tests patch it at
``src.orchestration.nodes.call_model._should_reset_summary_for_topic_switch``.
To preserve that contract, this module looks up the helper via the
``call_model`` module attribute (``_cm._should_reset_summary_for_topic_switch``)
at call time rather than importing the bound symbol — so patches stick.
The same pattern applies to ``_compute_search_effort``,
``_has_arithmetic_intent``, ``_has_numeric_tool_results``, and
``_has_substantive_search_results`` — but those are used only inside the
thinking-break body (``thinking_break_policy.py``), not here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cogtrix_core.orchestration.graph import _TOPIC_SWITCH_NUDGE

# Imported lazily inside the functions to preserve the test-patching
# contract (tests do ``patch("cogtrix_core.orchestration.nodes.call_model.
# _should_reset_summary_for_topic_switch", ...)``).

# #2054: the stuck-conclusion nudge (Bug G #1713) only makes sense for
# substantial research-style answers. Short conversational acknowledgments
# (e.g. "No rush, take your time.") trivially exceed the 90% similarity
# threshold and would otherwise trip the nudge, forcing the assistant to
# re-reply and producing duplicate/broken-record messages on chat channels.
# Require the (near-identical) prior finals to exceed this length before firing.
_STUCK_CONCLUSION_MIN_CHARS = 80


def apply_pre_invoke_directives(
    context: Any,
    state_messages: list[Any],
    repaired_state_messages: list[Any],
    msgs: list[Any],
    log: Any,
) -> list[Any]:
    """Run the P1 (post-bind message prep) and P0 (calibration +
    checkpoint accounting) directive phases.

    Mutates ``msgs`` in place — but also returns it for chaining
    convenience. Mutates list-cell fields on ``context`` in place
    (``call_count``, ``calls_since_last_checkpoint``,
    ``last_checkpoint_count``, ``rounds_since_checkpoint``,
    ``force_thinking_break``, ``stuck_threshold_calibrated``,
    ``stuck_no_checkpoint_threshold``).

    The ``force_thinking_break[0]`` cell may be **set** by P0 step 9
    (stuck via no-checkpoint accounting) or **cleared** by P0 step 9
    (a new checkpoint was recorded). The next phase
    (:func:`thinking_break_policy.maybe_apply_thinking_break`) consumes
    the flag in the *same* call_model round.
    """
    # Import inside the function so test patches on the call_model
    # module attributes resolve to the patched values, not the bound
    # symbols at import time.
    from cogtrix_core.orchestration.nodes import call_model as _cm

    msgs = _phase_p1_post_bind_prep(context, msgs, log, _cm)
    msgs = _phase_p0_calibration_and_checkpoint(context, msgs, log, _cm)
    return msgs


def apply_late_directives(
    context: Any,
    state_messages: list[Any],
    repaired_state_messages: list[Any],
    msgs: list[Any],
    log: Any,
) -> list[Any]:
    """Run the P2 (late directives) phase.

    Mutates ``msgs`` in place and returns it. May arm
    ``context.force_thinking_break[0]`` for the **next** call_model
    round via the polling-loop branch — the current round's
    thinking-break consumer has already run by the time this executes,
    so an arm here applies one round later.

    The polling-loop arm is suppressed when every recent same-tool
    result was a "not loaded" stub (bug #1510 fix) — the agent is
    still discovering the tool's state via ``request_tools`` and
    arming would punish the correct recovery move.
    """
    msgs = _phase_p2_late_directives(context, repaired_state_messages, msgs, log)
    return msgs


def _maybe_get_rolling_summary(memory_manager: Any, log: Any) -> str | None:
    """Return the memory layer's rolling summary text when available.

    Reads ``MemoryManager._summary`` under ``_hybrid_lock`` (non-blocking,
    short timeout) so a contended summarizer job never stalls the
    cascade.  Returns ``None`` on:

    - no manager (CLI direct path with no memory plumbing)
    - manager has no ``_summary`` attribute (very old subclass / mock)
    - lock contention beyond a short timeout
    - any exception (broad catch — memory layer must never break the
      orchestration cascade; the marker simply falls back to PR #1 prose)

    The string may be empty when the summarizer has not yet produced a
    summary; callers treat empty the same as ``None``.
    """
    if memory_manager is None:
        return None
    try:
        lock = getattr(memory_manager, "_hybrid_lock", None)
        if lock is not None and hasattr(lock, "acquire"):
            # Short timeout — never block the cascade waiting on the
            # background summarizer.
            if not lock.acquire(timeout=0.05):
                return None
            try:
                summary = getattr(memory_manager, "_summary", None)
            finally:
                lock.release()
        else:
            summary = getattr(memory_manager, "_summary", None)
        if isinstance(summary, str):
            return summary
        return None
    except Exception as exc:  # noqa: BLE001 — memory must never crash the cascade
        log.debug("Rolling-summary fetch skipped: %s", exc)
        return None


# Action-tier cap response signature — see ``cogtrix_core/orchestration/nodes/
# process_tools.py:343``.  The dispatcher emits a ToolMessage with this
# text when a tool has been called more than ``MAX_CONSECUTIVE_ACTION_CALLS``
# times in succession this turn.  When this signature appears in the
# recent N ToolMessages, the agent is being explicitly told to STOP
# calling the tool — the polling-loop detector must still fire so the
# advisory + thinking-break arm push the agent toward a final response.
# Without this guard, #1943 Fix #4's distinct-args exemption suppresses
# the polling-loop signal even while the agent thrashes the cap with
# different queries — observed as Gate 2's
# ``regression_web_search_no_external_url_recommendation_on_low_yield``
# recursing to the limit after 25 cap-hits.
_ACTION_TIER_CAP_SIGNATURE = "times in succession this turn"


def _recent_tool_responses_are_cap_hits(
    repaired_state_messages: list[Any],
    n: int,
) -> bool:
    """True when the latest *n* ToolMessages contain action-tier cap-hit
    responses (a dispatcher-emitted ``Further '<tool>' calls are blocked``
    message).  When the agent is repeatedly hitting the action-tier cap,
    the iteration is NOT making progress — it's thrashing — and the
    polling-loop signal must fire regardless of args distinctness."""
    if n <= 0:
        return False
    seen = 0
    for m in reversed(repaired_state_messages):
        if not hasattr(m, "tool_call_id"):
            continue
        seen += 1
        content = getattr(m, "content", "") or ""
        if isinstance(content, str) and _ACTION_TIER_CAP_SIGNATURE in content:
            return True
        if seen >= n:
            break
    return False


def _consecutive_same_tool_args_distinct(
    repaired_state_messages: list[Any],
    n: int,
) -> bool:
    """True when the last *n* ToolMessages came from same-tool calls with
    pairwise-distinct arguments — i.e. iteration, not polling (#1943 Fix #4).

    The polling-loop detector below trips on N consecutive ToolMessages
    that share a ``name`` (tool name) regardless of args.  But same-tool
    same-args is genuine polling, while same-tool DIFFERENT-args is
    legitimate iteration — sequential ``read_file`` of N distinct paths,
    diversified ``web_search`` queries, batched ``patch_file`` of
    different files.  This helper distinguishes the two so the
    detector's advisory + thinking-break arm fire only on the polling
    case.

    Walks the message list backwards collecting the latest *n*
    ToolMessages, matches each to its originating ``AIMessage.tool_calls``
    entry by ``tool_call_id``, and compares the resolved ``args`` dicts
    pairwise.  Returns ``False`` (i.e. NOT distinct → polling-loop
    detector should fire) when any of:

    * fewer than *n* ToolMessages found (insufficient evidence);
    * any args dict cannot be resolved (mock test object, AIMessage
      filtered out, etc.) — conservative default: fall through to
      legacy behaviour;
    * any pair of args dicts is structurally equal;
    * the recent responses contain an action-tier cap-hit signature —
      iteration that's thrashing the cap is NOT progress and the agent
      needs the polling-loop nudge to stop.

    Equality is by ``repr`` so unhashable values (lists, nested dicts)
    compare correctly; the args dicts are small so the cost is trivial.
    """
    if n <= 0:
        return False

    # Walk backwards collecting the latest n ToolMessages with their
    # tool_call_ids.
    tool_msgs: list[tuple[str, str | None]] = []  # [(tool_call_id, name)]
    for m in reversed(repaired_state_messages):
        if not hasattr(m, "tool_call_id"):
            continue
        tc_id = getattr(m, "tool_call_id", None) or ""
        tool_msgs.append((tc_id, getattr(m, "name", None)))
        if len(tool_msgs) >= n:
            break
    if len(tool_msgs) < n:
        return False

    # #1984 follow-up: if recent responses are action-tier cap-hits, the
    # iteration is thrashing — not progressing — and the polling-loop
    # signal must fire to push the agent toward a final response.
    if _recent_tool_responses_are_cap_hits(repaired_state_messages, n):
        return False

    # Build a lookup from tool_call_id -> args by walking AIMessages.
    args_by_id: dict[str, Any] = {}
    for m in repaired_state_messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_id = tc.get("id") or ""
                tc_args = tc.get("args")
            else:
                tc_id = getattr(tc, "id", None) or ""
                tc_args = getattr(tc, "args", None)
            if tc_id:
                args_by_id[tc_id] = tc_args

    # Resolve each ToolMessage's args.  Conservative default: any
    # unresolved entry → fall back to legacy polling-loop behaviour.
    resolved_args: list[str] = []
    for tc_id, _name in tool_msgs:
        if tc_id not in args_by_id:
            return False
        # ``repr`` so dicts with unhashable nested values still compare.
        resolved_args.append(repr(args_by_id[tc_id]))

    # Distinct iff the set has n unique entries.
    return len(set(resolved_args)) == n


# ─────────────────────────────────────────────────────────────────────
# P1 — post-bind message preparation
# ─────────────────────────────────────────────────────────────────────


def _phase_p1_post_bind_prep(
    context: Any,
    msgs: list[Any],
    log: Any,
    _cm: Any,
) -> list[Any]:
    """Transient-filter → context-cap → compress → topic-switch nudge
    → stuck-conclusion nudge.

    ``msgs`` is replaced rather than mutated for the filter / cap /
    compress steps (those return new lists). The topic-switch and
    stuck-conclusion nudges append to the new list.
    """
    # The caller already populated ``msgs`` with the repaired-state
    # messages prior to this phase. We re-apply the transient filter
    # here to make the function self-contained — it must produce the
    # same final ``msgs`` regardless of who built the input list.
    msgs = [
        m
        for m in msgs
        if not (
            hasattr(m, "response_metadata")
            and isinstance(m.response_metadata, dict)
            and m.response_metadata.get("transient")
        )
    ]

    # #1943 PR #1: compress BEFORE cap.  Eviction is destructive — it
    # drops messages without preserving their content.  Compression is
    # lossy-but-recoverable — it replaces verbose ToolMessage bodies
    # with one-paragraph summaries.  Running compression first means
    # the cap below has less to drop (often nothing).  Before this
    # change, the order was reversed: the cap evicted old ToolMessages
    # whose content would later be impossible to recover, then
    # compression ran on the survivors — which couldn't help with the
    # data that was already gone.  Reproducer: #1943 / verify-1919
    # context-overflow run.
    msgs = context.maybe_compress(msgs)
    if context.context_max_messages > 0 or context.context_max_tokens > 0:
        # #1943 PR #3: when the memory layer has a rolling summary, pass
        # it to the cap so the eviction marker can embed it (giving the
        # agent a semantic anchor for what was lost instead of just
        # ``data was lost`` + anti-fabrication nudge).  Tolerant of
        # missing manager (CLI direct path), missing attribute, or any
        # snapshot error — fall through to PR #1 prose unchanged.
        _evicted_summary: str | None = _maybe_get_rolling_summary(context.memory_manager, log)
        msgs = context.apply_context_message_cap(
            msgs,
            context.context_max_messages,
            context.context_max_tokens,
            evicted_summary=_evicted_summary,
        )

    if (
        context.topic_switch_detection_enabled
        and context.memory_manager is not None
        and _cm._should_reset_summary_for_topic_switch(msgs)
    ):
        _reset_summary_state = getattr(context.memory_manager, "reset_summary_state", None)
        if callable(_reset_summary_state):
            _reset_summary_state()
        else:
            _legacy_reset_summary = getattr(context.memory_manager, "_reset_summary_state", None)
            if callable(_legacy_reset_summary):
                _legacy_reset_summary()
        msgs.append(SystemMessage(content=_TOPIC_SWITCH_NUDGE))
        log.info("Topic switch detected — resetting summary state and nudging model")

    # ── Stuck-conclusion detector (Bug G #1713) ──────────────
    # If the last two assistant final responses (from prior turns)
    # are >= 90% similar, inject a HumanMessage nudge before the
    # LLM round so the model knows it's been repeating itself and
    # is steered toward either (a) honest "no new evidence" or
    # (b) a categorically different angle. Only fires at the
    # start of a fresh user turn (call_count == 1) so it doesn't
    # cross-talk with intra-turn rounds, and only when both
    # finals exist with non-empty content. Pairs with the
    # ``Forbidden`` system-prompt rule that bans the "You're
    # absolutely right" prefix when the answer is unchanged.
    if context.call_count[0] == 1:
        _prior_finals: list[str] = []
        for _m in reversed(msgs):
            if isinstance(_m, AIMessage):
                _c = getattr(_m, "content", "")
                if isinstance(_c, str) and _c.strip():
                    _prior_finals.append(_c)
                    if len(_prior_finals) == 2:
                        break
        if len(_prior_finals) == 2:
            # #2054: only fire for substantial answers — skip short
            # conversational acknowledgments that are near-identical by nature.
            # #2199: evaluate this cheap length guard BEFORE the O(n·m)
            # SequenceMatcher, so short prior responses (the common case on every
            # fresh user turn) short-circuit without paying the diff cost.
            _both_substantial = (
                min(len(_prior_finals[0]), len(_prior_finals[1])) > _STUCK_CONCLUSION_MIN_CHARS
            )
            if _both_substantial:
                from difflib import SequenceMatcher as _SeqMatch

                _sim = _SeqMatch(None, _prior_finals[0], _prior_finals[1]).ratio()
                if _sim >= 0.90:
                    msgs.append(
                        HumanMessage(
                            content=(
                                "[Stuck-conclusion check] Your prior two assistant "
                                "responses are near-identical "
                                f"(similarity {_sim:.2f}). Either acknowledge "
                                "honestly that you are unable to gather new "
                                "evidence (state what evidence would change your "
                                "answer), or pursue a categorically different "
                                "line of investigation — do NOT repeat the same "
                                "conclusion with a 'You're absolutely right' or "
                                "'I apologize' prefix. If you've considered the "
                                "user's input and your conclusion is unchanged, "
                                "say so plainly with the words "
                                "'my conclusion is unchanged' and explain what "
                                "would change it."
                            )
                        )
                    )
                    log.info(
                        "Stuck-conclusion nudge injected (similarity=%.2f) — "
                        "prior 2 final responses near-identical (Bug G #1713)",
                        _sim,
                    )

    return msgs


# ─────────────────────────────────────────────────────────────────────
# P0 — calibration + checkpoint accounting
# ─────────────────────────────────────────────────────────────────────


def _phase_p0_calibration_and_checkpoint(
    context: Any,
    msgs: list[Any],
    log: Any,
    _cm: Any,
) -> list[Any]:
    """Stuck-threshold calibration → checkpoint nudge → checkpoint
    summary → rounds-since-checkpoint accounting (which sets / clears
    ``force_thinking_break[0]`` for the current round)."""

    if not context.stuck_threshold_calibrated[0] and context.call_count[0] == 1:
        context.stuck_threshold_calibrated[0] = True
        from cogtrix_core.orchestration.intent import (
            TaskComplexity as _TC,
        )
        from cogtrix_core.orchestration.intent import (
            classify_task_complexity as _classify_tc,
        )

        _user_text = ""
        for _m in msgs:
            if hasattr(_m, "type") and _m.type == "human":
                _user_text = getattr(_m, "content", "")
                break
        _tc = _classify_tc(_user_text)
        if _tc == _TC.COMPLEX_ACTION:
            context.stuck_no_checkpoint_threshold[0] = 35
        elif _tc == _TC.COMPLEX_RESEARCH:
            context.stuck_no_checkpoint_threshold[0] = 20
        else:
            context.stuck_no_checkpoint_threshold[0] = 20
        log.debug(
            "Stuck threshold calibrated to %d (complexity=%s)",
            context.stuck_no_checkpoint_threshold[0],
            _tc.name,
        )

    if (
        context.calls_since_last_checkpoint[0] >= context.checkpoint_nudge_interval
        and context.call_count[0] > 3
    ):
        log.info(
            "Checkpoint nudge fired (calls_since=%d, round=%d)",
            context.calls_since_last_checkpoint[0],
            context.call_count[0],
        )
        msgs.append(
            SystemMessage(
                content=(
                    "[Checkpoint reminder] You've made several actions without "
                    "recording a checkpoint. Use the checkpoint tool now to record "
                    "what you've accomplished or learned since your last checkpoint."
                )
            )
        )
        context.calls_since_last_checkpoint[0] = 0

    if context.checkpoint_store is not None and len(context.checkpoint_store) > 0:
        _ckpt_summary = context.checkpoint_store.summary()
        if _ckpt_summary:
            msgs.append(HumanMessage(content=_ckpt_summary))

    if context.checkpoint_store is not None:
        current_ckpt_count = len(context.checkpoint_store)
        if current_ckpt_count > context.last_checkpoint_count[0]:
            context.last_checkpoint_count[0] = current_ckpt_count
            context.rounds_since_checkpoint[0] = 0
            context.calls_since_last_checkpoint[0] = 0
            # Bug #1717: when the agent records a new checkpoint, it has
            # demonstrably made progress. Clear any previously-armed
            # thinking break — e.g. one set by the "Temporal polling loop
            # detected" path (P2 below in the prior round). Without
            # this, the polling-loop arm fires even when the agent has
            # already heeded the advisory (synthesised + checkpointed),
            # truncating a substantive answer down to a degraded
            # re-summary on the next round. cogtrix57 reproducer: 579-
            # token answer + Checkpoint #1 recorded → "Stuck detected
            # — forcing thinking break" fires immediately → 131-token
            # re-summary → user asks "why have you posted this short
            # summary?".
            context.force_thinking_break[0] = False
        else:
            context.rounds_since_checkpoint[0] += 1
            _threshold = context.stuck_no_checkpoint_threshold[0]
            if (
                context.rounds_since_checkpoint[0] >= _threshold
                and context.call_count[0] > _threshold
            ):
                context.force_thinking_break[0] = True
                context.rounds_since_checkpoint[0] = 0
                log.info(
                    "No new checkpoints in %d rounds — forcing thinking break",
                    _threshold,
                )

    return msgs


# ─────────────────────────────────────────────────────────────────────
# P2 — late directives
# ─────────────────────────────────────────────────────────────────────


def _phase_p2_late_directives(
    context: Any,
    repaired_state_messages: list[Any],
    msgs: list[Any],
    log: Any,
) -> list[Any]:
    """Tool-state verification → reflection → polling-loop advisory →
    tool-output quality gate.

    The polling-loop branch may arm ``context.force_thinking_break[0]``
    for the **next** call_model round. The current round's
    thinking-break consumer has already run by the time this executes;
    a fresh arm here applies one round later (matched by the bug
    #1717 clear in :func:`_phase_p0_calibration_and_checkpoint` when
    the agent makes progress in the meantime).
    """

    if (
        context.tool_health_check_interval > 0
        and context.call_count[0] > 1
        and context.call_count[0] % context.tool_health_check_interval == 0
        and context.call_count[0] != context.last_tool_health_check_at[0]
    ):
        context.last_tool_health_check_at[0] = context.call_count[0]
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
        log.info(
            "Tool-state verification injected at turn %d (interval=%d)",
            context.call_count[0],
            context.tool_health_check_interval,
        )

    if (
        context.call_count[0] > 1
        and context.call_count[0] % context.reflection_interval == 0
        and context.call_count[0] != context.last_reflection_at[0]
    ):
        context.last_reflection_at[0] = context.call_count[0]
        if context.consecutive_errors[0] >= 2:
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
    # #1943 Fix #4: when the last N same-tool calls each used DISTINCT
    # arguments, the agent is iterating (sequential reads of N files,
    # diversified web_search queries, etc.), not polling.  Suppress the
    # polling-loop signal to avoid breaking legitimate iteration.  Same-
    # tool same-args genuinely is a poll and still trips the detector.
    _last_n_args_distinct = _consecutive_same_tool_args_distinct(
        repaired_state_messages, _MAX_CONSECUTIVE_SAME_TOOL
    )
    if (
        len(_recent_tool_names) >= _MAX_CONSECUTIVE_SAME_TOOL
        and len(set(_recent_tool_names[-_MAX_CONSECUTIVE_SAME_TOOL:])) == 1
        and _last_n_args_distinct
    ):
        # Iteration, not polling — suppress.  INFO so operators can see
        # the discriminator fired in case a legitimate-looking loop
        # turns out to be problematic.
        log.info(
            "Polling-loop signal suppressed — '%s' called %d times with distinct args "
            "(iteration, not polling).  No advisory injected.",
            _recent_tool_names[-1],
            _MAX_CONSECUTIVE_SAME_TOOL,
        )
    elif (
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
        #
        # However: if every consecutive call returned a "not loaded" stub,
        # the agent has not exhausted the tool — it is still discovering
        # that the tool is not active. Arming the thinking break here
        # punishes the correct recovery move (request_tools) and forces
        # the model into fabrication mode. Skip arming when all recent
        # tool calls were "not loaded" stubs (#1510).
        _consecutive_tool_msgs = [
            m
            for m in repaired_state_messages[-(max(len(repaired_state_messages), 0)) :]
            if hasattr(m, "tool_call_id")
        ][-_MAX_CONSECUTIVE_SAME_TOOL:]
        _all_stub_results = all(
            getattr(m, "content", "") and ("not loaded" in getattr(m, "content", "").lower())
            for m in _consecutive_tool_msgs
        )
        if not _all_stub_results:
            context.force_thinking_break[0] = True
            log.warning(
                "Temporal polling loop detected — '%s' called %d+ consecutive times; "
                "injecting advisory + arming thinking break for next round",
                _stuck_tool,
                _MAX_CONSECUTIVE_SAME_TOOL,
            )
        else:
            log.info(
                "Temporal polling loop detected — '%s' called %d+ times but all "
                "returned 'not loaded' stubs; advisory injected but thinking-break "
                "arm suppressed (agent may be recovering via request_tools)",
                _stuck_tool,
                _MAX_CONSECUTIVE_SAME_TOOL,
            )

    if context.tool_quality_gate_enabled and context.all_tool_results_substanceless(
        repaired_state_messages
    ):
        msgs.append(
            SystemMessage(
                content=(
                    "All tools returned no data this turn. Do not synthesise an answer "
                    "from prior context or memory. Report honestly that the tools "
                    "returned nothing and ask the user how to proceed."
                )
            )
        )
        log.info("Tool output quality gate injected — all tools returned empty")

    return msgs
