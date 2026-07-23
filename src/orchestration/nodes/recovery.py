"""Recovery node factories for phantom and action-intent retries."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage as _LangChainHumanMessage
from langchain_core.messages.modifier import RemoveMessage

from src.agent.core import CogtrixState
from src.logging_config import get_logger

# Marker key carried on every recovery-injected ``HumanMessage`` so detectors
# that walk to "the most recent user turn" can distinguish a real user turn
# from a recovery-cascade nudge.
#
# Bug context: ``_find_current_turn_start`` returns the index of the most
# recent ``HumanMessage`` in the conversation.  Recovery nodes (action-intent,
# fabricated-action, topic-substitution, …) inject a ``HumanMessage`` carrying
# a corrective nudge so the next ``call_model`` pass sees it as the active
# user instruction — that is the LangGraph-idiomatic way to steer the agent.
#
# The trouble: detectors that key on ``sources.user_prompt`` (in particular
# ``detect_topic_substitution`` after #1992) then read the *nudge text* as
# the user prompt and extract distinctive subjects from it.  The
# ``fabricated_action`` nudge text contains the literal substrings
# ``ToolMessage`` (CamelCase compound) and ``NOT`` (3-char acronym), so the
# topic-substitution detector flags them as missing subjects, deletes the
# assistant response, and injects another nudge — an infinite cascade that
# blows context and times out the turn.  Observed in CI on
# ``regression_persist_before_refusing × kimi-k2-5`` (PR #1996 run).
#
# Fix: tag every recovery-injected ``HumanMessage`` here and skip them in
# ``_find_current_turn_start``.  The marker lives in ``additional_kwargs``
# (LangChain's stable metadata channel) so it survives serialisation and
# does not leak into the LLM-visible content.
_RECOVERY_INJECTED_MARKER = "recovery_injected"


def HumanMessage(
    content: Any,
    **kwargs: Any,
) -> _LangChainHumanMessage:
    """Module-local ``HumanMessage`` factory that tags recovery nudges.

    Every recovery node in this module already imports ``HumanMessage`` from
    this namespace and constructs nudges as ``HumanMessage(content=...)``.
    Shadowing the import with this factory tags each one automatically — no
    per-callsite refactor is required and future recovery-node additions
    inherit the marker by default.

    Callers may pass ``additional_kwargs`` to supply other metadata;
    the recovery marker is added to whatever they passed.
    """
    additional_kwargs = dict(kwargs.pop("additional_kwargs", None) or {})
    additional_kwargs[_RECOVERY_INJECTED_MARKER] = True
    return _LangChainHumanMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        **kwargs,
    )


def _detect_unloaded_tool_in_last_turn(state: CogtrixState) -> str | None:
    """Find a tool that was called but reported as 'available but not loaded'.

    Scans the last few messages for a ToolMessage whose content matches the
    'tool not loaded' error pattern.  Returns the tool name so the nudge can
    tell the model exactly how to fix it.
    """
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return None
    msgs = state.get("messages") or []
    # Walk backwards through recent messages — only check the last 6 to
    # avoid surfacing stale errors from earlier in the run.
    for msg in reversed(msgs[-6:]):
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        if (
            "is in the catalog but not loaded" in content
            or "is available but not loaded" in content
        ):
            return getattr(msg, "name", None) or None
    return None


def _build_synthesized_giveup_message(state: CogtrixState) -> AIMessage:
    """Replace the last (likely meta-analysis) AI message with a clean answer.

    When recovery gives up, we don't want to leave the model's stuck-thinking
    output as the user-facing response.  Synthesize from checkpoints / tool
    results instead.  Falls back to a polite 'try rephrasing' message if no
    usable state was accumulated.
    """
    from src.orchestration.phases import (
        strip_foreign_tool_call_xml,
        synthesize_answer_from_state,
    )

    msgs = list(state.get("messages") or [])
    synthesized = synthesize_answer_from_state(msgs)
    if synthesized:
        # Defence in depth: even if synthesis pulled from a content path,
        # ensure no foreign XML leaks through.
        synthesized = strip_foreign_tool_call_xml(synthesized) or synthesized
        return AIMessage(content=str(synthesized))
    return AIMessage(
        content=(
            "I wasn't able to complete this request cleanly. "
            "Could you rephrase the question, or break it into smaller steps?"
        )
    )


def build_handle_phantom_node(
    phantom_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the phantom-recovery node bound to the run-local retry counter."""

    def handle_phantom(state: CogtrixState) -> dict:
        phantom_count[0] += 1
        msgs = state["messages"]
        last = msgs[-1]
        log = logger()
        log.warning(
            "Phantom tool call detected, attempt %d/%d. Injecting hint.",
            phantom_count[0],
            max_retries,
        )
        if phantom_count[0] > max_retries:
            log.info("Phantom retries exhausted — synthesizing final answer from accumulated state")
            synthesized = _build_synthesized_giveup_message(state)
            return {
                "messages": [
                    RemoveMessage(id=last.id),
                    synthesized,
                ]
            }
        return {
            "messages": [
                RemoveMessage(id=last.id),
                HumanMessage(
                    content=(
                        "Your last tool call could not be parsed by the server. "
                        "Please retry with the normal structured tool-call format, "
                        "or provide your answer directly if you do not need tools."
                    )
                ),
            ]
        }

    return handle_phantom


def build_handle_action_intent_node(
    action_intent_count: list[int],
    max_retries: int,
    incompleteness_check: Callable[[str], bool] | None = None,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the action-intent recovery node bound to the run-local retry counter.

    Args:
        action_intent_count: Mutable counter of how many times the node fired.
        max_retries: Maximum number of standard nudges before synthesising.
        incompleteness_check: Optional callable(str) -> bool that detects
            incomplete multi-step language ('first', 'to start').  When provided
            and the check passes, a more specific nudge is injected.
        logger: Logger factory.
    """

    def handle_action_intent(state: CogtrixState) -> dict:
        action_intent_count[0] += 1
        log = logger()
        log.warning(
            "Action-intent without tool call detected, attempt %d/%d. Injecting nudge.",
            action_intent_count[0],
            max_retries,
        )
        if action_intent_count[0] > max_retries:
            # Don't leave the model's stuck-thinking output as the final
            # response — synthesize a clean answer from what was accumulated.
            log.info(
                "Action-intent retries exhausted — synthesizing final answer from "
                "accumulated state instead of letting the meta-analysis stand"
            )
            msgs = state.get("messages") or []
            last = msgs[-1] if msgs else None
            synthesized = _build_synthesized_giveup_message(state)
            return {
                "messages": ([RemoveMessage(id=last.id)] if last is not None else [])
                + [synthesized]
            }

        # Context-aware nudge: if the prior turn called a tool that was
        # reported as 'available but not loaded', tell the model exactly
        # how to load it instead of issuing the generic nudge.
        unloaded = _detect_unloaded_tool_in_last_turn(state)
        if unloaded:
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            f"You tried to use '{unloaded}' but it was not loaded. "
                            f"Issue ONE structured tool call now: "
                            f'request_tools(add=["{unloaded}"])'
                            "  — then call the tool again on the next turn. "
                            "Do not describe the action; emit the tool_call directly."
                        )
                    )
                ]
            }

        # Incompleteness-specific nudge: when the model used sequential
        # language ('first', 'to start') and stopped, tell it the task
        # is not finished — more specific than the generic variant.
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        content = getattr(last, "content", "") if last is not None else ""
        if (
            incompleteness_check is not None
            and isinstance(content, str)
            and incompleteness_check(content)
        ):
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "You used language like 'first' or 'to start', "
                            "which implies there are more steps to complete. "
                            "The task is not finished — do not describe what "
                            "comes next. Call the appropriate tool(s) NOW."
                        )
                    )
                ]
            }

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

    return handle_action_intent


def build_handle_unverified_claim_node(
    unverified_claim_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the unverified-claim recovery node bound to a run-local counter.

    Fires when the model produced a final response containing a
    categorical claim about external state (today's date, latest
    version, file content, etc.) without first calling the matching
    verification tool. The recovery node deletes the unverified
    response and injects a nudge that explains *which* claim was
    caught and *which* tool to call to verify.

    After ``max_retries`` attempts the agent's answer ships as-is —
    we'd rather show a potentially-stale answer than spin forever.

    Args:
        unverified_claim_count: Mutable counter of how many times
            the node fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 2 (one revision attempt — anything more
            and the model is probably refusing to use the tool).
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        VerificationRule,
        collect_tool_names_this_turn,
        detect_unverified_claim,
    )

    def handle_unverified_claim(state: CogtrixState) -> dict:
        unverified_claim_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            # Defensive: if the content shape is unexpected, just let
            # the response through rather than risk a loop.
            return {"messages": []}

        # Recompute which rule matched (the routing function will have
        # done this already, but a second call is cheap and gives us
        # the rule object for the nudge text).
        turn_start = _find_current_turn_start(msgs)
        tools_called = collect_tool_names_this_turn(msgs, turn_start)
        rule: VerificationRule | None = detect_unverified_claim(last_content, tools_called)

        if rule is None:
            # Re-detection failed — possibly the response was already
            # revised by a concurrent path. Skip the nudge.
            return {"messages": []}

        log.warning(
            "Unverified-claim detected (rule=%s, attempt %d/%d). Injecting nudge.",
            rule.name,
            unverified_claim_count[0],
            max_retries,
        )

        if unverified_claim_count[0] > max_retries:
            log.info(
                "Unverified-claim retries exhausted (rule=%s) — accepting the "
                "agent's response as-is. The user may receive an unverified answer.",
                rule.name,
            )
            return {"messages": []}

        # Remove the unverified response so it doesn't accumulate in
        # context, and inject the nudge for the next call_model pass.
        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {"messages": [*removal, HumanMessage(content=rule.nudge_template)]}

    return handle_unverified_claim


def build_handle_unverified_entity_node(
    unverified_entity_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the unverified-entity recovery node bound to a run-local counter.

    cogtrix47 (Issues 5+6): when the model produced a final response
    that repeats high-specificity identifiers from the user's prompt
    (SKUs, store names, multi-word product names) WITHOUT any tool
    result confirming the entity exists, the response is operating
    on an unverified premise. This node deletes the response and
    injects a nudge listing the unverified entities + three explicit
    revision options (cite evidence, hedge, substitute the verified
    alternative).

    Args:
        unverified_entity_count: Mutable counter of how many times the
            node fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision attempt is enough; if
            the model still repeats unverified entities after the
            nudge it's actively ignoring the guard, not just bad
            luck.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        collect_grounded_sources,
        detect_unverified_entities,
        format_unverified_entity_nudge,
    )

    def handle_unverified_entity(state: CogtrixState) -> dict:
        unverified_entity_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        # #1964 Item C: detectors now consume a ``GroundedSources``
        # bundle (tool results + user prompt + system prompt).
        sources = collect_grounded_sources(msgs, turn_start)

        entities = detect_unverified_entities(last_content, sources=sources)

        if not entities:
            # Re-detection failed — possibly the response was already
            # revised by a concurrent path. Skip the nudge.
            return {"messages": []}

        log.warning(
            "Unverified-entity detected (entities=%s, attempt %d/%d). Injecting nudge.",
            entities,
            unverified_entity_count[0],
            max_retries,
        )

        if unverified_entity_count[0] > max_retries:
            log.info(
                "Unverified-entity retries exhausted (entities=%s) — accepting the "
                "agent's response as-is rather than spinning further.",
                entities,
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_unverified_entity_nudge(entities)),
            ]
        }

    return handle_unverified_entity


def build_handle_unsupported_quote_node(
    unsupported_quote_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the unsupported-quote recovery node bound to a run-local counter.

    #1841 (output-fidelity guard): when the model's final response contains
    a verbatim quote or explicit attribution that appears in NO tool result
    this turn — a fabricated quote / fabricated citation — this node deletes
    the response and injects a nudge listing the offending quote(s) with
    three revision options (quote verbatim from a real result, drop the
    attribution, or say the tools didn't establish it).

    Args:
        unsupported_quote_count: Mutable counter of how many times the node
            fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision attempt; a model still
            fabricating quotes after the nudge is actively ignoring the
            guard, not unlucky.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        collect_grounded_sources,
        detect_unsupported_quote,
        format_unsupported_quote_nudge,
    )

    def handle_unsupported_quote(state: CogtrixState) -> dict:
        unsupported_quote_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        # #1964 Item C: bundled grounding sources (tool results + user
        # prompt + system prompt).  System-prompt inclusion is the
        # structural fix for the false-fire class documented in #1962.
        sources = collect_grounded_sources(msgs, turn_start)

        quotes = detect_unsupported_quote(last_content, sources=sources)

        if not quotes:
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Unsupported-quote detected (quotes=%s, attempt %d/%d). Injecting nudge.",
            quotes,
            unsupported_quote_count[0],
            max_retries,
        )

        if unsupported_quote_count[0] > max_retries:
            log.info(
                "Unsupported-quote retries exhausted (quotes=%s) — accepting the "
                "agent's response as-is rather than spinning further.",
                quotes,
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_unsupported_quote_nudge(quotes)),
            ]
        }

    return handle_unsupported_quote


def build_handle_version_scope_node(
    version_scope_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the version-scope recovery node bound to a run-local counter.

    #1843 (version-scope-collapse guard): when the model attaches a
    lifecycle status (discontinued / deprecated / …) to a specific
    model-ID while the fetched evidence scopes that status only to a
    prefix-*parent* of that ID, this node deletes the response and injects
    a nudge naming the offending ``child → parent`` pairs.

    Unlike the other verification nodes, the evidence corpus is the WHOLE
    conversation's tool output (``turn_start=0``), not just the current
    turn. The next67 incident's worst fabrication surfaced on a *correction*
    turn where the model re-asserted a (wrong) status WITHOUT re-calling any
    tool — so the only ground truth is the research done on an earlier turn.
    Scoping to the current turn would leave that corpus empty and the guard
    blind. See ``src/orchestration/verification.py`` for the detector.

    Args:
        version_scope_count: Mutable counter of how many times the node
            fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision attempt; a model still
            collapsing the scope after the nudge is ignoring the guard.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        collect_tool_message_contents,
        detect_version_scope_mismatch,
        format_version_scope_nudge,
    )

    def handle_version_scope(state: CogtrixState) -> dict:
        version_scope_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        # Conversation-wide tool content — the misattribution may surface on
        # a correction turn that did no fresh research (see docstring).
        tool_contents = collect_tool_message_contents(msgs, 0)
        mismatches = detect_version_scope_mismatch(last_content, tool_contents)

        if not mismatches:
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Version-scope mismatch detected (pairs=%s, attempt %d/%d). Injecting nudge.",
            [(m.claimed_id, m.scoped_to_id) for m in mismatches],
            version_scope_count[0],
            max_retries,
        )

        if version_scope_count[0] > max_retries:
            log.info(
                "Version-scope retries exhausted (pairs=%s) — accepting the agent's "
                "response as-is rather than spinning further.",
                [(m.claimed_id, m.scoped_to_id) for m in mismatches],
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_version_scope_nudge(mismatches)),
            ]
        }

    return handle_version_scope


def build_handle_unsupported_attribution_node(
    unsupported_attribution_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the unsupported-attribution recovery node bound to a run-local
    counter.

    #1860 (attributed-prose-claim guard): when the model's response credits
    a source/authority ("as confirmed by", "according to", "officially …")
    for a paragraph whose distinctive content is not in any tool result
    this turn, the model is fabricating the source itself rather than
    paraphrasing. This node deletes the offending response and injects a
    nudge listing the unsupported attribution snippets, with three revision
    options (re-check + paraphrase faithfully, drop the attribution, or
    state plainly that the tools didn't establish it).

    Args:
        unsupported_attribution_count: Mutable counter of how many times
            the node fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision attempt; a model still
            fabricating attributed claims after the nudge is actively
            ignoring the guard, not unlucky.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        collect_grounded_sources,
        detect_unsupported_attribution,
        format_unsupported_attribution_nudge,
    )

    def handle_unsupported_attribution(state: CogtrixState) -> dict:
        unsupported_attribution_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        # #1964 Item C: bundled grounding sources (tool results + user
        # prompt + system prompt).
        sources = collect_grounded_sources(msgs, turn_start)

        snippets = detect_unsupported_attribution(last_content, sources=sources)

        if not snippets:
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Unsupported-attribution detected (snippets=%s, attempt %d/%d). Injecting nudge.",
            snippets,
            unsupported_attribution_count[0],
            max_retries,
        )

        if unsupported_attribution_count[0] > max_retries:
            log.info(
                "Unsupported-attribution retries exhausted (snippets=%s) — accepting "
                "the agent's response as-is rather than spinning further.",
                snippets,
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_unsupported_attribution_nudge(snippets)),
            ]
        }

    return handle_unsupported_attribution


def build_handle_entity_owner_mismatch_node(
    entity_owner_mismatch_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the entity-owner-mismatch recovery node bound to a run-local counter.

    #1988 (post-mortem #1987 Cluster A): when the model's response
    co-mentions a structured entity-ID (R-XX, DEC-…, CHG-…, NIMB-WBS-…)
    with a stakeholder name within a 240-char window, AND that
    (entity, name) pair does NOT co-appear in any tool result or in
    the system prompt this turn, the model is stitching a plausible-
    sounding owner onto the wrong entity.

    This node deletes the offending response and injects a nudge
    listing the unsupported pairs with three revision options
    (re-query for the entity's owner field, drop the owner attribution,
    or defer the question).

    Args:
        entity_owner_mismatch_count: Mutable counter of how many times
            the node fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision; a model still
            mis-attributing after the nudge is actively ignoring the
            guard.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        collect_grounded_sources,
        detect_entity_owner_mismatch,
        format_entity_owner_mismatch_nudge,
    )

    def handle_entity_owner_mismatch(state: CogtrixState) -> dict:
        entity_owner_mismatch_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        # #1964 Item C: bundled grounding sources (tool results + user
        # prompt + system prompt).
        sources = collect_grounded_sources(msgs, turn_start)

        mismatches = detect_entity_owner_mismatch(last_content, sources=sources)

        if not mismatches:
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Entity-owner mismatch detected (pairs=%s, attempt %d/%d). Injecting nudge.",
            mismatches,
            entity_owner_mismatch_count[0],
            max_retries,
        )

        if entity_owner_mismatch_count[0] > max_retries:
            log.info(
                "Entity-owner mismatch retries exhausted (pairs=%s) — accepting "
                "the agent's response as-is rather than spinning further.",
                mismatches,
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_entity_owner_mismatch_nudge(mismatches)),
            ]
        }

    return handle_entity_owner_mismatch


def build_handle_corpus_attribution_mismatch_node(
    corpus_attribution_mismatch_count: list[int],
    max_retries: int,
    corpus_attribution_detector: Callable[[str], list[str]],
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the corpus-aware attribution-mismatch recovery node.

    #2015 (post-mortem #2006 Cluster A root-cause): the existing
    ``detect_entity_owner_mismatch`` guard uses a *co-occurrence*
    grounding check — it asks "does (entity_id, name) appear within
    ~200 chars in any tool result this turn?".  That heuristic fails
    on dense corpora (e.g. the PM corpus stakeholder register lists 13
    people; the risk register has 20+ risks per chunk).  When the
    wrongly-attached stakeholder name DOES happen to co-occur in some
    retrieved chunk (because they own a DIFFERENT entity nearby in the
    text), the grounding check passes and the mismatch ships.  PM
    cycle 4 measured 9 detector fires but **15** final-response
    mismatches — the gap is the cases the loose detector missed.

    This node consumes a caller-supplied ``corpus_attribution_detector``
    callable that returns a list of human-readable mismatch
    descriptions (e.g. ``"R-13 attributed to 'Hyeon-Jin Park' but corpus
    owners are {Tomislav Hessford}"``).  The orchestration layer
    stays corpus-agnostic: the detector closure can be built over any
    deployment's curated owner index without ``src/orchestration/``
    knowing what's in it.

    When mismatches are found, the existing assistant response is
    deleted and a structurally specific nudge is injected naming each
    misattributed entity, the claimed owner, and the corpus-canonical
    owner — so the model has the right answer in front of it on the
    next call_model pass, not just a "fix it" instruction.

    Args:
        corpus_attribution_mismatch_count: Mutable per-run counter
            (lives on :class:`PerRunState`).
        max_retries: Maximum revision attempts before accepting the
            response.  Default 2 (see #2007 / PR #2012 rationale for
            the prose-fidelity detector family).
        corpus_attribution_detector: Caller-supplied callable that
            takes the assistant response text and returns a list of
            mismatch descriptions; empty list means clean.
        logger: Logger factory.
    """

    def handle_corpus_attribution_mismatch(state: CogtrixState) -> dict:
        corpus_attribution_mismatch_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None

        # No last message — nothing to scan, nothing to remove, no
        # nudge to inject.  Short-circuit before the detector is even
        # called so a misfiring detector can't manufacture a response
        # out of an empty string.
        if last is None:
            return {"messages": []}

        last_content = getattr(last, "content", "")

        if not isinstance(last_content, str):
            return {"messages": []}

        try:
            mismatches = corpus_attribution_detector(last_content)
        except Exception as exc:  # noqa: BLE001 — detector must not crash the run
            log.warning(
                "corpus_attribution_detector raised %s: %s — skipping nudge",
                type(exc).__name__,
                exc,
            )
            return {"messages": []}

        if not mismatches:
            return {"messages": []}

        log.warning(
            "Corpus attribution mismatch detected (mismatches=%s, attempt %d/%d). "
            "Injecting nudge.",
            mismatches,
            corpus_attribution_mismatch_count[0],
            max_retries,
        )

        if corpus_attribution_mismatch_count[0] > max_retries:
            log.info(
                "Corpus attribution-mismatch retries exhausted (mismatches=%s) — "
                "accepting the agent's response as-is rather than spinning further.",
                mismatches,
            )
            return {"messages": []}

        nudge = _format_corpus_attribution_mismatch_nudge(mismatches)

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=nudge),
            ]
        }

    return handle_corpus_attribution_mismatch


def _format_corpus_attribution_mismatch_nudge(mismatches: list[str]) -> str:
    """Render the corpus-aware mismatch nudge.

    Each item in *mismatches* is a structurally specific string like
    ``"R-13 attributed to 'Hyeon-Jin Park' but corpus owners are
    {Tomislav Hessford}"`` — already names the right answer.  The
    nudge wraps these lines with a clear directive on the two
    acceptable responses (correct each attribution OR omit the owner
    field for the entities in question).
    """
    bullets = "\n".join(f"  - {m}" for m in mismatches)
    # English plural is "mismatches" (-es), not "mismatchs" — naive
    # ``"s" if N > 1 else ""`` produced the wrong suffix in the first
    # cut and the test guards it.  Use the full word swap instead.
    noun = "mismatches" if len(mismatches) > 1 else "mismatch"
    # Cycle-10 post-mortem (#2006): the previous nudge wording was
    # too polite — it said "Revise the response to do ONE of …" and
    # the model often retried with the SAME wrong attribution.  This
    # rewrite leads with STOP / MUST and uses imperative verbs
    # (REPLACE / DROP) to push the model out of polite compliance
    # into actual substitution.  Intentionally abstract — no
    # corpus-specific entity-ids or stakeholder names in the
    # production nudge text (bias-leakage rule).
    return (
        f"STOP. Your response contains {len(mismatches)} entity-owner "
        f"attribution {noun} versus the authoritative corpus:\n\n"
        f"{bullets}\n\n"
        f"Each line above tells you exactly which name is wrong AND "
        f"which name(s) the corpus says are correct (in the "
        f"curly-brace set).  You MUST do ONE of the following for "
        f"each listed mismatch in your next response:\n\n"
        f"  (a) REPLACE the wrong name with the corpus owner shown "
        f"above, copied VERBATIM from the curly-brace set.  Do not "
        f"paraphrase; do not add or drop honorifics.\n"
        f"  (b) DROP the owner field for that entity entirely — "
        f"better to ship no owner than the wrong owner.\n\n"
        f"This is the role-association plausibility-substitution "
        f"failure mode catalogued in #2015 / #2006: the model "
        f"attaches a stakeholder to an entity because the "
        f"stakeholder's listed role topically matches the entity's "
        f"subject, even though the corpus assigns ownership to "
        f"someone else.  Stakeholder roles are CORPUS facts about "
        f"people, NOT inferences from entity topics.  Do NOT defend "
        f"the original attribution.  Do NOT substitute a DIFFERENT "
        f"'plausible-sounding' name — the corpus index caught the "
        f"pattern and will catch the same pattern again."
    )


def build_handle_topic_substitution_node(
    topic_substitution_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the topic-substitution recovery node bound to a run-local counter.

    #1989 (post-mortem #1987 Cluster C): when the user asks about a
    subject that does NOT appear in the corpus (e.g. *"CompactSync
    codebase tech debt"*), but the agent silently retitles its
    response to a related in-corpus subject (*"Project Nimbus
    Technical Debt Risks"*) and answers THAT, this node deletes the
    off-topic response and injects a nudge instructing the agent to
    either re-query specifically for the user's subject or defer
    cleanly.
    """
    from src.orchestration.verification import (
        collect_grounded_sources,
        detect_topic_substitution,
        format_topic_substitution_nudge,
    )

    def handle_topic_substitution(state: CogtrixState) -> dict:
        topic_substitution_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        sources = collect_grounded_sources(msgs, turn_start)

        missing = detect_topic_substitution(last_content, sources=sources)

        if not missing:
            return {"messages": []}

        log.warning(
            "Topic substitution detected (missing subjects=%s, attempt %d/%d). Injecting nudge.",
            missing,
            topic_substitution_count[0],
            max_retries,
        )

        if topic_substitution_count[0] > max_retries:
            log.info(
                "Topic-substitution retries exhausted (missing=%s) — accepting "
                "the agent's response as-is rather than spinning further.",
                missing,
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_topic_substitution_nudge(missing)),
            ]
        }

    return handle_topic_substitution


def build_handle_sycophancy_node(
    sycophancy_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the sycophancy recovery node bound to a run-local counter.

    Bug G (#1713): RLHF-tuned models bypass the system-prompt rule and
    open a final answer with *"You're absolutely right"* / *"I apologize"*
    even when their conclusion is unchanged — the social failure that
    magnifies the operational failure of repeated content. The earlier
    in-place-strip attempt (PR #1731) broke Gate 2 because it mutated
    the existing AIMessage body, leaving a half-formed remainder that
    downstream rounds referenced. This node uses the standard recovery
    pattern instead — fully remove the offending response and inject a
    nudge so the model regenerates from scratch on the next call_model
    round. None of the sibling recovery nodes have regressed Gate 2,
    because they replace the response rather than edit it in place.

    Args:
        sycophancy_count: Mutable counter of how many times the node
            fired this run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision attempt; if the model
            still opens with the validation prefix after the nudge, the
            response ships as-is (the existing logging-only path in
            ``call_model.py`` retains visibility).
        logger: Logger factory.
    """
    from src.orchestration.response_detectors import _is_sycophantic_prefix

    def handle_sycophancy(state: CogtrixState) -> dict:
        sycophancy_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None

        if last is None or not _is_sycophantic_prefix(last):
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Sycophantic-prefix detected (attempt %d/%d). Injecting nudge.",
            sycophancy_count[0],
            max_retries,
        )

        if sycophancy_count[0] > max_retries:
            log.info(
                "Sycophancy retries exhausted — accepting the agent's response "
                "as-is rather than spinning further. The logging-only path in "
                "call_model retains visibility."
            )
            return {"messages": []}

        removal: list[Any] = []
        if getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(
                    content=(
                        "Your response started with a sycophantic validation "
                        'prefix ("You\'re right" / "You\'re absolutely right" / '
                        '"I apologize" / similar). The system prompt forbids '
                        "this — validating the user and then giving the same "
                        "answer is dishonest, and validating before substantive "
                        "revision is unnecessary noise.\n\n"
                        "Re-emit your answer WITHOUT any validation or apology "
                        "prefix. Choose one:\n"
                        "  (a) If your conclusion is unchanged after considering "
                        'my input, say so plainly: "My conclusion is unchanged: '
                        '<conclusion>" — and state what evidence would change '
                        "it.\n"
                        "  (b) If you have new analysis, state it directly with "
                        "no preamble — open with the substantive content, not "
                        "with a validation phrase.\n\n"
                        'Do NOT begin with "You\'re right", "You\'re absolutely '
                        'right", "You\'re raising an important point", "I '
                        'apologize", "My apologies", or any equivalent '
                        "validation/apology opener."
                    )
                ),
            ]
        }

    return handle_sycophancy


def build_handle_fabricated_action_node(
    fabricated_action_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the fabricated-action-success recovery node.

    Bug #1869: when the user asks the agent to perform a destructive
    file operation and the corresponding tool is not in the active
    set (per #1870's upstream loadout fix, this should be rare —
    but the gap still exists for pure-delete intents, which have no
    matching Cogtrix tool), the model has been observed to silently
    fabricate a successful outcome — Q9 / Q10 of the holistic-test
    battery against ``cogtrix:release-next`` @ ``2bb52c7``:

        Q9:  "The file ...verification.py has been deleted from the
              codebase as requested." (no tool call, file intact)
        Q10: "The file ...text.py already contains the safe_divide
              function based on the successful write operations in
              this session." (no tool call, no prior writes)

    Sibling to the existing ``handle_fabrication`` node (which catches
    the success-after-tool-errors variant). This node fires when there
    were **zero** ToolMessages in the current user turn — the model
    went prose-only and confabulated the outcome.

    The nudge explicitly names both honest paths so the model can
    recognise its situation: either invoke the appropriate tool now,
    or report plainly that it cannot perform the action with its
    current tool inventory. The latter case is the right answer for
    pure-delete intents (Cogtrix has no ``delete_file`` tool).

    Mirrors the post-#1731 recovery pattern: remove the offending
    response wholesale (``RemoveMessage``) and inject the nudge —
    never mutate the existing AIMessage in place.

    Args:
        fabricated_action_count: Mutable per-run counter (lives on
            :class:`PerRunState`) of how many times this node has fired
            in the current run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1 — one revision; if the model still
            fabricates after the nudge, the response ships as-is
            with a logged warning.
        logger: Logger factory (overridable for testing).
    """
    from src.orchestration.response_detectors import (
        _looks_like_fabricated_action_success_without_tool_call,
    )

    def handle_fabricated_action(state: CogtrixState) -> dict:
        fabricated_action_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None

        if last is None or not _looks_like_fabricated_action_success_without_tool_call(msgs, last):
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Fabricated action-success without tool call (attempt %d/%d). " "Injecting correction.",
            fabricated_action_count[0],
            max_retries,
        )

        if fabricated_action_count[0] > max_retries:
            log.info(
                "Fabricated-action retries exhausted — accepting the agent's "
                "response as-is rather than spinning further."
            )
            return {"messages": []}

        removal: list[Any] = []
        if getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(
                    content=(
                        "Your response claims a file or system change was "
                        "completed (e.g. file deleted / written / created / "
                        "modified), but you did not invoke any tool in this "
                        "turn — no tool call was issued and no ToolMessage "
                        "appears in the conversation. You cannot have performed "
                        "the action.\n\n"
                        "Choose exactly one path and answer accordingly — do "
                        "NOT repeat the false-completion claim:\n"
                        "  (a) If a tool exists that can perform the requested "
                        "action, invoke it now via a structured tool call "
                        '(call `request_tools(add=["<tool_name>"])` first '
                        "if the tool is in the catalog but not yet loaded — "
                        "the catalog includes `write_file`, `patch_file`, "
                        "`append_file`, `read_file`, `execute_shell_command`, "
                        "and others).\n"
                        "  (b) If no available tool can perform the action "
                        "(for example, the user asked you to delete a file "
                        "and no delete tool is loaded), tell the user plainly "
                        "that you have NOT performed the action and explain "
                        "what tool would be needed.\n\n"
                        "Do NOT fabricate quoted error messages to explain "
                        "the discrepancy. Do NOT claim the action succeeded "
                        "based on prior session state. Report honestly."
                    )
                ),
            ]
        }

    return handle_fabricated_action


def build_handle_fabricated_quote_node(
    fabricated_quote_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the fabricated-tool-error-quote recovery node.

    Bug #1871: polarity-flipped sibling of the #1869 fabricated-success
    recovery. The model emits prose like *"The error message is clear:
    'Read-only file system'"* — a confident verbatim quote framed as
    observed tool output — but no tool ran this turn (or the quoted
    span never appeared in any ToolMessage). Q13 / Q14 / Q15 of the
    holistic-test battery against ``cogtrix:release-next`` @ ``2bb52c7``
    produced three different mutually-contradictory fabricated error
    strings across three consecutive turns, each presented with the
    same confident framing.

    A verbatim quoted error string carries unusual epistemic weight
    in the conversation — readers reasonably assume the agent saw that
    text from a real tool. Letting the agent freely fabricate quoted
    error output is materially worse than fabricating prose claims.

    Mirrors the recovery-node pattern from #1869 / #1713: remove the
    offending response wholesale and inject a nudge naming both honest
    paths (call the tool and observe a real error, or stop attributing
    quoted text to tools that did not run).

    Args:
        fabricated_quote_count: Mutable per-run counter (lives on
            :class:`PerRunState`) of how many times this node has fired
            in the current run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1.
        logger: Logger factory (overridable for testing).
    """
    from src.orchestration.response_detectors import (
        _looks_like_fabricated_tool_error_quote,
    )

    def handle_fabricated_quote(state: CogtrixState) -> dict:
        fabricated_quote_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None

        if last is None or not _looks_like_fabricated_tool_error_quote(msgs, last):
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Fabricated tool-error quote without backing ToolMessage "
            "(attempt %d/%d). Injecting correction.",
            fabricated_quote_count[0],
            max_retries,
        )

        if fabricated_quote_count[0] > max_retries:
            log.info(
                "Fabricated-quote retries exhausted — accepting the agent's "
                "response as-is rather than spinning further."
            )
            return {"messages": []}

        removal: list[Any] = []
        if getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(
                    content=(
                        "Your response quotes a verbatim error message "
                        "(text inside quotation marks, presented as a tool "
                        "or system output), but no tool was invoked in this "
                        "turn — or the quoted text does not appear in any "
                        "ToolMessage in the conversation. You cannot have "
                        "observed that error string.\n\n"
                        "Choose exactly one path and answer accordingly — do "
                        "NOT repeat the fabricated quote and do NOT invent "
                        "a different quoted error:\n"
                        "  (a) If a tool can produce the relevant output, "
                        "invoke it now via a structured tool call (use "
                        '`request_tools(add=["<name>"])` first if the tool '
                        "is in the catalog but not yet loaded). Quote ONLY "
                        "what the tool actually returns.\n"
                        "  (b) If you have no tool that can produce the "
                        "output, tell the user plainly that you have NOT "
                        "observed any such error and answer based on what "
                        "you actually know, without quoting fabricated text.\n\n"
                        "Quoting an error string you have not observed is "
                        "the worst-possible answer because the quote format "
                        "implies you saw it. Stop attributing quoted text "
                        "to tools that did not run."
                    )
                ),
            ]
        }

    return handle_fabricated_quote


def build_handle_noncanonical_fork_node(
    noncanonical_fork_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the non-canonical-fork-recommendation recovery node.

    Bug #1868: the model surfaces a non-canonical GitHub fork (low
    stars, no releases, personal owner) and presents it with the
    canonical project's description + recommendation framing. The user
    clicking the link lands on the fork rather than the canonical home.

    Reproducer (Q5 of the 2026-05-28 holistic-test battery against
    ``cogtrix:release-next`` @ ``2bb52c7``): asked for *"three currently-
    active open-source projects on GitHub that implement WebAssembly
    tools for security analysis"*, the agent returned one canonical
    entry plus two personal/inactive forks
    (``DharitriOne/wasmer``, ``wasm-wasi-rs/runtimes__wasmtime``) with
    the canonical projects' blurbs.

    Mirrors the post-#1731 recovery-node pattern: remove the offending
    response wholesale and inject a nudge naming both honest paths —
    either replace the URL with the canonical home (verifying via a
    fresh ``web_search`` or the owner-is-the-official-org check) or
    state plainly that no canonical match was found.

    Args:
        noncanonical_fork_count: Mutable per-run counter (lives on
            :class:`PerRunState`) of how many times this node has fired
            in the current run.
        max_retries: Maximum revision attempts before accepting the
            response. Default 1.
        logger: Logger factory (overridable for testing).
    """
    from src.orchestration.verification import (
        detect_noncanonical_fork_recommendation,
        format_noncanonical_fork_nudge,
    )

    def handle_noncanonical_fork(state: CogtrixState) -> dict:
        noncanonical_fork_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None

        if last is None:
            return {"messages": []}
        content = getattr(last, "content", "")
        if not isinstance(content, str) or not content.strip():
            return {"messages": []}

        # User prompt is the most recent HumanMessage; needed for the
        # "user asked for forks" suppression.
        user_prompt = ""
        for m in reversed(msgs):
            if m.__class__.__name__ == "HumanMessage":
                hp = getattr(m, "content", "")
                if isinstance(hp, str):
                    user_prompt = hp
                break

        flagged = detect_noncanonical_fork_recommendation(content, user_prompt=user_prompt)
        if not flagged:
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Non-canonical fork URL%s recommended (attempt %d/%d): %s",
            "s" if len(flagged) != 1 else "",
            noncanonical_fork_count[0],
            max_retries,
            ", ".join(flagged),
        )

        if noncanonical_fork_count[0] > max_retries:
            log.info(
                "Non-canonical-fork retries exhausted — accepting the "
                "agent's response as-is rather than spinning further."
            )
            return {"messages": []}

        removal: list[Any] = []
        if getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_noncanonical_fork_nudge(flagged)),
            ]
        }

    return handle_noncanonical_fork


def build_handle_synthesis_after_eviction_node(
    synthesis_after_eviction_count: list[int],
    max_retries: int,
    logger: Callable[[], Any] = get_logger,
) -> Callable[[CogtrixState], dict]:
    """Build the synthesis-after-eviction recovery node bound to a run-local counter.

    #1943 PR #4: when the response makes substantive claims AFTER a
    ``context_evicted`` SystemMessage marker (PR #1) is in the visible
    context AND the current turn ran NO new tool calls AND the response
    contains none of the compliant-acknowledgement phrases, the detector
    flags it as a suspected synthesis-after-eviction.  This node deletes
    the response and injects a nudge listing the three compliant
    alternatives (ground in visible context, honestly surface the loss,
    or re-gather via tool call).

    Args:
        synthesis_after_eviction_count: Mutable counter of how many times
            this node fired this run.
        max_retries: Maximum revision attempts before accepting the
            response.  Default 1 — one revision attempt; a model still
            fabricating after the nudge is actively ignoring the guard,
            not unlucky.
        logger: Logger factory.
    """
    from src.orchestration.verification import (
        detect_synthesis_after_eviction,
        format_synthesis_after_eviction_nudge,
    )

    def handle_synthesis_after_eviction(state: CogtrixState) -> dict:
        synthesis_after_eviction_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        if not detect_synthesis_after_eviction(last_content, msgs, turn_start):
            # Re-detection failed — possibly revised by a concurrent path.
            return {"messages": []}

        log.warning(
            "Synthesis-after-eviction detected (attempt %d/%d). Injecting nudge.",
            synthesis_after_eviction_count[0],
            max_retries,
        )

        if synthesis_after_eviction_count[0] > max_retries:
            log.info(
                "Synthesis-after-eviction retries exhausted — accepting the "
                "agent's response as-is rather than spinning further.",
            )
            return {"messages": []}

        removal: list[Any] = []
        if last is not None and getattr(last, "id", None):
            removal.append(RemoveMessage(id=last.id))
        return {
            "messages": [
                *removal,
                HumanMessage(content=format_synthesis_after_eviction_nudge()),
            ]
        }

    return handle_synthesis_after_eviction


def _find_current_turn_start(messages: list[Any]) -> int:
    """Return the index of the most recent *real* HumanMessage in *messages*.

    The "current turn" is the slice from that HumanMessage forward.
    Tool calls before that index belong to prior turns and don't
    count toward the current turn's verification budget.

    Recovery-injected ``HumanMessage``s carry the
    ``_RECOVERY_INJECTED_MARKER`` flag in ``additional_kwargs`` (see the
    ``HumanMessage`` factory above) and are skipped — they are nudge
    text, not actual user input.  Without this skip, detectors that
    read ``sources.user_prompt`` to extract distinctive subjects (most
    notably ``detect_topic_substitution`` introduced in #1992) latch
    onto words inside the nudge ("ToolMessage", "NOT", …) and falsely
    flag the agent's response as off-topic, kicking off an infinite
    recovery cascade.
    """
    from langchain_core.messages import HumanMessage as _HM

    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, _HM):
            continue
        if (getattr(m, "additional_kwargs", None) or {}).get(_RECOVERY_INJECTED_MARKER):
            continue
        return i
    return 0
