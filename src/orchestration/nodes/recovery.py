"""Recovery node factories for phantom and action-intent retries."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage

from src.agent.core import CogtrixState
from src.logging_config import get_logger


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
        collect_tool_message_contents,
        detect_unverified_entities,
        format_unverified_entity_nudge,
    )

    def handle_unverified_entity(state: CogtrixState) -> dict:
        from langchain_core.messages import HumanMessage as _HM

        unverified_entity_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        # Resolve the actual user prompt — the HumanMessage at turn_start.
        user_prompt = ""
        if turn_start < len(msgs) and isinstance(msgs[turn_start], _HM):
            up = msgs[turn_start].content
            if isinstance(up, str):
                user_prompt = up
        tool_contents = collect_tool_message_contents(msgs, turn_start)

        entities = detect_unverified_entities(last_content, user_prompt, tool_contents)

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
        collect_tool_message_contents,
        detect_unsupported_quote,
        format_unsupported_quote_nudge,
    )

    def handle_unsupported_quote(state: CogtrixState) -> dict:
        from langchain_core.messages import HumanMessage as _HM

        unsupported_quote_count[0] += 1
        log = logger()
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        last_content = getattr(last, "content", "") if last is not None else ""

        if not isinstance(last_content, str):
            return {"messages": []}

        turn_start = _find_current_turn_start(msgs)
        user_prompt = ""
        if turn_start < len(msgs) and isinstance(msgs[turn_start], _HM):
            up = msgs[turn_start].content
            if isinstance(up, str):
                user_prompt = up
        tool_contents = collect_tool_message_contents(msgs, turn_start)

        quotes = detect_unsupported_quote(last_content, tool_contents, user_prompt)

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


def _find_current_turn_start(messages: list[Any]) -> int:
    """Return the index of the most recent HumanMessage in *messages*.

    The "current turn" is the slice from that HumanMessage forward.
    Tool calls before that index belong to prior turns and don't
    count toward the current turn's verification budget.
    """
    from langchain_core.messages import HumanMessage as _HM

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], _HM):
            return i
    return 0
