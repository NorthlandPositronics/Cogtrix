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
