"""
Message handler for Cogtrix assistant mode.

Translates an IncomingMessage into an agent response and sends it back
via the originating channel.  Each invocation acquires the per-session lock
so rapid messages from the same chat are processed in order.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.agent.core import AgentRunner
from src.assistant.channel import Channel, IncomingMessage
from src.assistant.datamarking import apply_datamark as _apply_datamark
from src.assistant.datamarking import datamark_history as _datamark_history
from src.assistant.datamarking import datamark_instruction as _datamark_instruction
from src.assistant.datamarking import generate_datamark as _generate_datamark
from src.assistant.guardrails import _BLOCKED_RESPONSE, GuardrailPipeline
from src.assistant.scheduler import MessageScheduler, ScheduleReplyState, create_schedule_reply_tool
from src.orchestration.session_state import SessionState

log = logging.getLogger("cogtrix")

_UNSET: object = object()


def _load_prompt_value(value: str) -> str:
    """Return prompt text — reads from file path (starts with / or ~) or returns inline."""
    stripped = value.strip()
    if stripped.startswith("/") or stripped.startswith("~"):
        path = Path(stripped).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            log.warning("Failed to read contact prompt file %s: %s", path, exc)
            return ""
    return stripped


_DEFAULT_EXCLUDED: frozenset[str] = frozenset(
    {
        "whatsapp_send",
        "whatsapp_check",
        "whatsapp_send_image",
        "whatsapp_contacts",
        "telegram_send",
        "telegram_check",
        "telegram_send_photo",
        "telegram_contacts",
        "execute_shell_command",
        "execute_python",
        "write_file",
        "append_file",
        "read_file",
        "read_pdf",
        "list_directory",
        "file_info",
    }
)


class MessageHandler:
    """Core message-to-response pipeline for the assistant mode.

    Args:
        session_mgr: ChatSessionManager instance.
        config: Assistant-mode config dict (services.assistant section).
        llm: LLM instance.
        system_prompt: System prompt for the agent.
        registry: Tool registry.
        approvals: Set of tool names auto-approved without confirmation.
        available_tools: {name: tool} dict of on-demand tools.
        active_tools: List of initially active tool objects.
        max_context_tokens: Context window budget.
        compression_llm: Optional dedicated LLM for context compression.
        knowledge_store: Optional SharedKnowledgeStore (Sprint 2).
        datamarking_enabled: Enable Microsoft Spotlighting prompt injection defense.
    """

    def __init__(
        self,
        session_mgr: Any,
        config: dict[str, Any],
        llm: Any,
        system_prompt: str,
        registry: Any,
        approvals: set[str],
        available_tools: dict[str, Any],
        active_tools: list[Any],
        *,
        max_context_tokens: int | None = None,
        compression_llm: Any = None,
        knowledge_store: Any = None,
        guardrails: Any = None,
        agent_runner: AgentRunner,
        session_state: SessionState | None = None,
        parallel_tool_execution: bool = True,
        services_config: dict[str, Any] | None = None,
        scheduler: MessageScheduler | None = None,
        datamarking_enabled: Any = _UNSET,
    ) -> None:
        self._session_mgr = session_mgr
        self._llm = llm
        self._system_prompt = system_prompt
        self._registry = registry
        self._approvals = approvals
        self._max_context_tokens = max_context_tokens
        self._compression_llm = compression_llm
        self._knowledge_store = knowledge_store
        self._guardrails = guardrails if guardrails is not None else GuardrailPipeline({})
        self._agent_runner: AgentRunner = agent_runner
        self._session_state = session_state
        self._parallel_tool_execution = parallel_tool_execution
        self._services_config: dict[str, Any] = services_config or {}
        self._max_response_length: int = config.get("max_response_length", 4000)
        self._scheduler: MessageScheduler | None = scheduler

        if datamarking_enabled is _UNSET:
            guardrail_cfg = config.get("guardrails", {})
            self._datamarking_enabled: bool = guardrail_cfg.get("datamarking", True)
        else:
            self._datamarking_enabled = bool(datamarking_enabled)

        excluded = _DEFAULT_EXCLUDED | set(config.get("excluded_tools", []))
        self._excluded_tools = excluded

        self._available_tools = {
            name: tool for name, tool in available_tools.items() if name not in excluded
        }
        self._active_tools = [t for t in active_tools if getattr(t, "name", None) not in excluded]

    def _resolve_contact_prompt(self, msg: IncomingMessage) -> str | None:
        """Return per-contact instructions from config, or None if not configured."""
        channel_cfg = self._services_config.get(msg.channel, {})
        contact_prompts: dict[str, str] = channel_cfg.get("contact_prompts", {})
        if not contact_prompts:
            return None

        phonebook: dict[str, str] = channel_cfg.get("phonebook", {})
        contact_name: str | None = None

        identifiers: set[str] = set()
        if msg.resolved_phone:
            resolved = msg.resolved_phone.replace("@c.us", "").strip()
            identifiers.add(resolved)
            identifiers.add(resolved.lstrip("+"))
        identifiers.add(
            msg.chat_id.replace("@c.us", "")
            .replace("@lid", "")
            .replace("@s.whatsapp.net", "")
            .strip()
        )
        identifiers.add(msg.sender_id.strip())

        for name, value in phonebook.items():
            pb_digits = value.strip().replace("+", "")
            if pb_digits in identifiers:
                contact_name = name
                break

        if not contact_name or contact_name not in contact_prompts:
            return None

        loaded = _load_prompt_value(contact_prompts[contact_name])
        if not loaded:
            return None
        log.debug("Resolved contact prompt for '%s'", contact_name)
        return loaded

    def _check_guardrails(self, msg: IncomingMessage, session: Any, channel: Channel) -> bool:
        """Return True if input passes guardrails; on failure, send blocked response and return False."""
        result = self._guardrails.check_input(msg.text, msg.chat_id)
        if not result.is_safe:
            session.guardrail_violations += 1
            log.warning(
                "Guardrail blocked [%s] chat=%s: %s (violations=%d)",
                result.guard_name,
                msg.chat_id,
                result.reason,
                session.guardrail_violations,
            )
            channel.send(msg.chat_id, _BLOCKED_RESPONSE)
            return False
        return True

    def _prepare_context(self, msg: IncomingMessage, session: Any) -> tuple[Any, str | None]:
        """Prepare memory context and optionally augment with knowledge recall.

        Returns ``(context, combined_prefix)``.
        """
        context = session.memory_manager.prepare_context(msg.text)
        combined_prefix = context.context_prefix
        if self._knowledge_store:
            try:
                recall_k = 5
                knowledge = self._knowledge_store.recall(msg.text, k=recall_k)
                if knowledge:
                    section = f"Known facts (learned over time):\n{knowledge}"
                    combined_prefix = (
                        f"{combined_prefix}\n\n{section}" if combined_prefix else section
                    )
            except Exception as exc:
                log.debug("Knowledge recall failed: %s", exc)
        return context, combined_prefix

    def _prepare_agent_call(
        self,
        msg: IncomingMessage,
        context: Any,
        combined_prefix: str | None,
    ) -> tuple[str, str, list]:
        """Resolve effective prompt and apply datamarking to input and history.

        Returns ``(effective_prompt, user_input_for_agent, history_for_agent)``.
        """
        contact_prompt = self._resolve_contact_prompt(msg)
        effective_prompt = contact_prompt if contact_prompt else self._system_prompt

        dm_marker: str | None = None
        if self._datamarking_enabled:
            dm_marker = _generate_datamark()
            effective_prompt = _datamark_instruction(dm_marker) + "\n" + effective_prompt

        user_input_for_agent = _apply_datamark(msg.text, dm_marker) if dm_marker else msg.text
        history_for_agent = (
            _datamark_history(context.messages, dm_marker) if dm_marker else context.messages
        )
        return effective_prompt, user_input_for_agent, history_for_agent

    def _run_agent(
        self,
        *,
        user_input: str,
        history_messages: list,
        context_prefix: str | None,
        effective_prompt: str,
        active_tools: list[Any],
        session: Any,
    ) -> str:
        """Invoke the agent runner and return the response string."""
        try:
            runner = self._agent_runner
            call_session_state = SessionState(
                no_confirm=self._session_state.no_confirm if self._session_state else True,
            )
            response: str = runner(
                user_input=user_input,
                history_messages=history_messages,
                context_prefix=context_prefix,
                llm=self._llm,
                system_prompt=effective_prompt,
                registry=self._registry,
                approvals=set(self._approvals),
                available_tools=dict(self._available_tools),
                active_tools_list=active_tools,
                max_context_tokens=self._max_context_tokens,
                compression_llm=self._compression_llm,
                tool_call_guard=self._guardrails.check_tool_call,
                session_state=call_session_state,
                parallel_tool_execution=self._parallel_tool_execution,
            )
        except Exception as exc:
            log.error("Agent error for session %s: %s", session.session_key, exc)
            response = "I encountered an error processing your message. Please try again."
        return response

    def _route_response(
        self,
        msg: IncomingMessage,
        channel: Channel,
        response: str,
        schedule_state: ScheduleReplyState,
    ) -> str:
        """Route the response to scheduled or immediate delivery and return the text for memory."""
        if schedule_state.was_called and self._scheduler:
            reply_text = self._guardrails.sanitize_output(schedule_state.scheduled_text)
            if len(reply_text) > self._max_response_length:
                reply_text = reply_text[: self._max_response_length - 3] + "..."
            send_at = time.time() + schedule_state.delay_minutes * 60
            self._scheduler.schedule(msg.channel, msg.chat_id, reply_text, send_at)
            log.info("Reply scheduled for %s (%d min)", msg.chat_id, schedule_state.delay_minutes)
            return reply_text
        else:
            response = self._guardrails.sanitize_output(response)
            if len(response) > self._max_response_length:
                response = response[: self._max_response_length - 3] + "..."
            sent = channel.send(msg.chat_id, response)
            if not sent:
                log.warning("Failed to send reply to %s via %s", msg.chat_id, channel.name)
            return response

    def handle(self, msg: IncomingMessage, channel: Channel) -> None:
        """Process *msg* and send a response back via *channel*."""
        session = self._session_mgr.get_or_create(msg)
        with session.lock:
            if not self._check_guardrails(msg, session, channel):
                return

            if self._scheduler:
                cancelled = self._scheduler.cancel_pending(msg.channel, msg.chat_id)
                if cancelled:
                    log.debug("Cancelled %d pending reply(s) for %s", cancelled, msg.chat_id)

            session.last_activity = time.monotonic()
            context, combined_prefix = self._prepare_context(msg, session)

            schedule_state = ScheduleReplyState()
            active_tools: list[Any] = list(self._active_tools)
            if self._scheduler and "schedule_reply" not in self._excluded_tools:
                active_tools.append(create_schedule_reply_tool(schedule_state))

            effective_prompt, user_input, history = self._prepare_agent_call(
                msg, context, combined_prefix
            )
            response = self._run_agent(
                user_input=user_input,
                history_messages=history,
                context_prefix=combined_prefix,
                effective_prompt=effective_prompt,
                active_tools=active_tools,
                session=session,
            )
            response_for_memory = self._route_response(msg, channel, response, schedule_state)

            # Memory records the response regardless of delivery success
            # (at-least-once memory semantics).
            try:
                session.memory_manager.update(msg.text, response_for_memory)
                session.memory_manager.save()
            except Exception as exc:
                log.warning("Failed to update memory for session %s: %s", session.session_key, exc)

            do_extract = self._knowledge_store is not None and session.guardrail_violations == 0

        if do_extract:
            try:
                sanitized = self._guardrails.sanitize_output(msg.text)
                self._knowledge_store.extract_and_store(sanitized, response_for_memory)  # type: ignore[union-attr]
            except Exception as exc:
                log.debug("Knowledge extraction failed: %s", exc)
