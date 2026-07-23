"""
Message handler for Cogtrix assistant mode.

Translates an IncomingMessage into an agent response and sends it back
via the originating channel.  Each invocation acquires the per-session lock
so rapid messages from the same chat are processed in order.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.core import AgentRunner
from src.assistant.channel import Channel, IncomingMessage
from src.assistant.guardrails import _BLOCKED_RESPONSE, GuardrailPipeline
from src.orchestration.session_state import SessionState

log = logging.getLogger("cogtrix")

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
        self._max_response_length: int = config.get("max_response_length", 4000)

        excluded = _DEFAULT_EXCLUDED | set(config.get("excluded_tools", []))
        self._excluded_tools = excluded

        self._available_tools = {
            name: tool for name, tool in available_tools.items() if name not in excluded
        }
        self._active_tools = [t for t in active_tools if getattr(t, "name", None) not in excluded]

    def handle(self, msg: IncomingMessage, channel: Channel) -> None:
        """Process *msg* and send a response back via *channel*."""
        session = self._session_mgr.get_or_create(msg)
        with session.lock:
            result = self._guardrails.check_input(msg.text, msg.chat_id)
            if not result.is_safe:
                log.warning(
                    "Guardrail blocked [%s] chat=%s: %s",
                    result.guard_name,
                    msg.chat_id,
                    result.reason,
                )
                channel.send(msg.chat_id, _BLOCKED_RESPONSE)
                return

            session.last_activity = time.monotonic()
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

            try:
                runner = self._agent_runner
                call_session_state = SessionState(
                    no_confirm=self._session_state.no_confirm if self._session_state else True,
                )
                response = runner(
                    user_input=msg.text,
                    history_messages=context.messages,
                    context_prefix=combined_prefix,
                    llm=self._llm,
                    system_prompt=self._system_prompt,
                    registry=self._registry,
                    approvals=set(self._approvals),
                    available_tools=dict(self._available_tools),
                    active_tools_list=list(self._active_tools),
                    max_context_tokens=self._max_context_tokens,
                    compression_llm=self._compression_llm,
                    tool_call_guard=self._guardrails.check_tool_call,
                    session_state=call_session_state,
                    parallel_tool_execution=self._parallel_tool_execution,
                )
            except Exception as exc:
                log.error("Agent error for session %s: %s", session.session_key, exc)
                response = "I encountered an error processing your message. Please try again."

            response = self._guardrails.sanitize_output(response)

            if len(response) > self._max_response_length:
                response = response[: self._max_response_length - 3] + "..."

            sent = channel.send(msg.chat_id, response)
            if not sent:
                log.warning("Failed to send reply to %s via %s", msg.chat_id, channel.name)

            if self._knowledge_store:
                try:
                    self._knowledge_store.extract_and_store(msg.text, response)
                except Exception:
                    pass

            try:
                session.memory_manager.update(msg.text, response)
                session.memory_manager.save()
            except Exception as exc:
                log.warning("Failed to update memory for session %s: %s", session.session_key, exc)
