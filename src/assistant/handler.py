"""
Message handler for Cogtrix assistant mode.

Translates an IncomingMessage into an agent response and sends it back
via the originating channel.  Each invocation acquires the per-session lock
so rapid messages from the same chat are processed in order.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.core import AgentRunner
from src.agent.safety import UserCancelledRun
from src.assistant.channel import Channel, IncomingMessage
from src.assistant.datamarking import apply_datamark as _apply_datamark
from src.assistant.datamarking import datamark_history as _datamark_history
from src.assistant.datamarking import datamark_instruction as _datamark_instruction
from src.assistant.datamarking import generate_datamark as _generate_datamark
from src.assistant.deferral import (
    DeferralManager,
    DeferReplyState,
    SuppressReplyState,
    create_defer_processing_tool,
    create_suppress_reply_tool,
)
from src.assistant.guardrails import _BLOCKED_RESPONSE, GuardrailPipeline
from src.assistant.scheduler import (
    EditReplyState,
    MessageScheduler,
    QueueReplyState,
    ScheduleReplyState,
    create_cancel_scheduled_tool,
    create_edit_reply_tool,
    create_edit_scheduled_tool,
    create_list_scheduled_tool,
    create_queue_reply_tool,
    create_schedule_reply_tool,
)
from src.orchestration.phases import RECOVERY_FAILED_MESSAGE as _RECOVERY_FAILED_MESSAGE
from src.orchestration.session_state import SessionState

log = logging.getLogger("cogtrix")

_UNSET: object = object()
_PR_REF_RE = re.compile(r"\bPR\s*#(\d+)\b", re.IGNORECASE)

# Prefix of the internal error string returned by _run_agent's exception
# handler. Recognized so it is never delivered to an external contact in
# assistant/messaging mode (#2052).
_AGENT_ERROR_PREFIX = "I encountered a"


def _is_non_deliverable(response: Any) -> bool:
    """True when *response* is an internal control/error message that must
    NOT be delivered to an external messaging contact (#2052).

    Covers: empty/blank output (silence), the recovery-failed sentinel
    ("say continue…"), the agent-error string, and provider auth failures.
    In CLI mode these are shown to the operator; on a WhatsApp/Telegram
    channel they are meaningless or alarming to the contact and leak
    operational detail, so they are suppressed — the turn is dropped from
    memory so the agent recovers cleanly on the contact's next message.
    """
    if not isinstance(response, str):
        return not bool(response)
    stripped = response.strip()
    if not stripped:
        return True
    if stripped == _RECOVERY_FAILED_MESSAGE.strip():
        return True
    if stripped.startswith(_AGENT_ERROR_PREFIX):
        return True
    if stripped.lower().startswith(("authentication failed", "**authentication failed")):
        return True
    return False


def _load_prompt_value(value: str, allowed_roots: list[Path] | None = None) -> str:
    """Return prompt text — reads from file path (starts with / or ~) or returns inline.

    When *allowed_roots* is provided, the resolved path must be relative to one of
    the roots. Symlinks are followed via ``.resolve()`` before the containment check
    so a symlink pointing outside an allowed root is correctly rejected (BUG-096).
    """
    stripped = value.strip()
    if stripped.startswith("/") or stripped.startswith("~"):
        path = Path(stripped).expanduser().resolve()
        if allowed_roots:
            if not any(path.is_relative_to(root) for root in allowed_roots):
                log.warning(
                    "Contact prompt path %s is outside allowed roots %s — rejected",
                    path,
                    [str(r) for r in allowed_roots],
                )
                return ""
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                log.warning(
                    "Contact prompt file %s exists but is empty — returning None to caller",
                    path,
                )
            return content
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
        parallel_tool_execution: bool = True,
        services_config: dict[str, Any] | None = None,
        scheduler: MessageScheduler | None = None,
        deferral_mgr: DeferralManager | None = None,
        datamarking_enabled: Any = _UNSET,
        workflow_registry: Any = None,
        campaign_mgr: Any = None,
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
        self._parallel_tool_execution = parallel_tool_execution
        self._config = config  # Store config for recall threshold access
        self._services_config: dict[str, Any] = services_config or {}
        github_cfg = self._services_config.get("github", {}) if self._services_config else {}
        self._github_default_repo: str = str(github_cfg.get("default_repo", "")).strip()
        _mrl = config.get("max_response_length", 4000)
        if _mrl < 3:
            log.warning("max_response_length %d is below minimum (3); using 3", _mrl)
            _mrl = 3
        self._max_response_length: int = _mrl
        self._scheduler: MessageScheduler | None = scheduler
        self._deferral_mgr: DeferralManager | None = deferral_mgr
        self._workflow_registry: Any = workflow_registry
        self._campaign_mgr: Any = campaign_mgr

        if datamarking_enabled is _UNSET:
            guardrail_cfg = config.get("guardrails", {})
            self._datamarking_enabled: bool = guardrail_cfg.get("datamarking", True)
        else:
            self._datamarking_enabled = bool(datamarking_enabled)
        log.info(
            "Datamarking (Microsoft Spotlighting): %s",
            "enabled" if self._datamarking_enabled else "disabled",
        )

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

        _allowed = [Path.cwd(), Path.home()]
        loaded = _load_prompt_value(contact_prompts[contact_name], allowed_roots=_allowed)
        if not loaded:
            return None
        log.debug("Resolved contact prompt for '%s'", contact_name)
        return loaded

    def _resolve_recipient(self, msg: IncomingMessage) -> str | None:
        """Derive a human-readable recipient identifier from message metadata."""
        if msg.resolved_phone:
            return msg.resolved_phone
        if msg.sender_name:
            return msg.sender_name
        return msg.chat_id

    def _pr_reference_is_valid(self, pr_number: int) -> bool:
        """Return True when the referenced PR exists in the configured repo."""
        github_repo = getattr(self, "_github_default_repo", "")
        if not github_repo or shutil.which("gh") is None:
            return True

        cmd = [
            "gh",
            "api",
            f"repos/{github_repo}/pulls/{pr_number}",
            "--jq",
            ".number",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        except FileNotFoundError:
            log.debug("PR validation skipped — gh CLI not available")
            return True
        except subprocess.TimeoutExpired:
            log.debug("PR validation skipped — gh API timed out after 30s")
            return True

        if result.returncode != 0:
            return False
        return result.stdout.strip() == str(pr_number)

    def _prepare_outbound_text(self, text: str) -> str:
        """Validate PR refs in outbound text, then apply standard output sanitization."""
        github_repo = getattr(self, "_github_default_repo", "")
        if github_repo and _PR_REF_RE.search(text):
            seen: set[int] = set()

            def _replace(match: re.Match[str]) -> str:
                pr_number = int(match.group(1))
                if pr_number not in seen:
                    seen.add(pr_number)
                    if not self._pr_reference_is_valid(pr_number):
                        log.warning(
                            "Outbound message referenced missing PR #%d in %s; flagging it",
                            pr_number,
                            github_repo,
                        )
                        return f"PR #{pr_number} [not found]"
                return match.group(0)

            text = _PR_REF_RE.sub(_replace, text)

        return self._guardrails.sanitize_output(text)

    def _check_guardrails(
        self, msg: IncomingMessage, session: Any, channel: Channel, *, skip_trusted: bool = False
    ) -> bool:
        """Return True if input passes guardrails; on failure, send blocked response and return False.

        Args:
            skip_trusted: If True, bypass rate-limiting and blacklist checks (trusted operator).
        """
        result = self._guardrails.check_input(
            msg.text, msg.chat_id, skip_trusted_checks=skip_trusted
        )
        if not result.is_safe:
            session.guardrail_violations += 1
            log.warning(
                "Guardrail blocked [%s] chat=%s: %s (violations=%d)",
                result.guard_name,
                msg.chat_id,
                result.reason,
                session.guardrail_violations,
            )
            send_result = channel.send(msg.chat_id, _BLOCKED_RESPONSE)
            if send_result.ok and send_result.message_id:
                session.last_sent_message_id = send_result.message_id
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
                # Get threshold from config, fall back to store default (0.25)
                recall_threshold = self._config.get("vector_recall_threshold", 0.25)
                knowledge = self._knowledge_store.recall(
                    msg.text, k=recall_k, score_threshold=recall_threshold
                )
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
        session: Any = None,
    ) -> tuple[str, str, list, set[str], set[str]]:
        """Resolve effective prompt and apply datamarking to input and history.

        Returns ``(effective_prompt, user_input_for_agent, history_for_agent,
        workflow_excluded, workflow_approved)``.
        """
        workflow_excluded: set[str] = set()
        workflow_approved: set[str] = set()

        if self._workflow_registry is not None:
            resolved = self._workflow_registry.resolve(
                session_key=msg.session_key,
                msg_text=msg.text or "",
                sender_id=msg.sender_id or "",
                resolved_phone=msg.resolved_phone or "",
            )
            effective_prompt = resolved.system_prompt or self._system_prompt
            if session is not None and resolved.workflow_id:
                session.workflow_id = resolved.workflow_id
            if resolved.tool_policy is not None:
                workflow_excluded = set(resolved.tool_policy.excluded_tools)
                workflow_approved = set(resolved.tool_policy.additional_approved_tools)
        else:
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
        return (
            effective_prompt,
            user_input_for_agent,
            history_for_agent,
            workflow_excluded,
            workflow_approved,
        )

    def _run_agent(
        self,
        *,
        user_input: str,
        history_messages: list,
        context_prefix: str | None,
        effective_prompt: str,
        active_tools: list[Any],
        session: Any,
        extra_excluded: set[str] | None = None,
        extra_approvals: set[str] | None = None,
    ) -> tuple[str, set[str]]:
        """Invoke the agent runner and return (response, loaded_tools)."""
        from src.orchestration.run_config import AgentRunConfig

        try:
            call_session_state = SessionState(
                no_confirm=True,
            )
            available = dict(self._available_tools)
            if extra_excluded:
                available = {k: v for k, v in available.items() if k not in extra_excluded}
            call_approvals = set(self._approvals)
            if extra_approvals:
                call_approvals |= extra_approvals
            run_config = AgentRunConfig(
                llm=self._llm,
                system_prompt=effective_prompt,
                available_tools=available,
                active_tools_list=active_tools,
                max_context_tokens=self._max_context_tokens,
                compression_llm=self._compression_llm,
                tool_call_guard=self._guardrails.check_tool_call,
                session_state=call_session_state,
                memory_manager=session.memory_manager,
                parallel_tool_execution=self._parallel_tool_execution,
            )
            response: str = self._agent_runner(
                user_input=user_input,
                history_messages=history_messages,
                registry=self._registry,
                approvals=call_approvals,
                context_prefix=context_prefix,
                config=run_config,
            )
        except UserCancelledRun:
            log.info("Agent run cancelled for %s", session.session_key)
            return "", set()
        except Exception as exc:
            log.error("Agent error for session %s: %s", session.session_key, exc)
            response = f"{_AGENT_ERROR_PREFIX} {type(exc).__name__} error. Please try again or contact the administrator."
            return response, set()

        return response, call_session_state.loaded_tools

    def _route_response(
        self,
        msg: IncomingMessage,
        channel: Channel,
        response: str,
        schedule_state: ScheduleReplyState,
        edit_state: EditReplyState,
        queue_state: QueueReplyState,
        session: Any,
    ) -> str | None:
        """Route the response to scheduled or immediate delivery and return the text for memory."""
        edited_text: str | None = None
        if edit_state.was_called and session.last_sent_message_id:
            new_text = self._prepare_outbound_text(edit_state.new_text)
            if len(new_text) > self._max_response_length:
                new_text = new_text[: self._max_response_length - 3] + "..."
            result = channel.edit_message(msg.chat_id, session.last_sent_message_id, new_text)
            if result.ok:
                log.info("Edited last reply for %s", msg.chat_id)
                edited_text = new_text
            else:
                log.warning("Failed to edit message for %s: %s", msg.chat_id, result.error)
            # Fall through — schedule_reply may also have been called in the same turn.

        scheduled_text: str | None = None
        if schedule_state.was_called and self._scheduler:
            reply_text = self._prepare_outbound_text(schedule_state.scheduled_text)
            if len(reply_text) > self._max_response_length:
                reply_text = reply_text[: self._max_response_length - 3] + "..."
            send_at = time.time() + schedule_state.delay_minutes * 60
            recipient = self._resolve_recipient(msg)
            self._scheduler.schedule(
                msg.channel, msg.chat_id, reply_text, send_at, recipient=recipient
            )
            log.info("Reply scheduled for %s (%d min)", msg.chat_id, schedule_state.delay_minutes)
            scheduled_text = reply_text
            # Fall through — queue_reply may also have been called in the same turn.

        queued_texts: list[str] = []
        if queue_state.items and self._scheduler:
            recipient = self._resolve_recipient(msg)
            for item in queue_state.items:
                reply_text = self._prepare_outbound_text(item.text)
                if len(reply_text) > self._max_response_length:
                    reply_text = reply_text[: self._max_response_length - 3] + "..."
                self._scheduler.queue_after_tail(
                    msg.channel,
                    msg.chat_id,
                    reply_text,
                    gap_seconds=item.gap_minutes * 60,
                    recipient=recipient,
                    persist=False,
                )
                queued_texts.append(reply_text)
            self._scheduler.save()
            log.info("Queued %d message(s) for %s", len(queue_state.items), msg.chat_id)

        if scheduled_text is not None or queued_texts:
            all_texts: list[str] = []
            if edited_text is not None:
                all_texts.append(edited_text)
            if scheduled_text is not None:
                all_texts.append(scheduled_text)
            all_texts.extend(queued_texts)
            return "\n---\n".join(all_texts)

        if not edit_state.was_called:
            response = self._prepare_outbound_text(response)
            if len(response) > self._max_response_length:
                response = response[: self._max_response_length - 3] + "..."
            try:
                result = channel.send(msg.chat_id, response)
            except Exception as exc:
                log.warning(
                    "Failed to send reply to %s via %s: %s",
                    msg.chat_id,
                    channel.name,
                    exc,
                )
                return None
            if result.ok and result.message_id:
                session.last_sent_message_id = result.message_id
            elif not result.ok:
                log.warning("Failed to send reply to %s via %s", msg.chat_id, channel.name)
            return response

        # Edit was attempted.  If it succeeded, return the edited text for memory.
        if edited_text is not None:
            return edited_text

        # Edit failed (ok=False) and there is no schedule/queue activity.
        # Fall back to immediate send with the original agent response so the
        # user receives a reply rather than silence (BUG-110).
        log.info("Edit failed for %s — falling back to immediate send", msg.chat_id)
        fallback_text = self._prepare_outbound_text(edit_state.new_text)
        if len(fallback_text) > self._max_response_length:
            fallback_text = fallback_text[: self._max_response_length - 3] + "..."
        try:
            result = channel.send(msg.chat_id, fallback_text)
        except Exception as exc:
            log.warning(
                "Failed to send fallback reply to %s via %s: %s",
                msg.chat_id,
                channel.name,
                exc,
            )
            return None  # Don't record undelivered text in memory
        if result.ok and result.message_id:
            session.last_sent_message_id = result.message_id
        elif not result.ok:
            log.warning("Failed to send fallback reply to %s via %s", msg.chat_id, channel.name)
            return None  # Don't record undelivered text in memory
        return fallback_text

    def handle_batch(
        self,
        messages: list[IncomingMessage],
        channel: Channel,
        *,
        is_reprocessing: bool = False,
        deferral_depth: int = 0,
    ) -> None:
        """Process multiple rapid messages as a single agent turn."""
        if not messages:
            return

        # If a deferred record is pending for this chat and this is not a re-processing
        # pass, accumulate messages into the pending record instead of processing them.
        # BUG-094: evaluate all add_message calls eagerly (no short-circuit) to avoid
        # partial absorption under a race between add_message calls.
        if not is_reprocessing and self._deferral_mgr:
            primary = messages[0]
            absorbed = [self._deferral_mgr.add_message(m) for m in messages]
            if all(absorbed):
                log.info(
                    "Buffered %d message(s) for deferred session %s",
                    len(messages),
                    primary.session_key,
                )
                return
            if any(absorbed):
                # Partial absorption: at least one message was added to a record that
                # then fired between the two add_message calls. Cancel the deferred
                # record to prevent duplicate processing (BUG-094).
                log.warning(
                    "Partial deferral absorption for %s — cancelling deferred record "
                    "to prevent duplicate processing",
                    primary.session_key,
                )
                self._deferral_mgr.cancel(primary.session_key)

        if len(messages) == 1:
            self.handle(
                messages[0], channel, is_reprocessing=is_reprocessing, deferral_depth=deferral_depth
            )
            return
        combined_text = "\n".join(m.text for m in messages)
        primary = messages[-1]
        combined_msg = IncomingMessage(
            channel=primary.channel,
            chat_id=primary.chat_id,
            message_id=primary.message_id,
            sender_id=primary.sender_id,
            sender_name=primary.sender_name,
            text=combined_text,
            timestamp=primary.timestamp,
            metadata=primary.metadata,
            resolved_phone=primary.resolved_phone,
        )
        log.info(
            "Consolidated %d rapid messages for %s into single turn",
            len(messages),
            primary.chat_id,
        )
        self.handle(
            combined_msg, channel, is_reprocessing=is_reprocessing, deferral_depth=deferral_depth
        )

    def handle(
        self,
        msg: IncomingMessage,
        channel: Channel,
        *,
        is_reprocessing: bool = False,
        deferral_depth: int = 0,
    ) -> None:
        """Process *msg* and send a response back via *channel*."""
        session = self._session_mgr.get_or_create(msg)

        with session.lock:
            # Pre-record the user message for shutdown durability.  Uses
            # prerecord_user() which writes a lightweight pending file without
            # calling update(), so existing call-count assertions are
            # unaffected.  Cleaned up by update() on success or by
            # discard_prerecord() on deferral/suppress paths.  Called INSIDE
            # the lock to prevent concurrent _flush calls (debounce timer)
            # from racing on the same session's pending state.
            session.memory_manager.prerecord_user(msg.text)

            if not self._check_guardrails(msg, session, channel):
                session.memory_manager.discard_prerecord()
                return

            session.last_activity = time.monotonic()
            context, combined_prefix = self._prepare_context(msg, session)

            schedule_state = ScheduleReplyState()
            edit_state = EditReplyState()
            queue_state = QueueReplyState()
            defer_state = DeferReplyState()
            suppress_state = SuppressReplyState()
            active_tools: list[Any] = list(self._active_tools)
            if self._scheduler and "schedule_reply" not in self._excluded_tools:
                active_tools.append(create_schedule_reply_tool(schedule_state))
            if self._scheduler and "queue_reply" not in self._excluded_tools:
                active_tools.append(create_queue_reply_tool(queue_state))
            if "edit_last_reply" not in self._excluded_tools and session.last_sent_message_id:
                active_tools.append(create_edit_reply_tool(edit_state))
            if self._scheduler:
                # Scheduler tools are created per-call with caller_chat_id to enforce
                # per-session authorization (BUG-040, BUG-041).
                if "list_scheduled_messages" not in self._excluded_tools:
                    active_tools.append(
                        create_list_scheduled_tool(
                            self._scheduler,
                            self._services_config,
                            caller_chat_id=msg.chat_id,
                        )
                    )
                if "edit_scheduled_message" not in self._excluded_tools:
                    active_tools.append(
                        create_edit_scheduled_tool(
                            self._scheduler,
                            caller_chat_id=msg.chat_id,
                        )
                    )
                if "cancel_scheduled_message" not in self._excluded_tools:
                    active_tools.append(
                        create_cancel_scheduled_tool(
                            self._scheduler,
                            caller_chat_id=msg.chat_id,
                        )
                    )

            # Inject defer_processing if deferral manager is present and depth < max.
            # BUG-091: use the propagated deferral_depth parameter instead of querying
            # current_depth() — the record is already deleted from _records when
            # _fire_record calls this handler, so current_depth() would return 0.
            if self._deferral_mgr and "defer_processing" not in self._excluded_tools:
                if deferral_depth < self._deferral_mgr.max_depth:
                    active_tools.append(
                        create_defer_processing_tool(defer_state, schedule_state=schedule_state)
                    )

            # Inject suppress_reply in all turns so the agent can cleanly suppress
            # a reply without leaking reasoning text or tool-name artifacts into
            # the message body.  (Previously restricted to re-processing passes,
            # which caused the agent to emit "(No reply...)" or "suppress_reply"
            # as plain text when the tool was absent.)
            if "suppress_reply" not in self._excluded_tools:
                active_tools.append(create_suppress_reply_tool(suppress_state))

            # Inject report_campaign_outcome when this chat is an active campaign target
            campaign_outcome_state: Any = None
            _active_campaign: Any = None
            _active_target: Any = None
            if self._campaign_mgr is not None:
                match = self._campaign_mgr.get_active_campaign_for_chat(msg.channel, msg.chat_id)
                if match is not None:
                    _active_campaign, _active_target = match
                    from src.assistant.campaign import (
                        CampaignOutcomeState,
                        create_campaign_outcome_tool,
                    )

                    campaign_outcome_state = CampaignOutcomeState()
                    tool = create_campaign_outcome_tool(
                        campaign_outcome_state, _active_campaign.goal
                    )
                    if tool is not None:
                        active_tools.append(tool)
                    # Notify campaign manager that the contact replied
                    self._campaign_mgr.on_reply(msg.channel, msg.chat_id)

            effective_prompt, user_input, history, wf_excluded, wf_approved = (
                self._prepare_agent_call(msg, context, session=session)
            )
            response, turn_loaded_tools = self._run_agent(
                user_input=user_input,
                history_messages=history,
                context_prefix=combined_prefix,
                effective_prompt=effective_prompt,
                active_tools=active_tools,
                session=session,
                extra_excluded=wf_excluded or None,
                extra_approvals=wf_approved or None,
            )
            if turn_loaded_tools:
                log.debug(
                    "Session %s: tools auto-loaded this turn: %s",
                    session.session_key,
                    turn_loaded_tools,
                )

            # Check suppress_reply first — strongest signal: no delivery, no memory update.
            if suppress_state.was_called:
                log.info("Agent suppressed reply for %s", msg.chat_id)
                session.memory_manager.discard_prerecord()
                return

            # Check defer_processing — no delivery, no memory update; register deferral.
            # BUG-091: use deferral_depth (propagated from callback) rather than
            # current_depth() which returns 0 after _fire_record deletes the record.
            if defer_state.was_called and self._deferral_mgr:
                log.info(
                    "Agent deferred processing for %s by %.0fs (depth %d)",
                    msg.chat_id,
                    defer_state.delay_seconds,
                    deferral_depth,
                )
                self._deferral_mgr.defer(msg, defer_state.delay_seconds, depth=deferral_depth)
                session.memory_manager.discard_prerecord()
                return

            # #2052: never deliver an internal control/error message (empty
            # output, the recovery-failed sentinel, the agent-error string, or
            # a provider auth failure) to an external messaging contact. These
            # read as the agent being broken — or leak operational detail — to
            # a real contact. Stay silent and drop the turn from memory so the
            # agent recovers on the contact's next message.
            if _is_non_deliverable(response):
                log.warning(
                    "Non-deliverable agent response for %s — staying silent "
                    "(internal control/error message not sent to contact)",
                    msg.chat_id,
                )
                session.memory_manager.discard_prerecord()
                return

            response_for_memory = self._route_response(
                msg, channel, response, schedule_state, edit_state, queue_state, session
            )
            # _route_response returns None only when an edit fallback send fails and
            # no text was delivered.  Skip memory update in that case — storing a None
            # response would corrupt conversation history and crash knowledge extraction.
            if response_for_memory is None:
                log.warning(
                    "Skipping memory update for %s: no response was delivered", session.session_key
                )
                session.memory_manager.discard_prerecord()
                return

            try:
                session.memory_manager.update(msg.text, response_for_memory)
                session.memory_manager.save()
            except Exception as exc:
                log.warning(
                    "Failed to update memory for session %s: %s",
                    session.session_key,
                    exc,
                    exc_info=True,
                )

            # Process campaign outcome if the agent reported one
            if (
                campaign_outcome_state is not None
                and campaign_outcome_state.was_called
                and _active_campaign is not None
                and self._campaign_mgr is not None
            ):
                self._campaign_mgr.mark_target_outcome(
                    _active_campaign.id,
                    msg.chat_id,
                    campaign_outcome_state.outcome,
                    campaign_outcome_state.reason,
                )

            do_extract = session.guardrail_violations == 0
            sanitized_for_knowledge = (
                self._guardrails.sanitize_output(msg.text) if do_extract else None
            )

        if self._knowledge_store is not None and do_extract and sanitized_for_knowledge is not None:
            try:
                self._knowledge_store.extract_and_store(
                    sanitized_for_knowledge, response_for_memory, session.session_key
                )
            except Exception as exc:
                log.debug("Knowledge extraction failed: %s", exc)

    def handle_outbound(
        self,
        contact_name: str,
        instructions: str,
        channel: Channel,
        chat_id: str,
    ) -> tuple[str, str | None]:
        """Run agent for an operator-initiated outbound message.

        Operator instructions bypass input guardrails; output guardrails still apply.
        Memory records ``[Operator instruction] {instructions}`` as the user turn.

        Args:
            contact_name: Phonebook contact name (for logging and agent framing).
            instructions: Operator instructions — the agent's task.
            channel: Channel to send the response through.
            chat_id: Resolved chat identifier on the channel.

        Returns:
            ``(response_text, message_id)`` — the sanitized agent response and
            the channel message ID (None if delivery failed).
        """
        framed_text = (
            f"[Operator instruction — initiate conversation with {contact_name}]\n{instructions}"
        )
        synthetic_msg = IncomingMessage(
            channel=channel.name,
            chat_id=chat_id,
            message_id=f"outbound-{uuid.uuid4().hex[:12]}",
            sender_id="operator",
            sender_name="Operator",
            text=framed_text,
            timestamp=time.time(),
            metadata={"outbound": True},
        )

        session = self._session_mgr.get_or_create(synthetic_msg)

        memory_user_text = f"[Operator instruction] {instructions}"
        session.memory_manager.prerecord_user(memory_user_text)

        with session.lock:
            session.last_activity = time.monotonic()
            # Operator-originated messages skip rate-limit/blacklist (trusted) but
            # still run injection detection and encoding checks (#1076).
            if not self._check_guardrails(synthetic_msg, session, channel, skip_trusted=True):
                return "[Operator instruction blocked — injection or encoding detected]", None
            context, combined_prefix = self._prepare_context(synthetic_msg, session)

            effective_prompt, user_input, history, wf_excluded, wf_approved = (
                self._prepare_agent_call(synthetic_msg, context, session=session)
            )

            active_tools: list[Any] = list(self._active_tools)
            # Inject suppress_reply so the agent can cleanly decline without
            # leaking "suppress_reply" or "(No reply...)" into the sent text.
            outbound_suppress_state = SuppressReplyState()
            if "suppress_reply" not in self._excluded_tools:
                active_tools.append(create_suppress_reply_tool(outbound_suppress_state))

            response, turn_loaded_tools = self._run_agent(
                user_input=user_input,
                history_messages=history,
                context_prefix=combined_prefix,
                effective_prompt=effective_prompt,
                active_tools=active_tools,
                session=session,
                extra_excluded=wf_excluded or None,
                extra_approvals=wf_approved or None,
            )

            if turn_loaded_tools:
                log.debug(
                    "Outbound session %s: tools auto-loaded: %s",
                    session.session_key,
                    turn_loaded_tools,
                )

            if outbound_suppress_state.was_called:
                log.info(
                    "Agent suppressed outbound reply for %s (contact: %s)",
                    chat_id,
                    contact_name,
                )
                session.memory_manager.update(
                    f"[Operator instruction] {instructions}",
                    "[Outbound suppressed by agent]",
                )
                try:
                    session.memory_manager.save()
                except Exception as exc:
                    log.warning(
                        "Failed to save memory for suppressed outbound %s: %s",
                        session.session_key,
                        exc,
                    )
                return "[Outbound suppressed by agent]", None

            response = self._prepare_outbound_text(response)
            if len(response) > self._max_response_length:
                response = response[: self._max_response_length - 3] + "..."

            message_id: str | None = None
            result = channel.send(chat_id, response)
            if result.ok and result.message_id:
                session.last_sent_message_id = result.message_id
                message_id = result.message_id
            elif not result.ok:
                log.warning(
                    "Failed to send outbound message to %s via %s: %s",
                    chat_id,
                    channel.name,
                    result.error,
                )

            try:
                session.memory_manager.update(memory_user_text, response)
                session.memory_manager.save()
            except Exception as exc:
                log.warning(
                    "Failed to update memory for outbound session %s: %s",
                    session.session_key,
                    exc,
                    exc_info=True,
                )

        return response, message_id

    def simulate(
        self,
        *,
        channel_name: str,
        chat_id: str,
        message: str,
        direction: str = "inbound",
        instructions: str | None = None,
        sender_id: str = "simulator",
        sender_name: str = "Simulator",
        persist: bool = False,
    ) -> SimulateResult:
        """Run the full agent pipeline without delivering a message.

        Intended for testing system-prompt behaviour, workflow responses, and
        guardrail reactions without touching a live channel.

        Args:
            channel_name: Logical channel name (e.g. 'whatsapp', 'telegram').
            chat_id: Chat identifier on the channel.
            message: User message text (inbound) or context text (outbound).
            direction: 'inbound' (user → agent) or 'outbound' (operator → agent).
            instructions: Operator instructions for outbound simulation.
                          Falls back to *message* when absent.
            sender_id: Sender identifier inserted into the synthetic message.
            sender_name: Human-readable sender name.
            persist: When True, update and save session memory after the turn.

        Returns:
            SimulateResult with the agent response and pipeline metadata.
        """
        t_start = time.monotonic()

        if direction == "outbound":
            contact_label = sender_name
            if instructions and message:
                task = f"{instructions}\nOpening line to use: {message}"
            else:
                task = instructions or message
            framed_text = (
                f"[Operator instruction — initiate conversation with {contact_label}]\n{task}"
            )
            effective_sender_id = "operator"
            effective_sender_name = "Operator"
        else:
            framed_text = message
            effective_sender_id = sender_id
            effective_sender_name = sender_name

        synthetic_msg = IncomingMessage(
            channel=channel_name,
            chat_id=chat_id,
            message_id=f"sim-{uuid.uuid4().hex[:12]}",
            sender_id=effective_sender_id,
            sender_name=effective_sender_name,
            text=framed_text,
            timestamp=time.time(),
            metadata={"simulation": True},
        )

        session = self._session_mgr.get_or_create(synthetic_msg)
        with session.lock:
            # Input guardrails — skipped for outbound (mirrors handle_outbound behaviour).
            if direction == "inbound":
                guard_result = self._guardrails.check_input(framed_text, chat_id)
                if not guard_result.is_safe:
                    session.guardrail_violations += 1
                    return SimulateResult(
                        response=_BLOCKED_RESPONSE,
                        suppressed=False,
                        deferred=False,
                        blocked_by_guardrails=True,
                        guardrail_reason=guard_result.reason,
                        duration_ms=(time.monotonic() - t_start) * 1000,
                        memory_persisted=False,
                    )

            session.last_activity = time.monotonic()
            context, combined_prefix = self._prepare_context(synthetic_msg, session)

            schedule_state = ScheduleReplyState()
            edit_state = EditReplyState()
            queue_state = QueueReplyState()
            suppress_state = SuppressReplyState()
            defer_state = DeferReplyState()
            active_tools: list[Any] = list(self._active_tools)

            # Inject the same scheduler tools that handle() injects for inbound
            # turns so the agent can schedule, queue, or suppress replies during
            # a simulation (BUG-240).
            if direction == "inbound":
                if self._scheduler and "schedule_reply" not in self._excluded_tools:
                    active_tools.append(create_schedule_reply_tool(schedule_state))
                if self._scheduler and "queue_reply" not in self._excluded_tools:
                    active_tools.append(create_queue_reply_tool(queue_state))
                if "edit_last_reply" not in self._excluded_tools and session.last_sent_message_id:
                    active_tools.append(create_edit_reply_tool(edit_state))
                if self._scheduler:
                    if "list_scheduled_messages" not in self._excluded_tools:
                        active_tools.append(
                            create_list_scheduled_tool(
                                self._scheduler,
                                self._services_config,
                                caller_chat_id=chat_id,
                            )
                        )
                    if "edit_scheduled_message" not in self._excluded_tools:
                        active_tools.append(
                            create_edit_scheduled_tool(
                                self._scheduler,
                                caller_chat_id=chat_id,
                            )
                        )
                    if "cancel_scheduled_message" not in self._excluded_tools:
                        active_tools.append(
                            create_cancel_scheduled_tool(
                                self._scheduler,
                                caller_chat_id=chat_id,
                            )
                        )

            if "suppress_reply" not in self._excluded_tools:
                active_tools.append(create_suppress_reply_tool(suppress_state))

            if self._deferral_mgr and "defer_processing" not in self._excluded_tools:
                if 0 < self._deferral_mgr.max_depth:
                    active_tools.append(
                        create_defer_processing_tool(defer_state, schedule_state=None)
                    )

            # Wrap tool_call_guard to detect in-graph tool blocks so that
            # blocked_by_guardrails can be set accurately in SimulateResult
            # (BUG-241).
            _tool_guard_blocked = [False]
            _tool_guard_reason: list[str | None] = [None]

            def _tracking_tool_guard(tool_name: str, tool_args: dict[str, Any]) -> Any:
                result = self._guardrails.check_tool_call(tool_name, tool_args)
                if not result.is_safe and not _tool_guard_blocked[0]:
                    _tool_guard_blocked[0] = True
                    _tool_guard_reason[0] = result.reason
                return result

            effective_prompt, user_input, history, wf_excluded, wf_approved = (
                self._prepare_agent_call(synthetic_msg, context, session=session)
            )

            from src.orchestration.run_config import AgentRunConfig

            call_session_state = SessionState(no_confirm=True)
            available = dict(self._available_tools)
            if wf_excluded:
                available = {k: v for k, v in available.items() if k not in wf_excluded}
            call_approvals = set(self._approvals)
            if wf_approved:
                call_approvals |= wf_approved
            sim_run_config = AgentRunConfig(
                llm=self._llm,
                system_prompt=effective_prompt,
                available_tools=available,
                active_tools_list=active_tools,
                max_context_tokens=self._max_context_tokens,
                compression_llm=self._compression_llm,
                tool_call_guard=_tracking_tool_guard,
                session_state=call_session_state,
                memory_manager=session.memory_manager,
                parallel_tool_execution=self._parallel_tool_execution,
            )
            try:
                response = self._agent_runner(
                    user_input=user_input,
                    history_messages=history,
                    registry=self._registry,
                    approvals=call_approvals,
                    context_prefix=combined_prefix,
                    config=sim_run_config,
                )
            except UserCancelledRun:
                log.info("Simulate agent cancelled for %s", session.session_key)
                raise
            except Exception as exc:
                log.error("Simulate agent error for %s: %s", session.session_key, exc)
                response = "I encountered an error processing your message. Please try again."

            suppressed = suppress_state.was_called
            deferred = defer_state.was_called

            if not suppressed and not deferred:
                if direction == "outbound":
                    response = self._prepare_outbound_text(response)
                else:
                    response = self._guardrails.sanitize_output(response)
                if len(response) > self._max_response_length:
                    response = response[: self._max_response_length - 3] + "..."

            memory_persisted = False
            if persist and not deferred:
                try:
                    if direction == "outbound":
                        user_mem = f"[Operator instruction] {instructions or message}"
                    else:
                        user_mem = message
                    agent_mem = "" if suppressed else response
                    session.memory_manager.update(user_mem, agent_mem)
                    session.memory_manager.save()
                    memory_persisted = True
                except Exception as exc:
                    log.warning(
                        "Simulate: memory persist failed for %s: %s",
                        session.session_key,
                        exc,
                    )

        return SimulateResult(
            response="" if suppressed else response,
            suppressed=suppressed,
            deferred=deferred,
            blocked_by_guardrails=_tool_guard_blocked[0],
            guardrail_reason=_tool_guard_reason[0] if _tool_guard_blocked[0] else None,
            duration_ms=(time.monotonic() - t_start) * 1000,
            memory_persisted=memory_persisted,
        )


@dataclass
class SimulateResult:
    """Result of a MessageHandler.simulate() call."""

    response: str
    """Agent-generated response text (empty when suppressed)."""
    suppressed: bool
    """True when the agent called suppress_reply — no message would have been sent."""
    deferred: bool
    """True when the agent called defer_processing — turn would have been re-queued."""
    blocked_by_guardrails: bool
    """True when the input was blocked by the guardrail pipeline before reaching the agent."""
    guardrail_reason: str | None
    """Human-readable guard reason when blocked_by_guardrails is True, else None."""
    duration_ms: float
    """Wall-clock milliseconds for the full pipeline (LLM call included)."""
    memory_persisted: bool
    """True when persist=True was passed and memory was successfully saved."""
