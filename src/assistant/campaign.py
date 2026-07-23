"""
Campaign management for Cogtrix assistant mode (Level 2 outbound).

A campaign targets one or more contacts with a goal and instructions.
The CampaignManager tracks per-target progress, sends follow-ups when
contacts don't reply, escalates after max attempts, and lets the agent
classify goal completion via the ``report_campaign_outcome`` tool.

Persistence: ``data/assistant/campaigns.json`` via ``atomic_write_json``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.assistant._security_patterns import CONFUSABLE_TRANS, INJECTION_PATTERNS
from src.utils.atomic_write import atomic_write_json

if TYPE_CHECKING:
    pass

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc, assignment]

log = logging.getLogger("cogtrix")

_DEFAULT_CHECK_INTERVAL: float = 60.0
_DEFAULT_MAX_FOLLOW_UPS: int = 3
_DEFAULT_FOLLOW_UP_INTERVAL_HOURS: float = 24.0


def _sanitize_campaign_text(text: str) -> str:
    """Strip prompt-injection patterns from campaign text before LLM submission.

    This is a defense-in-depth measure for campaign ``goal`` and
    ``instructions`` fields that originate from API input and are
    interpolated verbatim into outbound LLM prompts.  The sanitizer:

    0. Normalizes Unicode (NFKC) and folds Cyrillic/Greek homoglyphs to
       Latin so that confusable characters cannot bypass pattern matching.
    1. Removes known instruction-override phrases (same patterns used by
       the inbound InputGuard, adapted for campaign text).
    2. Escapes back-ticks, XML-like tags, brackets and newlines so the
       injected text cannot break out of the framing delimiters.

    The function is idempotent and preserves benign text.
    """
    if not text:
        return text

    # 0. Normalize Unicode and fold homoglyphs so that e.g. Cyrillic 'і'
    #    is treated as Latin 'i' for pattern matching.
    text = unicodedata.normalize("NFKC", text).translate(CONFUSABLE_TRANS)

    # 1. Strip injection patterns.
    for pattern in INJECTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    # 2. Escape delimiter-breaking characters so the sanitized text
    #    cannot close the framing block or inject new XML tags.
    text = text.replace("`", "'")
    text = text.replace("<", "⟨")
    text = text.replace(">", "⟩")
    text = text.replace("]", "⟩")
    text = text.replace("\n", " ")

    return text


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CampaignTarget:
    """Progress tracking for a single contact within a campaign."""

    contact_name: str
    channel: str
    chat_id: str
    status: str = "pending"  # pending | active | completed | failed | escalated
    follow_ups_sent: int = 0
    last_outbound_at: float | None = None
    last_reply_at: float | None = None
    completion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignTarget:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Campaign:
    """A multi-target outbound campaign."""

    id: str
    name: str
    goal: str
    instructions: str
    targets: list[CampaignTarget]
    max_follow_ups: int = _DEFAULT_MAX_FOLLOW_UPS
    follow_up_interval_hours: float = _DEFAULT_FOLLOW_UP_INTERVAL_HOURS
    status: str = "draft"  # draft | active | paused | completed | cancelled
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["targets"] = [t.to_dict() for t in self.targets]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Campaign:
        targets_raw = data.get("targets", [])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "targets"}
        for float_field in ("follow_up_interval_hours", "created_at", "updated_at"):
            if float_field in known:
                known[float_field] = float(known[float_field])
        for int_field in ("max_follow_ups",):
            if int_field in known:
                known[int_field] = int(known[int_field])
        campaign = cls(**known, targets=[])
        campaign.targets = [CampaignTarget.from_dict(t) for t in targets_raw]
        return campaign

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "cancelled")

    def _check_completion(self) -> None:
        """Auto-complete the campaign if all targets are in a terminal state."""
        if self.status != "active":
            return
        if all(t.status in ("completed", "failed", "escalated") for t in self.targets):
            self.status = "completed"
            self.updated_at = time.time()
            log.info("Campaign %s auto-completed (all targets resolved)", self.id)


# ---------------------------------------------------------------------------
# Tool: report_campaign_outcome
# ---------------------------------------------------------------------------


class _CampaignOutcomeInput(BaseModel):
    """Input schema for the report_campaign_outcome tool."""

    outcome: str = Field(
        ...,
        description=(
            "Campaign outcome classification: 'completed' (goal achieved), "
            "'failed' (contact declined or goal cannot be achieved), "
            "or 'in_progress' (conversation ongoing, goal not yet resolved)."
        ),
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Brief explanation of why this outcome was chosen.",
    )


@dataclass
class CampaignOutcomeState:
    """Mutable state passed into the report_campaign_outcome closure."""

    was_called: bool = False
    outcome: str = ""
    reason: str = ""


def create_campaign_outcome_tool(
    state: CampaignOutcomeState,
    goal: str,
) -> Any:
    """Create a per-call ``report_campaign_outcome`` tool."""
    if StructuredTool is None:
        return None  # pragma: no cover

    _lock = threading.Lock()

    def _report_outcome(outcome: str, reason: str) -> str:
        with _lock:
            if state.was_called:
                return "Campaign outcome already reported for this turn."
            state.was_called = True
            state.outcome = outcome
            state.reason = reason
        return f"Campaign outcome recorded: {outcome} — {reason}"

    return StructuredTool.from_function(
        func=_report_outcome,
        name="report_campaign_outcome",
        description=(
            f"Report the outcome of the current campaign conversation. "
            f'Campaign goal: "{goal}". '
            f"Call this when you can classify the conversation outcome as "
            f"'completed' (goal achieved), 'failed' (contact declined), or "
            f"'in_progress' (not yet resolved). Only call once per turn."
        ),
        args_schema=_CampaignOutcomeInput,
    )


# ---------------------------------------------------------------------------
# CampaignManager
# ---------------------------------------------------------------------------


class CampaignManager:
    """Thread-safe campaign lifecycle manager with background follow-up scheduling.

    Args:
        persist_path: Path to the JSON persistence file.
        check_interval: Seconds between follow-up check passes.
    """

    def __init__(
        self,
        persist_path: Path | str,
        *,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
    ) -> None:
        self._persist_path = Path(persist_path)
        self._check_interval = check_interval
        self._campaigns: dict[str, Campaign] = {}
        self._lock = threading.RLock()
        self._handler: Any = None  # Set via set_handler()
        self._channels: dict[str, Any] = {}  # Set via set_channels()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._load()

    # -- Dependency injection (break circular refs) -------------------------

    def set_handler(self, handler: Any) -> None:
        self._handler = handler

    def set_channels(self, channels: dict[str, Any]) -> None:
        self._channels = channels

    # -- Persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for cdata in raw:
                campaign = Campaign.from_dict(cdata)
                self._campaigns[campaign.id] = campaign
            log.info("Loaded %d campaign(s) from %s", len(self._campaigns), self._persist_path)
        except Exception as exc:
            log.warning("Failed to load campaigns from %s: %s", self._persist_path, exc)

    def save(self) -> None:
        """Persist all campaigns to disk."""
        with self._lock:
            data = [c.to_dict() for c in self._campaigns.values()]
        try:
            with atomic_write_json(self._persist_path) as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("Failed to save campaigns: %s", exc)

    # -- CRUD ---------------------------------------------------------------

    def create(self, campaign: Campaign) -> Campaign:
        with self._lock:
            self._campaigns[campaign.id] = campaign
        self.save()
        return campaign

    def get(self, campaign_id: str) -> Campaign | None:
        with self._lock:
            return self._campaigns.get(campaign_id)

    def list_all(self, *, status_filter: str | None = None) -> list[Campaign]:
        with self._lock:
            campaigns = list(self._campaigns.values())
        if status_filter:
            campaigns = [c for c in campaigns if c.status == status_filter]
        return sorted(campaigns, key=lambda c: c.created_at, reverse=True)

    def update(self, campaign_id: str, **kwargs: Any) -> Campaign | None:
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                return None
            for key, value in kwargs.items():
                if key in campaign.__dataclass_fields__ and key not in (
                    "id",
                    "created_at",
                    "targets",
                ):
                    setattr(campaign, key, value)
            campaign.updated_at = time.time()
        self.save()
        return campaign

    def delete(self, campaign_id: str) -> bool:
        with self._lock:
            if campaign_id not in self._campaigns:
                return False
            del self._campaigns[campaign_id]
        self.save()
        return True

    # -- Launch & follow-up -------------------------------------------------

    def launch(self, campaign_id: str) -> dict[str, str]:
        """Send initial outbound to all pending targets. Returns {contact: status}."""
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                return {}
            if campaign.status not in ("draft", "paused"):
                return {}
            campaign.status = "active"
            campaign.updated_at = time.time()
            targets = list(campaign.targets)
            instructions = _sanitize_campaign_text(campaign.instructions)
            goal = _sanitize_campaign_text(campaign.goal)

        results: dict[str, str] = {}
        futures: list[tuple[str, Any]] = []  # (contact_name, future)

        for target in targets:
            if target.status not in ("pending",):
                results[target.contact_name] = f"skipped ({target.status})"
                continue

            channel_obj = self._channels.get(target.channel)
            if channel_obj is None:
                results[target.contact_name] = "channel_not_available"
                continue

            if self._handler is None:
                results[target.contact_name] = "handler_not_available"
                continue

            # Set target active BEFORE sending so on_reply() can match
            # incoming replies that arrive during the send window.
            with self._lock:
                target.status = "active"

            def _do_send(
                _contact: str,
                _channel: Any,
                _chat_id: str,
                _instructions: str,
                _t: CampaignTarget,
            ) -> tuple[str, str]:
                try:
                    framed = f"[Campaign outbound — goal: {goal}]\n{_instructions}"
                    _response, msg_id = self._handler.handle_outbound(
                        contact_name=_contact,
                        instructions=framed,
                        channel=_channel,
                        chat_id=_chat_id,
                    )
                    with self._lock:
                        _t.last_outbound_at = time.time()
                    return (_contact, "sent" if msg_id else "send_failed")
                except Exception as exc:
                    log.warning("Failed to send outbound to %s: %s", _contact, exc)
                    with self._lock:
                        _t.status = "pending"
                    return (_contact, f"error: {exc}")

            if self._executor is not None:
                fut = self._executor.submit(
                    _do_send, target.contact_name, channel_obj, target.chat_id, instructions, target
                )
                futures.append((target.contact_name, fut))
            else:
                # Fallback: executor not started (campaign manager idle) — run synchronously
                contact_name, status = _do_send(
                    target.contact_name, channel_obj, target.chat_id, instructions, target
                )
                results[contact_name] = status

        # Wait for all async sends to complete
        for contact_name, fut in futures:
            try:
                _, status = fut.result()
                results[contact_name] = status
            except Exception as exc:
                results[contact_name] = f"error: {exc}"

        self.save()
        return results

    def on_reply(self, channel: str, chat_id: str) -> Campaign | None:
        """Notify that a reply was received from a contact.

        Updates target ``last_reply_at`` and returns the campaign if found.
        """
        found: Campaign | None = None
        with self._lock:
            for campaign in self._campaigns.values():
                if campaign.status != "active":
                    continue
                for target in campaign.targets:
                    if (
                        target.channel == channel
                        and target.chat_id == chat_id
                        and target.status == "active"
                    ):
                        target.last_reply_at = time.time()
                        found = campaign
                        break
                if found is not None:
                    break
        if found is not None:
            self.save()
        return found

    def get_active_campaign_for_chat(
        self, channel: str, chat_id: str
    ) -> tuple[Campaign, CampaignTarget] | None:
        """Return the active campaign and target for a chat, if any."""
        with self._lock:
            for campaign in self._campaigns.values():
                if campaign.status != "active":
                    continue
                for target in campaign.targets:
                    if (
                        target.channel == channel
                        and target.chat_id == chat_id
                        and target.status == "active"
                    ):
                        return campaign, target
        return None

    def mark_target_outcome(
        self,
        campaign_id: str,
        chat_id: str,
        outcome: str,
        reason: str,
    ) -> None:
        """Mark a target's outcome from agent classification."""
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                return
            for target in campaign.targets:
                if target.chat_id == chat_id and target.status == "active":
                    if outcome == "completed":
                        target.status = "completed"
                        target.completion_reason = reason
                        log.info(
                            "Campaign %s target %s completed: %s",
                            campaign_id,
                            target.contact_name,
                            reason,
                        )
                    elif outcome == "failed":
                        target.status = "failed"
                        target.completion_reason = reason
                        log.info(
                            "Campaign %s target %s failed: %s",
                            campaign_id,
                            target.contact_name,
                            reason,
                        )
                    # "in_progress" — no status change
                    campaign._check_completion()
                    break
        self.save()

    # -- Background follow-up thread ----------------------------------------

    def start(self) -> None:
        """Start the background follow-up check thread."""
        with self._lock:
            if self._thread is not None:
                return
            if self._handler is None:
                raise RuntimeError(
                    "CampaignManager.start() called before set_handler(); wire dependencies first"
                )
            self._stop_event.clear()
            self._executor = ThreadPoolExecutor(
                thread_name_prefix="campaign-outbound",
            )
            self._thread = threading.Thread(
                target=self._follow_up_loop,
                daemon=True,
                name="campaign-follow-up",
            )
            self._thread.start()
        log.info("CampaignManager started (check interval %.0fs)", self._check_interval)

    def stop(self) -> None:
        """Signal the follow-up thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        log.info("CampaignManager stopped")

    def _follow_up_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_follow_ups()
            except Exception as exc:
                log.warning("Campaign follow-up check failed: %s", exc, exc_info=True)
            self._stop_event.wait(self._check_interval)

    def _process_follow_ups(self) -> None:
        """Check all active campaigns for targets needing follow-up or escalation."""
        now = time.time()

        with self._lock:
            active_campaigns = [c for c in self._campaigns.values() if c.status == "active"]

        for campaign in active_campaigns:
            interval_secs = campaign.follow_up_interval_hours * 3600

            for target in campaign.targets:
                if target.status != "active":
                    continue
                if target.last_outbound_at is None:
                    continue

                time_since_outbound = now - target.last_outbound_at
                # Only follow up if enough time has passed AND no reply since last outbound
                if time_since_outbound < interval_secs:
                    continue
                if (
                    target.last_reply_at is not None
                    and target.last_reply_at > target.last_outbound_at
                ):
                    continue

                # Check escalation threshold — re-check under lock to avoid
                # racing with a concurrent on_reply / mark_target_outcome.
                if target.follow_ups_sent >= campaign.max_follow_ups:
                    with self._lock:
                        if target.status != "active":
                            continue
                        target.status = "escalated"
                        target.completion_reason = (
                            f"No reply after {target.follow_ups_sent} follow-ups"
                        )
                        campaign._check_completion()
                    self.save()
                    log.info(
                        "Campaign %s target %s escalated after %d follow-ups",
                        campaign.id,
                        target.contact_name,
                        target.follow_ups_sent,
                    )
                    continue

                # Send follow-up
                self._send_follow_up(campaign, target)

    def _send_follow_up(self, campaign: Campaign, target: CampaignTarget) -> None:
        """Send a follow-up message for a target that hasn't replied.

        Submits work to the shared ThreadPoolExecutor so the follow-up dispatch
        loop is not blocked by slow LLM calls. Matches the executor pattern used
        by DeferralManager._reprocess_callback in service.py.
        """
        channel_obj = self._channels.get(target.channel)
        if channel_obj is None or self._handler is None:
            return

        follow_up_num = target.follow_ups_sent + 1
        safe_goal = _sanitize_campaign_text(campaign.goal)
        safe_instructions = _sanitize_campaign_text(campaign.instructions)
        framed = (
            f"[Campaign follow-up #{follow_up_num} — goal: {safe_goal}]\n"
            f"The contact has not replied to your previous message. "
            f"Send an appropriate follow-up based on the conversation history "
            f"and the original instructions: {safe_instructions}"
        )
        if self._executor is not None:
            self._executor.submit(
                lambda c=campaign, t=target, ch=channel_obj, f=framed, n=follow_up_num: (
                    self._do_follow_up(c, t, ch, f, n)
                ),
            )
        else:
            # Fallback: executor not started (e.g. test calling _process_follow_ups
            # directly without start()) — run synchronously so existing callers are
            # not broken.  Matches the fallback pattern used in launch().
            self._do_follow_up(campaign, target, channel_obj, framed, follow_up_num)

    def _do_follow_up(
        self,
        campaign: Campaign,
        target: CampaignTarget,
        channel_obj: Any,
        framed: str,
        follow_up_num: int,
    ) -> None:
        """Work function for executor — performs the actual handle_outbound call."""
        try:
            # Re-check target status under lock to close the TOCTOU window
            # between _process_follow_ups eligibility check and dispatch.
            with self._lock:
                if target.status != "active":
                    log.info(
                        "Campaign %s: follow-up #%d to %s suppressed — target status is '%s'",
                        campaign.id,
                        follow_up_num,
                        target.contact_name,
                        target.status,
                    )
                    return
            _resp, msg_id = self._handler.handle_outbound(
                contact_name=target.contact_name,
                instructions=framed,
                channel=channel_obj,
                chat_id=target.chat_id,
            )
            with self._lock:
                target.follow_ups_sent += 1
                target.last_outbound_at = time.time()
            self.save()
            log.info(
                "Campaign %s: sent follow-up #%d to %s (msg_id=%s)",
                campaign.id,
                follow_up_num,
                target.contact_name,
                msg_id,
            )
        except Exception as exc:
            log.warning(
                "Campaign %s: failed to send follow-up to %s: %s",
                campaign.id,
                target.contact_name,
                exc,
            )
