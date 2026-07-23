"""Tests for cogtrix_core/api/schemas/assistant.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.assistant import (
    AssistantStartRequest,
    AssistantStatusOut,
    CampaignCreateRequest,
    CampaignOut,
    CampaignTargetIn,
    CampaignTargetOut,
    CampaignUpdateRequest,
    ChannelStatusOut,
    ChatSessionOut,
    ContactOut,
    DeferredRecordOut,
    GuardrailStatusOut,
    KnowledgeFactOut,
    KnowledgeSearchRequest,
    OutboundRequest,
    OutboundResponse,
    ScheduledMessageEditRequest,
    ScheduledMessageOut,
    SimulateOut,
    SimulateRequest,
    ViolationRecordOut,
)


class TestChannelStatusOut:
    """ChannelStatusOut schema construction and validation."""

    def test_channel_status_out_valid(self) -> None:
        """Valid input constructs without error."""
        ch = ChannelStatusOut(
            name="whatsapp",
            type="whatsapp",
            enabled=True,
            connected=True,
            active_chats=3,
            poll_interval_s=5.0,
        )
        assert ch.name == "whatsapp"
        assert ch.error is None

    def test_channel_status_out_with_error(self) -> None:
        """Channel with error message."""
        ch = ChannelStatusOut(
            name="whatsapp",
            type="whatsapp",
            enabled=True,
            connected=False,
            active_chats=0,
            poll_interval_s=5.0,
            error="Connection refused",
        )
        assert ch.error == "Connection refused"

    def test_channel_status_out_invalid_type(self) -> None:
        """Invalid channel type raises ValidationError."""
        with pytest.raises(ValidationError):
            ChannelStatusOut(
                name="sms",
                type="sms",  # invalid
                enabled=True,
                connected=True,
                active_chats=0,
                poll_interval_s=5.0,
            )


class TestAssistantStatusOut:
    """AssistantStatusOut schema construction and validation."""

    def test_assistant_status_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        status = AssistantStatusOut(
            status="running",
            channels=[
                ChannelStatusOut(
                    name="whatsapp",
                    type="whatsapp",
                    enabled=True,
                    connected=True,
                    active_chats=3,
                    poll_interval_s=5.0,
                )
            ],
            started_at=now,
            uptime_s=3600.0,
        )
        assert status.status == "running"
        assert len(status.channels) == 1
        assert status.uptime_s == 3600.0

    def test_assistant_status_out_empty_channels(self) -> None:
        """Empty channels list is valid."""
        status = AssistantStatusOut(status="stopped")
        assert status.channels == []
        assert status.started_at is None
        assert status.uptime_s is None

    def test_assistant_status_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            AssistantStatusOut(status="offline")


class TestAssistantStartRequest:
    """AssistantStartRequest schema construction and validation."""

    def test_assistant_start_request_default(self) -> None:
        """Default force_restart is False."""
        req = AssistantStartRequest()
        assert req.force_restart is False

    def test_assistant_start_request_force_restart(self) -> None:
        """force_restart can be set to True."""
        req = AssistantStartRequest(force_restart=True)
        assert req.force_restart is True


class TestChatSessionOut:
    """ChatSessionOut schema construction and validation."""

    def test_chat_session_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        session = ChatSessionOut(
            session_key="whatsapp::+1234567890",
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            display_name="Alice",
            message_count=42,
            last_activity=now,
            memory_mode="conversation",
            is_locked=False,
        )
        assert session.session_key == "whatsapp::+1234567890"
        assert session.display_name == "Alice"
        assert session.is_locked is False

    def test_chat_session_out_null_display_name(self) -> None:
        """Null display name is valid."""
        session = ChatSessionOut(
            session_key="whatsapp::+1234567890",
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            display_name=None,
            message_count=0,
            last_activity=None,
            memory_mode="conversation",
            is_locked=False,
        )
        assert session.display_name is None

    def test_chat_session_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatSessionOut(
                session_key="whatsapp::+1234567890",
                channel="whatsapp",
                chat_id="+1234567890@c.us",
                display_name=None,
                message_count=0,
                last_activity=None,
                memory_mode="conversation",
                # is_locked missing
            )


class TestScheduledMessageOut:
    """ScheduledMessageOut schema construction and validation."""

    def test_scheduled_message_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        msg = ScheduledMessageOut(
            id="msg-123",
            chat_id="+1234567890",
            channel="whatsapp",
            recipient="Alice",
            text="Hello!",
            send_at=now,
            created_at=now,
            attempts=0,
            max_attempts=3,
            status="pending",
        )
        assert msg.text == "Hello!"
        assert msg.attempts == 0

    def test_scheduled_message_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        msg = ScheduledMessageOut(
            id="msg-123",
            chat_id="+1234567890",
            channel="whatsapp",
            recipient="Alice",
            text="Hello!",
            send_at=naive,
            created_at=naive,
            attempts=0,
            max_attempts=3,
            status="pending",
        )
        assert msg.send_at.tzinfo is not None
        assert msg.created_at.tzinfo is not None

    def test_scheduled_message_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ScheduledMessageOut(
                id="msg-123",
                chat_id="+1234567890",
                channel="whatsapp",
                recipient="Alice",
                text="Hello!",
                send_at=now,
                created_at=now,
                attempts=0,
                max_attempts=3,
                status="unknown",
            )


class TestScheduledMessageEditRequest:
    """ScheduledMessageEditRequest schema construction and validation."""

    def test_scheduled_message_edit_request_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        req = ScheduledMessageEditRequest(text="Updated text", send_at=now)
        assert req.text == "Updated text"

    def test_scheduled_message_edit_request_empty(self) -> None:
        """All fields can be omitted."""
        req = ScheduledMessageEditRequest()
        assert req.text is None
        assert req.send_at is None

    def test_scheduled_message_edit_request_text_too_long(self) -> None:
        """Text over 4096 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            ScheduledMessageEditRequest(text="x" * 4097)


class TestDeferredRecordOut:
    """DeferredRecordOut schema construction and validation."""

    def test_deferred_record_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        record = DeferredRecordOut(
            session_key="whatsapp::+1234567890",
            fire_at=now,
            pending_messages=["Hello", "World"],
            depth=1,
            max_depth=3,
            status="pending",
            created_at=now,
        )
        assert record.depth == 1
        assert record.pending_messages == ["Hello", "World"]

    def test_deferred_record_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        record = DeferredRecordOut(
            session_key="whatsapp::+1234567890",
            fire_at=naive,
            pending_messages=["Hello"],
            depth=1,
            max_depth=3,
            status="pending",
            created_at=naive,
        )
        assert record.fire_at.tzinfo is not None

    def test_deferred_record_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DeferredRecordOut(
                session_key="whatsapp::+1234567890",
                fire_at=now,
                pending_messages=["Hello"],
                depth=1,
                max_depth=3,
                status="done",
                created_at=now,
            )


class TestContactOut:
    """ContactOut schema construction and validation."""

    def test_contact_out_valid(self) -> None:
        """Valid input constructs without error."""
        contact = ContactOut(
            name="Alice",
            identifiers=["+1234567890", "alice_tg"],
            channels=["whatsapp", "telegram"],
            prompt="Be helpful to Alice.",
            filter_mode="allow",
        )
        assert contact.name == "Alice"
        assert contact.identifiers == ["+1234567890", "alice_tg"]
        assert contact.filter_mode == "allow"

    def test_contact_out_defaults(self) -> None:
        """Defaults for optional fields."""
        contact = ContactOut(
            name="Bob",
            identifiers=["+9876543210"],
        )
        assert contact.channels == []
        assert contact.prompt is None
        assert contact.filter_mode is None

    def test_contact_out_invalid_filter_mode(self) -> None:
        """Invalid filter_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            ContactOut(
                name="Alice",
                identifiers=["+1234567890"],
                filter_mode="invalid",
            )


class TestViolationRecordOut:
    """ViolationRecordOut schema construction and validation."""

    def test_violation_record_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        record = ViolationRecordOut(
            chat_id="+1234567890",
            channel="whatsapp",
            violation_type="input",
            detail="Blocked inappropriate content.",
            timestamp=now,
        )
        assert record.violation_type == "input"

    def test_violation_record_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ViolationRecordOut(
                chat_id="+1234567890",
                channel="whatsapp",
                violation_type="input",
                detail="Blocked.",
                # timestamp missing
            )


class TestGuardrailStatusOut:
    """GuardrailStatusOut schema construction and validation."""

    def test_guardrail_status_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        status = GuardrailStatusOut(
            blacklisted_chats=["+1234567890"],
            total_violations=12,
            recent_violations=[
                ViolationRecordOut(
                    chat_id="+1234567890",
                    channel="whatsapp",
                    violation_type="rate_limit",
                    detail="Too many requests.",
                    timestamp=now,
                )
            ],
        )
        assert status.total_violations == 12
        assert len(status.recent_violations) == 1

    def test_guardrail_status_out_empty_violations(self) -> None:
        """Empty violations list is valid."""
        status = GuardrailStatusOut(
            blacklisted_chats=[],
            total_violations=0,
        )
        assert status.recent_violations == []


class TestKnowledgeFactOut:
    """KnowledgeFactOut schema construction and validation."""

    def test_knowledge_fact_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        fact = KnowledgeFactOut(
            id="fact-123",
            text="Alice prefers tea.",
            source_chat="whatsapp::+1234567890",
            source_channel="whatsapp",
            created_at=now,
            relevance_score=0.87,
        )
        assert fact.text == "Alice prefers tea."
        assert fact.relevance_score == 0.87

    def test_knowledge_fact_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        fact = KnowledgeFactOut(
            id="fact-123",
            text="Test.",
            created_at=naive,
        )
        assert fact.created_at.tzinfo is not None

    def test_knowledge_fact_out_defaults(self) -> None:
        """Defaults for optional fields."""
        now = datetime.now(UTC)
        fact = KnowledgeFactOut(
            id="fact-123",
            text="Test.",
            created_at=now,
        )
        assert fact.source_chat is None
        assert fact.source_channel is None
        assert fact.relevance_score is None


class TestKnowledgeSearchRequest:
    """KnowledgeSearchRequest schema construction and validation."""

    def test_knowledge_search_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = KnowledgeSearchRequest(query="What does Alice prefer?")
        assert req.query == "What does Alice prefer?"
        assert req.top_k == 10

    def test_knowledge_search_request_custom_top_k(self) -> None:
        """Custom top_k is valid."""
        req = KnowledgeSearchRequest(query="test", top_k=50)
        assert req.top_k == 50

    def test_knowledge_search_request_empty_query(self) -> None:
        """Empty query raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="")

    def test_knowledge_search_request_query_too_long(self) -> None:
        """Query over 1024 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="x" * 1025)

    def test_knowledge_search_request_top_k_too_low(self) -> None:
        """top_k below 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="test", top_k=0)

    def test_knowledge_search_request_top_k_too_high(self) -> None:
        """top_k above 100 raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSearchRequest(query="test", top_k=101)


class TestSimulateRequest:
    """SimulateRequest schema construction and validation."""

    def test_simulate_request_valid_inbound(self) -> None:
        """Valid inbound simulation request constructs without error."""
        req = SimulateRequest(
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            message="Hello!",
        )
        assert req.direction == "inbound"
        assert req.persist is False

    def test_simulate_request_valid_outbound(self) -> None:
        """Valid outbound simulation request constructs without error."""
        req = SimulateRequest(
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            message="Context text.",
            direction="outbound",
            instructions="Greet the user.",
        )
        assert req.direction == "outbound"

    def test_simulate_request_invalid_direction(self) -> None:
        """Invalid direction raises ValidationError."""
        with pytest.raises(ValidationError):
            SimulateRequest(
                channel="whatsapp",
                chat_id="+1234567890@c.us",
                message="Hello!",
                direction="sideways",
            )

    def test_simulate_request_empty_message(self) -> None:
        """Empty message raises ValidationError."""
        with pytest.raises(ValidationError):
            SimulateRequest(
                channel="whatsapp",
                chat_id="+1234567890@c.us",
                message="",
            )

    def test_simulate_request_message_too_long(self) -> None:
        """Message over 8192 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            SimulateRequest(
                channel="whatsapp",
                chat_id="+1234567890@c.us",
                message="x" * 8193,
            )


class TestSimulateOut:
    """SimulateOut schema construction and validation."""

    def test_simulate_out_valid(self) -> None:
        """Valid input constructs without error."""
        out = SimulateOut(
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            session_key="whatsapp::+1234567890@c.us",
            direction="inbound",
            response="Hello! How can I help?",
            suppressed=False,
            deferred=False,
            blocked_by_guardrails=False,
            duration_ms=1234.5,
            memory_persisted=True,
        )
        assert out.response == "Hello! How can I help?"
        assert out.guardrail_reason is None

    def test_simulate_out_guardrail_blocked(self) -> None:
        """Guardrail blocked with reason."""
        out = SimulateOut(
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            session_key="whatsapp::+1234567890@c.us",
            direction="inbound",
            response="",
            suppressed=False,
            deferred=False,
            blocked_by_guardrails=True,
            guardrail_reason="Inappropriate content",
            duration_ms=100.0,
            memory_persisted=False,
        )
        assert out.blocked_by_guardrails is True
        assert out.guardrail_reason == "Inappropriate content"

    def test_simulate_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            SimulateOut(
                channel="whatsapp",
                chat_id="+1234567890@c.us",
                session_key="whatsapp::+1234567890@c.us",
                direction="inbound",
                response="",
                suppressed=False,
                deferred=False,
                blocked_by_guardrails=False,
                # duration_ms missing
                memory_persisted=False,
            )


class TestOutboundRequest:
    """OutboundRequest schema construction and validation."""

    def test_outbound_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = OutboundRequest(
            contact_name="Alice",
            instructions="Ask about availability.",
        )
        assert req.contact_name == "Alice"
        assert req.channel is None

    def test_outbound_request_with_channel(self) -> None:
        """Outbound request with preferred channel."""
        req = OutboundRequest(
            contact_name="Alice",
            instructions="Ask about availability.",
            channel="whatsapp",
        )
        assert req.channel == "whatsapp"

    def test_outbound_request_empty_contact_name(self) -> None:
        """Empty contact_name raises ValidationError."""
        with pytest.raises(ValidationError):
            OutboundRequest(contact_name="", instructions="Ask.")

    def test_outbound_request_empty_instructions(self) -> None:
        """Empty instructions raises ValidationError."""
        with pytest.raises(ValidationError):
            OutboundRequest(contact_name="Alice", instructions="")


class TestOutboundResponse:
    """OutboundResponse schema construction and validation."""

    def test_outbound_response_valid(self) -> None:
        """Valid input constructs without error."""
        resp = OutboundResponse(
            session_key="whatsapp::+1234567890@c.us",
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            contact_name="Alice",
            response_text="Hi Alice, are you available?",
            message_id="msg-123",
        )
        assert resp.response_text == "Hi Alice, are you available?"
        assert resp.message_id == "msg-123"

    def test_outbound_response_null_message_id(self) -> None:
        """Null message_id is valid."""
        resp = OutboundResponse(
            session_key="whatsapp::+1234567890@c.us",
            channel="whatsapp",
            chat_id="+1234567890@c.us",
            contact_name="Alice",
            response_text="Message sent.",
        )
        assert resp.message_id is None


class TestCampaignTargetIn:
    """CampaignTargetIn schema construction and validation."""

    def test_campaign_target_in_valid(self) -> None:
        """Valid input constructs without error."""
        target = CampaignTargetIn(contact_name="Alice", channel="whatsapp")
        assert target.contact_name == "Alice"
        assert target.channel == "whatsapp"

    def test_campaign_target_in_no_channel(self) -> None:
        """Channel is optional."""
        target = CampaignTargetIn(contact_name="Alice")
        assert target.channel is None

    def test_campaign_target_in_empty_name(self) -> None:
        """Empty contact_name raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignTargetIn(contact_name="")


class TestCampaignTargetOut:
    """CampaignTargetOut schema construction and validation."""

    def test_campaign_target_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        target = CampaignTargetOut(
            contact_name="Alice",
            channel="whatsapp",
            chat_id="+123@c.us",
            status="pending",
            follow_ups_sent=0,
            last_outbound_at=now,
            last_reply_at=None,
            completion_reason=None,
        )
        assert target.status == "pending"

    def test_campaign_target_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignTargetOut(
                contact_name="Alice",
                channel="whatsapp",
                chat_id="+123@c.us",
                status="unknown",
                follow_ups_sent=0,
            )


class TestCampaignCreateRequest:
    """CampaignCreateRequest schema construction and validation."""

    def test_campaign_create_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = CampaignCreateRequest(
            name="Spring sale",
            goal="Schedule meetings.",
            instructions="Reach out about sale.",
            targets=[CampaignTargetIn(contact_name="Alice")],
        )
        assert req.name == "Spring sale"
        assert req.max_follow_ups == 3
        assert req.follow_up_interval_hours == 24.0
        assert req.auto_launch is False

    def test_campaign_create_request_custom_values(self) -> None:
        """Custom max_follow_ups and interval."""
        req = CampaignCreateRequest(
            name="Spring sale",
            goal="Schedule meetings.",
            instructions="Reach out.",
            targets=[CampaignTargetIn(contact_name="Alice")],
            max_follow_ups=5,
            follow_up_interval_hours=48.0,
            auto_launch=True,
        )
        assert req.max_follow_ups == 5
        assert req.auto_launch is True

    def test_campaign_create_request_empty_name(self) -> None:
        """Empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignCreateRequest(
                name="",
                goal="Schedule meetings.",
                instructions="Reach out.",
                targets=[CampaignTargetIn(contact_name="Alice")],
            )

    def test_campaign_create_request_empty_targets(self) -> None:
        """Empty targets list raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignCreateRequest(
                name="Spring sale",
                goal="Schedule meetings.",
                instructions="Reach out.",
                targets=[],
            )

    def test_campaign_create_request_too_many_targets(self) -> None:
        """Over 100 targets raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignCreateRequest(
                name="Spring sale",
                goal="Schedule meetings.",
                instructions="Reach out.",
                targets=[CampaignTargetIn(contact_name=f"Person{i}") for i in range(101)],
            )

    def test_campaign_create_request_max_follow_ups_too_high(self) -> None:
        """max_follow_ups over 20 raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignCreateRequest(
                name="Spring sale",
                goal="Schedule meetings.",
                instructions="Reach out.",
                targets=[CampaignTargetIn(contact_name="Alice")],
                max_follow_ups=21,
            )

    def test_campaign_create_request_interval_too_low(self) -> None:
        """follow_up_interval_hours below 0.5 raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignCreateRequest(
                name="Spring sale",
                goal="Schedule meetings.",
                instructions="Reach out.",
                targets=[CampaignTargetIn(contact_name="Alice")],
                follow_up_interval_hours=0.4,
            )


class TestCampaignUpdateRequest:
    """CampaignUpdateRequest schema construction and validation."""

    def test_campaign_update_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = CampaignUpdateRequest(
            name="New name",
            status="paused",
            max_follow_ups=2,
        )
        assert req.name == "New name"
        assert req.status == "paused"

    def test_campaign_update_request_empty(self) -> None:
        """All fields can be omitted."""
        req = CampaignUpdateRequest()
        assert req.name is None
        assert req.status is None

    def test_campaign_update_request_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            CampaignUpdateRequest(status="unknown")


class TestCampaignOut:
    """CampaignOut schema construction and validation."""

    def test_campaign_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        campaign = CampaignOut(
            id="camp-123",
            name="Spring sale",
            goal="Schedule meetings.",
            instructions="Reach out.",
            targets=[
                CampaignTargetOut(
                    contact_name="Alice",
                    channel="whatsapp",
                    chat_id="+123@c.us",
                    status="pending",
                    follow_ups_sent=0,
                )
            ],
            max_follow_ups=3,
            follow_up_interval_hours=24.0,
            status="draft",
            created_at=now,
            updated_at=now,
        )
        assert campaign.name == "Spring sale"
        assert len(campaign.targets) == 1

    def test_campaign_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached via validator."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        campaign = CampaignOut(
            id="camp-123",
            name="Spring sale",
            goal="Schedule meetings.",
            instructions="Reach out.",
            targets=[],
            max_follow_ups=3,
            follow_up_interval_hours=24.0,
            status="draft",
            created_at=naive,
            updated_at=naive,
        )
        assert campaign.created_at.tzinfo is not None
        assert campaign.updated_at.tzinfo is not None

    def test_campaign_out_timestamp_from_int(self) -> None:
        """Integer timestamp is coerced to datetime."""
        campaign = CampaignOut(
            id="camp-123",
            name="Spring sale",
            goal="Schedule meetings.",
            instructions="Reach out.",
            targets=[],
            max_follow_ups=3,
            follow_up_interval_hours=24.0,
            status="draft",
            created_at=1700000000,
            updated_at=1700000000,
        )
        assert isinstance(campaign.created_at, datetime)
        assert campaign.created_at.year == 2023
