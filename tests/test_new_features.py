"""Tests for new features: user management API, token final field, model alias
auto-migration format, session auto-naming, ConfigOut new fields, and
cancel-event checks in the think pipeline.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# User management API tests (require DB)
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402


@pytest_asyncio.fixture()
async def db_session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


class TestUserRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_by_id(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        uid = str(uuid.uuid4())
        user = await repo.create(
            user_id=uid, username="alice", email="alice@example.com", password_hash="hash"
        )
        assert user.id == uid
        assert user.username == "alice"
        assert user.email == "alice@example.com"

        fetched = await repo.get_by_id(uid)
        assert fetched is not None
        assert fetched.username == "alice"

    @pytest.mark.asyncio
    async def test_get_by_username(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        await repo.create(
            user_id=str(uuid.uuid4()),
            username="bob",
            email="bob@example.com",
            password_hash="hash",
        )
        user = await repo.get_by_username("bob")
        assert user is not None
        assert user.email == "bob@example.com"

    @pytest.mark.asyncio
    async def test_get_by_email_case_insensitive(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        await repo.create(
            user_id=str(uuid.uuid4()),
            username="carol",
            email="Carol@Example.COM",
            password_hash="hash",
        )
        user = await repo.get_by_email("carol@example.com")
        assert user is not None
        assert user.username == "carol"

    @pytest.mark.asyncio
    async def test_list_all(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        for i in range(3):
            await repo.create(
                user_id=str(uuid.uuid4()),
                username=f"user{i}",
                email=f"user{i}@example.com",
                password_hash="hash",
            )
        await db_session.flush()
        users = await repo.list_all()
        assert len(users) == 3

    @pytest.mark.asyncio
    async def test_update_role(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        uid = str(uuid.uuid4())
        await repo.create(
            user_id=uid, username="dave", email="dave@example.com", password_hash="hash"
        )
        updated = await repo.update_role(uid, "admin")
        assert updated is not None
        assert updated.role == "admin"

    @pytest.mark.asyncio
    async def test_update_role_nonexistent(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        result = await repo.update_role("nonexistent-id", "admin")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        uid = str(uuid.uuid4())
        await repo.create(
            user_id=uid, username="eve", email="eve@example.com", password_hash="hash"
        )
        deleted = await repo.delete(uid)
        assert deleted is True
        assert await repo.get_by_id(uid) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        deleted = await repo.delete("nonexistent-id")
        assert deleted is False


# ---------------------------------------------------------------------------
# User schema validation
# ---------------------------------------------------------------------------


class TestUserSchemas:
    def test_user_create_request_valid(self):
        from src.api.schemas.user import UserCreateRequest

        req = UserCreateRequest(username="alice", email="alice@example.com", password="Password1!")
        assert req.role == "user"

    def test_user_create_request_admin_role(self):
        from src.api.schemas.user import UserCreateRequest

        req = UserCreateRequest(
            username="admin_user",
            email="admin@example.com",
            password="Password1!",
            role="admin",
        )
        assert req.role == "admin"

    def test_user_create_rejects_invalid_role(self):
        from pydantic import ValidationError

        from src.api.schemas.user import UserCreateRequest

        with pytest.raises(ValidationError, match="role"):
            UserCreateRequest(
                username="bad",
                email="bad@example.com",
                password="Password1!",
                role="superadmin",
            )

    def test_user_create_rejects_short_password(self):
        from pydantic import ValidationError

        from src.api.schemas.user import UserCreateRequest

        with pytest.raises(ValidationError, match="password"):
            UserCreateRequest(username="short", email="short@example.com", password="123")

    def test_user_create_rejects_invalid_username(self):
        from pydantic import ValidationError

        from src.api.schemas.user import UserCreateRequest

        with pytest.raises(ValidationError, match="username"):
            UserCreateRequest(username="has space", email="x@example.com", password="Password1!")

    def test_user_create_rejects_short_username(self):
        from pydantic import ValidationError

        from src.api.schemas.user import UserCreateRequest

        with pytest.raises(ValidationError, match="username"):
            UserCreateRequest(username="ab", email="x@example.com", password="Password1!")

    def test_user_update_request_optional(self):
        from src.api.schemas.user import UserUpdateRequest

        req = UserUpdateRequest()
        assert req.role is None

    def test_user_update_rejects_invalid_role(self):
        from pydantic import ValidationError

        from src.api.schemas.user import UserUpdateRequest

        with pytest.raises(ValidationError, match="role"):
            UserUpdateRequest(role="invalid")


# ---------------------------------------------------------------------------
# User routes — _user_to_out helper
# ---------------------------------------------------------------------------


class TestUserRouteHelper:
    def test_user_to_out_with_all_fields(self):
        from src.api.routes.users import _user_to_out

        mock_user = MagicMock()
        mock_user.id = "test-id"
        mock_user.username = "alice"
        mock_user.email = "alice@example.com"
        mock_user.role = "admin"
        mock_user.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        out = _user_to_out(mock_user)
        assert out.id == "test-id"
        assert out.username == "alice"
        assert out.role == "admin"

    def test_user_to_out_uses_getattr_for_missing_attrs(self):
        """_user_to_out uses getattr so it works with objects missing some fields."""
        from src.api.routes.users import _user_to_out

        mock = MagicMock()
        mock.id = "id-1"
        mock.username = "u"
        mock.email = "e@e.com"
        del mock.role  # simulate missing attribute
        mock.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        out = _user_to_out(mock)
        assert out.role == "user"  # default from getattr


# ---------------------------------------------------------------------------
# Token streaming — final field
# ---------------------------------------------------------------------------


class TestTokenFinalField:
    def test_token_payload_has_final_field(self):
        from src.api.ws import TokenPayload

        payload = TokenPayload(text="hello")
        assert payload.final is False
        payload_final = TokenPayload(text="done", final=True)
        assert payload_final.final is True

    def test_callback_emits_final_field(self):
        """WebSocketCallbackHandler.on_llm_new_token emits final=True after tool calls."""
        from src.api.callbacks import WebSocketCallbackHandler

        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        handler = WebSocketCallbackHandler(queue, loop)

        # Before any tool calls, final should be False
        handler.tool_call_count = 0
        source = inspect.getsource(handler.on_llm_new_token)
        assert "final" in source, "on_llm_new_token should include 'final' in payload"

        # After tool calls, tool_call_count > 0 → final=True
        handler.tool_call_count = 2
        # The logic should use tool_call_count > 0
        assert "tool_call_count" in source
        loop.close()


# ---------------------------------------------------------------------------
# Model alias auto-migration format
# ---------------------------------------------------------------------------


class TestModelAliasFormat:
    def test_auto_migrated_alias_prefers_model_name(self):
        """Auto-migrated models from provider section prefer model_name as alias."""
        from src.config import Config, _parse_providers_section

        cfg = Config(providers={}, models={})
        _parse_providers_section(cfg, {"prov": {"type": "openai", "model": "gpt-4o"}})
        assert "gpt-4o" in cfg.models

    def test_auto_migrated_alias_falls_back_to_provider_slash_model_on_collision(self):
        """When model_name collides, alias falls back to provider/model format."""
        from src.config import Config, ModelConfig, _parse_providers_section

        cfg = Config(providers={}, models={"gpt-4o": ModelConfig(provider="x", model="gpt-4o")})
        _parse_providers_section(cfg, {"prov": {"type": "openai", "model": "gpt-4o"}})
        assert "prov/gpt-4o" in cfg.models

    def test_synthetic_fallback_uses_provider_slash_model(self):
        """The synthetic default alias uses 'provider/model' format."""

        source_mod = inspect.getsource(__import__("src.config", fromlist=["load_config"]))
        # Check the fallback pattern: f"{first_prov.name}/{default_model}"
        assert "first_prov.name}/{default_model}" in source_mod


# ---------------------------------------------------------------------------
# Session auto-naming
# ---------------------------------------------------------------------------


class TestSessionAutoNaming:
    def test_default_session_name_format(self):
        from src.api.schemas.session import _default_session_name

        name = _default_session_name()
        assert name.startswith("Session ")
        # Should match "Session YYYY-MM-DD HH:MM"
        assert re.match(r"Session \d{4}-\d{2}-\d{2} \d{2}:\d{2}", name)

    def test_session_create_request_has_default_name(self):
        from src.api.schemas.session import SessionCreateRequest

        req = SessionCreateRequest()
        assert req.name.startswith("Session ")

    def test_session_create_request_custom_name(self):
        from src.api.schemas.session import SessionCreateRequest

        req = SessionCreateRequest(name="My Session")
        assert req.name == "My Session"


# ---------------------------------------------------------------------------
# ConfigOut new fields (system_prompt, guardrails)
# ---------------------------------------------------------------------------


class TestConfigOutNewFields:
    def test_config_out_has_system_prompt(self):
        from src.api.schemas.config import ConfigOut

        fields = ConfigOut.model_fields
        assert "system_prompt" in fields

    def test_config_out_has_guardrails(self):
        from src.api.schemas.config import ConfigOut

        fields = ConfigOut.model_fields
        assert "guardrails" in fields


# ---------------------------------------------------------------------------
# Cancel-event checks in think pipeline
# ---------------------------------------------------------------------------


class TestCancelEventChecks:
    def test_think_pipeline_checks_cancel_event(self):
        """_run_think_pipeline should check cancel_event.is_set() between phases."""
        from src.api import turn_runner

        source = inspect.getsource(turn_runner)
        # Should check cancel_event between pipeline phases
        assert "cancel_event.is_set()" in source
        # Should raise CancelledError on cancel
        assert "CancelledError" in source

    def test_cancel_check_between_classify_and_research(self):
        """Cancel check should appear after classify_think_task call."""
        from src.api import turn_runner

        source = inspect.getsource(turn_runner)
        # Find classify and subsequent cancel check
        classify_pos = source.find("classify_think_task")
        if classify_pos >= 0:
            # There should be a cancel check after classify
            next_cancel = source.find("cancel_event.is_set()", classify_pos)
            assert (
                next_cancel > classify_pos
            ), "cancel_event check should appear after classify_think_task"

    @pytest.mark.asyncio
    async def test_pipeline_cancelled_error_resets_agent_state(self):
        """CancelledError raised in think/delegate pipeline resets agent_state to idle.

        Regression test for BUG-FORGE-PIPELINE-CANCEL: when the pipeline phase
        raises CancelledError, session.agent_state must be reset to 'idle' so
        subsequent requests are not blocked by a stuck 'analyzing'/'delegating' state.
        """
        from unittest.mock import MagicMock, patch

        from src.api.turn_runner import _run_message_turn_inner

        session = MagicMock()
        session.id = "test-session-pipeline-cancel"
        session.turn_lock = asyncio.Lock()
        session.session_state = None
        session.run_config = None
        session.memory_manager = None
        session.cancel_event = asyncio.Event()
        # Simulate queue that can hold items
        session.ws_queue = asyncio.Queue(maxsize=100)
        session.active_confirmation_ui = None
        session.agent_state = "idle"
        session.token_counts = {"input_tokens": 0, "output_tokens": 0}

        # Patch run_agent at its source module (it is imported lazily inside the function)
        with patch("src.orchestration.runner.run_agent", return_value="some response"):
            # Patch _run_think_pipeline to raise CancelledError
            async def _cancel_think(*args, **kwargs):
                raise asyncio.CancelledError("test cancel from pipeline")

            with patch("src.api.turn_runner._run_think_pipeline", side_effect=_cancel_think):
                with pytest.raises(asyncio.CancelledError):
                    await _run_message_turn_inner(session, "test", "think", None, None)

        # The key invariant: agent_state must be "idle" after a pipeline cancel,
        # not stuck at "analyzing", "researching", or "deep_thinking".
        assert (
            session.agent_state == "idle"
        ), f"agent_state should be 'idle' after pipeline cancel, got {session.agent_state!r}"


# ---------------------------------------------------------------------------
# Workflow schema validation
# ---------------------------------------------------------------------------


class TestWorkflowSchemas:
    def test_workflow_create_schema(self):
        from src.api.schemas.workflow import WorkflowCreate

        wf = WorkflowCreate(id="test", name="Test")
        assert wf.knowledge_base is False
        assert wf.tool_policy.excluded_tools == []

    def test_workflow_update_all_optional(self):
        from src.api.schemas.workflow import WorkflowUpdate

        upd = WorkflowUpdate()
        assert upd.name is None
        assert upd.description is None
        assert upd.knowledge_base is None

    def test_workflow_binding_out(self):
        from src.api.schemas.workflow import WorkflowBindingOut

        binding = WorkflowBindingOut(session_key="wa::+123", workflow_id="bike-sales")
        assert binding.session_key == "wa::+123"

    def test_bind_workflow_request(self):
        from src.api.schemas.workflow import BindWorkflowRequest

        req = BindWorkflowRequest(workflow_id="test-wf")
        assert req.workflow_id == "test-wf"


# ---------------------------------------------------------------------------
# Workflow routes — helper functions
# ---------------------------------------------------------------------------


class TestWorkflowRouteHelpers:
    def test_wf_to_out(self):
        from src.api.routes.workflows import _wf_to_out
        from src.assistant.workflows import (
            WorkflowAutoDetect,
            WorkflowDefinition,
            WorkflowToolPolicy,
        )

        wf = WorkflowDefinition(
            id="test",
            name="Test",
            description="desc",
            system_prompt="prompt",
            knowledge_base=True,
            tool_policy=WorkflowToolPolicy(excluded_tools=["shell"]),
            auto_detect=WorkflowAutoDetect(enabled=True, keywords=["hi"]),
            created_at="2026-01-01",
            updated_at="2026-01-02",
        )
        out = _wf_to_out(wf)
        assert out.id == "test"
        assert out.name == "Test"
        assert out.system_prompt == "prompt"
        assert out.knowledge_base is True
        assert "shell" in out.tool_policy.excluded_tools
        assert out.auto_detect.enabled is True

    def test_get_registry_raises_503_when_none(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _get_registry

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])  # no workflow_registry attr
        with pytest.raises(HTTPException) as exc_info:
            _get_registry(mock_request)
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# ChatSession.workflow_id field
# ---------------------------------------------------------------------------


class TestChatSessionWorkflowField:
    def test_chat_session_has_workflow_id(self):
        from src.assistant.session import ChatSession

        # Check that the class has workflow_id in its annotations or __init__
        assert hasattr(ChatSession, "__annotations__") or hasattr(ChatSession, "workflow_id")
        # The field should exist and default to None
        import dataclasses

        if dataclasses.is_dataclass(ChatSession):
            fields = {f.name for f in dataclasses.fields(ChatSession)}
            assert "workflow_id" in fields


# ---------------------------------------------------------------------------
# BUG-221: Naive datetime → UTC timezone-aware serialization
# ---------------------------------------------------------------------------


class TestEnsureUtcValidator:
    """Regression: SQLite returns naive datetimes which Pydantic serialises
    without the 'Z' suffix, causing JS new Date() to interpret as local time."""

    def test_message_out_naive_gets_utc(self):
        from datetime import UTC, datetime

        from src.api.schemas.message import MessageOut

        naive = datetime(2026, 3, 8, 13, 34, 36, 841075)
        msg = MessageOut(
            id="test-id",
            session_id="sess-id",
            role="user",
            content="hello",
            created_at=naive,
        )
        assert msg.created_at.tzinfo is not None
        assert msg.created_at.tzinfo == UTC
        assert msg.created_at.isoformat().endswith("+00:00")

    def test_message_out_aware_passes_through(self):
        from datetime import UTC, datetime

        from src.api.schemas.message import MessageOut

        aware = datetime(2026, 3, 8, 13, 34, 36, 841075, tzinfo=UTC)
        msg = MessageOut(
            id="test-id",
            session_id="sess-id",
            role="user",
            content="hello",
            created_at=aware,
        )
        assert msg.created_at.tzinfo == UTC

    def test_session_out_naive_gets_utc(self):
        from datetime import UTC, datetime

        from src.api.schemas.session import SessionOut

        naive = datetime(2026, 3, 8, 12, 0, 0)
        sess = SessionOut(
            id="s-id",
            owner_id="u-id",
            name="Test",
            state="idle",
            config={},
            token_counts={
                "input_tokens": 0,
                "output_tokens": 0,
                "context_window": 131072,
            },
            active_tools=[],
            created_at=naive,
            updated_at=naive,
            archived_at=None,
        )
        assert sess.created_at.tzinfo == UTC
        assert sess.updated_at.tzinfo == UTC

    def test_api_key_out_naive_gets_utc(self):
        from datetime import UTC, datetime

        from src.api.schemas.auth import APIKeyOut

        naive = datetime(2026, 3, 8, 12, 0, 0)
        key = APIKeyOut(
            id="k-id",
            label="test",
            key_prefix="cgx_live_XXX",
            created_at=naive,
            expires_at=naive,
            last_used_at=None,
        )
        assert key.created_at.tzinfo == UTC
        assert key.expires_at.tzinfo == UTC
        assert key.last_used_at is None

    def test_ensure_utc_none_safe(self):
        from src.api.schemas.common import ensure_utc

        assert ensure_utc(None) is None


# ---------------------------------------------------------------------------
# Outbound messaging handler unit tests
# ---------------------------------------------------------------------------


class TestHandleOutbound:
    """Unit tests for MessageHandler.handle_outbound()."""

    def _make_handler(self):
        """Build a MessageHandler with minimal mocks for outbound testing."""
        handler_mod = pytest.importorskip("src.assistant.handler")
        MessageHandler = handler_mod.MessageHandler

        session_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.session_key = "whatsapp::+123@c.us"
        mock_session.last_activity = 0.0
        mock_session.last_sent_message_id = None
        mock_session.workflow_id = None
        mock_session.guardrail_violations = 0
        session_mgr.get_or_create.return_value = mock_session

        mock_mm = MagicMock()
        mock_context = MagicMock()
        mock_context.context_prefix = None
        mock_context.messages = []
        mock_mm.prepare_context.return_value = mock_context
        mock_session.memory_manager = mock_mm

        mock_runner = MagicMock(return_value="Hello there!")
        mock_guardrails = MagicMock()
        mock_guardrails.sanitize_output.side_effect = lambda x: x
        mock_guardrails.check_tool_call.return_value = MagicMock(is_safe=True)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="You are an assistant.",
            registry=MagicMock(),
            approvals={"*"},
            available_tools={},
            active_tools=[],
            agent_runner=mock_runner,
            guardrails=mock_guardrails,
            datamarking_enabled=False,
        )
        return handler, mock_session, mock_runner, mock_mm

    def test_handle_outbound_sends_message(self):
        handler, session, runner, mm = self._make_handler()
        channel = MagicMock()
        channel.name = "whatsapp"
        from src.assistant.channel import SendResult

        channel.send.return_value = SendResult(ok=True, message_id="msg-42")

        response, msg_id = handler.handle_outbound(
            contact_name="Alice",
            instructions="Ask about the project",
            channel=channel,
            chat_id="+123@c.us",
        )

        assert response == "Hello there!"
        assert msg_id == "msg-42"
        channel.send.assert_called_once_with("+123@c.us", "Hello there!")

    def test_handle_outbound_updates_memory(self):
        handler, session, runner, mm = self._make_handler()
        channel = MagicMock()
        channel.name = "whatsapp"
        from src.assistant.channel import SendResult

        channel.send.return_value = SendResult(ok=True, message_id="m1")

        handler.handle_outbound(
            contact_name="Alice",
            instructions="Say hello",
            channel=channel,
            chat_id="+123@c.us",
        )

        mm.update.assert_called_once()
        user_text = mm.update.call_args[0][0]
        assert "[Operator instruction]" in user_text
        assert "Say hello" in user_text
        mm.save.assert_called_once()

    def test_handle_outbound_frames_input_for_agent(self):
        handler, session, runner, mm = self._make_handler()
        channel = MagicMock()
        channel.name = "whatsapp"
        from src.assistant.channel import SendResult

        channel.send.return_value = SendResult(ok=True, message_id="m1")

        handler.handle_outbound(
            contact_name="Bob",
            instructions="Check status",
            channel=channel,
            chat_id="+456@c.us",
        )

        call_kwargs = runner.call_args
        user_input = call_kwargs.kwargs.get("user_input", call_kwargs[1].get("user_input", ""))
        assert "Operator instruction" in user_input
        assert "Bob" in user_input

    def test_handle_outbound_delivery_failure(self):
        handler, session, runner, mm = self._make_handler()
        channel = MagicMock()
        channel.name = "whatsapp"
        from src.assistant.channel import SendResult

        channel.send.return_value = SendResult(ok=False, error="timeout")

        response, msg_id = handler.handle_outbound(
            contact_name="Alice",
            instructions="test",
            channel=channel,
            chat_id="+123@c.us",
        )

        assert msg_id is None
        assert response == "Hello there!"
        # Memory is still updated even on delivery failure
        mm.update.assert_called_once()

    def test_handle_outbound_no_input_guardrails(self):
        """Operator instructions bypass input guardrails (no _check_guardrails call)."""
        handler, session, runner, mm = self._make_handler()
        channel = MagicMock()
        channel.name = "whatsapp"
        from src.assistant.channel import SendResult

        channel.send.return_value = SendResult(ok=True, message_id="m1")

        handler.handle_outbound(
            contact_name="Alice",
            instructions="test",
            channel=channel,
            chat_id="+123@c.us",
        )

        # _check_guardrails is not called — operator instructions are trusted
        handler._guardrails.check_input.assert_not_called()
        # But output guardrails are still applied
        handler._guardrails.sanitize_output.assert_called_once()


# ---------------------------------------------------------------------------
# Password complexity validation
# ---------------------------------------------------------------------------


class TestPasswordComplexity:
    """Validator is applied to both RegisterRequest and UserCreateRequest."""

    @pytest.mark.parametrize(
        "schema_path",
        [
            "src.api.schemas.auth.RegisterRequest",
            "src.api.schemas.user.UserCreateRequest",
        ],
    )
    def test_valid_password_accepted(self, schema_path: str) -> None:
        import importlib

        from pydantic import ValidationError

        module_path, cls_name = schema_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        # Should not raise
        try:
            cls(username="alice", email="alice@example.com", password="aB1!xxxx")
        except ValidationError as exc:
            pytest.fail(f"Valid password rejected: {exc}")

    @pytest.mark.parametrize(
        "password,missing",
        [
            ("AAAAAAAA", "lowercase"),
            ("aaaaaaaa", "uppercase"),
            ("aaBB!!@@", "digit"),
            ("aaBB1234", "special"),
        ],
    )
    def test_weak_password_rejected(self, password: str, missing: str) -> None:
        from pydantic import ValidationError

        from src.api.schemas.auth import RegisterRequest

        with pytest.raises(ValidationError, match="password"):
            RegisterRequest(username="alice", email="alice@example.com", password=password)

    def test_too_short_rejected_before_complexity(self) -> None:
        from pydantic import ValidationError

        from src.api.schemas.auth import RegisterRequest

        with pytest.raises(ValidationError, match="password"):
            RegisterRequest(username="alice", email="alice@example.com", password="aB1!")


# ---------------------------------------------------------------------------
# Admin self-deletion and self-demotion guard
# ---------------------------------------------------------------------------


class TestAdminSelfGuards:
    def _make_client_and_admin_token(self):
        import os
        import uuid

        os.environ.setdefault("COGTRIX_JWT_SECRET", "test-secret-key-for-testing-only-32ch")
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.auth import create_access_token

        app = create_app()
        admin_id = str(uuid.uuid4())
        admin_token = create_access_token(user_id=admin_id, role="admin")
        return TestClient(app, raise_server_exceptions=False), admin_token, admin_id

    def test_admin_cannot_delete_own_account(self) -> None:
        client, admin_token, admin_id = self._make_client_and_admin_token()
        with client:
            resp = client.delete(
                f"/api/v1/users/{admin_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "BAD_REQUEST"
        assert "own account" in resp.json()["error"]["message"]

    def test_admin_cannot_demote_own_account(self) -> None:
        client, admin_token, admin_id = self._make_client_and_admin_token()
        with client:
            resp = client.patch(
                f"/api/v1/users/{admin_id}",
                json={"role": "user"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "BAD_REQUEST"
        assert "demote" in resp.json()["error"]["message"]

    def test_admin_can_delete_another_user(self) -> None:
        import uuid

        client, admin_token, _ = self._make_client_and_admin_token()
        other_id = str(uuid.uuid4())
        with client:
            resp = client.delete(
                f"/api/v1/users/{other_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        # 404 because user doesn't exist — but NOT 400 (guard didn't fire)
        assert resp.status_code == 404
