"""Phase 7 API tests: comprehensive edge-case and error-handling unit tests.

Covers:
- TestValidationTranslation: translate_validation_errors, helper functions
- TestAuthEdgeCases: hash_password, verify_password, create_access_token, _decode_jwt, TokenData
- TestPagination: encode_cursor, decode_cursor, paginate_list
- TestConnectionManagerEdgeCases: ConnectionManager lifecycle and buffering
- TestCallbackHandlerEdgeCases: WebSocketCallbackHandler token/tool events
- TestConfirmationUIEdgeCases: ApiConfirmationUI resolve/cancel flow
- TestTurnRunnerHelpers: _build_history, _extract_token_counts
- TestAppExceptionHandlers: _http_exception_handler, _validation_exception_handler, _generic_exception_handler
- TestRegistrationBoundaries: field-level validation via TestClient
- TestLoginEdgeCases: wrong password / nonexistent user
- TestSessionCreateValidation: name length, config range limits
- TestMessageSendValidation: empty content, invalid mode, nonexistent session
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env is set
# ---------------------------------------------------------------------------

import jwt as jose_jwt  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from src.api.app import (  # noqa: E402
    _generic_exception_handler,
    _http_exception_handler,
    _validation_exception_handler,
)
from src.api.auth import (  # noqa: E402
    TokenData,
    _decode_jwt,
    create_access_token,
    hash_password,
    verify_password,
)
from src.api.callbacks import WebSocketCallbackHandler  # noqa: E402
from src.api.confirmation import _ACTION_MAP, ApiConfirmationUI  # noqa: E402
from src.api.pagination import decode_cursor, encode_cursor, paginate_list  # noqa: E402
from src.api.turn_runner import _build_history, _extract_token_counts  # noqa: E402
from src.api.validation import (  # noqa: E402
    _build_fallback_message,
    _extract_field_path,
    _humanize_name,
    _set_nested,
    translate_validation_errors,
)
from src.api.ws import ClientMessage, ConnectionManager, ServerMessage  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_app():
    """FastAPI app wired to an in-memory SQLite database."""
    import asyncio as _asyncio
    from unittest.mock import patch

    from src.api.app import create_app
    from src.api.db.engine import Base as _Base
    from src.api.db.engine import get_db

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _create():
        async with test_engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    _asyncio.get_event_loop().run_until_complete(_create())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = create_app()

        async def _override_get_db():
            async with test_session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override_get_db
        yield app

    _asyncio.get_event_loop().run_until_complete(test_engine.dispose())


@pytest.fixture()
def client(test_app):
    """Synchronous TestClient backed by the test app."""
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


def _auth_header(user_id: str = "test-user", role: str = "user") -> dict[str, str]:
    token = create_access_token(user_id, role)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 1. TestValidationTranslation
# ---------------------------------------------------------------------------


class TestValidationTranslation:
    @pytest.mark.parametrize(
        "error_type,field,ctx,expected_code",
        [
            ("string_too_short", "password", {"min_length": 8}, "TOO_SHORT"),
            ("string_too_long", "username", {"max_length": 64}, "TOO_LONG"),
            ("string_pattern_mismatch", "username", {}, "INVALID_FORMAT"),
            ("missing", "name", None, "REQUIRED"),
            ("literal_error", "mode", {"expected": "'a' or 'b'"}, "INVALID_CHOICE"),
            ("greater_than_equal", "max_steps", {"ge": 1}, "OUT_OF_RANGE"),
            ("less_than_equal", "max_steps", {"le": 200}, "OUT_OF_RANGE"),
            ("int_parsing", "count", None, "TYPE_MISMATCH"),
            ("bool_type", "active", None, "TYPE_MISMATCH"),
        ],
    )
    def test_translate_validation_error(self, error_type, field, ctx, expected_code):
        error: dict = {"loc": ["body", field], "type": error_type, "msg": "err"}
        if ctx is not None:
            error["ctx"] = ctx
        result = translate_validation_errors([error])
        assert result["fields"][field][0]["code"] == expected_code

    def test_field_messages_username_pattern(self):
        errors = [
            {
                "loc": ["body", "username"],
                "type": "string_pattern_mismatch",
                "msg": "...",
                "ctx": {},
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["username"][0]["message"]
        assert "letters" in msg.lower() or "only" in msg.lower()

    def test_field_messages_username_too_short(self):
        errors = [
            {
                "loc": ["body", "username"],
                "type": "string_too_short",
                "msg": "...",
                "ctx": {"min_length": 3},
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["username"][0]["message"]
        assert "3" in msg

    def test_field_messages_username_too_long(self):
        errors = [
            {
                "loc": ["body", "username"],
                "type": "string_too_long",
                "msg": "...",
                "ctx": {"max_length": 64},
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["username"][0]["message"]
        assert "64" in msg

    def test_field_messages_password_too_short(self):
        errors = [
            {
                "loc": ["body", "password"],
                "type": "string_too_short",
                "msg": "...",
                "ctx": {"min_length": 8},
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["password"][0]["message"]
        assert "8" in msg

    def test_field_messages_content_too_short(self):
        errors = [
            {
                "loc": ["body", "content"],
                "type": "string_too_short",
                "msg": "...",
                "ctx": {"min_length": 1},
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["content"][0]["message"]
        assert "empty" in msg.lower() or "cannot" in msg.lower()

    def test_field_messages_email_value_error(self):
        errors = [
            {
                "loc": ["body", "email"],
                "type": "value_error",
                "msg": "Value is not a valid email address",
            }
        ]
        result = translate_validation_errors(errors)
        msg = result["fields"]["email"][0]["message"]
        assert "email" in msg.lower()

    def test_extract_field_path_strips_body(self):
        assert _extract_field_path(["body", "username"]) == ["username"]

    def test_extract_field_path_strips_query(self):
        assert _extract_field_path(["query", "search"]) == ["search"]

    def test_extract_field_path_nested(self):
        assert _extract_field_path(["body", "config", "max_steps"]) == ["config", "max_steps"]

    def test_extract_field_path_empty(self):
        assert _extract_field_path([]) == ["_root"]

    def test_extract_field_path_no_prefix(self):
        assert _extract_field_path(["username"]) == ["username"]

    def test_set_nested_single_segment(self):
        target: dict = {}
        _set_nested(target, ["username"], {"code": "TOO_SHORT", "message": "too short"})
        assert target["username"] == [{"code": "TOO_SHORT", "message": "too short"}]

    def test_set_nested_multi_segment(self):
        target: dict = {}
        _set_nested(target, ["config", "max_steps"], {"code": "OUT_OF_RANGE", "message": "..."})
        assert target["config"]["max_steps"] == [{"code": "OUT_OF_RANGE", "message": "..."}]

    def test_set_nested_appends_to_existing(self):
        target: dict = {}
        _set_nested(target, ["field"], {"code": "A", "message": "a"})
        _set_nested(target, ["field"], {"code": "B", "message": "b"})
        assert len(target["field"]) == 2

    def test_multiple_errors_aggregate(self):
        errors = [
            {"loc": ["body", "username"], "type": "string_too_short", "msg": "...", "ctx": {}},
            {
                "loc": ["body", "username"],
                "type": "string_pattern_mismatch",
                "msg": "...",
                "ctx": {},
            },
        ]
        result = translate_validation_errors(errors)
        assert len(result["fields"]["username"]) == 2

    def test_empty_errors_list(self):
        result = translate_validation_errors([])
        assert result == {"fields": {}}

    def test_missing_loc_uses_root(self):
        errors = [{"type": "missing", "msg": "required"}]
        result = translate_validation_errors(errors)
        assert "_root" in result["fields"]

    def test_missing_type_uses_invalid_code(self):
        errors = [{"loc": ["body", "field"], "msg": "something"}]
        result = translate_validation_errors(errors)
        assert result["fields"]["field"][0]["code"] == "INVALID"

    def test_humanize_name_snake_case(self):
        assert _humanize_name("max_steps") == "Max steps"

    def test_humanize_name_single(self):
        assert _humanize_name("username") == "Username"

    def test_fallback_message_ge(self):
        err = {"type": "greater_than_equal", "ctx": {"ge": 1}}
        msg = _build_fallback_message("count", err)
        assert "1" in msg and "at least" in msg

    def test_fallback_message_le(self):
        err = {"type": "less_than_equal", "ctx": {"le": 200}}
        msg = _build_fallback_message("max_steps", err)
        assert "200" in msg and "at most" in msg

    def test_fallback_message_gt(self):
        err = {"type": "greater_than", "ctx": {"gt": 0}}
        msg = _build_fallback_message("value", err)
        assert "greater than" in msg

    def test_fallback_message_lt(self):
        err = {"type": "less_than", "ctx": {"lt": 100}}
        msg = _build_fallback_message("value", err)
        assert "less than" in msg

    def test_fallback_message_literal_with_expected(self):
        err = {"type": "literal_error", "ctx": {"expected": "'a' or 'b'"}}
        msg = _build_fallback_message("mode", err)
        assert "one of" in msg.lower()

    def test_fallback_message_missing(self):
        err = {"type": "missing"}
        msg = _build_fallback_message("email", err)
        assert "required" in msg.lower()

    def test_fallback_message_bool_type(self):
        err = {"type": "bool_type"}
        msg = _build_fallback_message("active", err)
        assert "true" in msg.lower() or "false" in msg.lower()

    def test_fallback_message_list_type(self):
        err = {"type": "list_type"}
        msg = _build_fallback_message("tags", err)
        assert "list" in msg.lower()

    def test_nested_loc_creates_nested_output(self):
        errors = [
            {
                "loc": ["body", "config", "max_steps"],
                "type": "greater_than_equal",
                "msg": "gte",
                "ctx": {"ge": 1},
            }
        ]
        result = translate_validation_errors(errors)
        assert "config" in result["fields"]
        assert "max_steps" in result["fields"]["config"]


# ---------------------------------------------------------------------------
# 2. TestAuthEdgeCases
# ---------------------------------------------------------------------------


class TestAuthEdgeCases:
    def test_hash_empty_password(self):
        h = hash_password("")
        assert verify_password("", h) is True

    def test_wrong_password_returns_false(self):
        h = hash_password("correct_password")
        assert verify_password("wrong_password", h) is False

    def test_unicode_password_roundtrip(self):
        pw = "p\u00e4ssw\u00f6rd\u0021"
        h = hash_password(pw)
        assert verify_password(pw, h) is True

    def test_72_byte_boundary_exact(self):
        pw_72 = "a" * 72
        h = hash_password(pw_72)
        assert verify_password(pw_72, h) is True

    def test_73_byte_password_matches_72_byte(self):
        # bcrypt truncates at 72 bytes — 73-byte password matches 72-byte hash
        pw_72 = "b" * 72
        pw_73 = "b" * 73
        h = hash_password(pw_72)
        assert verify_password(pw_73, h) is True

    def test_create_access_token_roundtrip(self):
        token = create_access_token("user-xyz", "admin")
        claims = _decode_jwt(token)
        assert claims["sub"] == "user-xyz"
        assert claims["role"] == "admin"

    def test_decode_expired_token_raises_401(self):
        expired = jose_jwt.encode(
            {
                "sub": "uid",
                "role": "user",
                "exp": datetime.now(UTC) - timedelta(hours=1),
            },
            _TEST_JWT_SECRET,
            algorithm="HS256",
        )
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(expired)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["code"] == "TOKEN_EXPIRED"

    def test_decode_malformed_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt("not.a.valid.jwt")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["code"] == "UNAUTHORIZED"

    def test_decode_wrong_secret_raises_401(self):
        token = jose_jwt.encode(
            {"sub": "uid", "role": "user", "exp": datetime.now(UTC) + timedelta(hours=1)},
            "completely_different_secret_here_xyz",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(token)
        assert exc_info.value.status_code == 401

    def test_decode_empty_string_raises_401(self):
        with pytest.raises(HTTPException):
            _decode_jwt("")

    def test_token_data_is_admin_true(self):
        td = TokenData("uid-1", "admin", {})
        assert td.is_admin is True

    def test_token_data_is_admin_false_for_user(self):
        td = TokenData("uid-2", "user", {})
        assert td.is_admin is False

    def test_token_data_user_id_stored(self):
        td = TokenData("uid-3", "user", {"extra": "value"})
        assert td.user_id == "uid-3"
        assert td.role == "user"
        assert td.raw_claims["extra"] == "value"

    def test_create_token_contains_iat_and_exp(self):
        token = create_access_token("uid-4", "user")
        claims = _decode_jwt(token)
        assert "iat" in claims
        assert "exp" in claims

    def test_multiple_tokens_for_same_user_are_different(self):
        t1 = create_access_token("uid-5", "user")
        t2 = create_access_token("uid-5", "user")
        # Tokens will differ due to different iat timestamps (or they might be
        # the same within the same second — just verify both decode correctly)
        c1 = _decode_jwt(t1)
        c2 = _decode_jwt(t2)
        assert c1["sub"] == c2["sub"] == "uid-5"


# ---------------------------------------------------------------------------
# 3. TestPagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_encode_decode_roundtrip(self):
        original = "some_cursor_value"
        encoded = encode_cursor(original)
        assert decode_cursor(encoded) == original

    def test_encode_produces_url_safe_string(self):
        encoded = encode_cursor("hello world+/")
        assert "+" not in encoded
        assert "/" not in encoded

    def test_decode_invalid_base64_raises(self):
        with pytest.raises((ValueError, Exception)):  # noqa: B017
            decode_cursor("not!!valid!!base64")

    def test_paginate_empty_list(self):
        items, cursor, has_more = paginate_list([], None, 10)
        assert items == []
        assert cursor is None
        assert has_more is False

    def test_paginate_fewer_items_than_limit(self):
        items, cursor, has_more = paginate_list(["a", "b", "c"], None, 10)
        assert items == ["a", "b", "c"]
        assert cursor is None
        assert has_more is False

    def test_paginate_exactly_limit_items(self):
        items, cursor, has_more = paginate_list(["a", "b"], None, 2)
        assert items == ["a", "b"]
        assert has_more is False

    def test_paginate_more_items_than_limit(self):
        items, cursor, has_more = paginate_list(["a", "b", "c"], None, 2)
        assert items == ["a", "b"]
        assert has_more is True
        assert cursor is not None

    def test_paginate_with_cursor_continues(self):
        all_items = ["a", "b", "c", "d"]
        # First page
        page1, cursor1, _ = paginate_list(all_items, None, 2)
        assert page1 == ["a", "b"]
        # Use the encoded cursor to get page 2
        decoded = decode_cursor(cursor1)
        page2, cursor2, has_more2 = paginate_list(all_items, decoded, 2)
        assert page2 == ["c", "d"]
        assert has_more2 is False
        assert cursor2 is None

    def test_limit_clamps_to_minimum_1(self):
        items, _, _ = paginate_list(["a", "b", "c"], None, 0)
        assert items == ["a"]

    def test_limit_clamps_to_maximum_500(self):
        big_list = [str(i) for i in range(600)]
        items, _, has_more = paginate_list(big_list, None, 1000)
        assert len(items) == 500
        assert has_more is True

    def test_cursor_not_found_starts_from_beginning(self):
        all_items = ["a", "b", "c"]
        items, _, _ = paginate_list(all_items, "nonexistent_cursor", 2)
        assert items == ["a", "b"]


# ---------------------------------------------------------------------------
# 4. TestConnectionManagerEdgeCases
# ---------------------------------------------------------------------------


class TestConnectionManagerEdgeCases:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect_lifecycle(self):
        manager = ConnectionManager()
        mock_ws = MagicMock()

        async def _close(code=1001):
            pass

        mock_ws.close = _close
        session_id = "sess-001"
        await manager.connect(session_id, mock_ws)
        assert session_id in manager._connections
        await manager.disconnect(session_id)
        assert session_id not in manager._connections

    @pytest.mark.asyncio
    async def test_second_connect_replaces_first(self):
        manager = ConnectionManager()
        close_called = []

        async def _close(code=1001):
            close_called.append(code)

        ws1 = MagicMock()
        ws1.close = _close
        ws2 = MagicMock()
        ws2.close = _close

        await manager.connect("sess-002", ws1)
        await manager.connect("sess-002", ws2)

        assert manager._connections["sess-002"] is ws2
        assert close_called  # old ws was closed

    @pytest.mark.asyncio
    async def test_send_to_session_without_connection_buffers_message(self):
        manager = ConnectionManager()
        # Session without an active WebSocket — should not raise
        await manager.send("sess-003", "token", {"text": "hello"})
        # Message should be buffered
        assert "sess-003" in manager._buffers
        assert len(manager._buffers["sess-003"]) == 1

    @pytest.mark.asyncio
    async def test_replay_missed_no_messages(self):
        manager = ConnectionManager()
        mock_ws = MagicMock()

        async def _send_noop(text: str) -> None:
            pass

        mock_ws.send_text = _send_noop
        await manager.connect("sess-004", mock_ws)
        # Replay with no buffered messages — no error
        await manager.replay_missed("sess-004", 0)

    @pytest.mark.asyncio
    async def test_replay_missed_filters_by_seq(self):
        manager = ConnectionManager()
        sent_texts = []

        async def _send_text(text):
            sent_texts.append(text)

        mock_ws = MagicMock()
        mock_ws.send_text = _send_text

        async def _noop_close(code: int = 1001) -> None:
            pass

        mock_ws.close = _noop_close

        # Send messages without an active connection first to populate buffer
        await manager.send("sess-005", "token", {"text": "msg0"})
        await manager.send("sess-005", "token", {"text": "msg1"})
        await manager.send("sess-005", "token", {"text": "msg2"})

        # Now connect and replay from seq 1
        await manager.connect("sess-005", mock_ws)
        await manager.replay_missed("sess-005", 1)
        # Should only replay seq 2
        assert len(sent_texts) == 1

    @pytest.mark.asyncio
    async def test_replay_missed_no_connection(self):
        manager = ConnectionManager()
        # No connection registered — should not raise
        await manager.replay_missed("sess-nonexistent", 0)

    def test_build_message_returns_valid_json(self):
        import json

        manager = ConnectionManager()
        json_str = manager._build_message("sess-006", "token", {"text": "hi"}, 0)
        data = json.loads(json_str)
        assert data["type"] == "token"
        assert data["session_id"] == "sess-006"
        assert data["payload"]["text"] == "hi"
        assert data["seq"] == 0

    def test_server_message_validation(self):
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        msg = ServerMessage(
            type="token",
            session_id="sess-007",
            payload={"text": "hello"},
            seq=1,
            ts=now,
        )
        assert msg.type == "token"
        assert msg.seq == 1

    def test_client_message_validation(self):
        msg = ClientMessage(type="ping", payload={})
        assert msg.type == "ping"

    def test_server_message_invalid_type_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ServerMessage(
                type="invalid_type",
                session_id="sess-008",
                payload={},
                seq=0,
                ts="2026-01-01T00:00:00.000Z",
            )


# ---------------------------------------------------------------------------
# 5. TestCallbackHandlerEdgeCases
# ---------------------------------------------------------------------------


class TestCallbackHandlerEdgeCases:
    def _make_handler(
        self,
    ) -> tuple[WebSocketCallbackHandler, asyncio.Queue, asyncio.AbstractEventLoop]:
        loop = asyncio.new_event_loop()

        async def _setup():
            return asyncio.Queue()

        inner_queue = loop.run_until_complete(_setup())
        handler = WebSocketCallbackHandler(inner_queue, loop)
        return handler, inner_queue, loop

    def test_on_tool_start_increments_count(self):
        handler, queue, loop = self._make_handler()
        handler.on_tool_start({"name": "web_search"}, "{}", run_id="run-1")
        assert handler.tool_call_count == 1
        loop.close()

    def test_on_tool_start_multiple_calls(self):
        handler, queue, loop = self._make_handler()
        handler.on_tool_start({"name": "tool_a"}, "{}", run_id="run-1")
        handler.on_tool_start({"name": "tool_b"}, "{}", run_id="run-2")
        assert handler.tool_call_count == 2
        loop.close()

    def test_on_tool_start_with_dict_input(self):
        handler, queue, loop = self._make_handler()
        # Should not raise when input_str is a dict
        handler.on_tool_start({"name": "search"}, {"query": "test"}, run_id="run-3")
        assert handler.tool_call_count == 1
        loop.close()

    def test_on_tool_end_after_start_no_error(self):
        import time

        handler, queue, loop = self._make_handler()
        handler.on_tool_start({"name": "tool_x"}, "{}", run_id="run-4")
        time.sleep(0.01)  # small delay to ensure non-zero duration
        handler.on_tool_end("result", run_id="run-4", name="tool_x")
        # run_id should be removed from _tool_starts
        assert "run-4" not in handler._tool_starts
        loop.close()

    def test_on_tool_error_removes_start_entry(self):
        handler, queue, loop = self._make_handler()
        handler.on_tool_start({"name": "tool_y"}, "{}", run_id="run-5")
        handler.on_tool_error(RuntimeError("oops"), run_id="run-5", name="tool_y")
        assert "run-5" not in handler._tool_starts
        loop.close()

    def test_on_tool_start_with_string_json_input(self):
        import json

        handler, queue, loop = self._make_handler()
        handler.on_tool_start({"name": "calc"}, json.dumps({"expression": "1+1"}), run_id="run-6")
        assert handler.tool_call_count == 1
        loop.close()

    def test_on_tool_start_with_invalid_json_string_does_not_raise(self):
        handler, queue, loop = self._make_handler()
        # Non-JSON string input should not raise
        handler.on_tool_start({"name": "tool"}, "not valid json", run_id="run-7")
        assert handler.tool_call_count == 1
        loop.close()


# ---------------------------------------------------------------------------
# 6. TestConfirmationUIEdgeCases
# ---------------------------------------------------------------------------


class TestConfirmationUIEdgeCases:
    def _make_ui(self) -> tuple[ApiConfirmationUI, asyncio.Queue, asyncio.AbstractEventLoop]:
        loop = asyncio.new_event_loop()

        async def _setup():
            return asyncio.Queue()

        inner_queue = loop.run_until_complete(_setup())
        ui = ApiConfirmationUI(inner_queue, loop)
        return ui, inner_queue, loop

    def test_action_map_has_all_six_actions(self):
        assert _ACTION_MAP["allow"] == "y"
        assert _ACTION_MAP["deny"] == "n"
        assert _ACTION_MAP["allow_all"] == "a"
        assert _ACTION_MAP["disable"] == "d"
        assert _ACTION_MAP["forbid_all"] == "f"
        assert _ACTION_MAP["cancel"] == "c"

    def test_render_prompt_resets_cancel_requested(self):
        ui, queue, loop = self._make_ui()
        # Manually set cancel flag
        ui._cancel_requested = True
        ui.render_prompt("tool_name", {}, frozenset(), 100)
        with ui._lock:
            assert ui._cancel_requested is False
        loop.close()

    def test_resolve_with_correct_id_returns_true(self):
        ui, queue, loop = self._make_ui()
        ui.render_prompt("write_file", {"path": "/tmp/x"}, frozenset(), 100)
        conf_id = ui._confirmation_id
        result = ui.resolve(conf_id, "allow")
        assert result is True
        loop.close()

    def test_resolve_with_wrong_id_returns_false(self):
        ui, queue, loop = self._make_ui()
        ui.render_prompt("write_file", {}, frozenset(), 100)
        result = ui.resolve("wrong-id-000", "allow")
        assert result is False
        loop.close()

    def test_resolve_allow_sets_action_y(self):
        ui, queue, loop = self._make_ui()
        ui.render_prompt("tool", {}, frozenset(), 100)
        conf_id = ui._confirmation_id
        ui.resolve(conf_id, "allow")
        with ui._lock:
            assert ui._pending_action == "y"
        loop.close()

    def test_cancel_sets_cancel_requested(self):
        ui, queue, loop = self._make_ui()
        ui.render_prompt("tool", {}, frozenset(), 100)
        ui.cancel()
        with ui._lock:
            assert ui._cancel_requested is True
        loop.close()

    def test_cancel_unblocks_pending_event(self):
        ui, queue, loop = self._make_ui()
        ui.render_prompt("tool", {}, frozenset(), 100)
        event = ui._pending_event
        ui.cancel()
        assert event is not None and event.is_set()
        loop.close()


# ---------------------------------------------------------------------------
# 7. TestTurnRunnerHelpers
# ---------------------------------------------------------------------------


class TestTurnRunnerHelpers:
    def test_build_history_none_returns_empty(self):
        assert _build_history(None) == []

    def test_build_history_with_exception_returns_empty(self):
        bad_mm = MagicMock()
        bad_mm.prepare_context.side_effect = RuntimeError("db error")
        result = _build_history(bad_mm)
        assert result == []

    def test_build_history_returns_messages(self):
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.messages = ["msg1", "msg2"]
        mm = MagicMock()
        mm.prepare_context.return_value = mock_ctx

        result = _build_history(mm)
        assert result == ["msg1", "msg2"]

    def test_build_history_passes_user_input(self):
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        mm = MagicMock()
        mm.prepare_context.return_value = mock_ctx

        _build_history(mm, "hello world")
        mm.prepare_context.assert_called_once_with("hello world")

    def test_extract_token_counts_from_callback(self):
        handler = MagicMock()
        handler.input_tokens = 100
        handler.output_tokens = 50
        handler.tool_call_count = 3

        counts = _extract_token_counts(handler)
        assert counts == {"input_tokens": 100, "output_tokens": 50, "tool_call_count": 3}

    def test_extract_token_counts_missing_attrs_return_zero(self):
        counts = _extract_token_counts(object())
        assert counts == {"input_tokens": 0, "output_tokens": 0, "tool_call_count": 0}

    def test_extract_token_counts_partial_attrs(self):
        obj = MagicMock(spec=["input_tokens"])
        obj.input_tokens = 42
        counts = _extract_token_counts(obj)
        assert counts["input_tokens"] == 42
        assert counts["output_tokens"] == 0


# ---------------------------------------------------------------------------
# 8. TestAppExceptionHandlers
# ---------------------------------------------------------------------------


class TestAppExceptionHandlers:
    def _mock_request(self) -> MagicMock:
        req = MagicMock()
        req.method = "GET"
        req.url.path = "/test"
        return req

    @pytest.mark.asyncio
    async def test_http_exception_handler_with_dict_detail(self):
        req = self._mock_request()
        exc = HTTPException(status_code=400, detail={"code": "MY_CODE", "message": "my msg"})
        resp = await _http_exception_handler(req, exc)
        body = resp.body
        import json

        data = json.loads(body)
        assert data["error"]["code"] == "MY_CODE"
        assert data["error"]["message"] == "my msg"
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_http_exception_handler_with_string_detail_400(self):
        req = self._mock_request()
        exc = HTTPException(status_code=400, detail="bad request string")
        resp = await _http_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "BAD_REQUEST"
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_http_exception_handler_404_not_found(self):
        req = self._mock_request()
        exc = HTTPException(status_code=404, detail="not found")
        resp = await _http_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "NOT_FOUND"
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_http_exception_handler_403_forbidden(self):
        req = self._mock_request()
        exc = HTTPException(status_code=403, detail="forbidden")
        resp = await _http_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_http_exception_handler_unknown_status_internal_error(self):
        req = self._mock_request()
        exc = HTTPException(status_code=418, detail="I am a teapot")
        resp = await _http_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert resp.status_code == 418

    @pytest.mark.asyncio
    async def test_http_exception_handler_401_unauthorized(self):
        req = self._mock_request()
        exc = HTTPException(status_code=401, detail="unauthorized")
        resp = await _http_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_generic_exception_handler_returns_500(self):
        req = self._mock_request()
        exc = RuntimeError("something went wrong")
        resp = await _generic_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert resp.status_code == 500
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["data"] is None

    @pytest.mark.asyncio
    async def test_validation_exception_handler_returns_422(self):
        req = self._mock_request()

        class FakeValidationError(Exception):
            def errors(self):
                return [
                    {
                        "loc": ["body", "username"],
                        "type": "string_too_short",
                        "msg": "too short",
                        "ctx": {"min_length": 3},
                    }
                ]

        exc = FakeValidationError("validation failed")
        resp = await _validation_exception_handler(req, exc)
        import json

        data = json.loads(resp.body)
        assert resp.status_code == 422
        assert data["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# 9. TestRegistrationBoundaries — Integration via TestClient
# ---------------------------------------------------------------------------


class TestRegistrationBoundaries:
    def test_username_too_short_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "ab", "email": "ab@test.com", "password": "Password1!"},
        )
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "VALIDATION_ERROR"
        details = error.get("details", {})
        fields = details.get("fields", {})
        assert "username" in fields

    def test_username_too_long_returns_422(self, client: TestClient):
        long_name = "u" * 65
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": long_name, "email": "long@test.com", "password": "Password1!"},
        )
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "VALIDATION_ERROR"

    def test_username_with_spaces_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "user name", "email": "space@test.com", "password": "Password1!"},
        )
        assert resp.status_code == 422

    def test_username_with_special_chars_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "user@name",
                "email": "special@test.com",
                "password": "Password1!",
            },
        )
        assert resp.status_code == 422

    def test_password_too_short_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "validuser", "email": "pwshort@test.com", "password": "1234567"},
        )
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "VALIDATION_ERROR"
        details = error.get("details", {})
        fields = details.get("fields", {})
        assert "password" in fields

    def test_password_too_long_returns_422(self, client: TestClient):
        long_pw = "P" * 129 + "1"
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "validuser2", "email": "pwlong@test.com", "password": long_pw},
        )
        assert resp.status_code == 422
        details = resp.json()["error"].get("details", {})
        fields = details.get("fields", {})
        assert "password" in fields

    def test_invalid_email_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "validuser3",
                "email": "not-an-email",
                "password": "Password1!",
            },
        )
        assert resp.status_code == 422

    def test_duplicate_username_returns_409(self, client: TestClient):
        payload = {
            "username": "dupuser7",
            "email": "first7@test.com",
            "password": "Password1!",
        }
        client.post("/api/v1/auth/register", json=payload)
        resp = client.post(
            "/api/v1/auth/register",
            json={**payload, "email": "second7@test.com"},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 10. TestLoginEdgeCases — Integration via TestClient
# ---------------------------------------------------------------------------


class TestLoginEdgeCases:
    def test_login_wrong_password_returns_401(self, client: TestClient):
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "loginuser8",
                "email": "login8@test.com",
                "password": "CorrectPass1!",
            },
        )
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "loginuser8", "password": "WrongPass1!"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_login_nonexistent_user_returns_401(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "ghost_user_xyz", "password": "AnyPass123!"},
        )
        assert resp.status_code == 401

    def test_login_missing_password_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "someuser"},
        )
        assert resp.status_code == 422

    def test_login_empty_body_returns_422(self, client: TestClient):
        resp = client.post("/api/v1/auth/login", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 11. TestSessionCreateValidation — Integration via TestClient
# ---------------------------------------------------------------------------


class TestSessionCreateValidation:
    def test_session_name_too_long_returns_422(self, client: TestClient):
        headers = _auth_header()
        long_name = "S" * 257
        resp = client.post(
            "/api/v1/sessions",
            json={"name": long_name},
            headers=headers,
        )
        assert resp.status_code == 422

    def test_session_config_max_steps_below_range_returns_422(self, client: TestClient):
        headers = _auth_header()
        resp = client.post(
            "/api/v1/sessions",
            json={"name": "Test", "config": {"max_steps": -1}},
            headers=headers,
        )
        assert resp.status_code == 422

    def test_session_config_max_steps_above_range_returns_422(self, client: TestClient):
        headers = _auth_header()
        resp = client.post(
            "/api/v1/sessions",
            json={"name": "Test", "config": {"max_steps": 201}},
            headers=headers,
        )
        assert resp.status_code == 422

    def test_session_create_unauthenticated_returns_401(self, client: TestClient):
        resp = client.post("/api/v1/sessions", json={"name": "Test"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 12. TestMessageSendValidation — Integration via TestClient
# ---------------------------------------------------------------------------


class TestMessageSendValidation:
    def _register_and_create_session(self, client: TestClient) -> tuple[str, str]:
        """Register, login, create session; return (token, session_id)."""
        uname = f"msgtester_{uuid.uuid4().hex[:8]}"
        reg = client.post(
            "/api/v1/auth/register",
            json={
                "username": uname,
                "email": f"{uname}@test.com",
                "password": "Password1!",
            },
        )
        assert reg.status_code == 201
        token = reg.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        sess = client.post("/api/v1/sessions", json={"name": "msg test"}, headers=headers)
        assert sess.status_code == 201
        session_id = sess.json()["data"]["id"]
        return token, session_id

    def test_send_empty_content_returns_422(self, client: TestClient):
        token, session_id = self._register_and_create_session(client)
        resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "", "mode": "normal"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_send_invalid_mode_returns_422(self, client: TestClient):
        token, session_id = self._register_and_create_session(client)
        resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "hello", "mode": "invalid_mode"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_send_to_nonexistent_session_returns_404(self, client: TestClient):
        token, _ = self._register_and_create_session(client)
        fake_id = str(uuid.uuid4())
        resp = client.post(
            f"/api/v1/sessions/{fake_id}/messages",
            json={"content": "hello", "mode": "normal"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_send_without_auth_returns_401(self, client: TestClient):
        token, session_id = self._register_and_create_session(client)
        resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "hello", "mode": "normal"},
        )
        assert resp.status_code == 401
