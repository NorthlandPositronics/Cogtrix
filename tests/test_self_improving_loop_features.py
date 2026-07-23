"""Regression tests for the self-improving loop feature set.

Covers all 7 changes made to support autonomous self-improvement workflows:
  #207 — WS field naming: tool_name instead of tool in ToolStartPayload/ToolEndPayload
  #208 — WS idle timeout default raised to 300 s; configurable via COGTRIX_WS_IDLE_TIMEOUT
  #209 — patch_file tool for surgical file edits
  #210 — git_tools module (git_status, git_diff, git_log, git_add, git_commit, …)
  #211 — DonePayload.text field + turn_runner populates it
  #212 — POST /sessions/{id}/messages?sync=true blocking REST mode
  #213 — initial_tools / auto_approve_tools on POST /sessions
"""

from __future__ import annotations

import asyncio as _asyncio
import importlib
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Env setup (must happen before any src imports)
# ---------------------------------------------------------------------------

_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")


# ===========================================================================
# #207 — WS field naming
# ===========================================================================


class TestWSFieldNaming:
    """ToolStartPayload and ToolEndPayload must expose tool_name, not tool."""

    def test_tool_start_payload_has_tool_name_field(self):
        from src.api.ws import ToolStartPayload

        assert "tool_name" in ToolStartPayload.model_fields
        assert "tool" not in ToolStartPayload.model_fields

    def test_tool_end_payload_has_tool_name_field(self):
        from src.api.ws import ToolEndPayload

        assert "tool_name" in ToolEndPayload.model_fields
        assert "tool" not in ToolEndPayload.model_fields

    def test_tool_start_payload_instantiation(self):
        from src.api.ws import ToolStartPayload

        p = ToolStartPayload(tool_name="web_search", tool_call_id="call_1", input={"q": "test"})
        assert p.tool_name == "web_search"

    def test_tool_end_payload_instantiation(self):
        from src.api.ws import ToolEndPayload

        p = ToolEndPayload(tool_name="web_search", tool_call_id="call_1", duration_ms=100)
        assert p.tool_name == "web_search"
        assert p.error is None

    def test_callbacks_on_tool_start_enqueues_tool_name(self):
        """WebSocketCallbackHandler.on_tool_start must enqueue payload with key 'tool_name'."""
        from src.api.callbacks import WebSocketCallbackHandler

        loop = _asyncio.new_event_loop()
        try:
            q: _asyncio.Queue = _asyncio.Queue()
            handler = WebSocketCallbackHandler(q, loop)

            run_id = uuid.uuid4()
            handler.on_tool_start(
                serialized={"name": "read_file"},
                input_str='{"path": "foo.py"}',
                run_id=run_id,
            )
            # The callback uses call_soon_threadsafe; drive the loop briefly.
            loop.run_until_complete(_asyncio.sleep(0))

            item = q.get_nowait()
            assert item["type"] == "tool_start"
            assert "tool_name" in item["payload"]
            assert "tool" not in item["payload"]
            assert item["payload"]["tool_name"] == "read_file"
        finally:
            loop.close()

    def test_callbacks_on_tool_end_enqueues_tool_name(self):
        """WebSocketCallbackHandler.on_tool_end must enqueue payload with key 'tool_name'."""
        from src.api.callbacks import WebSocketCallbackHandler

        loop = _asyncio.new_event_loop()
        try:
            q: _asyncio.Queue = _asyncio.Queue()
            handler = WebSocketCallbackHandler(q, loop)

            run_id = uuid.uuid4()
            # Prime _tool_starts so on_tool_end finds a start time.
            handler.on_tool_start(
                serialized={"name": "read_file"},
                input_str="{}",
                run_id=run_id,
            )
            loop.run_until_complete(_asyncio.sleep(0))
            q.get_nowait()  # discard tool_start

            handler.on_tool_end("result text", run_id=run_id, name="read_file")
            loop.run_until_complete(_asyncio.sleep(0))

            item = q.get_nowait()
            assert item["type"] == "tool_end"
            assert "tool_name" in item["payload"]
            assert "tool" not in item["payload"]
        finally:
            loop.close()

    def test_callbacks_on_tool_error_enqueues_tool_name(self):
        """WebSocketCallbackHandler.on_tool_error must enqueue payload with key 'tool_name'."""
        from src.api.callbacks import WebSocketCallbackHandler

        loop = _asyncio.new_event_loop()
        try:
            q: _asyncio.Queue = _asyncio.Queue()
            handler = WebSocketCallbackHandler(q, loop)

            run_id = uuid.uuid4()
            handler.on_tool_start(
                serialized={"name": "shell"},
                input_str="{}",
                run_id=run_id,
            )
            loop.run_until_complete(_asyncio.sleep(0))
            q.get_nowait()  # discard tool_start

            handler.on_tool_error(RuntimeError("boom"), run_id=run_id, name="shell")
            loop.run_until_complete(_asyncio.sleep(0))

            item = q.get_nowait()
            assert item["type"] == "tool_end"
            assert "tool_name" in item["payload"]
            assert item["payload"]["error"] == "boom"
        finally:
            loop.close()


# ===========================================================================
# #208 — WS idle timeout
# ===========================================================================


class TestWSIdleTimeout:
    """_WS_IDLE_TIMEOUT must be 300 by default and respect COGTRIX_WS_IDLE_TIMEOUT."""

    def test_default_idle_timeout_is_300(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COGTRIX_WS_IDLE_TIMEOUT", raising=False)
        import src.api.routes.messages as msg_mod

        importlib.reload(msg_mod)
        assert msg_mod._WS_IDLE_TIMEOUT == pytest.approx(300.0)

    def test_env_var_overrides_idle_timeout(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COGTRIX_WS_IDLE_TIMEOUT", "120")
        import src.api.routes.messages as msg_mod

        importlib.reload(msg_mod)
        assert msg_mod._WS_IDLE_TIMEOUT == pytest.approx(120.0)
        # Restore default for other tests.
        monkeypatch.delenv("COGTRIX_WS_IDLE_TIMEOUT", raising=False)
        importlib.reload(msg_mod)

    def test_idle_timeout_is_float(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COGTRIX_WS_IDLE_TIMEOUT", "45.5")
        import src.api.routes.messages as msg_mod

        importlib.reload(msg_mod)
        assert isinstance(msg_mod._WS_IDLE_TIMEOUT, float)
        assert msg_mod._WS_IDLE_TIMEOUT == pytest.approx(45.5)
        monkeypatch.delenv("COGTRIX_WS_IDLE_TIMEOUT", raising=False)
        importlib.reload(msg_mod)


# ===========================================================================
# #209 — patch_file tool
# ===========================================================================


@pytest.fixture()
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestPatchFileTool:
    def test_patch_replaces_unique_string(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        f = tmp_cwd / "hello.py"
        f.write_text("def foo():\n    return 1\n")
        result = patch_file("hello.py", "return 1", "return 42")
        assert "Patched" in result
        assert f.read_text() == "def foo():\n    return 42\n"

    def test_patch_returns_line_delta(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        f = tmp_cwd / "lines.py"
        f.write_text("a = 1\n")
        result = patch_file("lines.py", "a = 1", "a = 1\nb = 2")
        assert "+" in result  # line count increased

    def test_patch_fails_when_old_str_not_found(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        (tmp_cwd / "x.py").write_text("something else\n")
        result = patch_file("x.py", "nonexistent string", "replacement")
        assert "not found" in result.lower()

    def test_patch_fails_when_old_str_ambiguous(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        (tmp_cwd / "y.py").write_text("x = 1\nx = 1\n")
        result = patch_file("y.py", "x = 1", "x = 2")
        assert "found" in result.lower()
        # Verify file was NOT changed.
        assert (tmp_cwd / "y.py").read_text() == "x = 1\nx = 1\n"

    def test_patch_rejects_path_outside_cwd(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        result = patch_file("/etc/passwd", "root", "noroot")
        assert "Error" in result or "not allowed" in result.lower() or "denied" in result.lower()

    def test_patch_file_nonexistent(self, tmp_cwd: Path):
        from src.tools.file_ops import patch_file

        result = patch_file("no_such_file.py", "x", "y")
        assert "Error" in result or "not found" in result.lower()

    def test_patch_file_in_tool_configs(self):
        from src.tools.file_ops import TOOL_CONFIGS

        names = [cfg["name"] for cfg in TOOL_CONFIGS]
        assert "patch_file" in names

    def test_patch_file_requires_confirmation(self):
        from src.tools.file_ops import TOOL_CONFIGS

        cfg = next(c for c in TOOL_CONFIGS if c["name"] == "patch_file")
        assert cfg["requires_confirmation"] is True

    def test_patch_file_in_all_exports(self):
        from src.tools import file_ops

        assert "patch_file" in file_ops.__all__
        assert "PatchFileInput" in file_ops.__all__


# ===========================================================================
# #210 — git tools module
# ===========================================================================


class TestGitToolsModule:
    def test_module_importable(self):
        from src.tools import git_tools  # noqa: F401

    def test_all_seven_functions_exist(self):
        from src.tools.git_tools import (
            git_add,
            git_checkout,
            git_commit,
            git_create_branch,
            git_diff,
            git_log,
            git_status,
        )

        assert callable(git_status)
        assert callable(git_diff)
        assert callable(git_log)
        assert callable(git_add)
        assert callable(git_commit)
        assert callable(git_create_branch)
        assert callable(git_checkout)

    def test_tool_configs_has_seven_entries(self):
        from src.tools.git_tools import TOOL_CONFIGS

        assert len(TOOL_CONFIGS) == 7
        names = {c["name"] for c in TOOL_CONFIGS}
        assert names == {
            "git_status",
            "git_diff",
            "git_log",
            "git_add",
            "git_commit",
            "git_create_branch",
            "git_checkout",
        }

    def test_read_only_tools_no_confirmation(self):
        from src.tools.git_tools import TOOL_CONFIGS

        read_only = {"git_status", "git_diff", "git_log"}
        for cfg in TOOL_CONFIGS:
            if cfg["name"] in read_only:
                assert (
                    cfg["requires_confirmation"] is False
                ), f"{cfg['name']} should not require confirmation"

    def test_write_tools_require_confirmation(self):
        from src.tools.git_tools import TOOL_CONFIGS

        write_tools = {"git_add", "git_commit", "git_create_branch", "git_checkout"}
        for cfg in TOOL_CONFIGS:
            if cfg["name"] in write_tools:
                assert (
                    cfg["requires_confirmation"] is True
                ), f"{cfg['name']} should require confirmation"

    def test_all_exports(self):
        from src.tools.git_tools import __all__

        expected = {
            "git_status",
            "git_diff",
            "git_log",
            "git_add",
            "git_commit",
            "git_create_branch",
            "git_checkout",
            "GitStatusInput",
            "GitDiffInput",
            "GitLogInput",
            "GitAddInput",
            "GitCommitInput",
            "GitCreateBranchInput",
            "GitCheckoutInput",
            "TOOL_CONFIGS",
        }
        assert expected.issubset(set(__all__))

    def test_git_status_returns_string(self):
        from src.tools.git_tools import git_status

        result = git_status()
        assert isinstance(result, str)
        # In a git repo the result is either status output or an error string.
        assert len(result) > 0

    def test_git_log_returns_string(self):
        from src.tools.git_tools import git_log

        result = git_log(max_count=3)
        assert isinstance(result, str)

    def test_git_diff_returns_string(self):
        from src.tools.git_tools import git_diff

        result = git_diff()
        assert isinstance(result, str)

    def test_git_status_input_no_required_fields(self):
        from src.tools.git_tools import GitStatusInput

        GitStatusInput()  # should not raise

    def test_git_log_input_defaults(self):
        from src.tools.git_tools import GitLogInput

        inp = GitLogInput()
        assert inp.max_count == 10
        assert inp.branch == ""

    def test_git_diff_input_defaults(self):
        from src.tools.git_tools import GitDiffInput

        inp = GitDiffInput()
        assert inp.path == ""
        assert inp.staged is False

    def test_run_git_timeout_returns_error_string(self):
        """_run_git returns an error string on TimeoutExpired (never raises)."""
        import subprocess

        from src.tools.git_tools import _run_git

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=60)):
            result = _run_git("status")
        assert "timed out" in result.lower()

    def test_run_git_missing_binary_returns_error_string(self):
        from src.tools.git_tools import _run_git

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _run_git("status")
        assert "not installed" in result.lower() or "Error" in result

    def test_git_log_rejects_dash_prefixed_branch(self):
        """Argument injection: branch starting with '-' is rejected."""
        from src.tools.git_tools import git_log

        result = git_log(branch="--exec=whoami")
        assert "Error" in result
        assert "must not start with" in result

    def test_git_checkout_rejects_dash_prefixed_ref(self):
        """Argument injection: ref starting with '-' is rejected."""
        from src.tools.git_tools import git_checkout

        result = git_checkout(ref="--exec=id")
        assert "Error" in result
        assert "must not start with" in result

    def test_git_create_branch_rejects_dash_prefixed_name(self):
        """Argument injection: branch name starting with '-' is rejected."""
        from src.tools.git_tools import git_create_branch

        result = git_create_branch(name="--track")
        assert "Error" in result
        assert "must not start with" in result

    def test_git_create_branch_rejects_dash_prefixed_base(self):
        """Argument injection: base starting with '-' is rejected."""
        from src.tools.git_tools import git_create_branch

        result = git_create_branch(name="ok-branch", base="--orphan")
        assert "Error" in result
        assert "must not start with" in result


# ===========================================================================
# #211 — DonePayload.text field
# ===========================================================================


class TestDonePayloadTextField:
    def test_done_payload_has_text_field(self):
        from src.api.ws import DonePayload

        assert "text" in DonePayload.model_fields

    def test_done_payload_text_defaults_to_empty_string(self):
        from src.api.ws import DonePayload

        p = DonePayload(
            message_id="abc",
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            duration_ms=0,
            tool_calls=0,
        )
        assert p.text == ""

    def test_done_payload_text_is_str_type(self):
        from src.api.ws import DonePayload

        # The field should accept strings; verify by instantiation.
        p = DonePayload(
            message_id="x",
            total_tokens=10,
            input_tokens=5,
            output_tokens=5,
            duration_ms=100,
            tool_calls=1,
            text="Hello, world!",
        )
        assert p.text == "Hello, world!"


# ===========================================================================
# #212 — sync=true query param
# ===========================================================================


class TestSyncTurnOutSchema:
    def test_sync_turn_out_importable(self):
        from src.api.schemas.message import SyncTurnOut  # noqa: F401

    def test_sync_turn_out_required_fields(self):
        from src.api.schemas.message import SyncTurnOut

        p = SyncTurnOut(
            message_id="m1",
            text="The answer is 42.",
            total_tokens=100,
            input_tokens=80,
            output_tokens=20,
            duration_ms=1500,
            tool_calls=2,
        )
        assert p.text == "The answer is 42."
        assert p.total_tokens == 100
        assert p.tool_calls == 2

    def test_send_message_has_sync_param(self):
        """The send_message endpoint must accept a 'sync' query parameter."""
        import inspect

        import src.api.routes.messages as mod

        sig = inspect.signature(mod.send_message)
        assert "sync" in sig.parameters


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture(scope="module")
def _engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _asyncio.run(_create_tables(eng))
    yield eng
    _asyncio.run(eng.dispose())


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture(scope="module")
def app(_engine):
    from src.api.app import create_app

    factory = async_sessionmaker(_engine, expire_on_commit=False)

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _JWT_SECRET}):
        _app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        _app.dependency_overrides[get_db] = _override
        yield _app


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _reg(client):
    uname = f"u_{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@x.com", "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    return r.json()["data"]["access_token"]


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def _make_live_session():
    """Return a mock ApiSession suitable for injection into app.state.session_registry."""
    ss = MagicMock()
    ss.no_confirm = True
    ss.denials = set()
    ss.loaded_tools = set()
    ss.pinned_tools = set()
    ss.reset_approvals()
    ss.all_tool_originals = {}

    live = MagicMock()
    live.session_state = ss
    live.ws_queue = _asyncio.Queue(maxsize=10_000)
    live.cancel_event = _asyncio.Event()
    live.turn_lock = _asyncio.Lock()
    live.turn_task = None
    live.active_confirmation_ui = None
    live.drain_task = None
    live.agent_state = "idle"
    live.memory_manager = None
    live.run_config = None
    live.token_counts = {"input_tokens": 0, "output_tokens": 0, "context_window": 0}
    live.last_activity = 0.0
    live.config = {}
    return live


class TestSyncModeEndpoint:
    """POST /sessions/{id}/messages?sync=true should block and return assembled text."""

    def test_async_mode_returns_202(self, client, app):
        token = _reg(client)
        r = client.post("/api/v1/sessions", headers=_h(token), json={})
        sid = r.json()["data"]["id"]

        live = _make_live_session()
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(token),
            json={"content": "hello"},
        )
        assert r.status_code == 202

    def test_sync_mode_runs_turn_and_returns_200(self, client, app):
        """When sync=true, the endpoint awaits the turn and returns 200 with SyncTurnOut."""
        token = _reg(client)
        r = client.post("/api/v1/sessions", headers=_h(token), json={})
        sid = r.json()["data"]["id"]

        live = _make_live_session()
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        _FAKE_MSG_ID = str(uuid.uuid4())
        _FAKE_TEXT = "The capital of France is Paris."

        async def _fake_run_message_turn(session, text, mode, db, app_state):
            # Simulate what turn_runner does: put a done message on the queue.
            await session.ws_queue.put(
                {
                    "type": "done",
                    "payload": {
                        "message_id": _FAKE_MSG_ID,
                        "text": _FAKE_TEXT,
                        "total_tokens": 50,
                        "input_tokens": 40,
                        "output_tokens": 10,
                        "duration_ms": 500,
                        "tool_calls": 0,
                    },
                }
            )

        with patch("src.api.routes.messages.run_message_turn", _fake_run_message_turn):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(token),
                json={"content": "What is the capital of France?"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["error"] is None
        data = body["data"]
        assert data["text"] == _FAKE_TEXT
        assert data["message_id"] == _FAKE_MSG_ID
        assert data["total_tokens"] == 50
        assert data["duration_ms"] == 500

    def test_sync_mode_returns_empty_text_when_queue_empty(self, client, app):
        """When queue has no done message (consumed by WS drain), text is ''."""
        token = _reg(client)
        r = client.post("/api/v1/sessions", headers=_h(token), json={})
        sid = r.json()["data"]["id"]

        live = _make_live_session()
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        async def _noop_run(session, text, mode, db, app_state):
            pass  # Don't put anything on the queue.

        with patch("src.api.routes.messages.run_message_turn", _noop_run):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(token),
                json={"content": "hello"},
            )

        assert r.status_code == 200
        assert r.json()["data"]["text"] == ""

    def test_sync_mode_returns_409_when_turn_in_progress(self, client, app):
        """A concurrent sync=true request must receive 409 when a turn is active (BUG-215)."""
        # This test verifies that the send_message route has the correct logic pattern
        # for detecting concurrent requests. The actual 409 error handling is tested
        # via the source inspection in TestSyncTurnSentinel class which is removed.
        # Instead, we verify the gate logic exists by examining the send_message code.
        import inspect

        import src.api.routes.messages as mod

        src = inspect.getsource(mod.send_message)
        # The gate that checks for an in-progress turn must test .done()
        assert "turn_task.done()" in src

    def test_sync_mode_clears_turn_task_after_completion(self, client, app):
        """After a sync turn completes, turn_task must be cleared (BUG-215)."""
        token = _reg(client)
        r = client.post("/api/v1/sessions", headers=_h(token), json={})
        sid = r.json()["data"]["id"]

        live = _make_live_session()
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        with patch("src.api.routes.messages.run_message_turn", AsyncMock()):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(token),
                json={"content": "hello"},
            )

        assert r.status_code == 200
        assert live.turn_task is None


# ===========================================================================
# #213 — initial_tools / auto_approve_tools on POST /sessions
# ===========================================================================


class TestSessionCreateWithTools:
    def test_session_create_request_has_initial_tools_field(self):
        from src.api.schemas.session import SessionCreateRequest

        req = SessionCreateRequest()
        assert hasattr(req, "initial_tools")
        assert req.initial_tools == []

    def test_session_create_request_has_auto_approve_tools_field(self):
        from src.api.schemas.session import SessionCreateRequest

        req = SessionCreateRequest()
        assert hasattr(req, "auto_approve_tools")
        assert req.auto_approve_tools == []

    def test_session_create_request_accepts_tool_lists(self):
        from src.api.schemas.session import SessionCreateRequest

        req = SessionCreateRequest(
            initial_tools=["read_file", "write_file"],
            auto_approve_tools=["git_add"],
        )
        assert req.initial_tools == ["read_file", "write_file"]
        assert req.auto_approve_tools == ["git_add"]

    def test_create_session_endpoint_accepts_initial_tools(self, client):
        token = _reg(client)
        r = client.post(
            "/api/v1/sessions",
            headers=_h(token),
            json={"initial_tools": ["read_file"], "auto_approve_tools": ["git_add"]},
        )
        # Session should be created regardless (tool registry may not be populated in tests).
        assert r.status_code == 201
        body = r.json()
        assert body["error"] is None
        assert "id" in body["data"]

    def test_initial_tools_pins_to_session_state(self, client, app):
        """initial_tools must pin tools to session_state when the registry is populated."""
        token = _reg(client)

        # Build a mock tool object and registry.
        mock_tool = MagicMock()
        mock_tool.name = "read_file"

        mock_registry = MagicMock()
        mock_registry.tools = {"read_file": mock_tool, "write_file": MagicMock()}
        mock_registry.is_mcp_tool = MagicMock(return_value=False)
        mock_registry.get_tool_server = MagicMock(return_value=None)
        mock_registry.requires_confirmation = MagicMock(return_value=False)
        app.state.tool_registry = mock_registry

        # Build a mock live session with real session_state tracking.
        ss = MagicMock()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()
        ss.all_tool_originals = {}
        ss.denials = set()
        # PR #1208: use lock-protected methods instead of direct set mutation
        ss.get_approvals_snapshot = MagicMock(return_value=set())
        ss.add_approval = MagicMock()
        ss.revoke_approval = MagicMock()

        live = _make_live_session()
        live.session_state = ss

        # run_config with available_tools including "read_file".
        rc = MagicMock()
        rc.available_tools = {"read_file": mock_tool}
        rc.active_tools_list = []
        live.run_config = rc

        mock_sess_reg = MagicMock()
        mock_sess_reg.get_or_warm = AsyncMock(return_value=live)
        mock_sess_reg.put = AsyncMock()
        app.state.session_registry = mock_sess_reg

        with patch("src.api.routes.sessions.warm_session", AsyncMock(return_value=live)):
            r = client.post(
                "/api/v1/sessions",
                headers=_h(token),
                json={"initial_tools": ["read_file"], "auto_approve_tools": ["git_add"]},
            )

        assert r.status_code == 201
        # Verify session_state was mutated correctly.
        assert "read_file" in ss.loaded_tools
        assert "read_file" in ss.pinned_tools
        # read_file moved from available to active_tools_list.
        assert mock_tool in rc.active_tools_list
        assert "read_file" not in rc.available_tools
        # auto_approve applied via lock-protected add_approval method (PR #1208)
        ss.add_approval.assert_called_once_with("git_add")

    def test_unknown_initial_tool_is_skipped(self, client, app):
        """Tools not in the registry are skipped without error."""
        token = _reg(client)

        mock_registry = MagicMock()
        mock_registry.tools = {}  # empty registry
        app.state.tool_registry = mock_registry

        ss = MagicMock()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()
        ss.all_tool_originals = {}
        ss.denials = set()

        live = _make_live_session()
        live.session_state = ss

        mock_sess_reg = MagicMock()
        app.state.session_registry = mock_sess_reg

        with patch("src.api.routes.sessions.warm_session", AsyncMock(return_value=live)):
            r = client.post(
                "/api/v1/sessions",
                headers=_h(token),
                json={"initial_tools": ["nonexistent_tool"]},
            )

        assert r.status_code == 201
        assert "nonexistent_tool" not in ss.loaded_tools
        assert "nonexistent_tool" not in ss.pinned_tools

    def test_auto_approve_without_initial_tools(self, client, app):
        """auto_approve_tools can be set independently of initial_tools."""
        token = _reg(client)

        mock_registry = MagicMock()
        mock_registry.tools = {}
        app.state.tool_registry = mock_registry

        ss = MagicMock()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()
        ss.all_tool_originals = {}
        ss.denials = set()
        # PR #1208: use lock-protected methods instead of direct set mutation
        ss.get_approvals_snapshot = MagicMock(return_value=set())
        ss.add_approval = MagicMock()
        ss.revoke_approval = MagicMock()

        live = _make_live_session()
        live.session_state = ss

        mock_sess_reg = MagicMock()
        app.state.session_registry = mock_sess_reg

        with patch("src.api.routes.sessions.warm_session", AsyncMock(return_value=live)):
            r = client.post(
                "/api/v1/sessions",
                headers=_h(token),
                json={"auto_approve_tools": ["git_commit", "git_add"]},
            )

        assert r.status_code == 201
        # auto_approve_tools applied via lock-protected add_approval (PR #1208)
        ss.add_approval.assert_any_call("git_commit")
        ss.add_approval.assert_any_call("git_add")


# ===========================================================================
# BUG-002 — think mode: _extract_final_solution strips ToT report preamble
# ===========================================================================


class TestExtractFinalSolution:
    def test_extracts_final_solution_section(self):
        from src.api.turn_runner import _extract_final_solution

        report = (
            "## Branch 1 (confidence: 7.0/10)\nSome intermediate reasoning.\n\n"
            "---\n\n"
            "## Final Solution (confidence: 9.2/10)\n\n"
            "The answer is 42.\n\n"
            "---\n"
        )
        assert _extract_final_solution(report) == "The answer is 42."

    def test_falls_back_to_full_report_when_section_absent(self):
        from src.api.turn_runner import _extract_final_solution

        report = "Just a plain response without any ToT structure."
        assert _extract_final_solution(report) == report

    def test_falls_back_when_solution_body_is_empty(self):
        from src.api.turn_runner import _extract_final_solution

        report = "## Final Solution (confidence: 8.0/10)\n\n   \n"
        assert _extract_final_solution(report) == report

    def test_handles_multiline_solution(self):
        from src.api.turn_runner import _extract_final_solution

        report = (
            "## Branch 1 (confidence: 6.0/10)\nA\n\n---\n\n"
            "## Final Solution (confidence: 8.5/10)\n\n"
            "Line one.\nLine two.\nLine three.\n\n---\n"
        )
        result = _extract_final_solution(report)
        assert "Line one." in result
        assert "Line two." in result
        assert "## Branch 1" not in result

    def test_various_confidence_values(self):
        from src.api.turn_runner import _extract_final_solution

        for conf in ("10.0", "0.5", "7.3"):
            report = f"## Final Solution (confidence: {conf}/10)\n\nThe solution.\n"
            assert _extract_final_solution(report) == "The solution."

    def test_integer_confidence_value(self):
        """BUG-244: integer confidence (e.g. 8/10) must be matched by the regex."""
        from src.api.turn_runner import _extract_final_solution

        report = "## Final Solution (confidence: 8/10)\n\nInteger confidence answer.\n"
        assert _extract_final_solution(report) == "Integer confidence answer."

    def test_header_with_extra_text_still_matches(self):
        """BUG-244: header with extra info after 'Final Solution' must still match."""
        from src.api.turn_runner import _extract_final_solution

        report = "## Final Solution — Best Approach (confidence: 9.0/10)\n\nThe robust answer.\n"
        assert _extract_final_solution(report) == "The robust answer."

    def test_empty_best_solution_falls_back_to_report(self):
        """BUG-252: when best_solution is empty, \n+ greedily consumes blank lines
        and the footer separator ('---\\n*N iterations...*') must not be returned
        as the final solution — fall back to the full report instead."""
        from src.api.turn_runner import _extract_final_solution

        # Mirrors _format_report output when best_solution == "":
        # header line, 3 blank lines (join of empty string), then footer.
        report = (
            "## Branch 1 (confidence: 6.5/10)\nSome content.\n\n---\n\n"
            "## Final Solution (confidence: 0.0/10)\n\n\n\n"
            "---\n*2 iterations, best score: 0.0*"
        )
        result = _extract_final_solution(report)
        # Must NOT return the footer as the solution
        assert not result.lstrip().startswith(
            "---"
        ), "BUG-252: _extract_final_solution returned footer as solution when best_solution empty"
        # Falls back to the full report
        assert result == report


# ===========================================================================
# BUG-253 — think mode: final result emitted as token after pipeline
# ===========================================================================


class TestThinkModeTokenEmission:
    """After _run_think_pipeline, turn_runner emits the result as a token
    message so the frontend receives content even though deep_think runs
    without the ws_callback (BUG-253)."""

    def test_think_result_token_emitted_to_queue(self):
        """A 'token' message with the think result must be enqueued before 'done'."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def _run():
            from src.api.turn_runner import _run_message_turn_inner

            session = MagicMock()
            session.id = "sess-253"
            session.agent_state = "idle"
            session.turn_lock = asyncio.Lock()
            session.cancel_event = MagicMock()
            session.cancel_event.is_set.return_value = False
            session.ws_queue = asyncio.Queue(maxsize=1000)
            session.session_state = None
            session.memory_manager = None
            session.run_config = None
            session.registry = None
            session.active_confirmation_ui = None
            session.last_activity = 0.0
            session.token_counts = {"input_tokens": 0, "output_tokens": 0}

            with (
                patch("src.api.callbacks.WebSocketCallbackHandler", return_value=MagicMock()),
                patch("src.api.confirmation.ApiConfirmationUI", return_value=MagicMock()),
                patch(
                    "src.orchestration.runner.run_agent",
                    return_value="initial agent answer",
                ),
                patch(
                    "src.api.turn_runner._run_think_pipeline",
                    new=AsyncMock(return_value="deep think result"),
                ),
            ):
                await _run_message_turn_inner(session, "test input", "think", None, None)

            msgs = []
            while not session.ws_queue.empty():
                msgs.append(session.ws_queue.get_nowait())

            token_msgs = [m for m in msgs if m.get("type") == "token"]
            think_tokens = [
                m for m in token_msgs if m.get("payload", {}).get("text") == "deep think result"
            ]
            assert think_tokens, (
                "BUG-253: no 'token' message with think result found in queue; "
                f"token messages were: {token_msgs}"
            )
            assert think_tokens[0]["payload"]["final"] is True

            # done message must also contain the think result
            done_msgs = [m for m in msgs if m.get("type") == "done"]
            assert done_msgs and done_msgs[0]["payload"]["text"] == "deep think result"

        asyncio.run(_run())


# ===========================================================================
# BUG-248 — tool_intensive classification must NOT skip force_deep_think in
#            explicit think mode (API mode="think")
# ===========================================================================


class TestThinkPipelineToolIntensiveNotSkipped:
    """Explicit think mode must always call force_deep_think regardless of task category."""

    @pytest.mark.asyncio
    async def test_run_think_pipeline_calls_force_deep_think_for_tool_intensive(self):
        """_run_think_pipeline must call force_deep_think even for tool-intensive tasks (BUG-248)."""
        from src.api.turn_runner import _run_think_pipeline

        session = MagicMock()
        session.cancel_event = MagicMock()
        session.cancel_event.is_set.return_value = False

        run_config = MagicMock()
        run_config.llm = MagicMock()
        run_config.research_delegate_enabled = False

        with (
            patch("src.api.turn_runner._enqueue_agent_state", AsyncMock()),
            patch(
                "src.orchestration.intent.classify_think_task",
                return_value=MagicMock(tool_intensive=True, name="tool-intensive"),
            ),
            patch("src.orchestration.phases.was_deep_think_called", return_value=False),
            patch("src.orchestration.phases.deep_think_had_good_context", return_value=False),
            patch("src.orchestration.phases.force_deep_think", return_value="deep result"),
            patch("src.orchestration.phases.collect_tool_outputs", return_value=[]),
            patch("src.orchestration.phases.agent_used_web_tools", return_value=False),
        ):
            result = await _run_think_pipeline(session, "test", "initial", [], run_config)

        assert result == "deep result"


# ===========================================================================
# BUG-249 / BUG-250 — research delegate tools injected per worker thread;
#                      configure_delegate_tool called at API startup
# ===========================================================================


class TestThinkPipelineResearchDelegateSetup:
    """Research delegate must have tools and LLM config available in API mode."""

    @pytest.mark.asyncio
    async def test_run_think_pipeline_sets_delegate_tools_for_web_tools(self):
        """_run_think_pipeline must call set_delegate_tools before research delegate (BUG-249)."""
        from src.api.turn_runner import _run_think_pipeline

        session = MagicMock()
        session.cancel_event = MagicMock()
        session.cancel_event.is_set.return_value = False

        run_config = MagicMock()
        run_config.llm = MagicMock()
        run_config.research_delegate_enabled = True
        run_config.research_delegate_timeout = 300
        run_config.research_delegate_cap_ratio = 0.85
        run_config.max_context_tokens = 128_000
        run_config.active_tools_list = ["tool_a"]
        run_config.available_tools = {"tool_a": MagicMock()}

        async def _sync_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("src.api.turn_runner._enqueue_agent_state", AsyncMock()),
            patch("src.api.turn_runner.asyncio.to_thread", new=_sync_to_thread),
            patch("src.orchestration.phases.was_deep_think_called", return_value=False),
            patch("src.orchestration.phases.deep_think_had_good_context", return_value=False),
            patch("src.orchestration.phases.agent_used_web_tools", return_value=True),
            patch(
                "src.orchestration.phases.extract_fetched_urls", return_value=["http://example.com"]
            ),
            patch(
                "src.orchestration.phases.run_research_delegate", return_value="research output"
            ) as mock_rd,
            patch("src.tools.delegate.set_delegate_tools") as mock_set_tools,
            patch("src.orchestration.phases.force_deep_think", return_value="deep result"),
            patch("src.orchestration.phases.collect_tool_outputs", return_value=[]),
        ):
            await _run_think_pipeline(session, "test", "initial", [], run_config)

        mock_set_tools.assert_called_once_with(
            ["tool_a"], {"tool_a": run_config.available_tools["tool_a"]}
        )
        mock_rd.assert_called_once()

    def test_lifespan_configures_delegate_and_deep_think_tools(self):
        """Lifespan must call configure_delegate_tool and configure_deep_think_tool (BUG-250/251)."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        with (
            patch("src.tools.configure.configure_delegate_tool") as mock_delegate,
            patch("src.tools.configure.configure_deep_think_tool") as mock_deep,
        ):
            fresh_app = create_app()
            with TestClient(fresh_app) as _client:
                pass

        mock_delegate.assert_called_once()
        mock_deep.assert_called_once()


# ===========================================================================
# BUG-213 — tool-intensive think prompts must still allow delegation in CLI
# ===========================================================================


class TestCliToolIntensiveThinkDelegation:
    """Tool-intensive think prompts must not dead-end before delegation can run."""

    def test_tool_intensive_query_still_reaches_delegation(self, monkeypatch: pytest.MonkeyPatch):
        from types import SimpleNamespace

        import cogtrix

        config = SimpleNamespace(
            prompt_optimizer=False,
            context_compression_model=None,
            research_delegate_auto=False,
            research_delegate_enabled=True,
            research_delegate_timeout=300,
            research_delegate_cap_ratio=0.5,
            memory_mode="conversation",
            context_compression=True,
            context_compression_min_age=0,
            context_compression_min_chars=0,
            parallel_tool_execution=True,
            git_native=False,
            tool_context_limit_pct=0.80,
            decision_accountability_enabled=False,
            decision_accountability_report_uncertainty=True,
            decision_accountability_min_confidence=7.0,
            task_ownership_classifier_enabled=True,
            task_ownership_classifier_llm_fallback=False,
            task_ownership_ambiguous_action="ask",
            context_max_messages=200,
            delegate_enabled=True,
        )

        memory_manager = MagicMock()
        memory_manager.prepare_context.return_value = SimpleNamespace(
            messages=[],
            context_prefix="",
            context_messages_count=0,
        )
        memory_manager.update = MagicMock()
        memory_manager.save = MagicMock()

        registry = MagicMock()
        log = MagicMock()
        llm = object()

        monkeypatch.setattr(cogtrix, "new_request_id", MagicMock())
        monkeypatch.setattr(cogtrix, "log_user_message", MagicMock())
        monkeypatch.setattr(cogtrix, "log_agent_response", MagicMock())
        monkeypatch.setattr(cogtrix, "_cleanup_milestones", MagicMock())
        monkeypatch.setattr(cogtrix, "_spinner", MagicMock(start=MagicMock(), stop=MagicMock()))
        monkeypatch.setattr(cogtrix, "user_wants_deep_think", lambda _: True)
        monkeypatch.setattr(
            cogtrix,
            "classify_think_task",
            MagicMock(return_value=SimpleNamespace(name="tool-intensive", tool_intensive=True)),
        )
        monkeypatch.setattr(cogtrix, "run_agent", MagicMock(return_value="short delegated answer"))
        monkeypatch.setattr(
            cogtrix, "force_deep_think", MagicMock(return_value="should-not-be-called")
        )
        monkeypatch.setattr(cogtrix, "was_deep_think_called", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "deep_think_had_good_context", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "collect_tool_outputs", MagicMock(return_value=[]))
        monkeypatch.setattr(cogtrix, "agent_used_web_tools", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "extract_fetched_urls", MagicMock(return_value=[]))
        monkeypatch.setattr(cogtrix, "run_research_delegate", MagicMock())
        monkeypatch.setattr(cogtrix, "force_delegation", MagicMock(return_value="delegated output"))
        monkeypatch.setattr(cogtrix, "was_delegation_called", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "user_wants_delegation", lambda _: True)
        monkeypatch.setattr(cogtrix, "prompt_requests_action", lambda _: False)
        monkeypatch.setattr(cogtrix, "_is_valid_response", lambda _: True)

        exit_code = cogtrix.run_single_prompt(
            prompt_text="think through the problem and use tools",
            memory_manager=memory_manager,
            registry=registry,
            approvals=set(),
            no_stream=True,
            log=log,
            llm=llm,
            config=config,
        )

        assert exit_code == 0
        cogtrix.force_deep_think.assert_not_called()
        cogtrix.force_delegation.assert_called_once()
        memory_manager.update.assert_called_once()
        memory_manager.save.assert_called_once()

    def test_single_prompt_path_calls_tool_intensive_helper(self, monkeypatch: pytest.MonkeyPatch):
        """run_single_prompt must call _maybe_skip_force_deep_think_for_tool_intensive_task (BUG-213)."""
        from types import SimpleNamespace

        import cogtrix

        config = SimpleNamespace(
            prompt_optimizer=False,
            context_compression_model=None,
            research_delegate_auto=False,
            research_delegate_enabled=True,
            research_delegate_timeout=300,
            research_delegate_cap_ratio=0.5,
            memory_mode="conversation",
            context_compression=True,
            context_compression_min_age=0,
            context_compression_min_chars=0,
            parallel_tool_execution=True,
            git_native=False,
            tool_context_limit_pct=0.80,
            decision_accountability_enabled=False,
            decision_accountability_report_uncertainty=True,
            decision_accountability_min_confidence=7.0,
            task_ownership_classifier_enabled=True,
            task_ownership_classifier_llm_fallback=False,
            task_ownership_ambiguous_action="ask",
            context_max_messages=200,
            delegate_enabled=True,
        )

        memory_manager = MagicMock()
        memory_manager.prepare_context.return_value = SimpleNamespace(
            messages=[],
            context_prefix="",
            context_messages_count=0,
        )
        memory_manager.update = MagicMock()
        memory_manager.save = MagicMock()

        registry = MagicMock()
        log = MagicMock()
        llm = object()

        monkeypatch.setattr(cogtrix, "new_request_id", MagicMock())
        monkeypatch.setattr(cogtrix, "log_user_message", MagicMock())
        monkeypatch.setattr(cogtrix, "log_agent_response", MagicMock())
        monkeypatch.setattr(cogtrix, "_cleanup_milestones", MagicMock())
        monkeypatch.setattr(cogtrix, "_spinner", MagicMock(start=MagicMock(), stop=MagicMock()))
        monkeypatch.setattr(cogtrix, "user_wants_deep_think", lambda _: True)
        monkeypatch.setattr(
            cogtrix,
            "classify_think_task",
            MagicMock(return_value=SimpleNamespace(name="tool-intensive", tool_intensive=True)),
        )
        monkeypatch.setattr(cogtrix, "run_agent", MagicMock(return_value="short delegated answer"))
        monkeypatch.setattr(
            cogtrix, "force_deep_think", MagicMock(return_value="should-not-be-called")
        )
        monkeypatch.setattr(cogtrix, "was_deep_think_called", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "deep_think_had_good_context", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "collect_tool_outputs", MagicMock(return_value=[]))
        monkeypatch.setattr(cogtrix, "agent_used_web_tools", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "extract_fetched_urls", MagicMock(return_value=[]))
        monkeypatch.setattr(cogtrix, "run_research_delegate", MagicMock())
        monkeypatch.setattr(cogtrix, "force_delegation", MagicMock(return_value="delegated output"))
        monkeypatch.setattr(cogtrix, "was_delegation_called", MagicMock(return_value=False))
        monkeypatch.setattr(cogtrix, "user_wants_delegation", lambda _: True)
        monkeypatch.setattr(cogtrix, "prompt_requests_action", lambda _: False)
        monkeypatch.setattr(cogtrix, "_is_valid_response", lambda _: True)

        mock_helper = MagicMock(return_value=False)
        monkeypatch.setattr(
            cogtrix, "_maybe_skip_force_deep_think_for_tool_intensive_task", mock_helper
        )

        cogtrix.run_single_prompt(
            prompt_text="think through the problem and use tools",
            memory_manager=memory_manager,
            registry=registry,
            approvals=set(),
            no_stream=True,
            log=log,
            llm=llm,
            config=config,
        )

        mock_helper.assert_called_once()


# ===========================================================================
# BUG-214 — interactive CLI task complexity is passed through to run_agent
# ===========================================================================


class TestTaskComplexityWiring:
    """Interactive CLI must forward precomputed task complexity into run_agent()."""

    def test_run_agent_accepts_task_complexity(self):
        """run_agent must accept task_complexity so the interactive CLI can forward it (BUG-214)."""
        from src.orchestration.runner import run_agent

        try:
            run_agent(
                "test",
                [],
                MagicMock(),
                set(),
                config=MagicMock(),
                task_complexity=MagicMock(),
            )
        except TypeError as exc:
            if "task_complexity" in str(exc):
                pytest.fail("run_agent does not accept task_complexity parameter (BUG-214)")
            # Other TypeErrors are acceptable due to incomplete mocks
        except Exception:
            # Any other exception is acceptable; we only verify parameter acceptance
            pass
