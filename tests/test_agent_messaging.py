"""Tests for src/tools/agent_messaging.py."""

from __future__ import annotations

import json
import pathlib
import time

import pytest


@pytest.fixture()
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "data"


@pytest.fixture(autouse=True)
def reset_module(data_dir: pathlib.Path):
    """Patch _data_dir and restore module state after each test."""
    import src.tools.agent_messaging as _mod

    orig = _mod._data_dir
    _mod._data_dir = data_dir
    yield
    _mod._data_dir = orig


# ── send_to_agent ─────────────────────────────────────────────────────────────


class TestSendToAgent:
    def test_creates_inbox_file(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        send_to_agent("alice", "hello")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        assert inbox.exists()

    def test_appends_message_content(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        send_to_agent("alice", "first message")
        send_to_agent("alice", "second message")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        messages = json.loads(inbox.read_text())
        assert len(messages) == 2
        assert messages[0]["message"] == "first message"
        assert messages[1]["message"] == "second message"

    def test_returns_success_string(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        result = send_to_agent("bob", "hi")
        assert result == "Message sent to agent 'bob'"

    def test_prunes_expired_on_write(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        inbox.parent.mkdir(parents=True, exist_ok=True)
        # Write an expired message manually
        old_ts = time.time() - 90000  # 25 hours ago
        inbox.write_text(
            json.dumps([{"from_agent": "x", "message": "old", "sent_at": old_ts, "read": False}]),
            encoding="utf-8",
        )
        send_to_agent("alice", "new message")
        messages = json.loads(inbox.read_text())
        # Old message must be pruned, only new one remains
        assert len(messages) == 1
        assert messages[0]["message"] == "new message"

    def test_from_agent_stored(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        send_to_agent("alice", "hi there", from_agent="manager")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        messages = json.loads(inbox.read_text())
        assert messages[0]["from_agent"] == "manager"

    def test_message_initially_unread(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        send_to_agent("alice", "check this")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        messages = json.loads(inbox.read_text())
        assert messages[0]["read"] is False

    def test_multiple_sends_append_correctly(self, data_dir):
        from src.tools.agent_messaging import send_to_agent

        for i in range(5):
            send_to_agent("alice", f"msg {i}")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        messages = json.loads(inbox.read_text())
        assert len(messages) == 5
        assert [m["message"] for m in messages] == [f"msg {i}" for i in range(5)]

    def test_invalid_name_path_traversal(self):
        from src.tools.agent_messaging import send_to_agent

        result = send_to_agent("../evil", "attack")
        assert "Invalid" in result

    def test_invalid_name_too_long(self):
        from src.tools.agent_messaging import send_to_agent

        result = send_to_agent("a" * 65, "hi")
        assert "Invalid" in result

    def test_invalid_name_slash(self):
        from src.tools.agent_messaging import send_to_agent

        result = send_to_agent("foo/bar", "hi")
        assert "Invalid" in result

    def test_invalid_name_empty(self):
        from src.tools.agent_messaging import send_to_agent

        result = send_to_agent("", "hi")
        assert "Invalid" in result


# ── read_agent_inbox ──────────────────────────────────────────────────────────


class TestReadAgentInbox:
    def test_empty_agent_name_returns_error(self):
        from src.tools.agent_messaging import read_agent_inbox

        assert read_agent_inbox("") == "agent_name is required"

    def test_nonexistent_inbox_returns_empty(self):
        from src.tools.agent_messaging import read_agent_inbox

        assert read_agent_inbox("nobody") == "Inbox empty."

    def test_returns_formatted_messages(self, data_dir):
        from src.tools.agent_messaging import read_agent_inbox, send_to_agent

        send_to_agent("alice", "hello world", from_agent="manager")
        result = read_agent_inbox("alice")
        assert "[1]" in result
        assert "From: manager" in result
        assert "hello world" in result

    def test_read_label_no_before_read(self, data_dir):
        from src.tools.agent_messaging import read_agent_inbox, send_to_agent

        send_to_agent("alice", "test")
        # Peek at file before reading — should be unread
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        msgs_before = json.loads(inbox.read_text())
        assert msgs_before[0]["read"] is False

        result = read_agent_inbox("alice")
        assert "READ: no" not in result  # after read, all are marked yes
        assert "READ: yes" in result

    def test_marks_messages_as_read_in_file(self, data_dir):
        from src.tools.agent_messaging import read_agent_inbox, send_to_agent

        send_to_agent("alice", "one")
        send_to_agent("alice", "two")
        read_agent_inbox("alice")
        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        messages = json.loads(inbox.read_text())
        assert all(m["read"] is True for m in messages)

    def test_prunes_expired_on_read(self, data_dir):
        from src.tools.agent_messaging import read_agent_inbox

        inbox = data_dir / "tasks" / "inbox" / "alice.json"
        inbox.parent.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - 90000
        inbox.write_text(
            json.dumps([{"from_agent": "x", "message": "stale", "sent_at": old_ts, "read": False}]),
            encoding="utf-8",
        )
        result = read_agent_inbox("alice")
        assert result == "Inbox empty."
        # File should be written back as empty list
        messages = json.loads(inbox.read_text())
        assert messages == []

    def test_multiple_messages_formatted_correctly(self, data_dir):
        from src.tools.agent_messaging import read_agent_inbox, send_to_agent

        send_to_agent("alice", "msg one", from_agent="agent_a")
        send_to_agent("alice", "msg two", from_agent="agent_b")
        result = read_agent_inbox("alice")
        assert "[1]" in result
        assert "[2]" in result
        assert "agent_a" in result
        assert "agent_b" in result

    def test_invalid_name_path_traversal(self):
        from src.tools.agent_messaging import read_agent_inbox

        result = read_agent_inbox("../evil")
        assert "Invalid" in result

    def test_invalid_name_too_long(self):
        from src.tools.agent_messaging import read_agent_inbox

        result = read_agent_inbox("x" * 65)
        assert "Invalid" in result


# ── TOOL_CONFIGS ──────────────────────────────────────────────────────────────


class TestToolConfigs:
    def test_has_two_entries(self):
        from src.tools.agent_messaging import TOOL_CONFIGS

        assert len(TOOL_CONFIGS) == 2

    def test_neither_requires_confirmation(self):
        from src.tools.agent_messaging import TOOL_CONFIGS

        for entry in TOOL_CONFIGS:
            assert entry["requires_confirmation"] is False

    def test_correct_tool_names(self):
        from src.tools.agent_messaging import TOOL_CONFIGS

        names = {e["name"] for e in TOOL_CONFIGS}
        assert names == {"send_to_agent", "read_agent_inbox"}


# ── TOOL_SETUP ────────────────────────────────────────────────────────────────


class TestToolSetup:
    def test_sets_data_dir_from_config(self, tmp_path):
        import src.tools.agent_messaging as _mod
        from src.tools.agent_messaging import TOOL_SETUP

        class FakeConfig:
            data_dir = str(tmp_path / "custom")

        TOOL_SETUP(FakeConfig())  # type: ignore[arg-type]
        assert _mod._data_dir == tmp_path / "custom"

    def test_no_data_dir_attr_leaves_default(self, tmp_path):
        import src.tools.agent_messaging as _mod
        from src.tools.agent_messaging import TOOL_SETUP

        class FakeConfig:
            pass

        prev = _mod._data_dir
        TOOL_SETUP(FakeConfig())  # type: ignore[arg-type]
        # _data_dir was already patched to data_dir fixture; should remain unchanged
        assert _mod._data_dir == prev
