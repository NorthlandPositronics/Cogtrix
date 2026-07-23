"""Tests for src/memory/json_store.py — JsonFileMemoryStore and helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.memory.json_store import (
    JsonFileMemoryStore,
    _dict_to_message,
    _message_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers to create message objects when LangChain is available
# ---------------------------------------------------------------------------

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    _LC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LC_AVAILABLE = False
    AIMessage = None  # type: ignore[assignment, misc]
    HumanMessage = None  # type: ignore[assignment, misc]
    ToolMessage = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# _message_to_dict
# ---------------------------------------------------------------------------


class TestMessageToDict:
    def test_human_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = HumanMessage(content="hello")
        d = _message_to_dict(msg)
        assert d["type"] == "human"
        assert d["content"] == "hello"

    def test_ai_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = AIMessage(content="response")
        d = _message_to_dict(msg)
        assert d["type"] == "ai"
        assert d["content"] == "response"

    def test_ai_message_with_tool_calls(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        tool_calls = [{"id": "tc1", "name": "my_tool", "args": {"x": 1}}]
        msg = AIMessage(content="", tool_calls=tool_calls)
        d = _message_to_dict(msg)
        assert d["type"] == "ai"
        assert "tool_calls" in d
        assert d["tool_calls"][0]["name"] == "my_tool"
        assert d["tool_calls"][0]["args"] == {"x": 1}
        assert d["tool_calls"][0]["id"] == "tc1"

    def test_tool_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = ToolMessage(content="result", name="my_tool", tool_call_id="tc1")
        d = _message_to_dict(msg)
        assert d["type"] == "tool"
        assert d["content"] == "result"
        assert d["name"] == "my_tool"
        assert d["tool_call_id"] == "tc1"

    def test_tool_message_non_string_content(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = ToolMessage(content=["list", "content"], name="t", tool_call_id="tc2")
        d = _message_to_dict(msg)
        assert isinstance(d["content"], str)

    def test_plain_dict_human(self):
        d = _message_to_dict({"type": "human", "content": "hi"})
        assert d["type"] == "human"
        assert d["content"] == "hi"

    def test_plain_dict_ai_with_tool_calls(self):
        tc = [{"id": "x", "name": "fn", "args": {}}]
        d = _message_to_dict({"type": "ai", "content": "ok", "tool_calls": tc})
        assert d["tool_calls"] == tc

    def test_plain_dict_with_timestamp(self):
        d = _message_to_dict({"type": "human", "content": "ts", "timestamp": "2024-01-01"})
        assert d["timestamp"] == "2024-01-01"

    def test_plain_dict_tool_fields_forwarded(self):
        d = _message_to_dict(
            {"type": "tool", "content": "out", "name": "fn", "tool_call_id": "abc"}
        )
        assert d["name"] == "fn"
        assert d["tool_call_id"] == "abc"

    def test_fallback_non_dict_non_message(self):
        d = _message_to_dict("raw string")
        assert d["type"] == "human"
        assert d["content"] == "raw string"

    def test_ai_message_timestamp_preserved(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = AIMessage(content="x", additional_kwargs={"_ts": "2024-06-01T12:00:00"})
        d = _message_to_dict(msg)
        assert d.get("timestamp") == "2024-06-01T12:00:00"


# ---------------------------------------------------------------------------
# _dict_to_message
# ---------------------------------------------------------------------------


class TestDictToMessage:
    def test_human_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = _dict_to_message({"type": "human", "content": "hello"})
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"

    def test_ai_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = _dict_to_message({"type": "ai", "content": "reply"})
        assert isinstance(msg, AIMessage)
        assert msg.content == "reply"

    def test_ai_message_with_tool_calls(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        tc = [{"id": "tc1", "name": "fn", "args": {"k": "v"}}]
        msg = _dict_to_message({"type": "ai", "content": "", "tool_calls": tc})
        assert isinstance(msg, AIMessage)
        # tool_calls preserved in the reconstructed message
        assert msg.tool_calls is not None

    def test_tool_message(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = _dict_to_message(
            {"type": "tool", "content": "result", "name": "fn", "tool_call_id": "tc1"}
        )
        assert isinstance(msg, ToolMessage)
        assert msg.content == "result"
        assert msg.tool_call_id == "tc1"

    def test_fallback_without_langchain(self):
        with patch("src.memory.json_store.HumanMessage", None):
            with patch("src.memory.json_store.AIMessage", None):
                with patch("src.memory.json_store.ToolMessage", None):
                    result = _dict_to_message({"type": "human", "content": "hello"})
                    assert isinstance(result, dict)
                    assert result["content"] == "hello"

    def test_timestamp_preserved(self):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        msg = _dict_to_message(
            {"type": "human", "content": "x", "timestamp": "2024-01-01T00:00:00"}
        )
        assert isinstance(msg, HumanMessage)
        assert msg.additional_kwargs.get("_ts") == "2024-01-01T00:00:00"


# ---------------------------------------------------------------------------
# JsonFileMemoryStore
# ---------------------------------------------------------------------------


class TestJsonFileMemoryStore:
    def test_load_returns_empty_for_nonexistent_session(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        result = store.load_history("no-such-session")
        assert result == []

    def test_save_and_load_roundtrip(self, tmp_path):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        store.save_history("my-session", msgs)
        loaded = store.load_history("my-session")
        assert len(loaded) == 2
        assert isinstance(loaded[0], HumanMessage)
        assert loaded[0].content == "hi"
        assert loaded[1].content == "hello"

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        session_file = tmp_path / "corrupt.json"
        session_file.write_text("{bad json", encoding="utf-8")
        result = store.load_history("corrupt")
        assert result == []

    def test_load_ioerror_returns_empty(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        # Create a directory where the file would be (causes read error)
        session_dir = tmp_path / "ioerror.json"
        session_dir.mkdir()
        result = store.load_history("ioerror")
        assert result == []

    def test_save_disabled_when_base_dir_creation_fails(self):
        with patch("pathlib.Path.mkdir", side_effect=OSError("no permission")):
            store = JsonFileMemoryStore(base_dir="/nonexistent/path/xyz")
            assert store._save_disabled is True

    def test_save_no_op_when_disabled(self, tmp_path):
        if not _LC_AVAILABLE:
            pytest.skip("langchain not installed")
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        store._save_disabled = True
        store.save_history("s", [HumanMessage(content="hi")])
        # File should not exist
        assert not (tmp_path / "s.json").exists()

    def test_save_failure_increments_counter(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        with patch("tempfile.mkstemp", side_effect=OSError("disk full")):
            store.save_history("s", [])
        assert store._consecutive_save_failures == 1

    def test_save_disables_after_three_consecutive_failures(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        with patch("tempfile.mkstemp", side_effect=OSError("disk full")):
            store.save_history("s", [])
            store.save_history("s", [])
            store.save_history("s", [])
        assert store._save_disabled is True

    def test_save_resets_failure_counter_on_success(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        store._consecutive_save_failures = 2
        store.save_history("s", [])
        assert store._consecutive_save_failures == 0

    def test_session_path_sanitizes_traversal(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        # Should not raise; traversal sequences are sanitized
        path = store._session_path("../evil")
        assert ".." not in str(path)

    def test_session_path_truncates_long_id(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        long_id = "a" * 300
        path = store._session_path(long_id)
        assert len(path.stem) <= 128
