"""Tests for cogtrix_core/tools/checkpoint.py — CheckpointStore and create_checkpoint_tool."""

import pytest

from cogtrix_core.tools.checkpoint import CheckpointStore, create_checkpoint_tool


class TestCheckpointStore:
    def test_add_valid_finding_returns_index(self) -> None:
        store = CheckpointStore()
        assert store.add("found it") == (1, False)
        assert store.add("second") == (2, False)

    def test_add_strips_whitespace(self) -> None:
        store = CheckpointStore()
        store.add("  finding  ")
        assert store._findings == ["finding"]

    def test_add_empty_string_raises(self) -> None:
        store = CheckpointStore()
        with pytest.raises(ValueError, match="Checkpoint not recorded: finding must be non-empty"):
            store.add("")

    def test_add_whitespace_only_raises(self) -> None:
        store = CheckpointStore()
        with pytest.raises(ValueError, match="Checkpoint not recorded: finding must be non-empty"):
            store.add("   \t\n  ")

    def test_max_checkpoints_evicts_oldest(self) -> None:
        store = CheckpointStore(max_checkpoints=3)
        store.add("a")
        store.add("b")
        store.add("c")
        idx, evicted = store.add("d")
        assert store._findings == ["b", "c", "d"]
        assert idx == 4
        assert evicted is True

    def test_summary_empty_returns_empty(self) -> None:
        store = CheckpointStore()
        assert store.summary() == ""

    def test_summary_builds_correctly(self) -> None:
        store = CheckpointStore()
        store.add("first")
        store.add("second")
        summary = store.summary()
        assert "  1. first" in summary
        assert "  2. second" in summary

    def test_clear(self) -> None:
        store = CheckpointStore()
        store.add("x")
        store.clear()
        assert len(store) == 0

    def test_len(self) -> None:
        store = CheckpointStore()
        assert len(store) == 0
        store.add("a")
        assert len(store) == 1


class TestCreateCheckpointTool:
    def test_checkpoint_tool_records_finding(self) -> None:
        store = CheckpointStore()
        tool = create_checkpoint_tool(store)
        result = tool.invoke({"finding": "discovery"})
        assert "Checkpoint #1 recorded." in result
        assert "evicted" not in result

    def test_checkpoint_tool_warns_on_eviction(self) -> None:
        store = CheckpointStore(max_checkpoints=2)
        tool = create_checkpoint_tool(store)
        tool.invoke({"finding": "a"})
        tool.invoke({"finding": "b"})
        result = tool.invoke({"finding": "c"})
        assert "Checkpoint #3 recorded." in result
        assert "oldest checkpoint evicted" in result

    def test_checkpoint_tool_rejects_empty(self) -> None:
        store = CheckpointStore()
        tool = create_checkpoint_tool(store)
        result = tool.invoke({"finding": ""})
        assert "Checkpoint not recorded: finding must be non-empty" in result

    def test_checkpoint_tool_rejects_whitespace(self) -> None:
        store = CheckpointStore()
        tool = create_checkpoint_tool(store)
        result = tool.invoke({"finding": "   "})
        assert "Checkpoint not recorded: finding must be non-empty" in result

    def test_checkpoint_tool_strips_whitespace(self) -> None:
        store = CheckpointStore()
        tool = create_checkpoint_tool(store)
        result = tool.invoke({"finding": "  stripped  "})
        assert "Checkpoint #1 recorded." in result
        assert store._findings == ["stripped"]
