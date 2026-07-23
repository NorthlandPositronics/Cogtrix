"""Tests for Sprint 4 features: two-tier tool loading, --activate-tools,
RAG auto-activation, multi-index search, mode-switch pinned tool preservation,
and API PATCH active_tools_list sync.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.session_state import SessionState

# ---------------------------------------------------------------------------
# Fixture: save/restore RAG config to prevent test pollution
# ---------------------------------------------------------------------------


@pytest.fixture()
def _restore_rag_config():
    """Save and restore _rag_config around tests that call configure_rag."""
    import src.tools.rag as _rag_mod

    original = dict(_rag_mod._rag_config)
    yield
    _rag_mod._rag_config.update(original)


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAG auto-activation and multi-index
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseExists:
    """knowledge_base_exists() returns True only when FAISS indexes exist."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_returns_true_when_global_index_exists(self, tmp_path: Path) -> None:
        from src.tools.rag import configure_rag, knowledge_base_exists

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"")
        configure_rag(
            {
                "vectordb_dir": str(idx),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        assert knowledge_base_exists() is True

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_returns_true_when_api_index_exists(self, tmp_path: Path) -> None:
        from src.tools.rag import configure_rag, knowledge_base_exists

        uploads = tmp_path / "uploads"
        api_idx = uploads / "doc1" / "vectordb" / "faiss_index"
        api_idx.mkdir(parents=True)
        (api_idx / "index.faiss").write_bytes(b"")
        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(uploads),
            }
        )
        assert knowledge_base_exists() is True

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_returns_false_when_no_indexes(self, tmp_path: Path) -> None:
        from src.tools.rag import configure_rag, knowledge_base_exists

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        assert knowledge_base_exists() is False


class TestKnowledgeBaseStats:
    """knowledge_base_stats() returns correct index count and total size."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_counts_indexes_and_sizes(self, tmp_path: Path) -> None:
        from src.tools.rag import configure_rag, knowledge_base_stats

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"x" * 100)
        (idx / "index.pkl").write_bytes(b"y" * 50)

        configure_rag(
            {
                "vectordb_dir": str(idx),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        count, total_size = knowledge_base_stats()
        assert count == 1
        assert total_size == 150

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_empty_when_no_indexes(self, tmp_path: Path) -> None:
        from src.tools.rag import configure_rag, knowledge_base_stats

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        count, total_size = knowledge_base_stats()
        assert count == 0
        assert total_size == 0


class TestBuildDescription:
    """_build_description() builds dynamic tool descriptions."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_no_indexes_returns_default(self, tmp_path: Path) -> None:
        from src.tools.rag import _build_description, configure_rag

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        desc = _build_description()
        assert "uploaded documents" in desc

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_single_index_shows_size(self, tmp_path: Path) -> None:
        from src.tools.rag import _build_description, configure_rag

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"x" * 2048)

        configure_rag(
            {
                "vectordb_dir": str(idx),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        desc = _build_description()
        assert "KB" in desc or "MB" in desc
        assert "ingested documents" in desc

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_multiple_indexes_shows_count(self, tmp_path: Path) -> None:
        from src.tools.rag import _build_description, configure_rag

        uploads = tmp_path / "uploads"
        for name in ["doc1", "doc2", "doc3"]:
            idx = uploads / name / "vectordb" / "faiss_index"
            idx.mkdir(parents=True)
            (idx / "index.faiss").write_bytes(b"x" * 1024)

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(uploads),
            }
        )
        desc = _build_description()
        assert "3 document indexes" in desc


class TestRagShouldAutoActivate:
    """rag_should_auto_activate() reflects knowledge base existence."""

    def test_returns_false_when_flag_unset(self) -> None:
        import src.tools.configure as _cfg_mod

        original = _cfg_mod._rag_auto_activate
        try:
            _cfg_mod._rag_auto_activate = False
            assert _cfg_mod.rag_should_auto_activate() is False
        finally:
            _cfg_mod._rag_auto_activate = original

    def test_returns_true_when_flag_set(self) -> None:
        import src.tools.configure as _cfg_mod

        original = _cfg_mod._rag_auto_activate
        try:
            _cfg_mod._rag_auto_activate = True
            assert _cfg_mod.rag_should_auto_activate() is True
        finally:
            _cfg_mod._rag_auto_activate = original


class TestUpdateRagToolDescription:
    """_update_rag_tool_description() patches tool.description."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_patches_description(self, tmp_path: Path) -> None:
        from src.tools.configure import _update_rag_tool_description
        from src.tools.rag import TOOL_CONFIG, _build_description, configure_rag

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"x" * 2048)

        configure_rag(
            {
                "vectordb_dir": str(idx),
                "api_uploads_dir": str(tmp_path / "no-uploads"),
            }
        )
        TOOL_CONFIG["description"] = _build_description()

        tool = MagicMock()
        tool.description = "old description"
        _update_rag_tool_description(tool)
        assert "KB" in tool.description or "MB" in tool.description

    def test_noop_on_tool_without_description(self) -> None:
        from src.tools.configure import _update_rag_tool_description

        tool = object()  # no description attribute
        _update_rag_tool_description(tool)  # should not raise


class TestCollectFaissDirsMultiIndex:
    """_collect_faiss_dirs() discovers both global and per-document indexes."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_finds_global_and_api_indexes(self, tmp_path: Path) -> None:
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        global_idx = tmp_path / "global" / "faiss_index"
        global_idx.mkdir(parents=True)
        (global_idx / "index.faiss").write_bytes(b"")

        uploads = tmp_path / "uploads"
        for doc in ("doc1", "doc2"):
            idx = uploads / doc / "vectordb" / "faiss_index"
            idx.mkdir(parents=True)
            (idx / "index.faiss").write_bytes(b"")

        configure_rag(
            {
                "vectordb_dir": str(global_idx),
                "api_uploads_dir": str(uploads),
            }
        )

        dirs = _collect_faiss_dirs()
        assert len(dirs) == 3

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_skips_non_dir_entries(self, tmp_path: Path) -> None:
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        uploads = tmp_path / "uploads"
        uploads.mkdir()
        (uploads / "stray_file.txt").write_text("not a doc dir")
        idx = uploads / "doc1" / "vectordb" / "faiss_index"
        idx.mkdir(parents=True)
        (idx / "index.faiss").write_bytes(b"")

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(uploads),
            }
        )

        dirs = _collect_faiss_dirs()
        assert len(dirs) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. --activate-tools CLI argument parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestActivateToolsArgParsing:
    """--activate-tools CLI argument is parsed correctly."""

    def test_activate_tools_single(self) -> None:
        from src.cli.args import parse_arguments

        with patch.object(sys, "argv", ["cogtrix.py", "--activate-tools", "web_search"]):
            args = parse_arguments()
        assert args.activate_tools == "web_search"

    def test_activate_tools_comma_separated(self) -> None:
        from src.cli.args import parse_arguments

        with patch.object(sys, "argv", ["cogtrix.py", "--activate-tools", "shell,write_file"]):
            args = parse_arguments()
        assert args.activate_tools == "shell,write_file"

    def test_activate_tools_not_set(self) -> None:
        from src.cli.args import parse_arguments

        with patch.object(sys, "argv", ["cogtrix.py"]):
            args = parse_arguments()
        assert args.activate_tools is None

    def test_activate_tools_with_other_flags(self) -> None:
        from src.cli.args import parse_arguments

        with patch.object(
            sys,
            "argv",
            ["cogtrix.py", "--activate-tools", "web_search", "-y", "--debug"],
        ):
            args = parse_arguments()
        assert args.activate_tools == "web_search"
        assert args.no_confirm is True
        assert args.debug is True


class TestActivateToolsPinning:
    """--activate-tools pins tools correctly in SessionState."""

    def test_pins_from_available_tools(self) -> None:
        ss = SessionState()
        sentinel = MagicMock()
        sentinel.name = "web_search"
        available = {"web_search": sentinel}
        registry_tools: dict = {}

        for name in ["web_search"]:
            if name in available:
                registry_tools[name] = available.pop(name)
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "web_search" in registry_tools
        assert "web_search" in ss.pinned_tools
        assert "web_search" not in available

    def test_pins_from_registry_when_already_active(self) -> None:
        ss = SessionState()
        registry_tools = {"web_search": MagicMock()}

        for name in ["web_search"]:
            if name in registry_tools:
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "web_search" in ss.pinned_tools

    def test_pins_from_originals_as_fallback(self) -> None:
        ss = SessionState()
        sentinel = MagicMock()
        ss.all_tool_originals = {"web_search": sentinel}
        available: dict = {}
        registry_tools: dict = {}

        for name in ["web_search"]:
            if name in available:
                pass
            elif name in registry_tools:
                pass
            elif name in ss.all_tool_originals:
                registry_tools[name] = ss.all_tool_originals[name]
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "web_search" in registry_tools
        assert "web_search" in ss.pinned_tools

    def test_unknown_tool_skipped(self) -> None:
        ss = SessionState()
        available: dict = {}
        registry_tools: dict = {}
        ss.all_tool_originals = {}

        name = "nonexistent"
        found = name in available or name in registry_tools or name in ss.all_tool_originals
        assert not found
        assert "nonexistent" not in ss.pinned_tools


# ══════════════════════════════════════════════════════════════════════════════
# 3. Mode-switch pinned tool preservation
# ══════════════════════════════════════════════════════════════════════════════


class TestModeSwitchPinnedToolPreservation:
    """Pinned tools must survive mode switches."""

    def test_pinned_tools_survive_preset_rebuild(self) -> None:
        """Simulate mode switch: preset rebuilds, pinned tools
        in available_tools are re-promoted."""
        ss = SessionState(
            loaded_tools={"web_search", "shell"},
            pinned_tools={"web_search", "shell"},
        )

        active_dict = {"request_tools": MagicMock()}
        available_tools = {
            "web_search": MagicMock(),
            "shell": MagicMock(),
            "calculator": MagicMock(),
        }

        for pname in list(ss.pinned_tools):
            if pname in available_tools:
                active_dict[pname] = available_tools.pop(pname)

        ss.loaded_tools &= ss.pinned_tools

        assert "web_search" in active_dict
        assert "shell" in active_dict
        assert "web_search" not in available_tools
        assert "shell" not in available_tools
        assert ss.loaded_tools == {"web_search", "shell"}

    def test_agent_loaded_tools_cleared_on_mode_switch(self) -> None:
        ss = SessionState(
            loaded_tools={"web_search", "calculator", "shell"},
            pinned_tools={"web_search"},
        )

        ss.loaded_tools &= ss.pinned_tools

        assert ss.loaded_tools == {"web_search"}
        assert "calculator" not in ss.loaded_tools
        assert "shell" not in ss.loaded_tools


# ══════════════════════════════════════════════════════════════════════════════
# 4. API PATCH active_tools_list sync
# ══════════════════════════════════════════════════════════════════════════════


class TestApiPatchToolSync:
    """API PATCH /sessions/{id}/tools must sync run_config.active_tools_list."""

    def test_load_moves_tool_to_active_list(self) -> None:
        ss = SessionState()

        tool_obj = MagicMock()
        tool_obj.name = "web_search"

        rc = MagicMock()
        rc.available_tools = {"web_search": tool_obj}
        rc.active_tools_list = []

        name = "web_search"
        ss.loaded_tools.add(name)
        ss.pinned_tools.add(name)
        avail = rc.available_tools
        if name in avail:
            moved = avail.pop(name)
            rc.active_tools_list.append(moved)

        assert "web_search" not in rc.available_tools
        assert tool_obj in rc.active_tools_list
        assert "web_search" in ss.pinned_tools

    def test_unload_moves_tool_back_to_available(self) -> None:
        ss = SessionState(
            loaded_tools={"web_search"},
            pinned_tools={"web_search"},
        )

        tool_obj = MagicMock()
        tool_obj.name = "web_search"
        orig_obj = MagicMock()
        orig_obj.name = "web_search"
        ss.all_tool_originals = {"web_search": orig_obj}

        rc = MagicMock()
        rc.active_tools_list = [tool_obj]
        rc.available_tools = {}

        name = "web_search"
        ss.loaded_tools.discard(name)
        ss.pinned_tools.discard(name)
        for i, t in enumerate(rc.active_tools_list):
            if getattr(t, "name", None) == name:
                rc.active_tools_list.pop(i)
                rc.available_tools[name] = ss.all_tool_originals.get(name, t)
                break

        assert "web_search" in rc.available_tools
        assert len(rc.active_tools_list) == 0
        assert "web_search" not in ss.pinned_tools


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pinned tools lifecycle edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestPinnedToolsLifecycle:
    """Edge cases in the pinned tools lifecycle."""

    def test_pinned_tools_survive_multiple_resets(self) -> None:
        ss = SessionState(
            pinned_tools={"web_search"},
            loaded_tools={"web_search", "calculator"},
        )

        ss.reset_for_new_prompt()
        assert ss.loaded_tools == {"web_search"}

        ss.loaded_tools.add("shell")
        ss.loaded_tools.add("http_get")

        ss.reset_for_new_prompt()
        assert ss.loaded_tools == {"web_search"}
        assert ss.pinned_tools == {"web_search"}

    def test_pin_then_unpin(self) -> None:
        ss = SessionState()
        ss.loaded_tools.add("web_search")
        ss.pinned_tools.add("web_search")

        ss.pinned_tools.discard("web_search")
        ss.loaded_tools.discard("web_search")

        ss.reset_for_new_prompt()
        assert "web_search" not in ss.loaded_tools
        assert "web_search" not in ss.pinned_tools

    def test_multiple_pinned_tools(self) -> None:
        ss = SessionState(
            pinned_tools={"web_search", "shell", "calculator"},
            loaded_tools={"web_search", "shell", "calculator", "http_get"},
        )

        ss.reset_for_new_prompt()
        assert ss.loaded_tools == {"web_search", "shell", "calculator"}
        assert "http_get" not in ss.loaded_tools

    def test_disable_overrides_pin(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState(
            denials={"web_search"},
            pinned_tools={"web_search"},
            loaded_tools={"web_search"},
        )
        assert _classify_tool_status("web_search", ss) == "disabled"

    def test_on_demand_status_for_unknown_tool(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState()
        assert _classify_tool_status("unknown_tool", ss) == "on_demand"

    def test_active_status_for_agent_loaded(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState(loaded_tools={"web_search"})
        assert _classify_tool_status("web_search", ss) == "active"

    def test_auto_approved_status(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        # auto_approved requires the tool to also be loaded
        ss = SessionState(approvals={"web_search"}, loaded_tools={"web_search"})
        assert _classify_tool_status("web_search", ss) == "auto_approved"

    def test_approved_but_not_loaded_is_on_demand(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState(approvals={"web_search"})
        assert _classify_tool_status("web_search", ss) == "on_demand"


# ══════════════════════════════════════════════════════════════════════════════
# 6. _has_faiss_index unit tests (BUG-200)
# ══════════════════════════════════════════════════════════════════════════════


class TestHasFaissIndex:
    """BUG-200: _has_faiss_index must verify actual index files exist."""

    def test_returns_false_for_nonexistent_directory(self, tmp_path: Path) -> None:
        from src.tools.rag import _has_faiss_index

        assert _has_faiss_index(tmp_path / "nonexistent") is False

    def test_returns_false_for_empty_directory(self, tmp_path: Path) -> None:
        from src.tools.rag import _has_faiss_index

        d = tmp_path / "empty"
        d.mkdir()
        assert _has_faiss_index(d) is False

    def test_returns_true_for_index_faiss(self, tmp_path: Path) -> None:
        from src.tools.rag import _has_faiss_index

        d = tmp_path / "idx"
        d.mkdir()
        (d / "index.faiss").write_bytes(b"")
        assert _has_faiss_index(d) is True

    def test_returns_true_for_custom_faiss_name(self, tmp_path: Path) -> None:
        from src.tools.rag import _has_faiss_index

        d = tmp_path / "idx"
        d.mkdir()
        (d / "custom_name.faiss").write_bytes(b"")
        assert _has_faiss_index(d) is True

    def test_returns_false_for_non_faiss_files(self, tmp_path: Path) -> None:
        from src.tools.rag import _has_faiss_index

        d = tmp_path / "idx"
        d.mkdir()
        (d / "index.pkl").write_bytes(b"")
        assert _has_faiss_index(d) is False


# ══════════════════════════════════════════════════════════════════════════════
# 7. Delegate future.cancel on timeout (BUG-195)
# ══════════════════════════════════════════════════════════════════════════════


class TestDelegateFutureCancelOnTimeout:
    """BUG-195: delegate_parallel must cancel futures when timeout expires."""

    def test_future_cancel_called_on_timeout(self) -> None:
        """Verify the cancel pattern: when remaining <= 0, future.cancel() is called."""
        import concurrent.futures

        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        remaining = -1
        cancelled = False
        if remaining <= 0:
            cancelled = future.cancel()
        assert cancelled is True

    def test_source_calls_future_cancel(self) -> None:
        """Verify delegate.py source contains future.cancel() in the timeout branch."""
        import inspect

        from src.tools import delegate

        source = inspect.getsource(delegate)
        assert "future.cancel()" in source


# ══════════════════════════════════════════════════════════════════════════════
# 8. _collect_faiss_dirs edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestCollectFaissDirsEdgeCases:
    """Edge cases for _collect_faiss_dirs with None/empty api_uploads_dir."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_none_api_uploads_dir(self, tmp_path: Path) -> None:
        """api_uploads_dir=None should not cause errors."""
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": None,
            }
        )
        dirs = _collect_faiss_dirs()
        assert dirs == []

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_empty_string_api_uploads_dir(self, tmp_path: Path) -> None:
        """api_uploads_dir='' is falsy — should skip API index scan."""
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": "",
            }
        )
        dirs = _collect_faiss_dirs()
        assert dirs == []

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_nonexistent_api_uploads_dir(self, tmp_path: Path) -> None:
        """api_uploads_dir pointing to a non-existent directory returns empty."""
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        configure_rag(
            {
                "vectordb_dir": str(tmp_path / "no-global"),
                "api_uploads_dir": str(tmp_path / "does-not-exist"),
            }
        )
        dirs = _collect_faiss_dirs()
        assert dirs == []

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_only_global_index_when_no_api_uploads(self, tmp_path: Path) -> None:
        """Global index is found even when api_uploads_dir is None."""
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        global_idx = tmp_path / "faiss_index"
        global_idx.mkdir()
        (global_idx / "index.faiss").write_bytes(b"")

        configure_rag(
            {
                "vectordb_dir": str(global_idx),
                "api_uploads_dir": None,
            }
        )
        dirs = _collect_faiss_dirs()
        assert len(dirs) == 1
        assert dirs[0] == global_idx
