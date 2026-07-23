"""Tests for save_to_knowledge_base and agent-notes integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: save/restore RAG config
# ---------------------------------------------------------------------------


@pytest.fixture()
def _restore_rag_config():
    import cogtrix_core.tools.rag as _rag_mod

    original = dict(_rag_mod._rag_config)
    yield
    _rag_mod._rag_config.update(original)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_faiss_index(directory: Path) -> None:
    """Create a minimal directory structure that _has_faiss_index considers valid."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.faiss").write_bytes(b"")
    (directory / "index.pkl").write_bytes(b"")


# ============================================================================
# save_to_knowledge_base — JSONL fallback (no FAISS)
# ============================================================================


class TestSaveToKnowledgeBaseFallback:
    """save_to_knowledge_base writes to JSONL when FAISS is not available."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_writes_jsonl_entry(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})
        jsonl_path = tmp_path / "agent_notes.jsonl"

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            result = save_to_knowledge_base("The sky is blue.")

        assert result == "Saved to knowledge base."
        assert jsonl_path.exists()

        entries = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["content"] == "The sky is blue."
        assert entries[0]["type"] == "agent_note"

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            save_to_knowledge_base("Fact A")
            save_to_knowledge_base("Fact B")

        jsonl_path = tmp_path / "agent_notes.jsonl"
        lines = [json.loads(raw) for raw in jsonl_path.read_text().splitlines()]
        assert len(lines) == 2
        assert lines[0]["content"] == "Fact A"
        assert lines[1]["content"] == "Fact B"

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_metadata_stored_correctly(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            save_to_knowledge_base("Important finding.", source="research", tags=["ai", "nlp"])

        jsonl_path = tmp_path / "agent_notes.jsonl"
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["source"] == "research"
        assert entry["tags"] == ["ai", "nlp"]
        assert "timestamp" in entry

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_default_source_is_agent(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            save_to_knowledge_base("Some note.")

        entry = json.loads((tmp_path / "agent_notes.jsonl").read_text().strip())
        assert entry["source"] == "agent"

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_empty_content_returns_error(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            result = save_to_knowledge_base("")

        assert "Error" in result

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_whitespace_only_content_returns_error(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", False):
            result = save_to_knowledge_base("   \n  ")

        assert "Error" in result


# ============================================================================
# save_to_knowledge_base — FAISS path
# ============================================================================


class TestSaveToKnowledgeBaseFaiss:
    """save_to_knowledge_base creates/updates a FAISS sub-index."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_creates_new_index(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        mock_store = MagicMock()
        mock_embeddings = MagicMock()

        with (
            patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", True),
            patch("cogtrix_core.tools.rag._get_embeddings", return_value=mock_embeddings),
            patch("cogtrix_core.tools.rag.FAISS") as mock_faiss_cls,
            patch("cogtrix_core.tools.rag.save_faiss_store") as mock_save,
        ):
            mock_faiss_cls.from_documents.return_value = mock_store

            result = save_to_knowledge_base("New fact for FAISS.")

        assert result == "Saved to knowledge base."
        mock_faiss_cls.from_documents.assert_called_once()
        mock_save.assert_called_once()

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_appends_to_existing_index(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import (
            _AGENT_NOTES_SUBDIR,
            configure_rag,
            save_to_knowledge_base,
        )

        vectordb_dir = tmp_path / "faiss_index"
        configure_rag({"vectordb_dir": str(vectordb_dir)})

        # Create a pre-existing agent_notes FAISS dir with an index file
        notes_dir = tmp_path / _AGENT_NOTES_SUBDIR
        _make_faiss_index(notes_dir)

        mock_store = MagicMock()
        mock_embeddings = MagicMock()

        with (
            patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", True),
            patch("cogtrix_core.tools.rag._get_embeddings", return_value=mock_embeddings),
            patch(
                "cogtrix_core.tools.rag.load_faiss_store_safe", return_value=mock_store
            ) as mock_load,
            patch("cogtrix_core.tools.rag.save_faiss_store") as mock_save,
        ):
            result = save_to_knowledge_base("Appended fact.")

        assert result == "Saved to knowledge base."
        mock_load.assert_called_once()
        mock_store.add_documents.assert_called_once()
        mock_save.assert_called_once()

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_exception_returns_error_string(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        with (
            patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", True),
            patch("cogtrix_core.tools.rag._get_embeddings", side_effect=RuntimeError("embed fail")),
        ):
            result = save_to_knowledge_base("Some fact.")

        assert "Error saving" in result
        assert "embed fail" in result

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_metadata_passed_to_document(self, tmp_path: Path) -> None:
        from langchain_core.documents import Document

        from cogtrix_core.tools.rag import configure_rag, save_to_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index")})

        captured_docs: list[Document] = []
        mock_store = MagicMock()

        def _capture_from_documents(docs, embeddings):
            captured_docs.extend(docs)
            return mock_store

        with (
            patch("cogtrix_core.tools.rag.FAISS_AVAILABLE", True),
            patch("cogtrix_core.tools.rag._get_embeddings", return_value=MagicMock()),
            patch("cogtrix_core.tools.rag.FAISS") as mock_faiss_cls,
            # save_faiss_store dives into the real faiss C extension via
            # dependable_faiss_import() and calls write_index() on the
            # store's index attribute.  When the store is a MagicMock, the
            # MagicMock-typed argument sends faiss-cpu's SWIG layer into a
            # CPU-bound loop with no I/O — the source of the shard-D 94%
            # hang.  Skipping the persistence step here is consistent with
            # the sibling tests (test_creates_new_index,
            # test_appends_to_existing_index) which already patch it.
            patch("cogtrix_core.tools.rag.save_faiss_store"),
        ):
            mock_faiss_cls.from_documents.side_effect = _capture_from_documents

            save_to_knowledge_base("Tagged fact.", source="research_agent", tags=["nlp"])

        assert len(captured_docs) == 1
        meta = captured_docs[0].metadata
        assert meta["source"] == "research_agent"
        assert meta["tags"] == ["nlp"]
        assert meta["type"] == "agent_note"
        assert "timestamp" in meta


# ============================================================================
# _collect_faiss_dirs — agent-notes sub-index included
# ============================================================================


class TestCollectFaissDirsAgentNotes:
    """_collect_faiss_dirs includes the agent-notes sub-index when it exists."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_agent_notes_included_when_present(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _AGENT_NOTES_SUBDIR, _collect_faiss_dirs, configure_rag

        vectordb_dir = tmp_path / "faiss_index"
        notes_dir = tmp_path / _AGENT_NOTES_SUBDIR
        _make_faiss_index(notes_dir)

        configure_rag(
            {
                "vectordb_dir": str(vectordb_dir),
                "api_uploads_dir": None,
            }
        )

        dirs = _collect_faiss_dirs()
        assert notes_dir in dirs

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_agent_notes_excluded_when_absent(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _AGENT_NOTES_SUBDIR, _collect_faiss_dirs, configure_rag

        vectordb_dir = tmp_path / "faiss_index"
        configure_rag(
            {
                "vectordb_dir": str(vectordb_dir),
                "api_uploads_dir": None,
            }
        )

        dirs = _collect_faiss_dirs()
        for d in dirs:
            assert _AGENT_NOTES_SUBDIR not in str(d)

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_agent_notes_alongside_global_index(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _AGENT_NOTES_SUBDIR, _collect_faiss_dirs, configure_rag

        global_idx = tmp_path / "faiss_index"
        _make_faiss_index(global_idx)
        notes_dir = tmp_path / _AGENT_NOTES_SUBDIR
        _make_faiss_index(notes_dir)

        configure_rag(
            {
                "vectordb_dir": str(global_idx),
                "api_uploads_dir": None,
            }
        )

        dirs = _collect_faiss_dirs()
        assert global_idx in dirs
        assert notes_dir in dirs
        assert len(dirs) == 2


# ============================================================================
# TOOL_CONFIGS — registry contract
# ============================================================================


class TestToolConfigs:
    """TOOL_CONFIGS exports both tools; TOOL_CONFIG remains backward-compatible."""

    def test_tool_configs_has_two_entries(self) -> None:
        from cogtrix_core.tools.rag import TOOL_CONFIGS

        names = [c["name"] for c in TOOL_CONFIGS]
        assert "query_knowledge_base" in names
        assert "save_to_knowledge_base" in names

    def test_tool_config_is_query_tool(self) -> None:
        from cogtrix_core.tools.rag import TOOL_CONFIG

        assert TOOL_CONFIG["name"] == "query_knowledge_base"

    def test_save_tool_requires_no_confirmation(self) -> None:
        from cogtrix_core.tools.rag import TOOL_CONFIGS

        save_cfg = next(c for c in TOOL_CONFIGS if c["name"] == "save_to_knowledge_base")
        assert save_cfg["requires_confirmation"] is False

    def test_save_tool_has_input_schema(self) -> None:
        from pydantic import BaseModel

        from cogtrix_core.tools.rag import TOOL_CONFIGS

        save_cfg = next(c for c in TOOL_CONFIGS if c["name"] == "save_to_knowledge_base")
        schema = save_cfg["input_schema"]
        assert issubclass(schema, BaseModel)

    def test_save_tool_function_is_callable(self) -> None:
        from cogtrix_core.tools.rag import TOOL_CONFIGS, save_to_knowledge_base

        save_cfg = next(c for c in TOOL_CONFIGS if c["name"] == "save_to_knowledge_base")
        assert save_cfg["function"] is save_to_knowledge_base
