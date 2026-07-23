"""Comprehensive RAG-retrieval path-consistency tests (#2216).

Background
----------
The FAISS index directory is derived from ``config.rag.vectordb_dir`` by two
independent callers:

* ``cogtrix.run_ingest`` — the CLI ``--ingest`` path (writes the index).
* ``src.tools.configure.configure_rag_tool`` — configures the query tool
  (``query_knowledge_base`` / ``_collect_faiss_dirs``) (reads the index).

Post-#1951 they drifted: ingest wrote straight to ``<vectordb_dir>`` while the
query side read ``<vectordb_dir>/faiss_index`` — so ``query_knowledge_base``
never found a CLI-ingested index ("no knowledge base found"), even though the
index was built fine.

The fix routes BOTH through ``Config.resolve_rag_index_dir()`` — a single
source of truth that returns ``<vectordb_dir>/faiss_index``. These tests pin:
the helper's contract, ingest⇄query agreement, the end-to-end ingest→query
round-trip, the "old location is not silently searched" guard, the #1951
no-double-segment guard, and the (unchanged) agent-notes layout.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _restore_rag_config():
    """Snapshot/restore the global ``_rag_config`` so tests don't leak state."""
    import cogtrix_core.tools.rag as rag_mod

    saved = dict(rag_mod._rag_config)
    try:
        yield
    finally:
        rag_mod._rag_config.clear()
        rag_mod._rag_config.update(saved)


def _make_config(vectordb_dir: str, data_dir: str = "/tmp/cogtrix-data"):
    from cogtrix_core.config import Config

    cfg = Config()
    cfg.data_dir = data_dir
    cfg.rag.vectordb_dir = vectordb_dir
    cfg.embedding_provider_override = None
    cfg.embedding_model_override = None
    return cfg


_EMB = ("openai", "model", "http://emb", "key")  # stub resolve_embedding_config


# ───────────────────────── helper: single source of truth ──────────────────


class TestResolveRagIndexDir:
    def test_absolute_vectordb_dir(self):
        cfg = _make_config("/data/vectordb")
        assert cfg.resolve_rag_index_dir() == Path("/data/vectordb/faiss_index")

    def test_relative_vectordb_dir_resolved_under_data_dir(self):
        cfg = _make_config("vectordb", data_dir="/srv/data")
        assert cfg.resolve_rag_index_dir() == Path("/srv/data/vectordb/faiss_index")

    def test_cli_override_takes_precedence(self):
        cfg = _make_config("/data/vectordb")
        assert cfg.resolve_rag_index_dir("/other/kb") == Path("/other/kb/faiss_index")

    def test_legacy_data_prefix_is_normalized(self):
        cfg = _make_config("data/vectordb", data_dir="/srv/data")
        assert cfg.resolve_rag_index_dir() == Path("/srv/data/vectordb/faiss_index")

    def test_exactly_one_faiss_index_segment(self):
        # #1951 guard: never double the faiss_index segment.
        cfg = _make_config("/data/vectordb")
        assert cfg.resolve_rag_index_dir().parts.count("faiss_index") == 1


# ───────────────────────── ingest ⇄ query agreement (#2216) ─────────────────


class TestIngestQueryAgreement:
    def test_configure_rag_tool_targets_resolve_rag_index_dir(self):
        import cogtrix_core.tools.rag as rag_mod
        from cogtrix_core.tools.configure import configure_rag_tool

        cfg = _make_config("/data/vectordb")
        with patch.object(type(cfg), "resolve_embedding_config", return_value=_EMB):
            configure_rag_tool(cfg)
        assert Path(rag_mod._rag_config["vectordb_dir"]) == cfg.resolve_rag_index_dir()

    def test_run_ingest_targets_resolve_rag_index_dir(self):
        # The regression: pre-fix run_ingest used config.rag.vectordb_dir with NO
        # /faiss_index segment, so the captured path != resolve_rag_index_dir().
        import cogtrix

        cfg = _make_config("/data/vectordb")
        args = SimpleNamespace(
            docs_dir=None, vectordb_dir=None, embedding_provider=None, embedding_model=None
        )
        captured: dict[str, Path] = {}

        def _capture(ingest_config):
            captured["dir"] = Path(ingest_config.vectordb_dir)
            return MagicMock(
                success=True,
                documents_loaded=1,
                chunks_created=1,
                vector_store_path=captured["dir"],
                errors=[],
            )

        with (
            patch("cogtrix_core.rag.ingest_documents", side_effect=_capture),
            patch.object(type(cfg), "resolve_embedding_config", return_value=_EMB),
            patch("cogtrix.console", None),
        ):
            cogtrix.run_ingest(args, cfg)

        assert captured["dir"] == cfg.resolve_rag_index_dir()


# ───────────────────────── end-to-end round-trip ───────────────────────────


class TestIngestQueryRoundTrip:
    def test_run_ingest_output_is_found_by_query_side(self, tmp_path: Path):
        """Full contract: where run_ingest WRITES is where the query side READS.

        Pre-fix this fails — ingest writes to ``<vectordb_dir>`` but
        ``_collect_faiss_dirs`` searches ``<vectordb_dir>/faiss_index``.
        """
        import cogtrix
        from cogtrix_core.tools.configure import configure_rag_tool
        from cogtrix_core.tools.rag import _collect_faiss_dirs

        cfg = _make_config(str(tmp_path / "vectordb"), data_dir=str(tmp_path))
        args = SimpleNamespace(
            docs_dir=None, vectordb_dir=None, embedding_provider=None, embedding_model=None
        )

        def _fake_ingest(ingest_config):
            d = Path(ingest_config.vectordb_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "index.faiss").write_bytes(b"")
            (d / "index.pkl").write_bytes(b"")
            return MagicMock(
                success=True, documents_loaded=1, chunks_created=1, vector_store_path=d, errors=[]
            )

        with (
            patch("cogtrix_core.rag.ingest_documents", side_effect=_fake_ingest),
            patch.object(type(cfg), "resolve_embedding_config", return_value=_EMB),
            patch("cogtrix.console", None),
        ):
            rc = cogtrix.run_ingest(args, cfg)
            assert rc == 0
            configure_rag_tool(cfg)

        assert cfg.resolve_rag_index_dir() in _collect_faiss_dirs()

    def test_old_shallower_location_is_not_searched(self, tmp_path: Path):
        """An index at ``<vectordb_dir>`` (the pre-fix CLI write location, one
        level above ``faiss_index``) must NOT be silently picked up — proving
        the two sides genuinely have to agree (they now do via the helper)."""
        from cogtrix_core.tools.configure import configure_rag_tool
        from cogtrix_core.tools.rag import _collect_faiss_dirs

        cfg = _make_config(str(tmp_path / "vectordb"), data_dir=str(tmp_path))
        old_loc = cfg.resolve_data_path(cfg.rag.vectordb_dir)  # no /faiss_index
        old_loc.mkdir(parents=True)
        (old_loc / "index.faiss").write_bytes(b"")

        with patch.object(type(cfg), "resolve_embedding_config", return_value=_EMB):
            configure_rag_tool(cfg)

        assert old_loc not in _collect_faiss_dirs()


# ───────────────────────── agent-notes layout (unchanged) ───────────────────


class TestAgentNotesLayout:
    def test_agent_notes_is_sibling_of_faiss_index_dir(self, tmp_path: Path):
        from cogtrix_core.tools.configure import configure_rag_tool
        from cogtrix_core.tools.rag import _agent_notes_faiss_dir

        cfg = _make_config(str(tmp_path / "vectordb"), data_dir=str(tmp_path))
        with patch.object(type(cfg), "resolve_embedding_config", return_value=_EMB):
            configure_rag_tool(cfg)

        index_dir = cfg.resolve_rag_index_dir()
        # Notes live beside the faiss_index dir (i.e. under <vectordb_dir>),
        # the layout the JSONL fallback + #1951 tests already assume.
        assert _agent_notes_faiss_dir() == index_dir.parent / "agent_notes"
