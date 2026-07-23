"""Regression tests for #1951 — ingest / query path-doubling contract.

Pre-fix, ``ingest_documents`` silently appended ``/faiss_index`` to
``IngestConfig.vectordb_dir`` and wrote ``index.faiss`` there, while
``configure_rag({"vectordb_dir": ...})`` (and ``_collect_faiss_dirs``)
treated ``vectordb_dir`` as the directory that *directly* holds
``index.faiss``.  The two conventions disagreed.  Because
``cogtrix_core/tools/rag.py:VECTOR_DIR`` was already ``data/vectordb/faiss_index``,
ingest wrote to ``data/vectordb/faiss_index/faiss_index/`` and the query
side could never find the index — every ``query_knowledge_base`` returned
"No knowledge base found."

The fix collapses both sides onto the same convention: ``vectordb_dir``
is the EXACT directory that holds the FAISS index files.  These tests
pin that contract so the regression cannot return.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# ── ingest_documents writes to vectordb_dir directly ──────────────────


class TestIngestWritesDirectlyToVectordbDir:
    """``ingest_documents`` must NOT append ``/faiss_index`` (#1951)."""

    def test_persist_path_equals_vectordb_dir(self, tmp_path: Path) -> None:
        from cogtrix_core.rag.ingest import IngestConfig, ingest_documents

        vectordb = tmp_path / "vectordb"
        config = IngestConfig(
            docs_dir=tmp_path / "docs",
            vectordb_dir=vectordb,
        )

        # Stub out the full ingest pipeline so we can observe the save target
        # without needing a real embedding provider or sample documents.
        fake_doc = MagicMock()
        fake_doc.page_content = "hello"

        with (
            patch("cogtrix_core.rag.ingest._load_documents", return_value=([fake_doc], [])),
            patch("cogtrix_core.rag.ingest._split_documents", return_value=[fake_doc]),
            patch("cogtrix_core.rag.ingest._create_embeddings", return_value=MagicMock()),
            patch("cogtrix_core.rag.ingest.FAISS.from_documents", return_value=MagicMock()),
            patch("cogtrix_core.rag.ingest.save_faiss_store") as mock_save,
        ):
            result = ingest_documents(config)

        assert result.success, f"ingest failed: {result.errors}"
        # The exact save target — the bug was an extra ``/faiss_index`` segment.
        assert mock_save.call_count == 1
        _store, persist_path = mock_save.call_args.args
        assert persist_path == vectordb, (
            f"ingest_documents wrote to {persist_path!r} instead of vectordb_dir "
            f"{vectordb!r} — #1951 path-doubling has regressed."
        )

    def test_ingest_many_persist_path_equals_vectordb_dir(self, tmp_path: Path) -> None:
        from cogtrix_core.rag.ingest import IngestConfig, ingest_many

        vectordb = tmp_path / "vectordb"
        config = IngestConfig(
            docs_dir=tmp_path / "docs",
            vectordb_dir=vectordb,
        )

        fake_doc = MagicMock()
        fake_doc.page_content = "hello"

        with (
            patch(
                "cogtrix_core.rag.ingest._prepare_ingest_file",
                side_effect=lambda p, c: (str(p), [fake_doc]),
            ),
            patch("cogtrix_core.rag.ingest._create_embeddings", return_value=MagicMock()),
            patch("cogtrix_core.rag.ingest.FAISS.from_documents", return_value=MagicMock()),
            patch("cogtrix_core.rag.ingest.save_faiss_store") as mock_save,
        ):
            paths: list[Path] = [tmp_path / "a.txt", tmp_path / "b.txt"]
            results = ingest_many(paths, config)

        assert all(results.values())
        assert mock_save.call_count == 1
        _store, persist_path = mock_save.call_args.args
        assert persist_path == vectordb, (
            f"ingest_many wrote to {persist_path!r} instead of vectordb_dir "
            f"{vectordb!r} — #1951 path-doubling has regressed."
        )


# ── ingest path = query path roundtrip ─────────────────────────────────


class TestIngestQueryPathAlignment:
    """The path ingest writes to must equal the path the query side reads from.

    This pins the cross-module contract that was broken pre-#1951.
    """

    def test_configure_rag_path_matches_ingest_save_path(self, tmp_path: Path) -> None:
        """Both ingest and ``_collect_faiss_dirs`` must agree on the index
        directory when handed the same ``vectordb_dir``."""
        from cogtrix_core.rag.ingest import IngestConfig
        from cogtrix_core.tools.rag import _has_faiss_index, configure_rag

        vectordb = tmp_path / "kb" / "faiss_index"
        vectordb.mkdir(parents=True)
        # Simulate the artefacts ingest would have written.
        (vectordb / "index.faiss").write_bytes(b"")
        (vectordb / "index.pkl").write_bytes(b"")

        config = IngestConfig(docs_dir=tmp_path / "docs", vectordb_dir=vectordb)
        # ingest_documents would call save_faiss_store(_, config.vectordb_dir).
        # Verify the query side accepts that same path.
        configure_rag({"vectordb_dir": str(config.vectordb_dir)})

        import cogtrix_core.tools.rag as _rag_mod

        configured = Path(_rag_mod._rag_config["vectordb_dir"])
        assert configured == vectordb
        assert _has_faiss_index(configured), (
            "The path ingest writes to (config.vectordb_dir) is not recognised by "
            "_has_faiss_index — ingest/query path contract has drifted."
        )

    def test_tools_rag_module_default_vector_dir_layout(self) -> None:
        """``cogtrix_core/tools/rag.py`` ships a default ``VECTOR_DIR`` that points at
        ``data/vectordb/faiss_index``.  Post-#1951 the ingest tool hands that
        same path to ``IngestConfig.vectordb_dir`` — ingest must NOT add
        another ``/faiss_index`` segment, otherwise we resurrect the
        original bug."""
        from cogtrix_core.tools.rag import VECTOR_DIR

        assert VECTOR_DIR == Path("data/vectordb/faiss_index"), (
            "The default VECTOR_DIR in cogtrix_core/tools/rag.py changed; update this "
            "regression to match the new layout."
        )
