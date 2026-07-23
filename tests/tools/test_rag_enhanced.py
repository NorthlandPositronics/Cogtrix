"""Tests for M4.3 multi-document RAG enhancements.

Covers:
- ingest_many parallel ingestion (success, partial failure, worker cap)
- entity extraction and entity index merging
- query_knowledge_base score_threshold filtering
- rag_find_entity tool
- rag_ingest tool
- RAGConfig score_threshold validation
- configure_rag_tool entity_index_path and score_threshold wiring
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ingest_config(tmp_path: Path, entity_index_path: Path | None = None):
    from src.rag.ingest import IngestConfig

    return IngestConfig(
        docs_dir=tmp_path / "docs",
        vectordb_dir=tmp_path / "vectordb",
        embedding_provider="ollama",
        entity_index_path=entity_index_path,
    )


def _fake_document(text: str = "hello world", source: str = "doc.txt"):
    from langchain_core.documents import Document

    return Document(page_content=text, metadata={"source": source})


# ---------------------------------------------------------------------------
# ingest_many — parallel ingestion
# ---------------------------------------------------------------------------


class TestIngestMany:
    def test_empty_paths_returns_empty(self):
        from src.rag.ingest import IngestConfig, ingest_many

        config = IngestConfig(docs_dir=Path("/tmp"), vectordb_dir=Path("/tmp"))
        result = ingest_many([], config)
        assert result == {}

    def test_all_succeed(self, tmp_path: Path):
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        paths: list[str | Path] = [tmp_path / f"file{i}.txt" for i in range(3)]

        prepared = {str(path): (str(path), [_fake_document(source=path.name)]) for path in paths}

        mock_store = MagicMock()

        def fake_prepare(path, cfg):
            return prepared.get(str(path))

        with (
            patch("src.rag.ingest._prepare_ingest_file", side_effect=fake_prepare) as mock_prepare,
            patch("src.rag.ingest._create_embeddings", return_value=MagicMock()),
            patch("src.rag.ingest.FAISS.from_documents", return_value=mock_store) as mock_faiss,
            patch("src.rag.ingest.save_faiss_store") as mock_save_store,
        ):
            result = ingest_many(paths, config)

        assert len(result) == 3
        assert all(result.values())
        assert mock_prepare.call_count == 3
        mock_faiss.assert_called_once()
        mock_save_store.assert_called_once_with(mock_store, config.vectordb_dir / "faiss_index")

    def test_partial_failure(self, tmp_path: Path):
        """When some files fail, successes and failures are reported correctly."""
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        call_count = 0

        def alternating(path, cfg):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return str(path), [_fake_document(source=Path(path).name)]
            return None

        mock_store = MagicMock()

        with (
            patch("src.rag.ingest._prepare_ingest_file", side_effect=alternating),
            patch("src.rag.ingest._create_embeddings", return_value=MagicMock()),
            patch("src.rag.ingest.FAISS.from_documents", return_value=mock_store),
            patch("src.rag.ingest.save_faiss_store") as mock_save_store,
        ):
            paths: list[str | Path] = [tmp_path / f"file{i}.txt" for i in range(4)]
            result = ingest_many(paths, config)

        successes = sum(result.values())
        failures = len(result) - successes
        assert successes == 2
        assert failures == 2
        mock_save_store.assert_called_once_with(mock_store, config.vectordb_dir / "faiss_index")

    def test_builds_single_index_from_all_chunks(self, tmp_path: Path):
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        paths: list[str | Path] = [tmp_path / "a.txt", tmp_path / "b.txt"]

        def fake_prepare(path, cfg):
            return str(path), [_fake_document(source=Path(path).name)]

        mock_store = MagicMock()

        with (
            patch("src.rag.ingest._prepare_ingest_file", side_effect=fake_prepare),
            patch("src.rag.ingest._create_embeddings", return_value=MagicMock()),
            patch("src.rag.ingest.FAISS.from_documents", return_value=mock_store) as mock_faiss,
            patch("src.rag.ingest.save_faiss_store") as mock_save_store,
        ):
            result = ingest_many(paths, config)

        assert all(result.values())
        mock_faiss.assert_called_once()
        docs_arg, _embeddings_arg = mock_faiss.call_args.args
        assert len(docs_arg) == 2
        mock_save_store.assert_called_once_with(mock_store, config.vectordb_dir / "faiss_index")

    def test_worker_cap_at_eight(self, tmp_path: Path):
        """Workers are capped at min(len(paths), workers, 8)."""
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        captured_max_workers: list[int] = []

        original_tpe = __import__(
            "concurrent.futures", fromlist=["ThreadPoolExecutor"]
        ).ThreadPoolExecutor

        class CapturingTPE(original_tpe):
            def __init__(self, max_workers=None, **kw):
                captured_max_workers.append(max_workers)
                super().__init__(max_workers=max_workers, **kw)

        with patch("src.rag.ingest.ThreadPoolExecutor", CapturingTPE):
            with patch("src.rag.ingest._prepare_ingest_file", return_value=None):
                paths: list[str | Path] = [tmp_path / f"f{i}.txt" for i in range(12)]
                ingest_many(paths, config, workers=20)

        assert captured_max_workers[0] == 8  # capped at 8

    def test_worker_capped_at_path_count(self, tmp_path: Path):
        """When fewer paths than workers, workers = len(paths)."""
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        captured: list[int] = []

        original_tpe = __import__(
            "concurrent.futures", fromlist=["ThreadPoolExecutor"]
        ).ThreadPoolExecutor

        class CapTPE(original_tpe):
            def __init__(self, max_workers=None, **kw):
                captured.append(max_workers)
                super().__init__(max_workers=max_workers, **kw)

        with patch("src.rag.ingest.ThreadPoolExecutor", CapTPE):
            with patch("src.rag.ingest._prepare_ingest_file", return_value=None):
                paths: list[str | Path] = [tmp_path / "only_one.txt"]
                ingest_many(paths, config, workers=8)

        assert captured[0] == 1

    def test_exception_in_worker_returns_false(self, tmp_path: Path):
        """If a worker raises an unexpected exception, path maps to False."""
        from src.rag.ingest import ingest_many

        config = _make_ingest_config(tmp_path)
        paths: list[str | Path] = [tmp_path / "bad.txt"]

        with patch("src.rag.ingest._prepare_ingest_file", side_effect=RuntimeError("boom")):
            result = ingest_many(paths, config)

        assert list(result.values()) == [False]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_capitalized_phrases_extracted(self):
        from langchain_core.documents import Document

        from src.rag.ingest import _extract_entities

        chunks = [Document(page_content="John Smith visited New York last week.")]
        entities = _extract_entities(chunks, "test.txt")
        keys_lower = {k.lower() for k in entities}
        assert "john smith" in keys_lower or "new york" in keys_lower

    def test_quoted_strings_extracted(self):
        from langchain_core.documents import Document

        from src.rag.ingest import _extract_entities

        chunks = [Document(page_content='The system said "operation complete" to the user.')]
        entities = _extract_entities(chunks, "test.txt")
        assert "operation complete" in entities

    def test_frequent_words_extracted(self):
        from langchain_core.documents import Document

        from src.rag.ingest import _extract_entities

        # Repeat a non-stop word 3 times
        text = "python python python is great for data processing"
        chunks = [Document(page_content=text)]
        entities = _extract_entities(chunks, "test.txt")
        assert "python" in entities

    def test_stop_words_excluded(self):
        from langchain_core.documents import Document

        from src.rag.ingest import _extract_entities

        # "their" appears 3+ times but is a stop word
        text = "their their their data is important"
        chunks = [Document(page_content=text)]
        entities = _extract_entities(chunks, "test.txt")
        assert "their" not in entities

    def test_chunk_refs_include_source_name(self):
        from langchain_core.documents import Document

        from src.rag.ingest import _extract_entities

        chunks = [Document(page_content='He said "hello world" to everyone.')]
        entities = _extract_entities(chunks, "myfile.txt")
        for refs in entities.values():
            for ref in refs:
                assert ref.startswith("myfile.txt:")


# ---------------------------------------------------------------------------
# _update_entity_index
# ---------------------------------------------------------------------------


class TestUpdateEntityIndex:
    def test_creates_index_when_absent(self, tmp_path: Path):
        from src.rag.ingest import _update_entity_index

        index_path = tmp_path / "entity_index.json"
        _update_entity_index({"Python": ["doc.txt:chunk_0"]}, index_path)
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert "Python" in data

    def test_merges_into_existing_index(self, tmp_path: Path):
        from src.rag.ingest import _update_entity_index

        index_path = tmp_path / "entity_index.json"
        index_path.write_text(json.dumps({"Existing": ["old.txt:chunk_0"]}))
        _update_entity_index({"NewEntity": ["new.txt:chunk_0"]}, index_path)
        data = json.loads(index_path.read_text())
        assert "Existing" in data
        assert "NewEntity" in data

    def test_concurrent_writes_are_safe(self, tmp_path: Path):
        """Multiple threads writing to the same index should not corrupt it."""
        from src.rag.ingest import _update_entity_index

        index_path = tmp_path / "entity_index.json"
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                _update_entity_index({f"Entity{i}": [f"doc{i}.txt:chunk_0"]}, index_path)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        data = json.loads(index_path.read_text())
        assert len(data) == 10


# ---------------------------------------------------------------------------
# query_knowledge_base score_threshold filtering
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    def _make_mock_store(self, docs_and_scores):
        """Build a FAISS mock that returns the given (doc, score) pairs."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = docs_and_scores
        return mock_store

    def test_results_below_threshold_excluded(self, tmp_path: Path):
        from langchain_core.documents import Document

        from src.tools.rag import configure_rag, query_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index"), "score_threshold": 0.0})

        # distance=1.0 → similarity=0.5; distance=0.0 → similarity=1.0
        doc_high = Document(page_content="high quality", metadata={"source": "a.txt"})
        doc_low = Document(page_content="low quality", metadata={"source": "b.txt"})
        pairs = [(doc_high, 0.0), (doc_low, 1.0)]  # lower L2 = higher similarity

        faiss_dir = tmp_path / "faiss_index"
        faiss_dir.mkdir(parents=True)
        (faiss_dir / "index.faiss").touch()

        mock_store = self._make_mock_store(pairs)

        with (
            patch("src.tools.rag.FAISS_AVAILABLE", True),
            patch("src.tools.rag.load_faiss_store_safe", return_value=mock_store),
            patch("src.tools.rag._get_embeddings", return_value=MagicMock()),
        ):
            # threshold 0.8 → only similarity >= 0.8 passes (distance <= 0.25)
            result = query_knowledge_base("test", k=5, score_threshold=0.8)

        assert "high quality" in result
        assert "low quality" not in result

    def test_no_results_below_threshold_returns_message(self, tmp_path: Path):
        from langchain_core.documents import Document

        from src.tools.rag import configure_rag, query_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index"), "score_threshold": 0.0})

        faiss_dir = tmp_path / "faiss_index"
        faiss_dir.mkdir(parents=True)
        (faiss_dir / "index.faiss").touch()

        doc = Document(page_content="mediocre result", metadata={"source": "c.txt"})
        pairs = [(doc, 10.0)]  # distance=10 → similarity ≈ 0.09, below 0.9

        mock_store = self._make_mock_store(pairs)

        with (
            patch("src.tools.rag.FAISS_AVAILABLE", True),
            patch("src.tools.rag.load_faiss_store_safe", return_value=mock_store),
            patch("src.tools.rag._get_embeddings", return_value=MagicMock()),
        ):
            result = query_knowledge_base("test", k=5, score_threshold=0.9)

        assert "threshold" in result.lower()

    def test_default_threshold_returns_all(self, tmp_path: Path):
        """score_threshold=0.0 (default) must not filter anything."""
        from langchain_core.documents import Document

        from src.tools.rag import configure_rag, query_knowledge_base

        configure_rag({"vectordb_dir": str(tmp_path / "faiss_index"), "score_threshold": 0.0})

        faiss_dir = tmp_path / "faiss_index"
        faiss_dir.mkdir(parents=True)
        (faiss_dir / "index.faiss").touch()

        doc = Document(page_content="any result", metadata={"source": "d.txt"})
        pairs = [(doc, 100.0)]  # very high distance, but threshold is 0

        mock_store = self._make_mock_store(pairs)

        with (
            patch("src.tools.rag.FAISS_AVAILABLE", True),
            patch("src.tools.rag.load_faiss_store_safe", return_value=mock_store),
            patch("src.tools.rag._get_embeddings", return_value=MagicMock()),
        ):
            result = query_knowledge_base("test", k=5, score_threshold=0.0)

        assert "any result" in result


# ---------------------------------------------------------------------------
# rag_find_entity tool
# ---------------------------------------------------------------------------


class TestRagFindEntity:
    def test_finds_exact_entity(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_find_entity

        index = {"Python": ["file.txt:chunk_0", "file.txt:chunk_3"]}
        index_path = tmp_path / "entity_index.json"
        index_path.write_text(json.dumps(index))
        configure_rag({"entity_index_path": str(index_path)})

        result = rag_find_entity("Python")
        assert "Python" in result
        assert "chunk_0" in result

    def test_case_insensitive_search(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_find_entity

        index = {"Machine Learning": ["ml.pdf:chunk_1"]}
        index_path = tmp_path / "entity_index.json"
        index_path.write_text(json.dumps(index))
        configure_rag({"entity_index_path": str(index_path)})

        result = rag_find_entity("machine learning")
        assert "Machine Learning" in result

    def test_not_found_message(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_find_entity

        index_path = tmp_path / "entity_index.json"
        index_path.write_text(json.dumps({"Python": ["f.txt:chunk_0"]}))
        configure_rag({"entity_index_path": str(index_path)})

        result = rag_find_entity("Ruby")
        assert "not found" in result.lower()

    def test_no_entity_index_configured(self):
        from src.tools.rag import configure_rag, rag_find_entity

        configure_rag({"entity_index_path": None})
        result = rag_find_entity("Anything")
        assert "not configured" in result.lower()

    def test_max_results_respected(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_find_entity

        refs = [f"file.txt:chunk_{i}" for i in range(20)]
        index_path = tmp_path / "entity_index.json"
        index_path.write_text(json.dumps({"LargeEntity": refs}))
        configure_rag({"entity_index_path": str(index_path)})

        result = rag_find_entity("LargeEntity", max_results=3)
        # Only 3 shown + "and N more" message
        shown = [line for line in result.splitlines() if "chunk_" in line]
        assert len(shown) == 3


# ---------------------------------------------------------------------------
# rag_ingest tool
# ---------------------------------------------------------------------------


class TestRagIngest:
    def test_no_paths_returns_message(self):
        from src.tools.rag import rag_ingest

        result = rag_ingest("  ,  ")
        assert "No file paths" in result

    def test_calls_ingest_many(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_ingest

        configure_rag({"vectordb_dir": str(tmp_path / "vectordb" / "faiss_index")})

        with patch(
            "src.rag.ingest.ingest_many", return_value={str(tmp_path / "a.txt"): True}
        ) as mock_im:
            result = rag_ingest(str(tmp_path / "a.txt"))

        assert mock_im.called
        assert "1/1" in result

    def test_reports_failures(self, tmp_path: Path):
        from src.tools.rag import configure_rag, rag_ingest

        configure_rag({"vectordb_dir": str(tmp_path / "vectordb" / "faiss_index")})

        paths_result = {str(tmp_path / "a.txt"): True, str(tmp_path / "b.txt"): False}
        with patch("src.rag.ingest.ingest_many", return_value=paths_result):
            result = rag_ingest(f"{tmp_path}/a.txt,{tmp_path}/b.txt")

        assert "1/2" in result
        assert "Failed" in result


# ---------------------------------------------------------------------------
# RAGConfig score_threshold validation
# ---------------------------------------------------------------------------


class TestRAGConfigScoreThreshold:
    def test_default_is_zero(self):
        from src.config import RAGConfig

        cfg = RAGConfig()
        assert cfg.score_threshold == 0.0

    def test_valid_threshold_accepted(self):
        from src.config import RAGConfig

        cfg = RAGConfig(score_threshold=0.75)
        assert cfg.score_threshold == 0.75

    def test_threshold_above_one_raises(self):
        from src.config import ConfigError, RAGConfig

        with pytest.raises(ConfigError, match="score_threshold"):
            RAGConfig(score_threshold=1.5)

    def test_threshold_below_zero_raises(self):
        from src.config import ConfigError, RAGConfig

        with pytest.raises(ConfigError, match="score_threshold"):
            RAGConfig(score_threshold=-0.1)

    def test_parsed_from_config_dict(self, tmp_path: Path):
        """score_threshold is correctly parsed from a JSON config file."""
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / "cogtrix.json"
        cfg_file.write_text(json.dumps({"rag": {"score_threshold": 0.6}}))

        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.rag.score_threshold == 0.6

    def test_invalid_config_value_uses_default(self, tmp_path: Path):
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / "cogtrix.json"
        cfg_file.write_text(json.dumps({"rag": {"score_threshold": "not-a-float"}}))

        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.rag.score_threshold == 0.0


# ---------------------------------------------------------------------------
# configure_rag_tool wiring
# ---------------------------------------------------------------------------


class TestConfigureRagToolWiring:
    def test_entity_index_path_wired(self, tmp_path: Path):
        """configure_rag_tool passes entity_index_path to configure_rag."""
        from src.config import Config
        from src.tools.configure import configure_rag_tool

        config = Config()
        config.data_dir = str(tmp_path)
        config.rag.vectordb_dir = "vectordb"

        captured: dict = {}

        def fake_configure_rag(cfg):
            captured.update(cfg)

        with (
            patch("src.tools.rag.configure_rag", fake_configure_rag),
            patch("src.tools.rag.TOOL_CONFIG", {}),
            patch("src.tools.rag._build_description", return_value="desc"),
            patch("src.tools.rag.knowledge_base_exists", return_value=False),
            patch(
                "src.config.Config.resolve_embedding_config",
                return_value=("ollama", "nomic-embed-text", "http://localhost:11434", None),
            ),
        ):
            configure_rag_tool(config)

        assert "entity_index_path" in captured
        assert captured["entity_index_path"] is not None

    def test_score_threshold_wired(self, tmp_path: Path):
        from src.config import Config
        from src.tools.configure import configure_rag_tool

        config = Config()
        config.data_dir = str(tmp_path)
        config.rag.vectordb_dir = "vectordb"
        config.rag.score_threshold = 0.8

        captured: dict = {}

        def fake_configure_rag(cfg):
            captured.update(cfg)

        with (
            patch("src.tools.rag.configure_rag", fake_configure_rag),
            patch("src.tools.rag.TOOL_CONFIG", {}),
            patch("src.tools.rag._build_description", return_value="desc"),
            patch("src.tools.rag.knowledge_base_exists", return_value=False),
            patch(
                "src.config.Config.resolve_embedding_config",
                return_value=("ollama", "nomic-embed-text", "http://localhost:11434", None),
            ),
        ):
            configure_rag_tool(config)

        assert captured.get("score_threshold") == 0.8
