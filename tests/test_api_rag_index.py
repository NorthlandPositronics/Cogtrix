"""Regression coverage for raw FAISS persistence used by the RAG API."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _DummyEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        text = text.lower()
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_raw_faiss_round_trip(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from src.api.rag_index import load_faiss_store, save_faiss_store

    persist_dir = tmp_path / "faiss_index"
    fake_index = object()
    docs = {
        "doc-a": Document(page_content="alpha document", metadata={"source": "a"}),
        "doc-b": Document(page_content="beta document", metadata={"source": "b"}),
    }
    store = MagicMock()
    store.index = fake_index
    store.docstore = MagicMock(_dict=docs)
    store.index_to_docstore_id = {0: "doc-a", 1: "doc-b"}

    fake_faiss = MagicMock()

    def _write_index(index, path):
        assert index is fake_index
        Path(path).write_bytes(b"raw-faiss-index")

    fake_faiss.write_index.side_effect = _write_index
    fake_faiss.read_index.return_value = fake_index

    embeddings = _DummyEmbeddings()

    with patch("src.api.rag_index.dependable_faiss_import", return_value=fake_faiss):
        save_faiss_store(store, persist_dir)

    assert (persist_dir / "index.faiss").exists()
    assert (persist_dir / "metadata.json").exists()
    assert not (persist_dir / "index.pkl").exists()

    fake_loaded_store = MagicMock()
    with (
        patch("src.api.rag_index.dependable_faiss_import", return_value=fake_faiss),
        patch("src.api.rag_index.FAISS", return_value=fake_loaded_store) as mock_faiss_cls,
    ):
        loaded = load_faiss_store(persist_dir, embeddings)

    assert loaded is fake_loaded_store
    mock_faiss_cls.assert_called_once()


def test_load_faiss_store_safe_rejects_malicious_pickle(tmp_path: Path) -> None:
    """Verify load_faiss_store_safe returns None when only a malicious .pkl exists.

    Regression test for issue #930 — the legacy pickle migration path used
    FAISS.load_local(allow_dangerous_deserialization=True), which would execute
    arbitrary code from a tampered index.pkl file. After removing the legacy
    migration path, load_faiss_store_safe must return None when only a .pkl file
    is present (no safe-format index.faiss + metadata.json).

    This test verifies the fail-closed behavior: a planted malicious index.pkl
    must NOT be deserialized under any circumstances.
    """
    from src.api.rag_index import load_faiss_store_safe

    persist_dir = tmp_path / "malicious_index"
    persist_dir.mkdir()
    # Plant a malicious pickle file (no safe-format files).
    # The content is arbitrary — we verify it is NOT loaded.
    (persist_dir / "index.pkl").write_bytes(b"\x80\x03}q\x00X\x07\x00\x00\x00maliciousq\x01.")

    embeddings = _DummyEmbeddings()
    result = load_faiss_store_safe(persist_dir, embeddings)

    # Must return None — malicious pickle must NOT be deserialized
    assert result is None, "load_faiss_store_safe must reject index.pkl (no pickle loading)"


def test_load_faiss_store_safe_returns_none_when_no_index(tmp_path: Path) -> None:
    """Verify load_faiss_store_safe returns None when the index directory is empty.

    Ensures fail-closed behavior: no files → None (not an exception).
    """
    from src.api.rag_index import load_faiss_store_safe

    persist_dir = tmp_path / "empty_index"
    persist_dir.mkdir()

    embeddings = _DummyEmbeddings()
    result = load_faiss_store_safe(persist_dir, embeddings)

    assert result is None
