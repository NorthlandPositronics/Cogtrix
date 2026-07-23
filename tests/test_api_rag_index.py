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
