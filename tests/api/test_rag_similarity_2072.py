"""Regression tests for #2072 — unify the RAG L2->similarity formula + apply
score_threshold in the API search.

The API used ``1 - L2/2`` while the agent used ``1/(1+L2)``, so scores were
incomparable; the API never applied a threshold. Both now use the shared
``l2_to_similarity`` helper, and the API honors an optional ``score_threshold``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.rag.scoring import l2_to_similarity

pytest.importorskip("fastapi")

import cogtrix_core.api.routes.rag as rag_routes  # noqa: E402


def test_l2_to_similarity_formula() -> None:
    assert l2_to_similarity(0.0) == 1.0
    assert l2_to_similarity(1.0) == 0.5
    assert l2_to_similarity(9.0) == 0.1
    assert l2_to_similarity(-5.0) == 1.0  # negative distance clamped to 0


class _Doc:
    def __init__(self, content: str) -> None:
        self.page_content = content


def _store_with(*pairs):
    store = MagicMock()
    store.similarity_search_with_score.return_value = list(pairs)
    return store


def test_search_faiss_uses_shared_similarity_and_threshold(tmp_path) -> None:
    doc = SimpleNamespace(id="doc1", filename="d.md")
    (tmp_path / "doc1" / "vectordb" / "faiss_index").mkdir(parents=True)
    store = _store_with(
        (_Doc("near"), 0.0),  # similarity 1.0
        (_Doc("mid"), 1.0),  # similarity 0.5
        (_Doc("far"), 9.0),  # similarity 0.1
    )
    with (
        patch.object(rag_routes, "_get_uploads_dir", return_value=tmp_path),
        patch.object(rag_routes, "load_faiss_store", return_value=store),
        patch("cogtrix_core.tools.rag._get_embeddings", return_value=MagicMock()),
    ):
        # No threshold -> all three, scored via the shared helper.
        chunks, n = rag_routes._search_faiss("q", 5, [doc])
        assert n == 1
        scores = {c.text: c.score for c in chunks}
        assert scores["near"] == 1.0
        assert scores["mid"] == 0.5
        assert scores["far"] == 0.1

        # threshold 0.4 -> "far" (0.1) is filtered out.
        chunks2, _ = rag_routes._search_faiss("q", 5, [doc], score_threshold=0.4)
        texts = {c.text for c in chunks2}
        assert texts == {"near", "mid"}
