"""Integration tests for the cross-encoder re-rank wiring in
``src.tools.rag._retrieve_from_index`` (#1952 Option A).

The CE module itself is covered by ``tests/rag/test_reranker.py`` —
this file tests the *plumbing* between ``configure_rag`` /
``_retrieve_from_index`` / ``_apply_cross_encoder_rerank`` / the CE
module.

The CE library is stubbed at ``sys.modules['sentence_transformers']``
so these tests do not require the ``[rag-rerank]`` extra and never
download the ~80 MB model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ── Shared fixtures / fakes ──────────────────────────────────────────────────


class _FakeDoc:
    """Minimal stand-in for a langchain ``Document``."""

    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"<Doc {self.page_content[:30]!r}>"


class _FakeStore:
    """Fake FAISS store with deterministic top-K behaviour.

    Returns ``rank_order[:k]`` so we can drive both the "pre-rerank
    ordering is wrong" and the "rerank fixes it" assertions in one
    fixture.
    """

    def __init__(self, rank_order: list[tuple[_FakeDoc, float]]) -> None:
        self._rank_order = rank_order
        self.last_k_requested: int | None = None

    def similarity_search_with_score(
        self, _question: str, k: int = 4
    ) -> list[tuple[_FakeDoc, float]]:
        # Capture the requested k so tests can pin the over-fetch
        # behaviour when reranking is enabled.
        self.last_k_requested = k
        return list(self._rank_order[:k])


@pytest.fixture()
def _restore_rag_config():
    """Save/restore ``_rag_config`` so per-test ``configure_rag`` calls
    don't leak across tests."""
    import src.tools.rag as _rag_mod

    original = dict(_rag_mod._rag_config)
    yield
    _rag_mod._rag_config.clear()
    _rag_mod._rag_config.update(original)


@pytest.fixture()
def _reset_reranker_cache():
    """Reset the CE singleton between tests so monkeypatched stubs
    don't leak."""
    from src.rag.reranker import _reset_model_cache_for_tests

    _reset_model_cache_for_tests()
    yield
    _reset_model_cache_for_tests()


def _install_stub_cross_encoder(
    monkeypatch: pytest.MonkeyPatch,
    score_by_text: dict[str, float],
) -> dict[str, Any]:
    """Inject a stub ``sentence_transformers.CrossEncoder`` whose scores
    are dictated by ``score_by_text``.  Returns a capture record so
    tests can assert on construction args.
    """
    record: dict[str, Any] = {"init_calls": [], "predict_calls": []}

    class _StubCE:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            record["init_calls"].append({"model_name": model_name, "device": device})

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            record["predict_calls"].append(list(pairs))
            return [score_by_text.get(text, 0.0) for _q, text in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=_StubCE),
    )
    return record


# ── Default-off contract ─────────────────────────────────────────────────────


class TestRerankOffByDefault:
    """``use_cross_encoder_rerank=False`` is the documented default —
    the pre-#1952 behaviour must be bit-identical when the flag is off.
    """

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_pure_vector_path_unchanged_when_flag_off(self, tmp_path: Path) -> None:
        from src.tools.rag import _rag_config, _retrieve_from_index

        # Belt-and-braces: even on a fresh ``_rag_config``, the flag
        # must be explicitly False — a future default flip would change
        # the contract and break the existing test_rag.py suite silently.
        assert _rag_config.get("use_cross_encoder_rerank") is False

        docs = [_FakeDoc(f"c{i}") for i in range(3)]
        store = _FakeStore([(d, float(i) + 0.5) for i, d in enumerate(docs)])

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=3,
            use_hybrid=False,
        )

        # Identical to pre-#1952 — order and scores untouched.
        assert [d.page_content for d, _ in result] == ["c0", "c1", "c2"]
        assert [s for _, s in result] == [0.5, 1.5, 2.5]
        # And the store was queried with the user's exact k, not an
        # over-fetched pool.
        assert store.last_k_requested == 3


class TestRerankFlagOnPureVector:
    """When the flag is on, the CE pass runs over an over-fetched pool
    from the FAISS store and the final ordering follows CE scores.
    """

    @pytest.mark.usefixtures("_restore_rag_config", "_reset_reranker_cache")
    def test_reorders_pure_vector_pool_by_ce_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.tools.rag import _retrieve_from_index, configure_rag

        # FAISS hands back docs in WRONG order — pre-CE rank #0 has the
        # lowest CE score; the right answer is at FAISS rank #4.
        docs = [_FakeDoc(t) for t in ["wrong-a", "wrong-b", "wrong-c", "wrong-d", "correct"]]
        store = _FakeStore([(d, float(i) + 0.1) for i, d in enumerate(docs)])

        # CE scores — "correct" gets the highest, "wrong-*" are low.
        _install_stub_cross_encoder(
            monkeypatch,
            score_by_text={
                "wrong-a": 0.1,
                "wrong-b": 0.2,
                "wrong-c": 0.0,
                "wrong-d": 0.3,
                "correct": 9.9,
            },
        )

        configure_rag({"use_cross_encoder_rerank": True})

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="which is the correct chunk?",
            k=2,
            use_hybrid=False,
        )

        # Top result is the CE-preferred chunk — even though FAISS ranked
        # it dead last in the over-fetched pool.
        assert [d.page_content for d, _ in result] == ["correct", "wrong-d"]

    @pytest.mark.usefixtures("_restore_rag_config", "_reset_reranker_cache")
    def test_over_fetches_when_rerank_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``rerank_over_k_multiplier`` controls the over-fetch size.
        At ``multiplier=3`` and ``k=4``, the FAISS store must be queried
        with ``k=12`` (3 × 4) so the CE has a wider pool.
        """
        from src.tools.rag import _retrieve_from_index, configure_rag

        docs = [_FakeDoc(f"d{i}") for i in range(20)]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(docs)])

        _install_stub_cross_encoder(monkeypatch, score_by_text={})

        configure_rag(
            {
                "use_cross_encoder_rerank": True,
                "rerank_over_k_multiplier": 3,
            }
        )

        _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=4,
            use_hybrid=False,
        )

        # max(k * multiplier, k + 4) = max(12, 8) = 12.
        assert store.last_k_requested == 12

    @pytest.mark.usefixtures("_restore_rag_config", "_reset_reranker_cache")
    def test_passes_configured_model_and_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.tools.rag import _retrieve_from_index, configure_rag

        docs = [_FakeDoc("a"), _FakeDoc("b")]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(docs)])

        record = _install_stub_cross_encoder(monkeypatch, score_by_text={})

        configure_rag(
            {
                "use_cross_encoder_rerank": True,
                "rerank_model": "BAAI/bge-reranker-large",
                "rerank_device": "cuda:1",
            }
        )

        _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=2,
            use_hybrid=False,
        )

        assert record["init_calls"], "CrossEncoder was not constructed"
        assert record["init_calls"][0]["model_name"] == "BAAI/bge-reranker-large"
        assert record["init_calls"][0]["device"] == "cuda:1"

    @pytest.mark.usefixtures("_restore_rag_config", "_reset_reranker_cache")
    def test_graceful_fallback_when_sentence_transformers_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the library is unavailable, the wiring must NOT raise —
        it falls back to the over-fetched pool's top-K.  Retrieval is
        never worse than the baseline.
        """
        from src.tools.rag import _retrieve_from_index, configure_rag

        # Force ImportError when reranker tries to ``import sentence_transformers``.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        docs = [_FakeDoc(f"d{i}") for i in range(10)]
        store = _FakeStore([(d, float(i) + 0.1) for i, d in enumerate(docs)])

        configure_rag({"use_cross_encoder_rerank": True})

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=3,
            use_hybrid=False,
        )

        # Fallback returns the first k from the over-fetched pool
        # in pre-CE order — i.e. the original FAISS ranking.
        assert [d.page_content for d, _ in result] == ["d0", "d1", "d2"]
