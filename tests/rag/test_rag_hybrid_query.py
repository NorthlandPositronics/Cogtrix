"""Integration tests for the hybrid query path (#1981).

Drives ``src.tools.rag._retrieve_from_index`` and the surrounding
``query_knowledge_base`` flow against synthetic ``FAISS`` stores +
``BM25Sidecar`` instances.  Pins:

1. When ``use_bm25_hybrid`` is OFF, ``_retrieve_from_index`` is the
   pure-vector path — bit-identical to pre-#1981 behaviour.
2. When ``use_bm25_hybrid`` is ON but no sidecar exists, the path
   falls back to pure-vector — graceful degradation.
3. When both are present, RRF fusion produces a top-K ordering that
   surfaces chunks that appear in BOTH ranked lists (the desired
   effect).  Reproducer for Regime B from #1952.
4. Back-compat: existing pure-vector tests in ``tests/test_rag.py``
   still pass (verified by running both suites together).
"""

from __future__ import annotations

from pathlib import Path

from cogtrix_core.rag.bm25 import save_sidecar


class _FakeDoc:
    """Minimal stand-in for a langchain ``Document``."""

    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"<Doc {self.page_content[:30]!r}>"


class _FakeStore:
    """Stand-in for a FAISS vector store.  ``similarity_search_with_score``
    returns a deterministic ``(doc, score)`` ranking driven by the
    fixture's ``_rank_order`` mapping."""

    def __init__(self, rank_order: list[tuple[_FakeDoc, float]]) -> None:
        # ``rank_order`` is the deterministic ranking the fake store
        # returns regardless of the query — sufficient for testing the
        # fusion plumbing without a real embedding model.
        self._rank_order = rank_order

    def similarity_search_with_score(
        self, _question: str, k: int = 4
    ) -> list[tuple[_FakeDoc, float]]:
        return list(self._rank_order[:k])


# ── _retrieve_from_index — pure-vector path ─────────────────────────────


class TestRetrieveFromIndexPureVector:
    """When ``use_hybrid=False``, ``_retrieve_from_index`` must be a
    no-op wrapper over ``store.similarity_search_with_score`` — exactly
    the pre-#1981 behaviour."""

    def test_returns_store_results_unchanged(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _retrieve_from_index

        docs = [_FakeDoc(f"chunk {i}") for i in range(5)]
        store = _FakeStore([(d, float(i) + 0.1) for i, d in enumerate(docs)])

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=3,
            use_hybrid=False,
        )
        assert len(result) == 3
        # Identical ordering to the fake store.
        assert [d.page_content for d, _ in result] == ["chunk 0", "chunk 1", "chunk 2"]
        # Identical scores.
        assert [s for _, s in result] == [0.1, 1.1, 2.1]


# ── _retrieve_from_index — hybrid path falls back when sidecar missing ─


class TestHybridFallsBackWithoutSidecar:
    """When the hybrid flag is on but no sidecar exists in
    ``vector_dir``, the pure-vector path runs."""

    def test_no_sidecar_means_pure_vector(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _retrieve_from_index

        docs = [_FakeDoc(f"c{i}") for i in range(3)]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(docs)])

        # tmp_path has no ``bm25.pkl`` — hybrid path falls through.
        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=3,
            use_hybrid=True,
        )
        assert [d.page_content for d, _ in result] == ["c0", "c1", "c2"]


# ── _retrieve_from_index — hybrid path with sidecar ────────────────────


class TestHybridRetrievalWithSidecar:
    """When both the flag and sidecar are present, the fused ranking
    must place chunks that appear on BOTH ranked lists ahead of
    chunks that appear on only one."""

    def test_fused_top1_is_shared_between_ranklists(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _retrieve_from_index

        # Build a sidecar with five chunks; one of them ("the budget chunk")
        # is the one that should rise to the top under fusion.  The text
        # contains the verbatim monetary token so BM25 picks it up.
        chunks_text = [
            "Project status update: schedule on track for M4 delivery.",
            "Steering minutes: agenda included roadmap and staffing.",
            "Approved envelope: $2,400,000 across M1 through M4 phases.",
            "Stakeholder register: Beatriz Cazadora-Olesen (Data Lead).",
            "Risk register R-12: AcmeDB cross-region replication lag.",
        ]
        from cogtrix_core.rag.bm25 import build_sidecar

        sidecar = build_sidecar([_FakeDoc(t) for t in chunks_text])
        assert sidecar is not None
        save_sidecar(sidecar, tmp_path)

        # FAISS-side ranking: the budget chunk is rank-3 (sub-optimal,
        # mirrors the regime-B behaviour from #1952).  The fake store
        # returns the docs in this order at any over-fetched k.
        fake_docs = [_FakeDoc(t) for t in chunks_text]
        vector_rank = [
            (fake_docs[0], 0.40),
            (fake_docs[3], 0.42),  # stakeholder register dominates (regime C)
            (fake_docs[2], 0.50),  # budget chunk: vector rank-3
            (fake_docs[1], 0.55),
            (fake_docs[4], 0.60),
        ]
        store = _FakeStore(vector_rank)

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="Approved envelope: $2,400,000",
            k=3,
            use_hybrid=True,
        )
        top_contents = [d.page_content for d, _ in result]
        assert top_contents, "hybrid query must return some results"
        # The budget chunk MUST be top-1 after fusion: BM25 ranks it
        # rank-1 on its side (exact monetary match), vector ranks it
        # rank-3 → fused score is highest of any chunk.
        assert "Approved envelope" in top_contents[0], (
            f"After RRF fusion the budget chunk must be top-1.  Got top-1 = "
            f"{top_contents[0]!r}.  Regime B reproducer has regressed."
        )

    def test_fused_result_count_capped_at_k(self, tmp_path: Path) -> None:
        from cogtrix_core.rag.bm25 import build_sidecar
        from cogtrix_core.tools.rag import _retrieve_from_index

        # Sidecar has 6 chunks; vector store has 6 chunks; both retrievers
        # over-fetch to 6, but ``k=2`` must clamp the final return.
        chunks_text = [f"chunk number {i} of six total." for i in range(6)]
        sidecar = build_sidecar([_FakeDoc(t) for t in chunks_text])
        assert sidecar is not None
        save_sidecar(sidecar, tmp_path)

        fake_docs = [_FakeDoc(t) for t in chunks_text]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(fake_docs)])

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="chunk number six",
            k=2,
            use_hybrid=True,
        )
        assert len(result) == 2

    def test_synthetic_scores_strictly_ascending(self, tmp_path: Path) -> None:
        """The hybrid path projects fused ranks into ascending synthetic
        scores so the caller's existing sort-ascending logic keeps working
        without special-casing.  Verify the scores are monotonically
        increasing across the returned list."""
        from cogtrix_core.rag.bm25 import build_sidecar
        from cogtrix_core.tools.rag import _retrieve_from_index

        chunks_text = [f"alpha beta gamma chunk {i}" for i in range(4)]
        sidecar = build_sidecar([_FakeDoc(t) for t in chunks_text])
        assert sidecar is not None
        save_sidecar(sidecar, tmp_path)

        fake_docs = [_FakeDoc(t) for t in chunks_text]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(fake_docs)])

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="alpha beta gamma",
            k=4,
            use_hybrid=True,
        )
        scores = [s for _, s in result]
        for prev, curr in zip(scores, scores[1:], strict=False):
            assert prev < curr, (
                f"Hybrid path must emit strictly ascending synthetic scores "
                f"(prev={prev}, curr={curr}) so the caller's sort-ascending "
                f"logic is deterministic."
            )


# ── Sidecar load is best-effort on schema/corruption ───────────────────


class TestHybridGracefulDegradation:
    """When the sidecar load fails for any reason — corrupt pickle,
    schema-version drift — the hybrid path falls through to pure-vector
    silently.  Retrieval is never WORSE than the baseline."""

    def test_corrupt_sidecar_falls_back(self, tmp_path: Path) -> None:
        from cogtrix_core.tools.rag import _retrieve_from_index

        # Garbage bytes — ``load_sidecar`` returns None, hybrid path
        # falls through.
        (tmp_path / "bm25.pkl").write_bytes(b"not a real pickle")

        docs = [_FakeDoc(f"c{i}") for i in range(3)]
        store = _FakeStore([(d, float(i)) for i, d in enumerate(docs)])

        result = _retrieve_from_index(
            store=store,
            vector_dir=tmp_path,
            question="anything",
            k=3,
            use_hybrid=True,
        )
        # Pure-vector fallback: ranking + scores match the store output.
        assert [d.page_content for d, _ in result] == ["c0", "c1", "c2"]
        assert [s for _, s in result] == [0.0, 1.0, 2.0]
