"""Unit tests for the BM25 sidecar primitives (#1981).

Covers the building blocks — tokenisation, sidecar build / save /
load roundtrip, the BM25 query helper, and the Reciprocal Rank
Fusion algorithm.  Integration tests against the full
``query_knowledge_base`` flow live in ``test_rag_hybrid_query.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogtrix_core.rag.bm25 import (
    SIDECAR_VERSION,
    BM25Sidecar,
    _tokenize,
    build_sidecar,
    load_sidecar,
    query_sidecar,
    reciprocal_rank_fusion,
    save_sidecar,
)

# ── Tokeniser ─────────────────────────────────────────────────────────


class TestTokenizer:
    def test_lowercases(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("Hello, world!") == ["hello", "world"]

    def test_splits_on_currency_punctuation(self) -> None:
        """``$2,400,000`` becomes three numeric tokens — exactly what BM25
        needs to deterministically hit a budget doc that uses that exact
        amount (Regime B from #1952)."""
        tokens = _tokenize("Approved envelope: $2,400,000")
        # Each digit run is its own token.
        assert "2" in tokens
        assert "400" in tokens
        assert "000" in tokens
        # Word tokens too.
        assert "approved" in tokens
        assert "envelope" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []
        assert _tokenize("   ") == []

    def test_pure_punctuation(self) -> None:
        assert _tokenize("!!!,.,") == []


# ── Sidecar shape / invariants ─────────────────────────────────────────


class _FakeChunk:
    """Minimal stand-in for a langchain ``Document``."""

    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


class TestBM25SidecarInvariants:
    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="must be the same length"):
            BM25Sidecar(
                corpus_tokens=[["a"], ["b"]],
                chunk_texts=["only one"],
            )

    def test_metadata_filled_to_match_corpus_length(self) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[["a"], ["b"]],
            chunk_texts=["a", "b"],
        )
        assert sidecar.chunk_metadata == [{}, {}]

    def test_explicit_metadata_preserved(self) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[["a"]],
            chunk_texts=["a"],
            chunk_metadata=[{"source": "doc.md"}],
        )
        assert sidecar.chunk_metadata == [{"source": "doc.md"}]


# ── build_sidecar ──────────────────────────────────────────────────────


class TestBuildSidecar:
    def test_builds_from_simple_chunks(self) -> None:
        chunks = [
            _FakeChunk("Approved envelope: $2,400,000", {"source": "budget.md"}),
            _FakeChunk("The schedule is on track.", {"source": "status.md"}),
        ]
        sidecar = build_sidecar(chunks)
        assert sidecar is not None
        assert len(sidecar.corpus_tokens) == 2
        assert sidecar.chunk_texts[0] == "Approved envelope: $2,400,000"
        assert sidecar.chunk_metadata[0] == {"source": "budget.md"}

    def test_filters_below_min_tokens(self) -> None:
        chunks = [
            _FakeChunk("only one"),  # 2 tokens — at default threshold
            _FakeChunk("hi"),  # 1 token — filtered out
            _FakeChunk(""),  # empty — filtered out
        ]
        sidecar = build_sidecar(chunks, min_tokens=2)
        assert sidecar is not None
        assert len(sidecar.corpus_tokens) == 1
        assert sidecar.chunk_texts == ["only one"]

    def test_returns_none_for_empty_corpus(self) -> None:
        assert build_sidecar([]) is None
        # Everything filtered out → also None.
        assert build_sidecar([_FakeChunk("")]) is None

    def test_handles_missing_metadata(self) -> None:
        # Chunks without a ``metadata`` attribute / with None metadata
        # produce empty-dict metadata entries — no crash.
        chunks = [_FakeChunk("two words", metadata=None)]
        sidecar = build_sidecar(chunks)
        assert sidecar is not None
        assert sidecar.chunk_metadata == [{}]


# ── Save / load roundtrip ──────────────────────────────────────────────


class TestSidecarPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[["alpha", "beta"], ["gamma"]],
            chunk_texts=["alpha beta", "gamma"],
            chunk_metadata=[{"src": "a.md"}, {"src": "b.md"}],
        )
        out = save_sidecar(sidecar, tmp_path)
        assert out.is_file()
        loaded = load_sidecar(tmp_path)
        assert loaded is not None
        assert loaded.corpus_tokens == sidecar.corpus_tokens
        assert loaded.chunk_texts == sidecar.chunk_texts
        assert loaded.chunk_metadata == sidecar.chunk_metadata
        assert loaded.schema_version == SIDECAR_VERSION

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert load_sidecar(tmp_path) is None

    def test_load_returns_none_for_corrupt_pickle(self, tmp_path: Path) -> None:
        # Garbage bytes — must not crash, must fall back to None so the
        # caller can take the pure-vector path.
        (tmp_path / "bm25.pkl").write_bytes(b"not a real pickle")
        assert load_sidecar(tmp_path) is None

    def test_load_returns_none_for_schema_mismatch(self, tmp_path: Path) -> None:
        import pickle

        bad = BM25Sidecar(
            corpus_tokens=[["a"]],
            chunk_texts=["a"],
            schema_version="v-from-the-future",
        )
        (tmp_path / "bm25.pkl").write_bytes(pickle.dumps(bad))
        assert load_sidecar(tmp_path) is None


# ── Query helper ───────────────────────────────────────────────────────


class TestQuerySidecar:
    def test_returns_chunk_idx_score_pairs_sorted_desc(self) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[
                ["budget", "approved", "envelope", "2", "400", "000"],
                ["schedule", "on", "track", "this", "quarter"],
                ["risk", "register", "high", "high", "data", "team"],
            ],
            chunk_texts=["budget chunk", "schedule chunk", "risk chunk"],
        )
        result = query_sidecar(sidecar, "approved envelope 2 400 000", k=2)
        assert result, "BM25 should hit the budget chunk"
        # Top result must be the budget chunk (index 0).
        assert result[0][0] == 0
        # Scores descending.
        for prev, curr in zip(result, result[1:], strict=False):
            assert prev[1] >= curr[1]

    def test_returns_empty_for_empty_query(self) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[["a", "b"]],
            chunk_texts=["a b"],
        )
        assert query_sidecar(sidecar, "", k=4) == []
        # All-punctuation query tokenises to nothing.
        assert query_sidecar(sidecar, "!!!,?", k=4) == []

    def test_returns_empty_when_no_match(self) -> None:
        sidecar = BM25Sidecar(
            corpus_tokens=[["alpha", "beta"]],
            chunk_texts=["alpha beta"],
        )
        # ``gamma`` doesn't appear in the corpus — BM25 score is 0,
        # filtered out by the query helper.
        assert query_sidecar(sidecar, "gamma", k=4) == []


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_single_list_pass_through(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"]])
        items = [item for item, _ in fused]
        assert items == ["a", "b", "c"]
        # Rank-1 score = 1 / (60 + 1).
        assert pytest.approx(fused[0][1], abs=1e-9) == 1.0 / 61.0
        assert pytest.approx(fused[1][1], abs=1e-9) == 1.0 / 62.0

    def test_item_in_both_lists_rises(self) -> None:
        # ``a`` appears at rank 2 in list-1 and rank 1 in list-2 →
        # should outrank ``b`` (rank 1 in list-1 only) and ``c``
        # (rank 2 in list-2 only).
        fused = reciprocal_rank_fusion([["b", "a"], ["a", "c"]])
        items = [item for item, _ in fused]
        assert items[0] == "a"
        assert set(items) == {"a", "b", "c"}

    def test_custom_k_constant(self) -> None:
        # Larger k_constant flattens the rank-1-vs-rank-K gap.
        fused = reciprocal_rank_fusion([["a", "b"]], k_constant=1)
        assert pytest.approx(fused[0][1], abs=1e-9) == 1.0 / 2.0
        assert pytest.approx(fused[1][1], abs=1e-9) == 1.0 / 3.0

    def test_rejects_non_positive_k_constant(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            reciprocal_rank_fusion([["a"]], k_constant=0)

    def test_key_function_groups_by_identity(self) -> None:
        # Items are dicts (unhashable); ``key=`` extracts the doc-id.
        list_a = [{"id": 1, "src": "vec"}, {"id": 2, "src": "vec"}]
        list_b = [{"id": 2, "src": "bm25"}, {"id": 1, "src": "bm25"}]
        fused = reciprocal_rank_fusion([list_a, list_b], key=lambda d: d["id"])
        # Same length as the union of ids (2), not the sum.
        assert len(fused) == 2
        # Both items appearing on both sides means the higher combined
        # rank wins — id=2 is rank-2 + rank-1, id=1 is rank-1 + rank-2,
        # so they tie at the same score.
        scores = [s for _, s in fused]
        assert pytest.approx(scores[0], abs=1e-9) == scores[1]


# ── Regime-B reproducer (synthetic) ────────────────────────────────────


class TestRegimeBMonetaryTokens:
    """Synthetic reproduction of #1952's Regime B failure.

    Builds a corpus with one budget doc + multiple unrelated docs.
    Queries an exact monetary token verbatim from the budget doc.
    BM25 must surface the budget doc top-1 — that's the property the
    embedding model fails on with qwen3-embedding.
    """

    def test_bm25_finds_budget_doc_by_exact_amount(self) -> None:
        chunks = [
            _FakeChunk(
                "Project status M4: schedule is on track, no risks "
                "flagged this cycle, team morale steady."
            ),
            _FakeChunk(
                "Steering committee Q1 minutes: agenda covered roadmap, "
                "staffing, vendor selection, exec sponsor sign-off."
            ),
            _FakeChunk(
                "Approved envelope: $2,400,000.  Budget allocation by "
                "phase: M1 $400,000; M2 $600,000; M3 $700,000; M4 $700,000."
            ),
            _FakeChunk(
                "Stakeholder register: Hyeon-Jin Park (Migration Squad "
                "Lead), Beatriz Cazadora-Olesen (Data Squad Lead), "
                "Tomislav Hessford (Steering Sponsor)."
            ),
            _FakeChunk(
                "Risk register R-12: AcmeDB cross-region replication lag.  "
                "Owner: Beatriz Cazadora-Olesen.  Severity high."
            ),
        ]
        sidecar = build_sidecar(chunks)
        assert sidecar is not None

        # Verbatim text from the budget doc.
        hits = query_sidecar(sidecar, "Approved envelope: $2,400,000", k=3)
        assert hits, "BM25 should produce at least one hit"
        top_idx = hits[0][0]
        # The budget doc is chunk index 2 in the corpus above.
        assert top_idx == 2, (
            f"BM25 top-1 for exact monetary query should be the budget "
            f"chunk (idx 2), got idx {top_idx}.  Regime B reproducer has regressed."
        )

    def test_bm25_handles_partial_monetary_match(self) -> None:
        """A partial query (just the dollar amount, no surrounding text)
        still surfaces the budget doc — BM25's IDF makes the unique
        amount a strong signal even on its own."""
        chunks = [
            _FakeChunk("Schedule status: on track."),
            _FakeChunk("Approved envelope: $2,400,000."),
            _FakeChunk("Stakeholder register."),
        ]
        sidecar = build_sidecar(chunks)
        assert sidecar is not None
        hits = query_sidecar(sidecar, "$2,400,000", k=2)
        assert hits, "BM25 should surface the budget chunk"
        assert hits[0][0] == 1
