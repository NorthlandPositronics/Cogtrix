"""#2008 — retrieval regression gate for the specific corpus-token misses.

#2008 catalogued deterministic (all-3-iters) retrieval misses in the PM harness:
a fact lives verbatim in the corpus, the query asks for it, `rag_consulted=True`
and the tools/format are all correct — yet the retrieval never puts the token in
front of the model. The issue's acceptance is *"after the CE re-ranker is enabled
and landed, re-run and check whether each token now surfaces."* This test codifies
that check so it stops being a manual ripgrep after every cycle.

It runs the **real** retrieval path — the same `ingest_corpus_idempotent` +
`configure_rag(use_cross_encoder_rerank=True)` + `query_knowledge_base` the harness
uses (`run.py`) — against the committed FAISS index, and asserts each gold token
appears in the top-k results.

**Regime split (end-to-end evidence against Spark qwen3-embedding, 2026-07-03):**
- **Regime B (numeric)** — the re-ranker alone does NOT fix it: qwen3-embedding
  ranks the ``$1,106,500`` gold chunk at FAISS rank ~165/296, far below the CE
  re-rank pool (24), so the CE never sees it (top-8 had no ``07_budget.md`` at
  all). **BM25 hybrid is the fix**: lexical retrieval ranks that chunk ~#8, RRF
  fuses it into the pool, and the CE promotes it into the top-k — verified
  end-to-end (``$1,106,500`` surfaces). Asserted (with ``use_bm25_hybrid`` on).
- **Regime C (proper-noun / role owner)** — ALSO fixed by BM25 hybrid (#2426).
  The earlier "unfixable" read was on standalone components (CE-only #26, BM25-only
  #33); the full pipeline fuses BM25+vector via RRF *then* CE-reranks, which
  surfaces the R-13 / AcmeCloud-TAM entry into the top-8 even for the generic
  query. Asserted (verified end-to-end). The residual `role_pm_02` miss is
  **response-shaping / attribution** (the model names the *formal* owner or
  hallucinates — not retrieval), tracked in #2426; regime-D (SC-cluster) likewise.

**Live retrieval is skipped without an embedding provider** (`ROLE_PM_EMBEDDING_*`
or `OPENAI_API_KEY`) — querying FAISS needs the same embedding model that built the
committed index. The pure-logic tests below always run (no embeddings, CI-safe).
Run live with, e.g.::

    OPENAI_API_KEY=… uv run --extra rag --extra rag-rerank \\
        pytest tests/role_pm/test_retrieval_regression_2008.py -q
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_HARNESS_DIR = Path(__file__).resolve().parent
_CORPUS_DIR = _HARNESS_DIR / "corpus"
_FAISS_INDEX_DIR = _HARNESS_DIR / "rag" / "faiss_index"

# Steer to k=8 like the harness (run.py) — qwen3-embedding needs more candidates.
_K = 8


@dataclass(frozen=True)
class RetrievalCase:
    """A single #2008 deterministic-miss reproducer."""

    case_id: str
    query: str
    # The gold surfaces if ANY of these token variants appears in the top-k text
    # (mirrors the scenarios' ``at_least_n_contains: 1 | …`` criteria).
    gold_any: tuple[str, ...]
    expected_source: str  # corpus doc the gold verbatim lives in
    regime: str  # "B" numeric | "C" proper-noun | "D" cluster
    xfail_reason: str | None = field(default=None)


# The deterministic (all-3-iters) misses from #2008. Queries are the scenarios'
# real user prompts (tests/role_pm/scenarios/*.yaml).
_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        case_id="regimeB_budget_actuals",
        query=(
            "Draft a budget variance memo: actuals through end of last month, the "
            "end-of-programme forecast, and the line items driving the variance with "
            "concrete numbers and line IDs."
        ),
        gold_any=("$1,106,500", "1,106,500", "1,106"),
        expected_source="07_budget.md",
        regime="B",
        # Fixed by BM25 hybrid (embedding buries it at rank ~165; BM25 ~#8 → pool
        # → CE promotes it). Verified end-to-end against Spark qwen3-embedding.
        xfail_reason=None,
    ),
    RetrievalCase(
        case_id="regimeC_vendor_risk_owner",
        query=(
            "What does our risk register say about our vendor dependencies? Give the "
            "risk IDs, the owners, and what mitigations are in flight."
        ),
        gold_any=("Yusuf Almasi", "AcmeCloud TAM", "Technical Account Manager"),
        expected_source="05_risk_register.md",
        regime="C",
        # #2426: ALSO fixed by BM25 hybrid — the earlier "unfixable" call was on
        # standalone components (CE-only #26, BM25-only #33); the full pipeline
        # fuses BM25+vector via RRF THEN CE-reranks, which surfaces the R-13 /
        # AcmeCloud-TAM entry (chunk 75, "…actions in flight") into the top-8 for
        # this generic query. Verified end-to-end against Spark qwen3-embedding.
        # (The residual agent-level miss is response-shaping/attribution — the
        # model names the *formal* owner or hallucinates — NOT retrieval; #2426.)
        xfail_reason=None,
    ),
)


def _embedding_provider_configured() -> bool:
    """True when an embedding endpoint the harness understands is available."""
    return bool(
        os.environ.get("ROLE_PM_EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ROLE_PM_EMBEDDING_BASE_URL")  # e.g. a local ollama
    )


def _gold_present(result_text: str, gold_any: tuple[str, ...]) -> bool:
    """Mirror the scenario ``at_least_n_contains: 1`` semantics."""
    return any(tok in result_text for tok in gold_any)


def _run_retrieval(query: str, k: int = _K) -> str:
    """Faithfully reproduce the harness retrieval path and return the top-k text.

    Mirrors run.py: idempotent corpus ingest (reuses the committed index when the
    hash matches) + configure_rag with the CE re-ranker ON + query_knowledge_base.
    """
    from tests.role_pm.corpus_ingest import ingest_corpus_idempotent

    provider = os.environ.get("ROLE_PM_EMBEDDING_PROVIDER", "openai")
    ingest_corpus_idempotent(
        corpus_dir=_CORPUS_DIR,
        vectordb_dir=_FAISS_INDEX_DIR,
        embedding_provider=provider,
        embedding_model=os.environ.get("ROLE_PM_EMBEDDING_MODEL"),
        base_url=os.environ.get("ROLE_PM_EMBEDDING_BASE_URL"),
        api_key=os.environ.get("ROLE_PM_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )

    from cogtrix_core.tools.rag import configure_rag, query_knowledge_base

    configure_rag(
        {
            "vectordb_dir": str(_FAISS_INDEX_DIR),
            "api_uploads_dir": None,
            "embedding_provider": provider,
            "embedding_model": os.environ.get("ROLE_PM_EMBEDDING_MODEL"),
            "base_url": os.environ.get("ROLE_PM_EMBEDDING_BASE_URL"),
            "api_key": os.environ.get("ROLE_PM_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            "score_threshold": 0.0,
            "use_cross_encoder_rerank": True,
            "use_bm25_hybrid": True,  # the #2008 regime-B fix — see module docstring
        }
    )
    return query_knowledge_base(question=query, k=k)


# ── Live retrieval gate (skipped without an embedding provider) ──────────────


@pytest.mark.skipif(
    not _embedding_provider_configured(),
    reason="no embedding provider configured (set ROLE_PM_EMBEDDING_* or OPENAI_API_KEY)",
)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
def test_gold_token_surfaces(case: RetrievalCase) -> None:
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)
    result = _run_retrieval(case.query)
    assert _gold_present(result, case.gold_any), (
        f"[{case.case_id}] none of {case.gold_any} surfaced in top-{_K} "
        f"(expected from {case.expected_source})"
    )


# ── Pure-logic tests (always run — no embeddings) ────────────────────────────


class TestCaseTable:
    def test_cases_are_well_formed(self) -> None:
        assert _CASES, "the #2008 regression corpus must not be empty"
        seen: set[str] = set()
        for c in _CASES:
            assert c.case_id and c.case_id not in seen, f"duplicate/empty id {c.case_id!r}"
            seen.add(c.case_id)
            assert c.query.strip()
            assert c.gold_any and all(t.strip() for t in c.gold_any)
            assert c.expected_source.endswith(".md")
            assert c.regime in {"B", "C", "D"}

    def test_regime_b_and_c_retrieval_are_asserted(self) -> None:
        by_regime = {c.regime: c for c in _CASES}
        # Both regime-B (numeric) and regime-C (proper-noun) retrieval are fixed by
        # BM25 hybrid (#2425/#2426) and verified end-to-end, so both are hard
        # asserts (not xfail). Any regime-C residual is response-shaping, not
        # retrieval, and lives at the agent level — out of this gate's scope.
        assert by_regime["B"].xfail_reason is None
        assert by_regime["C"].xfail_reason is None


class TestGoldPresence:
    def test_matches_any_variant(self) -> None:
        assert _gold_present(
            "… cumulative actuals of $1,106,500 through M4 …", ("$1,106,500", "1,106")
        )

    def test_absent_when_no_variant(self) -> None:
        assert not _gold_present(
            "approximately $1.1M, roughly on budget", ("$1,106,500", "1,106,500")
        )


class TestProviderDetection:
    def test_returns_bool(self) -> None:
        assert isinstance(_embedding_provider_configured(), bool)

    def test_true_when_openai_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ROLE_PM_EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("ROLE_PM_EMBEDDING_BASE_URL", raising=False)
        assert _embedding_provider_configured() is True

    def test_false_when_none_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in (
            "OPENAI_API_KEY",
            "ROLE_PM_EMBEDDING_API_KEY",
            "ROLE_PM_EMBEDDING_BASE_URL",
        ):
            monkeypatch.delenv(v, raising=False)
        assert _embedding_provider_configured() is False
