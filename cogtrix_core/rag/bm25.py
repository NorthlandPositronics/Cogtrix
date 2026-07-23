"""BM25 sparse-retrieval sidecar for hybrid RAG queries (#1981).

The FAISS index built by :mod:`src.rag.ingest` is a *dense* retriever —
it returns the K chunks whose vector representation is closest to the
query's embedding in L2 space.  Pure dense retrieval has two
well-documented failure modes that #1952 catalogued for the
qwen3-embedding model running on the cluster:

* **Regime B — numeric / monetary tokens are smoothed.** A query like
  ``Approved envelope: $2,400,000`` should surface ``07_budget.md``,
  but the embedding treats the dollar amount as semantically generic
  and the right chunk falls out of the top 10.
* **Regime C — one "dense" document dominates.** Queries with
  generic role-words (``budget variance memo``, ``schedule slip``)
  all retrieve chunks from the stakeholder-register document — its
  name+role+description chunks happen to sit at the centroid of the
  corpus's embedding space.

BM25 is the standard sparse-retrieval baseline.  It tokenises text,
computes term-frequency × inverse-document-frequency with a
saturation curve, and ranks chunks by the resulting score.  Crucially,
it treats every token equally — ``$2,400,000`` becomes the literal
token ``2400000`` (after punctuation stripping) and the budget doc
gets a hit; generic role-words don't pull toward the stakeholder
register because BM25's IDF weighting penalises frequent terms.

This module builds the sparse-retrieval *sidecar* — a pickled tuple
of the tokenised corpus + per-chunk metadata that
:mod:`src.tools.rag` can load alongside the FAISS index and use to
produce a second ranked list.  The two lists are then fused via
Reciprocal Rank Fusion (RRF) — see :func:`reciprocal_rank_fusion`.

Design properties
================================================================

* **Opt-in.**  ``IngestConfig.build_bm25_sidecar`` controls
  build-time creation; ``configure_rag({"use_bm25_hybrid": True})``
  controls query-time use.  Both default to ``False``, so existing
  pure-vector pipelines see zero behaviour change.
* **Sidecar, not replacement.**  The FAISS index is still authoritative
  for dense retrieval.  BM25 only contributes ranking signal.  Falling
  back to pure-vector when the sidecar is absent is the documented
  graceful-degradation path.
* **Schema versioning.**  ``SIDECAR_VERSION`` bumps invalidate the
  pickle on next ingest — keeps the format honest as the tokeniser
  or stored metadata evolves.
* **No vector-store coupling.**  This module knows nothing about
  FAISS or langchain ``Document`` shape.  It produces and consumes
  plain tuples and ``BM25Sidecar`` instances.  The fusion + langchain
  bridging lives in :mod:`src.tools.rag` where the dense path already
  sits.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("cogtrix.rag.bm25")

# File the pickle lives at, alongside the FAISS index files.
# Co-locating with ``index.faiss`` means a single ``vectordb_dir`` is
# all the operator has to wire — no separate path config.
_SIDECAR_FILENAME = "bm25.pkl"

# Bump this when ``BM25Sidecar``'s fields change in an incompatible
# way (added field, changed tokenisation rules, etc.).  Sidecars at
# an older schema are silently ignored on load — the operator gets a
# pure-vector path until ``--ingest`` rebuilds.
SIDECAR_VERSION = "v1"

# Tokeniser — lowercase, split on non-word runs.  ``re.findall(r"\\w+", ...)``
# is the standard ``rank_bm25`` recipe.  Keeping the regex as a module
# constant makes the contract explicit: numeric/monetary tokens
# (``$2,400,000`` → ``2``, ``400``, ``000``) become discrete terms.
# BM25's IDF weighting then handles their statistical importance —
# the dollar amount stays unique across the corpus while ``000`` is
# heavily downweighted.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Minimum tokens a chunk must have to be worth indexing.  Empty / very-
# short chunks (caption lines, single-word headings) add noise to IDF
# without contributing useful retrieval signal.  Conservative default;
# tune via :func:`build_sidecar`'s ``min_tokens`` kwarg if needed.
_DEFAULT_MIN_TOKENS = 2


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract word tokens.

    Matches the recipe used in :mod:`rank_bm25`'s own README examples.
    Punctuation, currency symbols, and whitespace are stripped; what
    remains are lowercased word tokens.
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass(slots=True)
class BM25Sidecar:
    """Tokenised corpus + per-chunk metadata sidecar for BM25 retrieval.

    Attributes
    ----------
    corpus_tokens:
        ``[chunk_idx -> list[str]]``  Tokenised chunks in ingest order.
        Aligned 1:1 with ``chunk_texts``.
    chunk_texts:
        Original chunk text strings.  Stored so :mod:`src.tools.rag` can
        reconstruct ``langchain_core.documents.Document`` objects for
        fusion without re-hitting the FAISS store.
    chunk_metadata:
        Per-chunk metadata dict (source filename, page, etc.) from
        the langchain ``Document.metadata`` slot.  Mirrors the FAISS
        side so fused results carry the same provenance.
    schema_version:
        ``SIDECAR_VERSION`` at build time.  ``load_sidecar`` returns
        ``None`` for older versions — graceful fall-back to pure-vector.
    """

    corpus_tokens: list[list[str]]
    chunk_texts: list[str]
    chunk_metadata: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SIDECAR_VERSION

    def __post_init__(self) -> None:
        # Defensive shape check — the build path always produces
        # aligned lists; if any caller hand-constructs a sidecar with
        # mismatched lengths we'd produce silently wrong results.
        n = len(self.corpus_tokens)
        if len(self.chunk_texts) != n:
            raise ValueError(
                f"BM25Sidecar: corpus_tokens ({n}) and chunk_texts "
                f"({len(self.chunk_texts)}) must be the same length"
            )
        if self.chunk_metadata and len(self.chunk_metadata) != n:
            raise ValueError(
                f"BM25Sidecar: chunk_metadata ({len(self.chunk_metadata)}) "
                f"must be empty or the same length as corpus_tokens ({n})"
            )
        # Fill metadata with empty dicts if caller didn't provide it —
        # the query path zips against this list and expects aligned
        # indexing.
        if not self.chunk_metadata:
            self.chunk_metadata = [{} for _ in range(n)]


def build_sidecar(
    chunks: list[Any],
    *,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
) -> BM25Sidecar | None:
    """Build a :class:`BM25Sidecar` from a list of langchain ``Document``.

    Returns ``None`` when the corpus is empty after filtering — the
    caller should skip the sidecar-save step in that case.

    Parameters
    ----------
    chunks:
        Iterable of langchain-style ``Document`` instances (or anything
        that quacks the same — ``page_content: str`` and
        ``metadata: dict``).  Duck-typed so callers don't have to import
        langchain just to construct a sidecar in a test.
    min_tokens:
        Chunks tokenising to fewer terms than this are skipped (they
        contribute noise to IDF without useful retrieval signal).
    """
    corpus_tokens: list[list[str]] = []
    chunk_texts: list[str] = []
    chunk_metadata: list[dict[str, Any]] = []

    skipped = 0
    for chunk in chunks:
        text = getattr(chunk, "page_content", None)
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        tokens = _tokenize(text)
        if len(tokens) < min_tokens:
            skipped += 1
            continue
        corpus_tokens.append(tokens)
        chunk_texts.append(text)
        metadata = getattr(chunk, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_metadata.append(dict(metadata))

    if not corpus_tokens:
        _log.warning(
            "BM25 sidecar build: every chunk filtered out (skipped=%d, min_tokens=%d) — "
            "no sidecar will be written",
            skipped,
            min_tokens,
        )
        return None

    if skipped:
        _log.info(
            "BM25 sidecar build: indexed %d chunks, skipped %d (below min_tokens=%d)",
            len(corpus_tokens),
            skipped,
            min_tokens,
        )

    return BM25Sidecar(
        corpus_tokens=corpus_tokens,
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
    )


def save_sidecar(sidecar: BM25Sidecar, vectordb_dir: Path) -> Path:
    """Pickle ``sidecar`` to ``vectordb_dir / bm25.pkl``.

    Creates ``vectordb_dir`` if needed.  Returns the on-disk path.
    """
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    out = vectordb_dir / _SIDECAR_FILENAME
    with out.open("wb") as fh:
        pickle.dump(sidecar, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_sidecar(vectordb_dir: Path) -> BM25Sidecar | None:
    """Load a sidecar from ``vectordb_dir / bm25.pkl``.

    Returns ``None`` when:

    * the file is absent (operator never built one);
    * the pickle is corrupt;
    * the schema version doesn't match :data:`SIDECAR_VERSION`.

    In every "missing or stale" case the caller falls back to
    pure-vector retrieval — graceful degradation is the contract.
    """
    path = vectordb_dir / _SIDECAR_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            # B301: the sidecar is a locally-produced artefact written by
            # ``save_sidecar`` into ``vectordb_dir`` — the same directory
            # that already holds FAISS's own ``index.pkl`` (which the
            # FAISS loader unpickles too).  Threat model is identical to
            # the existing dense-index load path: an attacker who can
            # write ``bm25.pkl`` can already write ``index.pkl``.  No
            # additional attack surface introduced.
            obj = pickle.load(fh)  # nosec B301
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
        _log.warning("BM25 sidecar load failed (%s): %s — falling back to vector-only", path, exc)
        return None

    if not isinstance(obj, BM25Sidecar):
        _log.warning(
            "BM25 sidecar load: unexpected pickle type %s — falling back to vector-only",
            type(obj).__name__,
        )
        return None
    if obj.schema_version != SIDECAR_VERSION:
        _log.info(
            "BM25 sidecar load: schema version %s != current %s — falling back to vector-only "
            "(run --ingest again to rebuild)",
            obj.schema_version,
            SIDECAR_VERSION,
        )
        return None
    return obj


def query_sidecar(
    sidecar: BM25Sidecar,
    question: str,
    k: int = 4,
) -> list[tuple[int, float]]:
    """Run a BM25 query against ``sidecar`` and return ``[(chunk_idx, score)]``.

    Top-K by descending score.  Empty result when the question
    tokenises to nothing (e.g. all punctuation) or no chunk has a
    positive score.

    Returns chunk indices into ``sidecar.corpus_tokens`` /
    ``sidecar.chunk_texts``; callers project those into ``Document``
    shape via the sidecar's parallel arrays.
    """
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    # Imported lazily so importing this module doesn't require
    # ``rank_bm25`` to be installed (sidecar-build path can still
    # run; only the query path needs the lib).
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        _log.warning(
            "BM25 hybrid retrieval requested but rank_bm25 is not installed; "
            "install it with: uv pip install 'cogtrix[rag]'"
        )
        return []

    bm25 = BM25Okapi(sidecar.corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    # Pair scores with indices, keep only positive-score hits (a score
    # of 0 means no query token matched the chunk — nothing to fuse),
    # and take top-K by descending score.
    scored = [(i, float(s)) for i, s in enumerate(scores) if s > 0.0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    *,
    k_constant: int = 60,
    key: Any = None,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Implementation of the algorithm from Cormack et al. 2009 ("Reciprocal
    Rank Fusion outperforms Condorcet and individual Rank Learning
    Methods").  For an item ``d`` appearing at rank ``r_i`` (1-based)
    in the ``i``-th list, contributes ``1 / (k_constant + r_i)`` to its
    score.  Items absent from a list contribute nothing.

    Parameters
    ----------
    ranked_lists:
        Outer list = one ranked list per retriever.  Inner lists are
        the ranked items, best-first.
    k_constant:
        The RRF tuning parameter.  The standard value (60) works
        across corpora without per-corpus tuning.
    key:
        Optional ``Callable[[item], Hashable]`` used to extract the
        identity key from each ranked item.  When omitted, items are
        used directly (they must already be hashable).  This lets us
        pass ranked ``Document`` lists and key by chunk-id.

    Returns
    -------
    A list of ``(item, fused_score)`` tuples sorted by descending
    score.  Items appearing in multiple lists rise to the top — that's
    the fusion effect.
    """
    if k_constant <= 0:
        raise ValueError(f"RRF k_constant must be positive (got {k_constant})")

    # Group items by their identity key while preserving first-seen
    # ordering so ties resolve deterministically.
    scores: dict[Any, float] = {}
    representatives: dict[Any, Any] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            ident = key(item) if key is not None else item
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k_constant + rank)
            representatives.setdefault(ident, item)

    fused = [(representatives[ident], score) for ident, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


__all__ = [
    "BM25Sidecar",
    "SIDECAR_VERSION",
    "build_sidecar",
    "load_sidecar",
    "query_sidecar",
    "reciprocal_rank_fusion",
    "save_sidecar",
]
