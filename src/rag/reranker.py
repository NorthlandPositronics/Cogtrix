"""Cross-encoder re-ranker for the RAG retrieval path (#1952).

The default FAISS retrieval (``similarity_search_with_score``) ranks
candidates by raw embedding-space L2 distance.  Diagnostic probing of
the ``qwen3-embedding`` index (issue #1952, PM role-test harness run
#1948) showed three failure regimes:

* **Regime A** — distinctive uncommon tokens work well (large rank gap
  between #1 and #2; the right chunk is clearly on top).
* **Regime B** — numeric / monetary queries get tokenised away.  Top-10
  band collapses to L2 ≈ 0.83-0.99; the document that contains the
  exact dollar figure does not appear at all.
* **Regime C** — generic role-based queries pull into the densest
  document in the corpus (a stakeholder register full of recurring
  names+roles).  Unrelated topics return the same wrong chunks.

A second-pass cross-encoder scores ``(query, chunk)`` pairs jointly
with a small fine-tuned model — much more discriminative than the
bi-encoder ranking from the embedding model alone.  This module adds
that pass as an opt-in stage between FAISS retrieval and the caller's
top-K cut.

Why a separate module
---------------------

* **Optional dependency.** ``sentence-transformers`` transitively pulls
  in ``torch`` and ``transformers`` — hundreds of MB.  Users who only
  want FAISS / BM25 should not pay that.  This module imports the
  library lazily and degrades gracefully when it is missing.
* **Process-singleton model.** The CE model weights (~80 MB for the
  default ``cross-encoder/ms-marco-MiniLM-L-6-v2``) are loaded once
  per process and cached at module scope, so subsequent queries reuse
  the warm model.  The first query after opt-in pays the load cost
  and downloads weights into the HuggingFace cache.
* **Graceful fallback contract.** Every failure mode (lib missing,
  model load fails, scoring raises) falls back to the input docs in
  their incoming order so retrieval is never *worse* than the
  unre-ranked baseline.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

# Default cross-encoder model name.  ``cross-encoder/ms-marco-MiniLM-L-6-v2``
# (~80 MB) is the recommendation from issue #1952 — small enough to load
# on CPU without GPU acceleration, trained on MS-MARCO question-passage
# pairs which match the RAG query shape closely.  Override via
# ``configure_rag({"rerank_model": "..."})`` if a heavier model fits
# the deployment.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-scope cache so the CE model loads exactly once per process per
# (model_name, device) pair.  Guarded by a lock because tool calls may
# fan out across threads in the assistant runtime.
_model_cache: dict[tuple[str, str | None], Any] = {}
_model_cache_lock = threading.Lock()

_logger = logging.getLogger(__name__)


def _try_load_cross_encoder(model_name: str, device: str | None) -> Any | None:
    """Lazy-load ``sentence_transformers.CrossEncoder`` for the given model.

    Returns ``None`` on any failure (lib missing, model not found,
    download blocked, OOM) so callers can fall back cleanly.

    The result is cached at module scope keyed by ``(model_name, device)``
    so the second call returns the already-loaded model without retrying
    the import or the download.
    """
    cache_key = (model_name, device)
    with _model_cache_lock:
        cached = _model_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        # Imported lazily — module load must not require sentence-transformers
        # to be installed.  Users on the ``[rag]`` extra (without
        # ``[rag-rerank]``) hit this path the first time the reranker
        # would fire; the ImportError is the documented fallback.
        #
        # CI does not install the ``[rag-rerank]`` extra (it pulls in
        # torch + transformers, hundreds of MB), so pyright in CI cannot
        # resolve the symbol.  Runtime behaviour is guarded by the
        # except below; the type-checker just needs to be told.
        from sentence_transformers import (  # pyright: ignore[reportMissingImports]
            CrossEncoder,
        )
    except ImportError:
        _logger.info(
            "Cross-encoder re-ranker unavailable: sentence-transformers is not "
            "installed.  Install the [rag-rerank] extra to enable.  Falling "
            "back to the un-re-ranked retrieval order."
        )
        return None

    try:
        model = CrossEncoder(model_name, device=device) if device else CrossEncoder(model_name)
    except Exception as exc:  # noqa: BLE001 — broad: model load can fail many ways
        _logger.warning(
            "Cross-encoder re-ranker failed to load model %r: %s.  Falling "
            "back to the un-re-ranked retrieval order.",
            model_name,
            exc,
        )
        return None

    with _model_cache_lock:
        # Race-safe insert: another thread may have populated the slot
        # while we were loading; if so, drop the duplicate and use theirs.
        existing = _model_cache.get(cache_key)
        if existing is not None:
            return existing
        _model_cache[cache_key] = model

    return model


def _document_text(doc: Any) -> str:
    """Return the text used for cross-encoder scoring.

    Defensive against non-langchain ``Document``-shaped objects: anything
    with a ``page_content`` attribute is fine; anything else is coerced
    via ``str()``.  Empty content is coerced to an empty string so the
    CE call shape stays valid (a paired score is still produced).
    """
    content = getattr(doc, "page_content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def rerank(
    query: str,
    docs: Iterable[Any],
    k: int,
    *,
    model_name: str = DEFAULT_RERANK_MODEL,
    device: str | None = None,
) -> list[Any]:
    """Re-rank ``docs`` against ``query`` and return the top ``k``.

    Pure function over the input iterable: the original order is never
    mutated; a new list is returned.  Documents are scored by the cross-
    encoder model named in ``model_name`` (default
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``) and sorted by descending
    score before the top-``k`` cut.

    Args:
        query: The user's question — passed verbatim to the CE.
        docs: An iterable of LangChain ``Document`` (or any object with a
            ``page_content`` attribute).  Drained once.
        k: Maximum number of documents to return.  Clamped to ``≥ 0``;
            ``k=0`` returns an empty list immediately.
        model_name: HuggingFace cross-encoder model identifier.  The
            default is small (~80 MB) and CPU-friendly.
        device: Optional torch device string (``"cpu"`` / ``"cuda"`` /
            ``"cuda:0"`` / ``"mps"``).  When ``None``, lets
            ``sentence-transformers`` choose.

    Returns:
        A list of at most ``k`` documents in descending re-ranked
        relevance order.  When the cross-encoder is unavailable (lib
        missing, model fails to load, scoring raises) the function
        falls back to returning the first ``k`` input documents in
        their incoming order — re-ranking never makes retrieval worse
        than the baseline.
    """
    if k <= 0:
        return []

    # Materialise the input so we can both pair-score and slice.  This
    # is cheap relative to the CE forward pass.
    candidates = list(docs)
    if not candidates:
        return []

    # No-op cases that skip the model load entirely.
    if not query or not query.strip():
        return candidates[:k]
    if len(candidates) == 1:
        return candidates[:k]

    model = _try_load_cross_encoder(model_name, device)
    if model is None:
        # Graceful fallback — preserve the caller's input order.
        return candidates[:k]

    pairs = [(query, _document_text(doc)) for doc in candidates]

    try:
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001 — any scoring failure → fallback
        _logger.warning(
            "Cross-encoder re-rank failed during predict(): %s.  Falling back "
            "to the un-re-ranked retrieval order.",
            exc,
        )
        return candidates[:k]

    # ``model.predict`` returns either a numpy array or a list of floats
    # depending on the sentence-transformers version.  Coerce to a list
    # of floats up front so downstream comparisons work uniformly.
    try:
        score_list = [float(s) for s in scores]
    except (TypeError, ValueError) as exc:
        _logger.warning(
            "Cross-encoder re-rank produced non-numeric scores (%s).  Falling "
            "back to the un-re-ranked retrieval order.",
            exc,
        )
        return candidates[:k]

    if len(score_list) != len(candidates):
        # Defensive: the model returned a different number of scores
        # than we sent pairs for.  Cannot trust the order — fall back.
        _logger.warning(
            "Cross-encoder re-rank produced %d scores for %d input pairs; "
            "falling back to the un-re-ranked retrieval order.",
            len(score_list),
            len(candidates),
        )
        return candidates[:k]

    # Sort by descending score.  Stable sort preserves the FAISS / hybrid
    # tie-break order when CE scores are equal.
    ordered = sorted(
        zip(candidates, score_list, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [doc for doc, _score in ordered[:k]]


def _reset_model_cache_for_tests() -> None:
    """Clear the module-scope CE model cache.

    Test-only helper.  Production code never calls this — the cache is
    a deliberate per-process singleton.
    """
    with _model_cache_lock:
        _model_cache.clear()


__all__ = [
    "DEFAULT_RERANK_MODEL",
    "rerank",
]
