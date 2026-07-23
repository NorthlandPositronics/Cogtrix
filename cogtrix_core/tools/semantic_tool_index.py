"""In-memory semantic index over tool descriptions for retrieval-augmented tool loading.

Provides embedding-based semantic search over the on-demand tool catalog so
agents can discover the right tools without browsing the full list.

Usage::

    from cogtrix_core.tools.semantic_tool_index import ToolIndex, configure

    # Once at startup (called automatically by TOOL_SETUP)
    configure(provider="ollama", model="nomic-embed-text")

    # Per request_tools call
    index = ToolIndex(catalog)          # lazy — no network calls yet
    matches = index.search("send email", k=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cogtrix_core.config import Config

from cogtrix_core.providers import create_embeddings

log = logging.getLogger("cogtrix.tools.semantic_tool_index")

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ── Module-level embedding configuration ──────────────────────────────────────

_config: dict[str, str | None] = {
    "provider": "ollama",
    "model": None,
    "base_url": None,
    "api_key": None,
}


def configure(
    provider: str = "ollama",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Set the embedding configuration used by all ToolIndex instances."""
    _config["provider"] = provider
    _config["model"] = model
    _config["base_url"] = base_url
    _config["api_key"] = api_key


def TOOL_SETUP(config: Config) -> None:
    """Read embedding config from Config and store for ToolIndex instances."""
    emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
    configure(
        provider=emb_type,
        model=emb_model,
        base_url=emb_base_url,
        api_key=emb_api_key,
    )


# ── ToolIndex ─────────────────────────────────────────────────────────────────


class ToolIndex:
    """In-memory semantic index over a tool catalog.

    Embeddings are computed lazily on the first ``search()`` call so
    startup is never slowed by network calls.

    Args:
        catalog: ``{tool_name: short_description}`` mapping.
    """

    def __init__(self, catalog: dict[str, str]) -> None:
        self._catalog = dict(catalog)
        self._names: list[str] = list(catalog.keys())
        self._descs: list[str] = list(catalog.values())
        self._matrix: Any = None  # np.ndarray once built
        self._embeddings: Any = None  # LangChain Embeddings instance
        self._ready = False

    # ── Public API ─────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if the embedding index has been built."""
        return self._ready

    def search(self, query: str, k: int = 5) -> list[str]:
        """Return up to k tool names most semantically similar to *query*.

        Falls back to keyword substring match when embeddings are unavailable.
        """
        if not self._catalog:
            return []

        if not self._ready:
            self._try_build_index()

        if self._ready:
            return self._semantic_search(query, k)
        return self._keyword_search(query, k)

    # ── Index construction ─────────────────────────────────────────

    def _try_build_index(self) -> None:
        """Attempt to build the embedding index; silently activates fallback on failure."""
        if not _HAS_NUMPY:
            log.warning(
                "numpy is not installed; semantic tool search falling back to keyword match"
            )
            return

        try:
            provider = (_config["provider"] or "ollama").lower()
            model = _config["model"]
            base_url = _config["base_url"]
            api_key = _config["api_key"]

            emb = create_embeddings(provider, model=model, base_url=base_url, api_key=api_key)
            self._embeddings = emb
            self._build_matrix()
        except Exception as exc:
            log.warning("Semantic tool index build failed (%s); using keyword fallback", exc)

    def _build_matrix(self) -> None:
        """Embed all descriptions and store a normalised matrix."""
        assert _np is not None
        vecs = self._embeddings.embed_documents(self._descs)
        arr = _np.array(vecs, dtype=_np.float32)  # (N, D)
        # L2-normalise each row so cosine similarity = dot product
        norms = _np.linalg.norm(arr, axis=1, keepdims=True)
        norms = _np.where(norms == 0, 1.0, norms)
        self._matrix = arr / norms
        self._ready = True

    # ── Search back-ends ───────────────────────────────────────────

    def _semantic_search(self, query: str, k: int) -> list[str]:
        assert _np is not None
        try:
            q_vec = self._embeddings.embed_query(query)
            q_arr = _np.array(q_vec, dtype=_np.float32)
            norm = float(_np.linalg.norm(q_arr))
            if norm > 0:
                q_arr = q_arr / norm
            scores = self._matrix @ q_arr  # (N,)
            top_k = min(k, len(self._names))
            indices = _np.argsort(scores)[::-1][:top_k]
            return [self._names[int(i)] for i in indices]
        except Exception as exc:
            log.warning("Semantic search failed (%s); falling back to keyword match", exc)
            return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> list[str]:
        """Substring keyword match across tool names and descriptions."""
        words = [w.lower() for w in query.split() if w]
        if not words:
            return self._names[:k]

        matches: list[str] = []
        for name, desc in zip(self._names, self._descs, strict=False):
            haystack = (name + " " + desc).lower()
            if any(w in haystack for w in words):
                matches.append(name)
                if len(matches) >= k:
                    break
        return matches
