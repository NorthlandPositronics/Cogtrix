"""Shared FAISS L2-distance → similarity conversion (#2072).

FAISS ``similarity_search_with_score`` returns an **L2 distance** (lower = more
similar). The agent search (``query_knowledge_base``) and the API search
(``/api/v1/rag/search``) must map distance → a 0–1 similarity the SAME way, so
their ``score`` fields and ``score_threshold`` semantics are comparable. The
agent's ``score_threshold`` gate already assumes this transform, so it is the
canonical one.
"""

from __future__ import annotations


def l2_to_similarity(distance: float) -> float:
    """Map a FAISS L2 distance to a 0–1 similarity score (1.0 = identical)."""
    return 1.0 / (1.0 + max(0.0, float(distance)))
