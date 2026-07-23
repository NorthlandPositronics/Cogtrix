"""#2485 — query_knowledge_base appends a pre-generation grounding footer.

Capable models (deepseek, kimi) plateau at ~46% on PM RAG tasks because they
mis-attribute owners to entities — the entity-owner-mismatch failure mode
(#1987), the single largest guardrail-firing category. The fix steers the model
*before* it composes: append an abstract grounding rule to non-empty retrieval
results (bind each attributed owner/fact to a single chunk, cite it). These
tests pin that the footer:

  * appears only when at least one chunk was surfaced (never on the
    no-results / no-KB / error paths, which would dilute an honest "nothing
    found" reply);
  * lands AFTER the evidence, so the chunk listing is unchanged;
  * does not pollute the #2218 source-digest parsing (no phantom source line);
  * stays corpus-agnostic (#2006 bias-leakage rule — no entity-ids/names/topics).
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import cogtrix_core.tools.rag as rag
from cogtrix_core.tools.rag import _RAG_GROUNDING_FOOTER, query_knowledge_base


def _fake_pairs() -> list[tuple[Document, float]]:
    """Two chunks whose owner lives in chunk [1] and whose look-alike
    stakeholder lives in chunk [2] — the exact shape that tempts a swap."""
    return [
        (
            Document(
                page_content="Risk R-12 | Owner: Bob Smith | status: on track",
                metadata={"source": "risk_register.md"},
            ),
            0.10,
        ),
        (
            Document(
                page_content="Alice Jones leads the Migration Squad.",
                metadata={"source": "team_directory.md", "page": 3},
            ),
            0.20,
        ),
    ]


def _run_query(
    question: str,
    retrieve_return: list[tuple[Document, float]],
    *,
    dirs: list[Path] | None = None,
    k: int = 4,
) -> str:
    """Drive query_knowledge_base down the format path with the retrieval
    layer mocked, returning the raw tool output string."""
    patches = [
        patch.object(rag, "FAISS_AVAILABLE", True),
        patch.object(
            rag,
            "_collect_faiss_dirs",
            return_value=dirs if dirs is not None else [Path("/fake/idx")],
        ),
        patch.object(rag, "_get_embeddings", return_value=MagicMock()),
        patch.object(rag, "load_faiss_store_safe", return_value=MagicMock()),
        patch.object(rag, "_retrieve_from_index", return_value=retrieve_return),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return query_knowledge_base(question, k=k)


class TestGroundingFooter:
    def test_footer_appended_after_chunks_when_results(self) -> None:
        out = _run_query("who owns R-12", _fake_pairs())

        assert "Found 2 relevant document(s):" in out
        assert "[1] Source: risk_register.md" in out
        assert "[2] Source: team_directory.md (page 3)" in out
        assert _RAG_GROUNDING_FOOTER in out
        # The nudge must trail the evidence, never precede or split it.
        assert out.index("[1] Source:") < out.index(_RAG_GROUNDING_FOOTER)
        assert out.index("[2] Source:") < out.index(_RAG_GROUNDING_FOOTER)

    def test_no_footer_when_zero_results(self) -> None:
        out = _run_query("nothing matches this", [])

        assert "No relevant documents found" in out
        assert _RAG_GROUNDING_FOOTER not in out

    def test_no_footer_when_no_knowledge_base(self) -> None:
        with (
            patch.object(rag, "FAISS_AVAILABLE", True),
            patch.object(rag, "_collect_faiss_dirs", return_value=[]),
        ):
            out = query_knowledge_base("q")

        assert "No knowledge base found" in out
        assert _RAG_GROUNDING_FOOTER not in out

    def test_footer_does_not_add_phantom_source(self) -> None:
        # The #2218 report digest parses source filenames with _KB_SOURCE_RE.
        # The footer must not introduce a line that regex mistakes for a source.
        from tests.role_pm.run import _KB_SOURCE_RE

        out = _run_query("owners", _fake_pairs())

        assert _KB_SOURCE_RE.findall(out) == ["risk_register.md", "team_directory.md (page 3)"]

    def test_footer_is_corpus_agnostic(self) -> None:
        # #2006 cycle-10 bias-leakage rule: no corpus-specific entity ids,
        # stakeholder names, or topics baked into the production string.
        for banned in ("R-12", "Bob", "Alice", "Migration", "risk_register", "Nimbus"):
            assert banned not in _RAG_GROUNDING_FOOTER
        # It must actually carry the grounding instruction.
        low = _RAG_GROUNDING_FOOTER.lower()
        assert "single result" in low
        assert "cite" in low
