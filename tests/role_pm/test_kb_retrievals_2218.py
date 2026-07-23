"""#2218 — the PM report must capture query_knowledge_base RESULTS, not just names.

A content-criterion miss (e.g. ``contains '51 minutes'``) is only classifiable as
MODEL (retrieval surfaced the chunk, model omitted it) vs CODE/CONFIG (retrieval
failed to surface it) if the report records what retrieval returned. These tests
cover ``_extract_kb_retrievals`` — the per-turn digest of each query_knowledge_base
call's query, k, returned source filenames, and a capped snippet.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.role_pm.run import _KB_DIGEST_SNIPPET_CHARS, _extract_kb_retrievals


def _kb_call(cid: str, question: str, k: int = 4) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "query_knowledge_base", "args": {"question": question, "k": k}, "id": cid}
        ],
    )


_SAMPLE_RESULT = (
    "Found 2 relevant document(s):\n\n"
    "[1] Source: 13_status_report_m4.md\n"
    "    Rehearsal 1 ran 51 minutes vs the 38-minute budget.\n\n"
    "[2] Source: 07_budget.md (page 3)\n"
    "    Total actuals came to $1,106,500.\n"
)


class TestExtractKbRetrievals:
    def test_pairs_call_with_result_and_parses_sources(self) -> None:
        msgs = [
            HumanMessage(content="status?"),
            _kb_call("c1", "rehearsal duration vs budget", k=8),
            ToolMessage(content=_SAMPLE_RESULT, tool_call_id="c1", name="query_knowledge_base"),
        ]
        got = _extract_kb_retrievals(msgs)

        assert len(got) == 1
        r = got[0]
        assert r["question"] == "rehearsal duration vs budget"
        assert r["k"] == 8
        assert r["sources"] == ["13_status_report_m4.md", "07_budget.md (page 3)"]
        assert "51 minutes" in r["result_snippet"]

    def test_snippet_is_capped_and_marked_truncated(self) -> None:
        big = "Found 1 relevant document(s):\n\n[1] Source: big.md\n    " + ("x" * 5000)
        msgs = [
            _kb_call("c1", "q"),
            ToolMessage(content=big, tool_call_id="c1", name="query_knowledge_base"),
        ]
        r = _extract_kb_retrievals(msgs)[0]
        assert len(r["result_snippet"]) <= _KB_DIGEST_SNIPPET_CHARS + len(" …[truncated]")
        assert r["result_snippet"].endswith("…[truncated]")

    def test_unanswered_call_records_empty_digest(self) -> None:
        # Retrieval call with no matching ToolMessage (e.g. cut mid-flight): the
        # digest is still recorded so the miss is visible, just with no sources.
        msgs = [_kb_call("c1", "orphaned query")]
        r = _extract_kb_retrievals(msgs)[0]
        assert r["question"] == "orphaned query"
        assert r["sources"] == []
        assert r["result_snippet"] == ""

    def test_non_kb_tool_calls_are_ignored(self) -> None:
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "checkpoint", "args": {}, "id": "x1"}]),
            ToolMessage(content="noted", tool_call_id="x1", name="checkpoint"),
        ]
        assert _extract_kb_retrievals(msgs) == []

    def test_multiple_calls_each_digested_in_order(self) -> None:
        msgs = [
            _kb_call("c1", "first"),
            ToolMessage(
                content="Found 1 relevant document(s):\n\n[1] Source: a.md\n    aaa",
                tool_call_id="c1",
                name="query_knowledge_base",
            ),
            _kb_call("c2", "second"),
            ToolMessage(
                content="Found 1 relevant document(s):\n\n[1] Source: b.md\n    bbb",
                tool_call_id="c2",
                name="query_knowledge_base",
            ),
        ]
        got = _extract_kb_retrievals(msgs)
        assert [r["question"] for r in got] == ["first", "second"]
        assert [r["sources"] for r in got] == [["a.md"], ["b.md"]]
