"""Tests for the PM role-test ``_criterion_matches`` operator set.

#2024: cycles 9, 16 (gpt-oss-20b) and 18 (gemma-4-26b) demonstrated
that hard ``contains:`` checks penalise honestly-terse models — they
hit all #1987 detector targets but produced 0–1/18 clean iterations
because they decline to enumerate tokens they have no grounded
evidence for.  The harness now also supports
``at_least_n_contains: N | opt1 | opt2 | ...`` for weighted
comprehensiveness checks that allow conservative honest models to
pass.
"""

from __future__ import annotations

from tests.role_pm.run import _criterion_matches


class TestContainsOperator:
    def test_contains_pass(self) -> None:
        ok, _ = _criterion_matches("contains: foo", "the foo is here", None)
        assert ok

    def test_contains_fail(self) -> None:
        ok, _ = _criterion_matches("contains: foo", "nothing relevant", None)
        assert not ok


class TestNotContainsOperator:
    def test_not_contains_pass(self) -> None:
        ok, _ = _criterion_matches("not_contains: foo", "nothing relevant", None)
        assert ok

    def test_not_contains_fail(self) -> None:
        ok, _ = _criterion_matches("not_contains: foo", "the foo is here", None)
        assert not ok


class TestAnyContainsOperator:
    def test_any_contains_pass_on_any_one(self) -> None:
        ok, _ = _criterion_matches("any_contains: foo | bar | baz", "we found bar", None)
        assert ok

    def test_any_contains_fail_when_none(self) -> None:
        ok, _ = _criterion_matches("any_contains: foo | bar | baz", "we found qux", None)
        assert not ok


class TestAtLeastNContainsOperator:
    """#2024 — weighted comprehensiveness criterion."""

    def test_passes_when_threshold_met_exactly(self) -> None:
        ok, desc = _criterion_matches(
            "at_least_n_contains: 2 | SC-1 | SC-2 | SC-3 | SC-4 | SC-5",
            "We surface SC-1 and SC-2 explicitly.",
            None,
        )
        assert ok
        assert "matched 2" in desc

    def test_passes_when_threshold_exceeded(self) -> None:
        ok, _ = _criterion_matches(
            "at_least_n_contains: 2 | SC-1 | SC-2 | SC-3",
            "All three: SC-1, SC-2, SC-3.",
            None,
        )
        assert ok

    def test_fails_when_below_threshold(self) -> None:
        ok, desc = _criterion_matches(
            "at_least_n_contains: 3 | SC-1 | SC-2 | SC-3 | SC-4 | SC-5",
            "Only SC-1 mentioned.",
            None,
        )
        assert not ok
        assert "matched 1" in desc

    def test_passes_with_min_threshold_one(self) -> None:
        # The equivalent of any_contains: but with the at_least_n form
        # — useful when migrating from contains: to a softer check.
        ok, _ = _criterion_matches(
            "at_least_n_contains: 1 | R-12 | R-19",
            "Discussed R-12 in detail.",
            None,
        )
        assert ok

    def test_fails_with_min_threshold_one_when_none_match(self) -> None:
        ok, _ = _criterion_matches(
            "at_least_n_contains: 1 | R-12 | R-19",
            "Nothing specific cited.",
            None,
        )
        assert not ok

    def test_malformed_missing_threshold_does_not_break_run(self) -> None:
        # No integer at position 0 — treat as a rubric pass rather
        # than silently failing a misconfigured criterion.
        ok, desc = _criterion_matches(
            "at_least_n_contains: foo | bar",
            "anything",
            None,
        )
        assert ok
        assert "integer threshold" in desc

    def test_malformed_empty_options_does_not_break_run(self) -> None:
        ok, desc = _criterion_matches(
            "at_least_n_contains: 2",
            "anything",
            None,
        )
        assert ok
        assert "malformed" in desc


class TestToolCalledOperator:
    def test_tool_called_pass(self) -> None:
        ok, _ = _criterion_matches(
            "tool_called: query_knowledge_base",
            "(response text)",
            ["query_knowledge_base", "checkpoint"],
        )
        assert ok

    def test_tool_called_fail(self) -> None:
        ok, _ = _criterion_matches(
            "tool_called: web_search",
            "(response text)",
            ["query_knowledge_base"],
        )
        assert not ok
