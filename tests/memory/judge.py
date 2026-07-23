"""LLM judge stub for memory recall test harness.

Issue #133 calls for a pinned LLM judge (e.g. gpt-4o) that compares a model
answer to ground truth and emits PASS/FAIL.  In the current test suite we use
exact-match scoring for determinism and speed, but this module provides the
judge interface so the harness is ready for an LLM-backed judge when CI
secrets and API budget allow.
"""

from __future__ import annotations

from typing import Protocol


class Judge(Protocol):
    """Protocol for a recall judge."""

    def evaluate(self, question: str, model_answer: str, ground_truth: str) -> bool:
        """Return True if the model answer passes, False otherwise."""
        ...


class ExactMatchJudge:
    """Deterministic judge: PASS if ground_truth appears in model_answer."""

    def evaluate(self, question: str, model_answer: str, ground_truth: str) -> bool:
        return ground_truth in model_answer


class NormalizedJudge:
    """Case-insensitive, whitespace-normalized exact match."""

    def evaluate(self, question: str, model_answer: str, ground_truth: str) -> bool:
        return ground_truth.lower() in model_answer.lower().strip()


# Default judge used by the harness.
DEFAULT_JUDGE: Judge = ExactMatchJudge()
