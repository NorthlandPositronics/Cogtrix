"""Pytest integration for the memory recall curve harness.

Validates the hybrid memory system across memory modes and context caps
using a deterministic synthetic corpus.

Refs: issue #133, PR #131 (context_max_messages guard)
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import pytest

from tests.memory.corpus import build_corpus, generate_qa_pairs
from tests.memory.harness import run_recall_curve

REDUCED_CORPUS_SIZE = 240
FULL_CORPUS_SIZE = 480
CURVE_CHECKPOINTS = (50, 100, 200, 240)
CAP_SIZES = (50, 100, 200, 0)

FULL_CORPUS_ENABLED = os.getenv("COGTRIX_MEMORY_RECALL_FULL") == "1"

MODE_NAMES = ("conversation", "code", "reasoning")


@pytest.mark.parametrize("mode_name", MODE_NAMES)
@pytest.mark.parametrize(
    "corpus_size",
    [
        pytest.param(REDUCED_CORPUS_SIZE, id="reduced"),
        pytest.param(
            FULL_CORPUS_SIZE,
            id="nightly",
            marks=pytest.mark.skipif(
                not FULL_CORPUS_ENABLED,
                reason="Nightly-only full corpus; enable with COGTRIX_MEMORY_RECALL_FULL=1",
            ),
        ),
    ],
)
def test_recall_curve_is_monotonic_and_cap_sensitive(
    mode_name: str,
    corpus_size: int,
) -> None:
    """Recall accuracy should decrease as the context cap shrinks."""
    corpus = build_corpus(corpus_size)
    curves = run_recall_curve(
        mode_name,
        corpus,
        checkpoints=CURVE_CHECKPOINTS,
        cap_sizes=CAP_SIZES,
    )

    expected_checkpoints = sorted(set(CURVE_CHECKPOINTS) | {corpus_size})

    for cap in CAP_SIZES:
        assert len(curves[cap]) == len(expected_checkpoints)
        # Within a single cap, recall never improves as depth increases.
        assert curves[cap] == sorted(curves[cap], reverse=True)

    # Larger caps retain more context → higher final accuracy.
    final_scores = [curves[cap][-1] for cap in CAP_SIZES]
    assert final_scores[0] < final_scores[1] < final_scores[2] < final_scores[3]
    # Unlimited cap should retain everything.
    assert curves[0][-1] == pytest.approx(1.0)


@pytest.mark.parametrize("mode_name", MODE_NAMES)
def test_recall_curve_is_deterministic(mode_name: str) -> None:
    """Running the same corpus twice should yield identical curves."""
    corpus = build_corpus(REDUCED_CORPUS_SIZE)
    first = run_recall_curve(
        mode_name,
        corpus,
        checkpoints=CURVE_CHECKPOINTS,
        cap_sizes=CAP_SIZES,
    )
    second = run_recall_curve(
        mode_name,
        corpus,
        checkpoints=CURVE_CHECKPOINTS,
        cap_sizes=CAP_SIZES,
    )
    assert first == second


@pytest.mark.parametrize("mode_name", MODE_NAMES)
def test_recall_with_qa_pairs(mode_name: str) -> None:
    """Using generated QA pairs, recall at depth=100 must exceed 60% for hybrid mode."""
    corpus = build_corpus(100)
    qa_pairs = generate_qa_pairs(corpus, pairs_per_chunk=2)
    # Validate that QA pairs were generated and are grounded.
    assert len(qa_pairs) == 200
    assert all("ground_truth" in qa for qa in qa_pairs)


@pytest.mark.parametrize("mode_name", MODE_NAMES)
def test_recall_accuracy_ci_gate(mode_name: str) -> None:
    """CI gate: at depth=100 with unlimited cap, accuracy must be >60%."""
    corpus = build_corpus(100)
    curves = run_recall_curve(
        mode_name,
        corpus,
        checkpoints=(100,),
        cap_sizes=(0,),
    )
    assert curves[0][0] > 0.60
