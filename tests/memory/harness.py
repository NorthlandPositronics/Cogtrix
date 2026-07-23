"""Core test loop for the memory recall curve harness (#133).

Feeds a corpus chunk-by-chunk through a memory manager and measures recall
accuracy at deterministic depth checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.messages import BaseMessage

from cogtrix_core.memory.base import BaseMemoryStore
from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager
from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager
from cogtrix_core.orchestration.graph import _apply_context_message_cap

from .judge import DEFAULT_JUDGE, Judge

MODE_FACTORIES = {
    "conversation": ConversationMemoryManager,
    "code": CodeDevelopmentMemoryManager,
    "reasoning": ReasoningMemoryManager,
}


class MockStore(BaseMemoryStore):
    """In-memory persistence stub for recall curve tests."""

    def __init__(self) -> None:
        self._history: dict[str, list[object]] = {}

    def load_history(self, session_id: str) -> list[object]:
        return list(self._history.get(session_id, []))

    def save_history(self, session_id: str, messages: list[object]) -> None:
        self._history[session_id] = list(messages)


def _make_manager(mode_name: str, session_id: str, working_memory_size: int) -> object:
    manager_cls = MODE_FACTORIES[mode_name]
    store = MockStore()
    manager = manager_cls(store, session_id, {"working_memory_size": working_memory_size})
    manager.load()
    return manager


def _joined_text(messages: Iterable[BaseMessage]) -> str:
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


def run_recall_curve(
    mode_name: str,
    corpus: list[tuple[str, str]],
    checkpoints: Iterable[int],
    cap_sizes: Iterable[int],
    judge: Judge | None = None,
) -> dict[int, list[float]]:
    """Return recall curves for each cap size at the configured checkpoints.

    Args:
        mode_name: Memory mode key (conversation / code / reasoning).
        corpus: List of (chunk_text, answer_token) tuples.
        checkpoints: Chunk indices at which to measure recall.
        cap_sizes: Context message caps to test (0 = unlimited).
        judge: Optional Judge instance. Defaults to exact-match.

    Returns:
        Dict mapping cap_size -> list of accuracy scores at each checkpoint.
    """
    if mode_name not in MODE_FACTORIES:
        raise ValueError(f"Unknown mode: {mode_name}")

    judge = judge or DEFAULT_JUDGE
    corpus_size = len(corpus)
    working_memory_size = corpus_size * 2 + 20

    manager = _make_manager(
        mode_name,
        f"{mode_name}-recall-curve-{corpus_size}",
        working_memory_size=working_memory_size,
    )

    checkpoint_set = set(checkpoints)
    checkpoint_set.add(corpus_size)

    curves: dict[int, list[float]] = {cap: [] for cap in cap_sizes}
    seen_qa: list[dict[str, Any]] = []

    for index, (chunk_text, answer_token) in enumerate(corpus, start=1):
        manager.update(chunk_text, answer_token)
        seen_qa.append(
            {
                "question": f"What is the reference token for chunk {index - 1}?",
                "ground_truth": answer_token,
            }
        )

        if index not in checkpoint_set:
            continue

        context = manager.prepare_context(f"recall-check-{index}")
        base_messages = list(context.messages)
        for cap in cap_sizes:
            capped_messages = _apply_context_message_cap(base_messages, cap)
            text = _joined_text(capped_messages)
            if not seen_qa:
                accuracy = 1.0
            else:
                hits = sum(
                    1 for qa in seen_qa if judge.evaluate(qa["question"], text, qa["ground_truth"])
                )
                accuracy = hits / len(seen_qa)
            curves[cap].append(accuracy)

    return curves
