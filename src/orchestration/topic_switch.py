"""Topic-switch heuristic for the orchestration graph.

Extracted from ``src/orchestration/graph.py`` as part of the /forge A1.4
extraction (2026-05-23). Pure heuristic, no graph-build /
langgraph-runtime dependency.

Detects when the latest user message switches topics relative to the
recent message window, so the orchestrator can drop the prior summary
and avoid contaminating the new turn with stale context.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TOPIC_SWITCH_MESSAGE_WINDOW = 8
_TOPIC_SWITCH_MAX_WORDS = 15
_TOPIC_SWITCH_MIN_SIMILARITY = 0.40
_TOPIC_SWITCH_NUDGE = (
    "The user has changed topic. Answer the new question directly without reference "
    "to the prior task."
)
_TOPIC_SWITCH_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "what's",
    "whats",
    "with",
    "you",
    "your",
}


def _topic_switch_tokens(text: str) -> list[str]:
    """Return normalized content tokens used by the topic-switch heuristic."""
    normalized = text.lower().replace("'s", "")
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2 and token not in _TOPIC_SWITCH_STOPWORDS
    ]


def _should_reset_summary_for_topic_switch(messages: list[Any]) -> bool:
    """Return True when the latest user message appears to switch topics."""
    if not messages:
        return False

    last_human_idx = -1
    last_human_text = ""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if getattr(msg, "type", None) == "human":
            last_human_idx = idx
            content = getattr(msg, "content", "")
            last_human_text = content if isinstance(content, str) else ""
            break

    if last_human_idx <= 0 or not last_human_text:
        return False

    if len(last_human_text.split()) >= _TOPIC_SWITCH_MAX_WORDS:
        return False

    current_tokens = _topic_switch_tokens(last_human_text)
    if not current_tokens:
        return False

    reference_messages = messages[
        max(0, last_human_idx - _TOPIC_SWITCH_MESSAGE_WINDOW) : last_human_idx
    ]
    reference_tokens: list[str] = []
    for msg in reference_messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            reference_tokens.extend(_topic_switch_tokens(content))

    if not reference_tokens:
        return False

    current_counts = Counter(current_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(count, reference_counts[token]) for token, count in current_counts.items())
    similarity = overlap / max(len(current_tokens), len(reference_tokens))

    has_question = "?" in last_human_text

    if has_question:
        return overlap <= 1 and similarity < _TOPIC_SWITCH_MIN_SIMILARITY

    # Imperative commands ("check slack", "look at github") are valid topic
    # switches even without a question mark.  Use a stricter threshold —
    # zero overlap and at least 2 meaningful tokens — to reduce false positives
    # on short continuations like "okay proceed".
    return overlap == 0 and len(current_tokens) >= 2


__all__ = [
    "_TOPIC_SWITCH_MAX_WORDS",
    "_TOPIC_SWITCH_MESSAGE_WINDOW",
    "_TOPIC_SWITCH_MIN_SIMILARITY",
    "_TOPIC_SWITCH_NUDGE",
    "_TOPIC_SWITCH_STOPWORDS",
    "_should_reset_summary_for_topic_switch",
    "_topic_switch_tokens",
]
