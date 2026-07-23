"""Distill durable facts from a rolling memory summary."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from cogtrix_core.concurrency import invoke_with_timeout

log = logging.getLogger("cogtrix")

_DISTILL_TIMEOUT_SECONDS = 60

_DISTILL_SYSTEM = (
    "You extract durable institutional memory from a session summary.\n\n"
    "Rules:\n"
    "- Preserve only facts that should remain true and relevant a week from now.\n"
    "- Include confirmed decisions, durable blockers, and verified entity states.\n"
    "- Exclude transient reasoning, already-resolved issues, and speculation.\n"
    "- Output a plain bulleted list with at most 15 items.\n"
    "- Keep each bullet to at most 20 words.\n"
)


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(_coerce_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return _coerce_text(content["text"])
        return json.dumps(content, sort_keys=True)
    return str(content)


def _parse_facts(text: str) -> list[str]:
    facts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*•]|\d+[).])\s*", "", line)
        if not line:
            continue
        words = line.split()
        if len(words) > 20:
            line = " ".join(words[:20])
        facts.append(line)
        if len(facts) >= 15:
            break
    return facts


def distill_summary(llm: Any | None, summary: str) -> list[str]:
    """Return durable facts extracted from *summary* using *llm*.

    If *llm* is ``None``, returns an empty list immediately — consistent
    with the defensive fallback used in :func:`compress_to_tier`.
    """
    if llm is None:
        return []

    if not summary.strip():
        return []

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        log.debug("LangChain not available — skipping fact distillation")
        return []

    prompt = [
        SystemMessage(content=_DISTILL_SYSTEM),
        HumanMessage(
            content=(
                "From the summary below, extract only durable facts. "
                "Return a plain bulleted list.\n\n"
                f"Summary:\n{summary}"
            )
        ),
    ]

    try:
        # Bounded-timeout LLM invocation via the centralized helper —
        # migrated under #1903; see docs/architecture/CONCURRENCY.md for
        # the policy.
        try:
            response = invoke_with_timeout(llm.invoke, prompt, timeout=_DISTILL_TIMEOUT_SECONDS)
        except TimeoutError:
            log.warning(
                "distill_summary: LLM call timed out after %ds — returning no facts",
                _DISTILL_TIMEOUT_SECONDS,
            )
            return []
        facts = _parse_facts(_coerce_text(getattr(response, "content", response)))
        return facts
    except Exception as exc:
        log.warning("Fact distillation failed (non-fatal): %s", exc)
        return []
