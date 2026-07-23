"""LLM-based incremental conversation summarization.

Provides a standalone ``generate_summary()`` function that uses a
chat-completion LLM to compress older conversation messages into a
concise rolling summary.  The summary is *incremental* — on each
call it merges newly evicted messages into the existing summary.

The function is intentionally stateless; all state (existing summary,
messages to summarize) is passed in and the result is returned.
"""

import concurrent.futures
import logging
from typing import Any

log = logging.getLogger("cogtrix")

_SUMMARIZE_TIMEOUT_SECONDS = 60

_SUMMARIZE_SYSTEM = (
    "You are a concise conversation summarizer.  Your output is "
    "injected into a chat agent's context window so it retains "
    "long-term awareness of earlier exchanges.\n\n"
    "Rules:\n"
    "• Preserve key facts, data, decisions, user preferences, and "
    "action items.\n"
    "• Drop small-talk, greetings, and verbose tool-call details.\n"
    "• Write in third person present tense ('The user asked …').\n"
    "• Keep the summary under 400 words.\n"
    "• Use bullet points for clarity."
)


def _format_messages_text(messages: list[Any]) -> str:
    """Convert a list of LangChain / dict messages to readable text."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("type", "unknown")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = type(msg).__name__.replace("Message", "").lower()
            content = msg.content or ""
        else:
            continue

        # Skip tool-call intermediate steps (low information density)
        if role in ("tool", "toolmessage"):
            continue

        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        content = str(content).strip()
        if not content:
            continue

        # Truncate very long individual messages to keep prompt small
        if len(content) > 2000:
            content = content[:1000] + " [...] " + content[-500:]

        label = "User" if role in ("human", "humanmessage") else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def generate_summary(
    llm: Any,
    messages: list[Any],
    existing_summary: str | None = None,
) -> str | None:
    """Generate or update a conversation summary.

    Parameters
    ----------
    llm:
        A LangChain-compatible chat model (must support ``.invoke()``).
    messages:
        New messages to incorporate into the summary.
    existing_summary:
        The current rolling summary (``None`` on first call).

    Returns
    -------
    str | None
        The updated summary text, or ``None`` if summarization failed.
    """
    if not messages:
        return existing_summary

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        log.debug("LangChain not available — skipping summarization")
        return existing_summary

    messages_text = _format_messages_text(messages)
    if not messages_text.strip():
        return existing_summary

    user_prompt_parts: list[str] = []
    if existing_summary:
        user_prompt_parts.append(
            "Here is the existing summary of earlier conversation:\n"
            f"---\n{existing_summary}\n---\n\n"
            "Incorporate the following new exchanges into the summary, "
            "merging overlapping topics and keeping the result concise:"
        )
    else:
        user_prompt_parts.append("Summarize the following conversation:")

    user_prompt_parts.append(f"\n\n{messages_text}")

    prompt = [
        SystemMessage(content=_SUMMARIZE_SYSTEM),
        HumanMessage(content="\n".join(user_prompt_parts)),
    ]

    try:
        # Use an explicit pool (not `with`) so that shutdown(wait=False) in the
        # timeout branch is not overridden by the context-manager __exit__, which
        # always calls shutdown(wait=True) and would block on a hung LLM thread.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(llm.invoke, prompt)
            try:
                response = future.result(timeout=_SUMMARIZE_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                log.warning(
                    "generate_summary: LLM call timed out after %ds — returning existing summary",
                    _SUMMARIZE_TIMEOUT_SECONDS,
                )
                return existing_summary
        finally:
            pool.shutdown(wait=False)
        summary = (response.content or "").strip()
        if not summary:
            log.warning("Summarizer returned empty response")
            return existing_summary
        return summary
    except Exception as exc:
        log.warning("Summarization failed (non-fatal): %s", exc)
        return existing_summary
