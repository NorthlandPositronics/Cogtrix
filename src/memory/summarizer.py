"""LLM-based incremental conversation summarization.

Provides a standalone ``generate_summary()`` function that uses a
chat-completion LLM to compress older conversation messages into a
concise rolling summary.  The summary is *incremental* — on each
call it merges newly evicted messages into the existing summary.

The function is intentionally stateless; all state (existing summary,
messages to summarize) is passed in and the result is returned.

Since ADR-0056 PR-D, the function also serves stage 5 of the
``web_search`` pipeline via the ``purpose="web_search_synthesis"``
keyword. See ``docs/optional/prompts/web-search-synthesis.md`` for the
synthesis prompt design + the 11 hard constraints it enforces.
"""

import logging
from typing import Any

from src.concurrency import invoke_with_timeout

log = logging.getLogger("cogtrix")

# Per-purpose default timeouts. The conversation constant keeps its
# historical name so test_hybrid.py's hung-LLM regression guard
# (issue #1340) can still monkey-patch it. The web_search_synthesis
# constant matches ADR-0056's stage-5 7s deadline; the synthesiser's
# retry path overrides to 5s for the smaller-model fallback.
_SUMMARIZE_TIMEOUT_SECONDS = 60
_WEB_SEARCH_SYNTHESIS_TIMEOUT_SECONDS = 7

_SUPPORTED_PURPOSES: tuple[str, ...] = ("conversation", "web_search_synthesis")


def _resolve_default_timeout(purpose: str) -> int | float:
    """Look up the default timeout for *purpose* with a *live* read so
    that ``monkeypatch.setattr(mod, "_SUMMARIZE_TIMEOUT_SECONDS", X)``
    affects subsequent calls.
    """
    if purpose == "conversation":
        return _SUMMARIZE_TIMEOUT_SECONDS
    return _WEB_SEARCH_SYNTHESIS_TIMEOUT_SECONDS


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

# System prompt for ``purpose="web_search_synthesis"``. Verbatim 11-rule
# text from ``docs/optional/prompts/web-search-synthesis.md`` — changing this
# constant requires updating that document and the regression tests
# in ``tests/tools/test_web_search_synthesis_prompt.py``.
_WEB_SEARCH_SYNTHESIS_SYSTEM = (
    "You are a research synthesiser. Multiple web sources have been "
    "retrieved for a user query. Your job is to produce a single "
    "coherent synthesis the user can read in 30 seconds.\n"
    "\n"
    "You will receive:\n"
    "  - The original user query.\n"
    "  - Numbered extracts from 1 to N. Each extract is preceded by a "
    "header containing its citation index (①②③… in "
    "display order), the registered domain, the domain-class, and a "
    "recency tag.\n"
    "\n"
    "Rules — these are absolute:\n"
    "\n"
    '1. CITATIONS ARE MANDATORY. Every factual statement in "Key '
    'findings" must be followed by one or more citation indices in '
    "square brackets, like [①] or [②③]. A statement "
    "without a citation is a bug. If you cannot cite a claim, do not "
    "write it.\n"
    "\n"
    "2. NO URLS in your output. Not in Key findings, not in "
    "Disagreements, not in Gaps. Citation indices replace URLs "
    "entirely. URLs are emitted separately by the formatting layer. "
    'Lines containing "http://", "https://", or "www." will be '
    "dropped by post-processing.\n"
    "\n"
    "3. NO CLAIMS BEYOND THE EXTRACTS. You have only the supplied "
    "extracts. You do not have prior knowledge of the topic. If the "
    "extracts do not say something, neither do you. When the extracts "
    'are silent on an aspect of the query, list it under "Gaps".\n'
    "\n"
    "4. SYNTHESISE, DO NOT LIST. Group facts by sub-topic of the "
    "query, not by source. Cross-reference: when multiple sources "
    "agree on a fact, cite all of them on one statement. A "
    '"disagreement" means sources state directly contradictory facts '
    "(e.g., one says version 1.5 was released in March, another says "
    "April). Different facets of the same topic are NOT disagreements; "
    "just present them as separate statements under Key findings. "
    "When sources do disagree in the strict sense, state both "
    'positions under "Disagreements" with their citations — do not '
    "pick a winner.\n"
    "\n"
    "5. OUTPUT SCHEMA IS FIXED, with section-omission allowed per "
    "Rule 8. The order, when sections are present, is always:\n"
    "\n"
    "    ## Key findings\n"
    "    ### <Sub-topic name>\n"
    "    <Statement.> [citation]\n"
    "    <Statement.> [citation]\n"
    "\n"
    "    ### <Next sub-topic>\n"
    "    <Statement.> [citation]\n"
    "\n"
    "    ## Disagreements\n"
    "    - <Issue>. <Position A> [citation]; <Position B> [citation].\n"
    "\n"
    "    ## Gaps\n"
    "    - <Aspect of query the extracts do not cover.>\n"
    "\n"
    'Omit "Disagreements" and "Gaps" sections entirely if and only '
    'if they are genuinely empty. Do not write "None" or "N/A". '
    "Rule 8 describes the special case where Key findings itself is "
    "omitted.\n"
    "\n"
    "6. LENGTH CAP. Keep total output under 600 words. Synthesis is "
    "a handoff to a downstream reader, not an essay.\n"
    "\n"
    '7. TONE. Neutral, declarative. No hedging phrases like "it '
    'seems" or "appears to" unless the extract itself hedges and '
    'the hedge is load-bearing for the user. No filler ("It is worth '
    'noting that…"). No headings beyond the schema above.\n'
    "\n"
    "8. WHEN THE QUERY CANNOT BE ANSWERED FROM EXTRACTS. If the "
    "extracts are off-topic, contradictory in ways that block any "
    "synthesis, or simply empty, output ONLY the Gaps section (Key "
    "findings and Disagreements omitted entirely — this is an "
    "explicit exception to Rule 5):\n"
    "\n"
    "    ## Gaps\n"
    "    - The retrieved sources do not contain information that "
    "answers the query.\n"
    "\n"
    "Do not invent. Do not extrapolate.\n"
    "\n"
    "9. ONE STATEMENT PER LINE. Each statement under a Key-findings "
    "sub-topic occupies exactly one line, ending with its citation. "
    "No wrap. No multi-line sentences. If a thought needs more than "
    "one sentence, write each sentence as its own line, each cited "
    "independently.\n"
    "\n"
    "10. CITATION-CORRECTNESS SELF-CHECK. Before emitting each line, "
    "re-read the cited extract and confirm the citation supports the "
    "claim. If you find yourself about to cite a source for something "
    "it doesn't actually say, do not write the line. This is the "
    "single most important quality bar — fabricating citations is "
    "worse than omitting facts.\n"
    "\n"
    "11. LANGUAGE. Reply in the language of the user query. If the "
    "query is in English, reply in English. If the query is in "
    "Russian, Spanish, German, Chinese, etc., reply in that language "
    "while keeping the section headers (## Key findings, ## "
    "Disagreements, ## Gaps) in English — they are structural markers "
    "consumed by post-processing."
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


def _build_conversation_prompt(
    messages_text: str,
    existing_summary: str | None,
) -> list[Any]:
    """Build the ``[SystemMessage, HumanMessage]`` for the conversation
    summariser path. Extracted so the dispatch is uniform."""
    from langchain_core.messages import HumanMessage, SystemMessage

    if existing_summary:
        intro = (
            "Here is the existing summary of earlier conversation:\n"
            f"---\n{existing_summary}\n---\n\n"
            "Incorporate the following new exchanges into the summary, "
            "merging overlapping topics and keeping the result concise:"
        )
    else:
        intro = "Summarize the following conversation:"
    return [
        SystemMessage(content=_SUMMARIZE_SYSTEM),
        HumanMessage(content=f"{intro}\n\n{messages_text}"),
    ]


def _build_synthesis_prompt(messages: list[Any]) -> list[Any]:
    """Build the ``[SystemMessage, *messages]`` for the web_search
    synthesis path.

    ``messages`` is expected to be a single ``HumanMessage`` whose
    content was formatted by the caller per the synthesis-prompt
    doc's "Human prompt" template. The synthesiser owns formatting
    because the citation indices and header structure are
    pipeline-specific.
    """
    from langchain_core.messages import SystemMessage

    return [SystemMessage(content=_WEB_SEARCH_SYNTHESIS_SYSTEM), *messages]


def generate_summary(
    llm: Any,
    messages: list[Any],
    existing_summary: str | None = None,
    *,
    purpose: str = "conversation",
    timeout_seconds: int | None = None,
) -> str | None:
    """Generate or update a conversation summary, or run web-search synthesis.

    Parameters
    ----------
    llm:
        A LangChain-compatible chat model (must support ``.invoke()``).
    messages:
        Conversation messages (purpose="conversation") OR a single
        pre-formatted ``HumanMessage`` (purpose="web_search_synthesis").
    existing_summary:
        The current rolling summary on the conversation path
        (``None`` on first call). Must be ``None`` on the synthesis
        path — passing a non-None value raises ``ValueError``.
    purpose:
        ``"conversation"`` (default; historical behaviour) or
        ``"web_search_synthesis"`` (ADR-0056 stage 5).
    timeout_seconds:
        Per-call LLM-invoke deadline. ``None`` (default) → look up the
        per-purpose default (60s for conversation, 7s for synthesis).

    Returns
    -------
    str | None
        The generated text. ``None`` if the call timed out, the LLM
        returned an empty response, or LangChain isn't importable.
        On the conversation path, ``existing_summary`` is returned
        instead of ``None`` for backwards compatibility — the
        synthesis path always returns ``None`` on failure because
        ``existing_summary`` must be ``None`` there.
    """
    if purpose not in _SUPPORTED_PURPOSES:
        raise ValueError(
            f"Unsupported purpose {purpose!r}; expected one of {list(_SUPPORTED_PURPOSES)}"
        )
    if purpose == "web_search_synthesis" and existing_summary is not None:
        raise ValueError(
            "existing_summary must be None when purpose='web_search_synthesis' "
            "(synthesis is single-shot per web_search call)"
        )

    if not messages:
        return existing_summary

    try:
        from langchain_core.messages import HumanMessage  # noqa: F401
    except ImportError:
        log.debug("LangChain not available — skipping summarization")
        return existing_summary

    if purpose == "conversation":
        messages_text = _format_messages_text(messages)
        if not messages_text.strip():
            return existing_summary
        prompt = _build_conversation_prompt(messages_text, existing_summary)
    else:  # web_search_synthesis
        prompt = _build_synthesis_prompt(messages)

    effective_timeout = (
        timeout_seconds if timeout_seconds is not None else _resolve_default_timeout(purpose)
    )

    try:
        # Bounded-timeout LLM invocation via the centralized helper —
        # migrated under #1903; see docs/architecture/CONCURRENCY.md
        # for the policy.
        try:
            response = invoke_with_timeout(llm.invoke, prompt, timeout=effective_timeout)
        except TimeoutError:
            log.warning(
                "generate_summary(purpose=%s): LLM call timed out after %ds",
                purpose,
                effective_timeout,
            )
            return existing_summary
        raw = getattr(response, "content", str(response)) or ""
        if isinstance(raw, list):
            raw = " ".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in raw)
        text = str(raw).strip()
        if not text:
            log.warning("generate_summary(purpose=%s) returned empty response", purpose)
            return existing_summary
        return text
    except Exception as exc:
        log.warning("generate_summary(purpose=%s) failed (non-fatal): %s", purpose, exc)
        return existing_summary
