"""Message-list repair and trimming utilities for the orchestration graph.

Extracted from ``src/orchestration/graph.py`` as the first step of the
graph.py 5-module split proposed by the /forge audit
(architect finding A1.1, 2026-05-23). All four helpers operate on a
``list[BaseMessage]`` and have no graph-build / langgraph-runtime
dependency — they can be tested in isolation and reused freely.

Functions:

* :func:`_detect_invalid_tool_calls` — scan history for "is not a valid
  tool" errors and return de-duplicated tool names the LLM attempted.
* :func:`_strip_failed_tool_messages` — remove those ToolMessage errors
  and the matching AIMessage tool_call entries.
* :func:`_repair_tool_message_pairs` — drop orphaned / misordered
  ToolMessages whose ``tool_call_id`` does not pair with a preceding
  AIMessage (OpenAI rejects this shape).
* :func:`_apply_context_message_cap` — chunk-aware history trim that
  respects AIMessage + ToolMessage pairing under both message-count
  and token caps.

The leading underscore on each name is preserved from the original
graph.py module for back-compat with the existing call sites in
``src.orchestration.nodes.call_model`` and the test suite.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.orchestration.compression import _CHARS_PER_TOKEN

# Regex used by ``_detect_invalid_tool_calls`` to recognise the tool-not-
# loaded error that the orchestrator emits before auto-activation.
_INVALID_TOOL_RE = re.compile(r"^Error:\s*(\S+)\s+is not a valid tool")


def _detect_invalid_tool_calls(
    messages: list,
    start_idx: int = 0,
) -> list[str]:
    """
    Scan *messages* from *start_idx* for **any** "is not a valid tool"
    ToolMessage error, regardless of whether the tool is in the on-demand
    pool.

    Returns a de-duplicated, ordered list of tool names the LLM tried.
    """
    from langchain_core.messages import ToolMessage

    found: list[str] = []
    seen: set[str] = set()
    for i in range(start_idx, len(messages)):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        m = _INVALID_TOOL_RE.match(content)
        if m:
            tool_name = m.group(1)
            if tool_name not in seen:
                found.append(tool_name)
                seen.add(tool_name)
    return found


def _strip_failed_tool_messages(messages: list, tool_names: set[str]) -> list:
    """
    Return a copy of *messages* with ToolMessage errors (and their matching
    AIMessage tool_calls) removed for tools in *tool_names*.

    This cleans up the conversation history after auto-activation so the
    resumed agent doesn't see the failed "is not a valid tool" attempts.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    tool_call_ids_to_remove: set[str] = set()
    cleaned: list = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = getattr(msg, "content", "")
            if name in tool_names and isinstance(content, str) and "is not a valid tool" in content:
                tcid = getattr(msg, "tool_call_id", "")
                if tcid:
                    tool_call_ids_to_remove.add(tcid)
                continue
        cleaned.append(msg)

    if not tool_call_ids_to_remove:
        return cleaned

    final: list = []
    for msg in cleaned:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                remaining = [tc for tc in tool_calls if tc.get("id") not in tool_call_ids_to_remove]
                if len(remaining) != len(tool_calls):
                    extra = dict(getattr(msg, "additional_kwargs", {}))
                    extra.pop("tool_calls", None)
                    new_msg = AIMessage(
                        content=getattr(msg, "content", ""),
                        tool_calls=remaining,
                        additional_kwargs=extra,
                    )
                    if not remaining and not (
                        isinstance(new_msg.content, str) and new_msg.content.strip()
                    ):
                        continue
                    final.append(new_msg)
                    continue
        final.append(msg)
    return final


def _repair_tool_message_pairs(messages: list) -> list:
    """Remove ToolMessages whose tool_call_id has no valid preceding AIMessage.

    OpenAI (and compatible providers) reject requests where a ToolMessage is not
    preceded by an AIMessage that contains a tool_call with a matching id.  This
    situation arises when:
    - An MCP/tool call raises an exception (e.g. ClosedResourceError) and the
      ToolMessage error is stored in state, but the triggering AIMessage was empty
      or had a malformed / truncated tool_calls list.
    - Message compression strips tool_calls from an AIMessage while retaining the
      paired ToolMessages.

    The repair pass collects every tool_call id that appears in an AIMessage
    (checking .tool_calls, additional_kwargs["tool_calls"], and Anthropic/Bedrock
    content blocks), then drops any ToolMessage whose tool_call_id is absent from
    that set or appears before the declaring AIMessage.  Truly empty AIMessages
    (no content, no tool_calls of any kind) that no longer serve as a pair anchor
    are also dropped.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    def _collect_tool_call_ids(msg: AIMessage) -> set[str]:
        """Return all tool_call ids declared by an AIMessage across all encoding styles."""
        ids: set[str] = set()
        # Standard LangChain attribute (OpenAI, Anthropic modern, etc.)
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tcid:
                ids.add(tcid)
        # OpenAI additional_kwargs encoding (some providers / older LangChain)
        for tc in (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []:
            tcid = tc.get("id") if isinstance(tc, dict) else None
            if tcid:
                ids.add(tcid)
        # Anthropic/Bedrock content-block encoding: content=[{type:tool_use, id:...}]
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tcid = block.get("id")
                    if tcid:
                        ids.add(tcid)
        return ids

    def _msg_has_content(msg: AIMessage) -> bool:
        """True when the message carries text, tool-calls, or content blocks."""
        if _collect_tool_call_ids(msg):
            return True
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(content)  # any content blocks (including tool_use) count
        return bool(content)

    # Pass 1 — collect declared tool_call ids and their first declaring position.
    declared_ids: set[str] = set()
    declared_positions: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_call_ids = _collect_tool_call_ids(msg)
        declared_ids |= tool_call_ids
        for tcid in tool_call_ids:
            declared_positions.setdefault(tcid, idx)

    # Pass 2 — identify orphaned and misordered ToolMessage tool_call_ids.
    orphaned_ids: set[str] = set()
    misordered_ids: set[str] = set()
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid and tcid not in declared_ids:
                orphaned_ids.add(tcid)
                continue
            if tcid and declared_positions.get(tcid) is not None:
                if msg_idx < declared_positions[tcid]:
                    misordered_ids.add(tcid)

    if not orphaned_ids and not misordered_ids:
        return messages

    # Logging via the original ``cogtrix.orchestration.graph`` logger so the
    # repair warnings keep landing in the same operator-facing stream that
    # existed before the extraction.
    logging.getLogger("cogtrix.orchestration.graph").warning(
        "Repairing %d orphaned and %d misordered ToolMessage(s) (orphans: %s; misordered: %s) — "
        "likely caused by ClosedResourceError, malformed tool_calls, or compressed history",
        len(orphaned_ids),
        len(misordered_ids),
        ", ".join(sorted(orphaned_ids)) if orphaned_ids else "none",
        ", ".join(sorted(misordered_ids)) if misordered_ids else "none",
    )

    repaired: list = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid in orphaned_ids or tcid in misordered_ids:
                continue  # drop orphaned or misordered ToolMessage
        elif isinstance(msg, AIMessage):
            # Drop truly empty AIMessages: no text, no tool_calls, no content blocks
            if not _msg_has_content(msg):
                continue
        repaired.append(msg)
    return repaired


def _apply_context_message_cap(
    messages: list,
    max_messages: int | None,
    max_tokens: int | None = None,
) -> list:
    """Trim oldest message pairs when history exceeds the configured cap(s).

    Consecutive AIMessage + ToolMessage runs are treated as a single logical
    chunk so tool-call pairs are never split.  Oldest chunks are dropped until
    both the message-count and token budgets fit.  The newest chunk is always
    preserved even if it exceeds the configured budget on its own.
    """
    if (not max_messages or max_messages <= 0) and (not max_tokens or max_tokens <= 0):
        return messages

    from langchain_core.messages import AIMessage, ToolMessage

    def _tool_ids(msg: Any) -> set[str]:
        ids: set[str] = set()
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tcid:
                ids.add(tcid)
        return ids

    def _msg_tokens(msg: Any) -> int:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return max(1, len(content) // _CHARS_PER_TOKEN)
        if isinstance(content, list):
            chars = 0
            for item in content:
                if isinstance(item, str):
                    chars += len(item)
                elif isinstance(item, dict):
                    chars += len(item.get("text", ""))
            return max(1, chars // _CHARS_PER_TOKEN)
        return 1

    chunks: list[list[Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            tc_ids = _tool_ids(msg)
            if tc_ids:
                chunk: list[Any] = [msg]
                j = i + 1
                while j < len(messages):
                    nxt = messages[j]
                    if not isinstance(nxt, ToolMessage):
                        break
                    if getattr(nxt, "tool_call_id", None) not in tc_ids:
                        break
                    chunk.append(nxt)
                    j += 1
                chunks.append(chunk)
                i = j
                continue
        chunks.append([msg])
        i += 1

    kept: list[list[Any]] = []
    kept_count = 0
    kept_tokens = 0
    for chunk in reversed(chunks):
        chunk_count = len(chunk)
        chunk_tokens = sum(_msg_tokens(msg) for msg in chunk)
        if not kept:
            kept.append(chunk)
            kept_count += chunk_count
            kept_tokens += chunk_tokens
            continue
        if max_messages and max_messages > 0 and kept_count + chunk_count > max_messages:
            break
        if max_tokens and max_tokens > 0 and kept_tokens + chunk_tokens > max_tokens:
            break
        kept.append(chunk)
        kept_count += chunk_count
        kept_tokens += chunk_tokens

    if not kept:
        return messages

    kept.reverse()
    truncated = [m for chunk in kept for m in chunk]
    dropped = len(messages) - len(truncated)
    if dropped > 0:
        logging.getLogger("cogtrix.orchestration.graph").warning(
            "context_max_messages=%s context_max_tokens=%s: dropped %d oldest message(s)",
            max_messages if max_messages is not None else 0,
            max_tokens if max_tokens is not None else 0,
            dropped,
        )
    return truncated


__all__ = [
    "_INVALID_TOOL_RE",
    "_apply_context_message_cap",
    "_detect_invalid_tool_calls",
    "_repair_tool_message_pairs",
    "_strip_failed_tool_messages",
]
