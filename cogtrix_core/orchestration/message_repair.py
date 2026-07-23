"""Message-list repair and trimming utilities for the orchestration graph.

Extracted from ``cogtrix_core/orchestration/graph.py`` as the first step of the
graph.py 5-module split proposed by the /forge audit
(architect finding A1.1, 2026-05-23). All four helpers operate on a
``list[BaseMessage]`` and have no graph-build / langgraph-runtime
dependency — they can be tested in isolation and reused freely.

Functions:

* :func:`_detect_invalid_tool_calls` — scan history for "is not a valid
  tool" errors and return de-duplicated tool names the LLM attempted.
* :func:`_strip_failed_tool_messages` — remove those ToolMessage errors
  and the matching AIMessage tool_call entries.
* :func:`_repair_tool_message_pairs` — reconcile AIMessage(tool_calls)
  ⇄ ToolMessage pairing both directions: drop orphaned / misordered
  ToolMessages, and inject synthetic answers for declared-but-unanswered
  tool_calls (OpenAI rejects either shape).
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

from cogtrix_core.orchestration.compression import _CHARS_PER_TOKEN, _content_len

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
    """Repair the AIMessage(tool_calls) ⇄ ToolMessage pairing both directions.

    OpenAI (and compatible providers) reject a request unless **every** tool_call
    an AIMessage declares is answered by a following ToolMessage, and no
    ToolMessage is orphaned/misordered. This guard reconciles both sides:

    - **Orphaned / misordered ToolMessages** — a ``tool`` result whose
      ``tool_call_id`` has no declaring AIMessage, or appears *before* it — are
      dropped (e.g. an MCP call raising ``ClosedResourceError`` left a result in
      state with no triggering AIMessage, or compression stripped the AIMessage's
      tool_calls while keeping the results). Truly empty AIMessages (no content,
      no tool_calls of any kind) are dropped too.
    - **Unanswered tool_calls** (#2238) — a ``tool_call_id`` an AIMessage
      *declares* that has no following ToolMessage — get a synthetic placeholder
      ``ToolMessage("[tool call not completed]", tool_call_id=...)`` injected right
      after the declaring AIMessage. Without this, a partially-fulfilled parallel
      batch (cut mid-flight by the recursion limit or per-tool budget, deduped, or
      half-trimmed by compression) reaches the provider unanswered → a hard 400
      that kills the turn and can wedge the session.

    Declared ids are harvested across all encodings (``.tool_calls``,
    ``additional_kwargs["tool_calls"]``, and Anthropic/Bedrock ``tool_use``
    content blocks).
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

    def _collect_tool_call_names(msg: AIMessage) -> dict[str, str]:
        """Map declared tool_call id -> tool name (best-effort) for synthetic answers."""
        names: dict[str, str] = {}
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if tcid and nm:
                names[tcid] = nm
        for tc in (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []:
            if isinstance(tc, dict):
                tcid = tc.get("id")
                nm = (tc.get("function") or {}).get("name") or tc.get("name")
                if tcid and nm:
                    names.setdefault(tcid, nm)
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tcid, nm = block.get("id"), block.get("name")
                    if tcid and nm:
                        names.setdefault(tcid, nm)
        return names

    # Pass 1 — collect declared tool_call ids, first declaring position, and names.
    declared_ids: set[str] = set()
    declared_positions: dict[str, int] = {}
    declared_names: dict[str, str] = {}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_call_ids = _collect_tool_call_ids(msg)
        declared_ids |= tool_call_ids
        for tcid in tool_call_ids:
            declared_positions.setdefault(tcid, idx)
        for tcid, nm in _collect_tool_call_names(msg).items():
            declared_names.setdefault(tcid, nm)

    # Pass 2 — identify orphaned and misordered ToolMessage tool_call_ids, and the
    # set of declared ids that ARE answered by a ToolMessage following their
    # declaration.
    orphaned_ids: set[str] = set()
    misordered_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if not tcid:
                continue
            if tcid not in declared_ids:
                orphaned_ids.add(tcid)
                continue
            decl_pos = declared_positions.get(tcid)
            if decl_pos is not None and msg_idx < decl_pos:
                misordered_ids.add(tcid)
            elif decl_pos is not None and msg_idx > decl_pos:
                answered_ids.add(tcid)

    # Pass 3 (#2238) — declared tool_call ids with NO answering ToolMessage after
    # their declaration. Left as-is, the AIMessage's tool_calls reach the provider
    # unanswered → 400 "insufficient tool messages following tool_calls" (kills the
    # turn and can wedge the session). This arises when a parallel tool batch is
    # cut mid-flight (recursion / per-tool budget), a dedup/skip emits no
    # ToolMessage, or compression drops a ToolMessage while keeping its AIMessage.
    # We inject a synthetic placeholder ToolMessage for each so the pairing holds.
    unanswered_ids: set[str] = declared_ids - answered_ids

    # Pass 3b (#2365 defect B) — redundant DUPLICATE declarations. Under heavy
    # compression / retry churn a tool-calling AIMessage can be re-emitted, so the
    # SAME tool_call_id is *declared* by a later AIMessage too. Pass 3's set math
    # (declared − answered) sees the id as answered (the first occurrence was) and
    # repairs nothing — but the provider validates PER assistant-message
    # occurrence and 400s on the second, unanswered declaration. A tool_call_id
    # can be answered only once, so injecting a second placeholder would itself
    # 400 (duplicate answer); the correct repair is to DROP the redundant
    # re-declaration. Removal-only, so it composes with call_model's id()-based
    # RemoveMessage state write-back (a modifying repair would be deleted from
    # state and never re-added — the #2276 coupling). We only drop a re-declaring
    # AIMessage whose declared ids are ALL earlier-declared duplicates AND which
    # carries no unique text (a pure re-emitted tool-call block); a content-
    # bearing duplicate or an id-form (colon/underscore) mismatch is left for the
    # shipped #2365 diagnostic (_format_tool_pair_diagnostic) to characterise.
    redundant_duplicate_idxs: set[int] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        ids = _collect_tool_call_ids(msg)
        if not ids:
            continue
        # Every declared id was FIRST declared by an earlier AIMessage → this whole
        # message is a redundant re-declaration (declared_positions keeps the first
        # position, so the original declarer never matches this).
        if any(declared_positions.get(tcid, idx) >= idx for tcid in ids):
            continue
        _content = getattr(msg, "content", None)
        if isinstance(_content, str) and _content.strip():
            continue  # carries unique text → don't drop (removal would lose it)
        if isinstance(_content, list) and any(
            isinstance(b, dict) and b.get("type") != "tool_use" for b in _content
        ):
            continue  # non-tool_use content blocks → unique content, keep
        redundant_duplicate_idxs.add(idx)

    # Pass 4 — contiguity. OpenAI/Azure reject a request unless every ToolMessage
    # *immediately* follows its declaring AIMessage(tool_calls) (or a sibling
    # ToolMessage in the same block). A directive/nudge/compression step can wedge
    # a non-tool message between an AIMessage's tool_calls and its answering
    # ToolMessage(s); declaration and order are both still valid, so Passes 1-3
    # leave it untouched, but the provider 400s ("messages with role 'tool' must
    # be a response to a preceeding message with 'tool_calls'"). Group each
    # answered ToolMessage under its declaring AIMessage; a foreign message inside
    # that span means the answers must be relocated to restore contiguity.
    answers_by_decl: dict[int, list[int]] = {}
    for ti, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        tcid = getattr(msg, "tool_call_id", None)
        if not tcid or tcid in orphaned_ids or tcid in misordered_ids:
            continue
        decl = declared_positions.get(tcid)
        if decl is not None and ti > decl:
            answers_by_decl.setdefault(decl, []).append(ti)
    displaced = False
    for decl, tis in answers_by_decl.items():
        run = set(tis)
        # A gap between the declaration and the last answer that is not itself an
        # answer for this declaration = a wedged foreign message → not contiguous.
        if any(k not in run for k in range(decl + 1, max(tis))):
            displaced = True
            break

    # Pass 5 — duplicate answers (#2276). More than one ToolMessage answering the
    # SAME tool_call_id: OpenAI/Azure require exactly one tool response per declared
    # tool_call, so the extra is unmatched → a hard 400 that kills the turn
    # (observed on gpt-4o via OpenRouter, role_pm_06: a tool node emitted two
    # results for one parallel call). Keep the first answer and drop the rest. This
    # is removal-only, so it composes with call_model's RemoveMessage write-back.
    seen_answer_tcids: set[str] = set()
    duplicate = False
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tc = getattr(msg, "tool_call_id", None)
        if not tc or tc in orphaned_ids or tc in misordered_ids:
            continue
        if tc in seen_answer_tcids:
            duplicate = True
            break
        seen_answer_tcids.add(tc)

    if (
        not orphaned_ids
        and not misordered_ids
        and not unanswered_ids
        and not displaced
        and not duplicate
        and not redundant_duplicate_idxs
    ):
        return messages

    # Logging via the original ``cogtrix.orchestration.graph`` logger so the
    # repair warnings keep landing in the same operator-facing stream that
    # existed before the extraction.
    logging.getLogger("cogtrix.orchestration.graph").warning(
        "Repairing %d orphaned, %d misordered, %d unanswered, %s displaced, "
        "%s duplicate, and %d redundant-declaration ToolMessage block(s) "
        "(orphans: %s; misordered: %s; unanswered: %s) — likely caused by "
        "ClosedResourceError, malformed tool_calls, a budget/recursion-cut tool "
        "batch, a duplicated tool result, a re-emitted tool-call declaration "
        "(#2365), an injected directive splitting a tool pair, or compressed history",
        len(orphaned_ids),
        len(misordered_ids),
        len(unanswered_ids),
        "some" if displaced else "no",
        "some" if duplicate else "no",
        len(redundant_duplicate_idxs),
        ", ".join(sorted(orphaned_ids)) if orphaned_ids else "none",
        ", ".join(sorted(misordered_ids)) if misordered_ids else "none",
        ", ".join(sorted(unanswered_ids)) if unanswered_ids else "none",
    )

    consumed: set[int] = set()
    emitted_tcids: set[str] = set()
    repaired: list = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            if idx in consumed:
                continue  # already emitted contiguously under its declaring AIMessage
            tcid = getattr(msg, "tool_call_id", None)
            if tcid in orphaned_ids or tcid in misordered_ids:
                continue  # drop orphaned or misordered ToolMessage
            if tcid in emitted_tcids:
                continue  # drop a duplicate answer for an already-answered tool_call
            if tcid:
                emitted_tcids.add(tcid)
            repaired.append(msg)  # defensive: answered tool whose declarer was dropped
        elif isinstance(msg, AIMessage):
            # #2365: drop a redundant duplicate-declaration AIMessage (all its
            # tool_calls were declared earlier + it carries no unique text), so the
            # provider never sees the second, unanswered occurrence.
            if idx in redundant_duplicate_idxs:
                continue
            # Drop truly empty AIMessages: no text, no tool_calls, no content blocks
            if not _msg_has_content(msg):
                continue
            repaired.append(msg)
            # Pull this AIMessage's answered ToolMessages into place (original
            # order) so the tool block sits immediately after it — restoring
            # contiguity when a foreign message had split the pair. Keep one answer
            # per tool_call_id (drop duplicates) so the block stays OpenAI-compliant.
            for ti in answers_by_decl.get(idx, []):
                consumed.add(ti)
                tc = getattr(messages[ti], "tool_call_id", None)
                if tc in emitted_tcids:
                    continue  # duplicate — one answer already emitted for this id
                if tc:
                    emitted_tcids.add(tc)
                repaired.append(messages[ti])
            # Inject a synthetic answer for each unanswered tool_call FIRST declared
            # by this AIMessage (so duplicate declarations don't double-inject),
            # keeping the ToolMessage block contiguous.
            to_fill = [
                tcid
                for tcid in _collect_tool_call_ids(msg)
                if tcid in unanswered_ids and declared_positions.get(tcid) == idx
            ]
            for tcid in to_fill:
                emitted_tcids.add(tcid)
                repaired.append(
                    ToolMessage(
                        content="[tool call not completed]",
                        tool_call_id=tcid,
                        name=declared_names.get(tcid),
                        additional_kwargs={"cogtrix.kind": "synthetic_tool_repair"},
                    )
                )
        else:
            repaired.append(msg)
    return repaired


def _format_tool_pair_diagnostic(messages: list) -> str:
    """Return an ordered, content-free dump of the AIMessage⇄ToolMessage pairing.

    Emitted by ``call_model`` right before re-raising a tool-pair provider 400
    (#2365) so one real occurrence reveals the exact mechanism the set-based
    repair missed — a duplicate ``tool_call_id`` declaration (declared by two
    AIMessages, answered once) vs a colon/underscore id-form mismatch
    (``name:idx`` native-Kimi vs ``name_idx`` Cogtrix-synth) between the
    declaration and its answer.

    One line per message: index, role, declared tool_call_ids harvested across
    ALL encodings (``.tool_calls``, ``additional_kwargs['tool_calls']``, and
    ``tool_use`` content blocks) for AIMessages, or the answered ``tool_call_id``
    for ToolMessages, plus a content length. **No message content** is included
    (privacy + log size) — only structure and ids. Also appends a summary of
    declared-but-never-answered ids computed per OCCURRENCE (not set), which is
    precisely the axis the current repair does not check.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    def _ids(msg: Any) -> list[str]:
        ids: list[str] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tcid:
                ids.append(str(tcid))
        for tc in (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                ids.append(str(tc["id"]))
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    ids.append(str(block["id"]))
        return ids

    lines: list[str] = []
    declared_occurrences: list[str] = []  # every declaration, with duplicates
    answered: list[str] = []
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        clen = _content_len(msg)
        if isinstance(msg, AIMessage):
            ids = _ids(msg)
            declared_occurrences.extend(ids)
            lines.append(f"  [{i}] {role} declares={ids or '-'} content_len={clen}")
        elif isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                answered.append(str(tcid))
            lines.append(f"  [{i}] {role} answers={tcid or '-'} content_len={clen}")
        else:
            lines.append(f"  [{i}] {role} content_len={clen}")

    # Per-occurrence unanswered: subtract answers one-for-one so a duplicate
    # declaration answered only once shows up (a plain set-difference hides it).
    remaining = list(answered)
    unanswered_by_occurrence: list[str] = []
    for tcid in declared_occurrences:
        if tcid in remaining:
            remaining.remove(tcid)
        else:
            unanswered_by_occurrence.append(tcid)
    dup_declarations = [
        tcid for tcid in set(declared_occurrences) if declared_occurrences.count(tcid) > 1
    ]

    header = (
        f"tool-pair diagnostic ({len(messages)} msgs): "
        f"unanswered_by_occurrence={unanswered_by_occurrence or 'none'}; "
        f"duplicate_declarations={sorted(dup_declarations) or 'none'}; "
        f"orphan_answers={sorted(set(answered) - set(declared_occurrences)) or 'none'}"
    )
    return header + "\n" + "\n".join(lines)


def _apply_context_message_cap(
    messages: list,
    max_messages: int | None,
    max_tokens: int | None = None,
    evicted_summary: str | None = None,
) -> list:
    """Trim oldest message pairs when history exceeds the configured cap(s).

    Consecutive AIMessage + ToolMessage runs are treated as a single logical
    chunk so tool-call pairs are never split.  Oldest chunks are dropped until
    both the message-count and token budgets fit.  The newest chunk is always
    preserved even if it exceeds the configured budget on its own.

    *evicted_summary* (#1943 PR #3): an optional rolling summary string from
    the memory layer covering the messages that may be evicted.  When provided
    and non-empty, the ``[CONTEXT NOTICE]`` marker prepended after eviction
    embeds the summary text — giving the agent a semantic anchor to honestly
    acknowledge what was lost, rather than just being told ``data was lost``
    and (often) fabricating from training-data knowledge instead.  Pass
    ``None`` or an empty string when no summary is available; the marker
    falls back to the PR #1 prose unchanged.
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
        # #1943 PR #1: emit a SystemMessage marker so the agent has a
        # recoverable signal that data was lost.  Without this marker the
        # agent sees a normal-looking message history with no indication
        # that earlier turns were dropped — and may confidently produce a
        # "synthesis" that references the dropped content from training-
        # data knowledge instead of recognising the data is gone.
        # See #1943 for the verify-1919 reproducer.
        #
        # The marker is prepended (not appended) so it appears at the
        # OLDEST surviving position, mimicking where the dropped span
        # used to live in the conversation order.  It carries
        # ``additional_kwargs["cogtrix.kind"] = "context_evicted"`` so
        # downstream detectors (PR #4 in the #1943 roadmap) can route on
        # the kind rather than substring-matching the prose.
        from langchain_core.messages import SystemMessage as _SystemMessage

        _base_prose = (
            f"[CONTEXT NOTICE] {dropped} older message(s) were removed "
            f"from this conversation because the configured limits "
            f"(max_messages={max_messages or 'unset'}, "
            f"max_tokens={max_tokens or 'unset'}) were exceeded.  "
        )
        # #1943 PR #3: when the memory layer has a rolling summary covering
        # the evicted span, embed it.  The agent gets a semantic anchor to
        # answer honestly about what was lost instead of either fabricating
        # specifics or returning an unhelpful refusal.  The anti-fabrication
        # nudge from PR #1 stays — the summary is broad-strokes context,
        # never a substitute for the verbatim originals.
        if evicted_summary and evicted_summary.strip():
            _marker_content = (
                _base_prose + "Rolling summary of older context (from memory layer; "
                "broad-strokes only, not verbatim):\n"
                + evicted_summary.strip()
                + "\n\nThe original messages are gone — do NOT invent "
                "specifics not in the summary above (names, quotes, "
                "numbers, file paths).  If your current task depends on "
                "verbatim detail, request a re-read or tell the user the "
                "relevant context was lost."
            )
        else:
            _marker_content = (
                _base_prose + "The dropped content is gone — do NOT claim or "
                "summarise anything that was in those messages from "
                "memory.  If your current task depends on the dropped "
                "content, request a re-read or tell the user the relevant "
                "context was lost."
            )

        marker = _SystemMessage(
            content=_marker_content,
            additional_kwargs={"cogtrix.kind": "context_evicted"},
        )
        truncated.insert(0, marker)
    return truncated


__all__ = [
    "_INVALID_TOOL_RE",
    "_apply_context_message_cap",
    "_detect_invalid_tool_calls",
    "_format_tool_pair_diagnostic",
    "_repair_tool_message_pairs",
    "_strip_failed_tool_messages",
]
