"""Context compression pipeline for long agent runs.

Summarizes old, large ToolMessages in-place before each LLM call to
reduce per-cycle token usage without losing important context.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import re
from typing import Any

from src.logging_config import get_logger

try:
    from langchain_core.messages import AIMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    AIMessage = None  # type: ignore[misc, assignment]
    ToolMessage = None  # type: ignore[misc, assignment]
    _HAS_LANGCHAIN = False

COMPRESSION_MIN_AGE_CYCLES = 3
COMPRESSION_MIN_CHARS = 2_000
_COMPRESSION_THRESHOLD_RATIO = 0.72
_FALLBACK_MAX_CHARS = 30_000

_COMPRESSION_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="compress"
)
atexit.register(_COMPRESSION_POOL.shutdown, wait=False)


def _content_len(msg: Any) -> int:
    """Return character count of a message's content without redundant str() calls."""
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(s) for s in c if isinstance(s, str))
    return 0


def truncate_tool_output(text: str, max_chars: int) -> str:
    """Middle-truncate *text* if it exceeds *max_chars*."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    removed = len(text) - max_chars
    return (
        text[:keep] + f"\n\n[... {removed:,} chars truncated to fit context budget — "
        f"use start_line/max_lines to page through, or search to "
        f"find specific sections ...]\n\n" + text[-keep:]
    )


def compress_tool_message(content: str, tool_name: str, llm: Any) -> str:
    """Compress a ToolMessage via one-shot LLM summarization.

    Preserves key artifacts (file paths, errors, line numbers, schemas,
    exact values) while stripping verbose prose and boilerplate.

    Falls back to middle-truncation on any failure.
    """
    log = get_logger()
    try:
        compress_prompt = (
            "You are a context compressor for an AI agent's working memory. "
            "Condense the tool output below, preserving ALL of:\n"
            "- File paths, URLs, directory names\n"
            "- Error messages and stack traces (exact text)\n"
            "- Line numbers and column numbers\n"
            "- Schema definitions, type signatures, data structures\n"
            "- Exact numeric values, IDs, hashes\n"
            "- Key findings, decisions, conclusions\n"
            "- Code snippets referenced later\n\n"
            "Remove:\n"
            "- Verbose explanatory prose restating obvious context\n"
            "- Redundant formatting, decoration, boilerplate\n"
            "- Raw HTML/XML markup (keep extracted content)\n"
            "- Duplicate information\n\n"
            "Output ONLY the compressed content. No preamble.\n\n"
            f"Tool: {tool_name}\n"
            f"Output to compress:\n{content}"
        )
        response = llm.invoke(compress_prompt)
        raw = getattr(response, "content", str(response))
        if isinstance(raw, list):
            raw = " ".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in raw)
        compressed = str(raw).strip()

        if not compressed or len(compressed) < 20:
            log.debug("Compression returned empty/tiny result, using truncation fallback")
            fallback_len = max(len(content) * 3 // 4, min(len(content), 200))
            return truncate_tool_output(content, min(fallback_len, _FALLBACK_MAX_CHARS))

        if len(compressed) >= len(content):
            log.debug("Compression did not reduce size, keeping original")
            return content

        log.debug(
            "Compressed tool output: %d chars -> %d chars (%.0f%% reduction)",
            len(content),
            len(compressed),
            (1 - len(compressed) / len(content)) * 100,
        )
        return compressed

    except Exception as exc:
        log.warning("Tool message compression failed: %s", exc, exc_info=True)
        fallback_len = max(len(content) * 3 // 4, min(len(content), 200))
        return truncate_tool_output(content, min(fallback_len, _FALLBACK_MAX_CHARS))


def apply_message_compression(
    messages: list,
    call_count: int,
    compression_cache: dict[str, str],
    llm: Any,
    max_context_tokens: int | None,
    min_age_cycles: int = COMPRESSION_MIN_AGE_CYCLES,
    min_chars: int = COMPRESSION_MIN_CHARS,
) -> list:
    """Build a compressed copy of messages for the LLM invocation.

    Does NOT mutate the input list.  Returns a new list where eligible
    ToolMessages have their content replaced with cached or freshly
    generated summaries.

    A ToolMessage is eligible when both conditions hold:
      1. More than *min_age_cycles* call_model outputs appear after it.
      2. Its content length >= *min_chars*.

    The pass itself only runs when total message chars reach 72 % of the
    context window, and is skipped entirely for providers with fewer than
    16 384 context tokens (where trimming is cheaper).
    """
    if not _HAS_LANGCHAIN:
        return messages

    if max_context_tokens is None:
        return messages

    if max_context_tokens < 16_384:
        return messages

    total_chars = sum(_content_len(m) for m in messages)
    context_chars = max_context_tokens * 4
    threshold_chars = int(context_chars * _COMPRESSION_THRESHOLD_RATIO)

    if total_chars < threshold_chars:
        return messages

    log = get_logger()
    log.debug(
        "Compression pass triggered at cycle %d (total_chars=%d, threshold=%d)",
        call_count,
        total_chars,
        threshold_chars,
    )

    # Calculate age of each ToolMessage (number of AIMessages after it).
    ai_count_from_end = 0
    msg_age: dict[int, int] = {}
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            ai_count_from_end += 1
        if isinstance(msg, ToolMessage):
            msg_age[i] = ai_count_from_end

    # First pass: identify eligible messages and separate cached vs. needs-LLM.
    eligible: dict[int, tuple[str, str, str]] = {}  # idx -> (content, tool_name, tcid)
    cached: dict[int, str] = {}  # idx -> cached compressed content
    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        tcid = getattr(msg, "tool_call_id", None)
        raw_content = getattr(msg, "content", "")
        if not isinstance(raw_content, str):
            continue
        content = raw_content or ""
        age = msg_age.get(i, 0)
        if age < min_age_cycles or len(content) < min_chars:
            continue
        if tcid and tcid in compression_cache:
            cached[i] = compression_cache[tcid]
        else:
            tool_name = getattr(msg, "name", "unknown_tool")
            tool_name = re.sub(r"[\r\n\x00]", "", tool_name)[:100]
            eligible[i] = (content, tool_name, tcid or "")

    # Compress eligible messages in parallel. LangChain LLM.invoke() makes
    # stateless HTTP calls that are safe for concurrent use.
    compressed_results: dict[int, str] = {}
    if eligible:

        def _compress_one(idx: int) -> tuple[int, str]:
            content, tool_name, _ = eligible[idx]
            try:
                return idx, compress_tool_message(content, tool_name, llm)
            except Exception:
                return idx, truncate_tool_output(
                    content, min(len(content) * 3 // 4, _FALLBACK_MAX_CHARS)
                )

        futures = {_COMPRESSION_POOL.submit(_compress_one, i): i for i in eligible}
        for future in concurrent.futures.as_completed(futures):
            idx, compressed = future.result()
            compressed_results[idx] = compressed

        # Update cache sequentially — no concurrent writes.
        for idx, compressed in compressed_results.items():
            tcid = eligible[idx][2]
            if tcid:
                compression_cache[tcid] = compressed

    # Assemble result list, tracking saved chars incrementally.
    result = []
    saved_chars = 0
    for i, msg in enumerate(messages):
        if i in compressed_results or i in cached:
            compressed_content = compressed_results[i] if i in compressed_results else cached[i]
            original_len = _content_len(msg)
            saved_chars += original_len - len(compressed_content)
            replacement = ToolMessage(
                content=compressed_content,
                tool_call_id=getattr(msg, "tool_call_id", "") or "",
                name=getattr(msg, "name", ""),
            )
            result.append(replacement)
        else:
            result.append(msg)

    compressed_count = len(compressed_results)
    if compressed_count > 0:
        new_total = total_chars - saved_chars
        log.info(
            "Compressed %d tool messages: %d chars -> %d chars (%.0f%% reduction)",
            compressed_count,
            total_chars,
            new_total,
            (1 - new_total / total_chars) * 100 if total_chars else 0,
        )

    return result


def create_compression_llm(model_ref: str, config: Any) -> Any:
    """Create a dedicated LLM for context compression.

    Resolves *model_ref* — a model registry name or ``"provider/model"``
    string — against the config's models registry and provider list,
    then builds a LangChain LLM via ``create_llm_from_provider_config``.

    Returns ``None`` on any failure (caller falls back to the main LLM).
    """
    log = get_logger()
    try:
        from copy import copy

        from src.agent.core import create_llm_from_provider_config

        provider_name: str | None = None
        model_name: str | None = None

        mc = config.get_model_config(model_ref)
        if mc is not None:
            provider_name = mc.provider
            model_name = mc.model
        elif "/" in model_ref:
            provider_name, model_name = model_ref.split("/", 1)
        else:
            provider_name = config.provider
            model_name = model_ref

        prov_cfg = copy(config.get_provider_config(provider_name))
        if model_name:
            prov_cfg.model = model_name
        if mc is not None:
            if mc.num_ctx is not None:
                prov_cfg.num_ctx = mc.num_ctx
            if mc.temperature is not None:
                prov_cfg.temperature = mc.temperature

        llm = create_llm_from_provider_config(prov_cfg)
        log.info("Compression LLM created: %s/%s", provider_name, model_name)
        return llm
    except Exception as exc:
        log.warning("Failed to create compression LLM '%s': %s", model_ref, exc)
        return None
