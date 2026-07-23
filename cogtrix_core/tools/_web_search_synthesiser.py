"""Synthesiser for the web_search tool (ADR-0056 stage 5 + PR-D).

Consumes the extractor's ``ExtractedSource`` list, formats the
extract block per ``docs/optional/prompts/web-search-synthesis.md``, calls
``cogtrix_core/memory/summarizer.py::generate_summary`` with
``purpose="web_search_synthesis"``, runs four post-call validators
(URL line-drop, citation-presence regex, schema check, length cap),
and on validation failure / timeout retries once against a smaller
model.

On unrecoverable failure the synthesiser returns
``SynthesisResult(text=None, …)``. The public ``web_search.py``
entry point (PR-E) treats that as the cue to emit the "Synthesis
unavailable" Sources-only fallback documented in the ADR.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from cogtrix_core.memory.summarizer import generate_summary
from cogtrix_core.tools._web_search_domain_class import detect_affiliation_disclaimer
from cogtrix_core.tools._web_search_extractor import ExtractedSource

log = logging.getLogger("cogtrix")

# Per-purpose deadlines per ADR-0056 stage-5 fallback strategy.
#
# Primary raised from 7s → 10s after the cogtrix41c real-world run
# showed qwen3-coder/spark consistently spending 8-10s on the
# ~4000-token synthesis prompt. The old 7s cap timed out almost every
# call and the agent saw "Synthesis: failed (empty-response)" even
# when the underlying LLM was actively making progress.
#
# The outer pipeline ceiling (``_WEB_SEARCH_HARD_DEADLINE_S = 25s`` in
# ``web_search.py``) still bounds the worst case. Typical stages 1-4
# total 6-9s, leaving 16-19s of headroom for stage 5+6 — comfortably
# above 10s primary + 5s fallback = 15s.
_PRIMARY_DEADLINE_S = 10
_FALLBACK_DEADLINE_S = 5

# Length cap — 600 words is the prompt's ask; we tolerate 20% over
# before truncating. ~5 chars/word average → ~3600 char soft cap,
# ~4320 char hard ceiling.
_LENGTH_WORD_CAP = 600
_LENGTH_HARD_WORD_CAP = 720

# Citation regex — one or more circled numbers (Unicode block 2460-2473
# covers ① through ⑳, the cap declared in ADR-0056).
_CITATION_RE = re.compile(r"\[([①-⑳]+(?:[①-⑳])*)\]\s*$")

# A line containing a URL marker. Whole-line drop applied during
# post-processing per ADR-0056 + the synthesis-prompt doc Rule 2.
_URL_MARKER_RE = re.compile(r"://|www\.", re.IGNORECASE)

# Section headers expected in the output schema. Order matters.
_SECTION_KEY_FINDINGS = "## Key findings"
_SECTION_DISAGREEMENTS = "## Disagreements"
_SECTION_GAPS = "## Gaps"


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of one synthesis attempt.

    ``text`` is the validated + post-processed synthesis on success,
    ``None`` when the synthesiser couldn't produce schema-valid output
    (caller falls back to Sources-only per ADR-0056).

    ``reason`` explains why text is None — surfaced in the Coverage
    block so operators see synthesis-failure rates.

    ``model_used`` records which model emitted the final text:
    ``"primary"`` on first-shot success, ``"fallback"`` after retry.
    ``None`` when text is None.

    ``elapsed_ms`` is the wall-clock cost across all attempts.
    """

    text: str | None
    reason: str | None
    model_used: str | None
    elapsed_ms: int


async def synthesise(
    llm_primary: Any,
    extracts: list[ExtractedSource],
    query: str,
    *,
    deadline_s: float = float(_PRIMARY_DEADLINE_S),
    llm_fallback: Any | None = None,
    fallback_deadline_s: float = float(_FALLBACK_DEADLINE_S),
) -> SynthesisResult:
    """Synthesise ``extracts`` into a topic-organised picture.

    Parameters
    ----------
    llm_primary:
        Production summariser model. ``llm.invoke(prompt)`` must work
        (the underlying ``generate_summary`` runs it in a thread pool).
    extracts:
        Output of stage 4 (extractor). Only sources whose status is
        ``"extracted"``, ``"extracted-truncated"``, or
        ``"extracted-raw-fallback"`` contribute content; the rest are
        recorded as citation indices but their text is empty.
    query:
        The user query. Echoed into the human-prompt template per the
        synthesis-prompt doc.
    deadline_s:
        Wall-clock deadline for the primary call (default
        ``_PRIMARY_DEADLINE_S`` — currently 10s, raised from 7s after
        cogtrix41c surfaced provider latency exceeding the old budget;
        see Bug H in the synthesiser-investigation report).
    llm_fallback:
        Optional smaller model for the retry path. ``None`` (the
        default) means "no retry" — single attempt, then None on
        failure. The caller resolves the smaller model from
        ``SUMMARIZER_FALLBACK_MODEL`` env var (default
        ``claude-haiku-4-5``) and passes it here.
    fallback_deadline_s:
        Wall-clock deadline for the retry call (default 5s).
    """
    started_at = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started_at) * 1000)

    # Build the human-side prompt once; both attempts use the same.
    human_prompt = _format_human_prompt(query, extracts)

    # Attempt 1 — primary model.
    primary_text, primary_attempt_reason = await _attempt(llm_primary, human_prompt, deadline_s)
    if primary_attempt_reason is None:
        # Call succeeded; run the post-call validators.
        primary_validated, primary_validation_reason = _validate(primary_text)
        if primary_validated is not None:
            return SynthesisResult(
                text=primary_validated,
                reason=None,
                model_used="primary",
                elapsed_ms=_elapsed_ms(),
            )
        primary_reason = primary_validation_reason
    else:
        # Call layer failed (timeout / exception / empty). Skip
        # validators — there's nothing valid to validate.
        primary_reason = primary_attempt_reason

    # Attempt 2 — fallback model if available.
    if llm_fallback is None:
        log.info(
            "synthesis primary attempt failed (%s); no fallback configured",
            primary_reason,
        )
        return SynthesisResult(
            text=None,
            reason=primary_reason or "primary-failed",
            model_used=None,
            elapsed_ms=_elapsed_ms(),
        )

    log.info("synthesis primary attempt failed (%s); trying fallback", primary_reason)

    fallback_text, fallback_attempt_reason = await _attempt(
        llm_fallback, human_prompt, fallback_deadline_s
    )
    if fallback_attempt_reason is None:
        fallback_validated, fallback_validation_reason = _validate(fallback_text)
        if fallback_validated is not None:
            return SynthesisResult(
                text=fallback_validated,
                reason=None,
                model_used="fallback",
                elapsed_ms=_elapsed_ms(),
            )
        fallback_reason: str | None = fallback_validation_reason
    else:
        fallback_reason = fallback_attempt_reason

    return SynthesisResult(
        text=None,
        reason=fallback_reason or "fallback-failed",
        model_used=None,
        elapsed_ms=_elapsed_ms(),
    )


# ── Prompt formatting ─────────────────────────────────────────────────


# Maximum extract citation index. Beyond ⑳ we'd need bracket-number
# fallback per ADR; not implemented in v1 since the depth cap is 6.
_CIRCLED_INDICES = [chr(0x2460 + i) for i in range(20)]  # ①..⑳


def _format_human_prompt(query: str, extracts: list[ExtractedSource]) -> Any:
    """Build the single HumanMessage that the synthesis prompt expects.

    Per the synthesis-prompt doc "Human prompt" section: includes the
    user query and one block per extracted source with citation index,
    domain, domain-class, recency tag, title, and the extracted text.
    """
    from langchain_core.messages import HumanMessage

    lines = [f"User query: {query}", "", "Extracts (cite by ①②③… index):", ""]

    for i, source in enumerate(extracts):
        if i >= len(_CIRCLED_INDICES):
            break  # cap at ⑳; the formatter handles the tail
        citation = _CIRCLED_INDICES[i]
        ranked = source.fetch_outcome.ranked
        domain = _domain_from_url(ranked.canonical_url)
        recency = ranked.published_date or "undated"
        header = f"【{citation}】 {domain} [{ranked.domain_class} · {recency}]"
        lines.append(header)
        # #1842: surface a content-declared affiliation disclaimer so the
        # synthesis does not present an unaffiliated/unofficial source as
        # authoritative. classify_domain only sees the URL — this catches
        # an official-LOOKING domain whose own text disclaims affiliation.
        disclaimer = detect_affiliation_disclaimer(source.extracted_text or "")
        if disclaimer:
            lines.append(
                f"⚠ AUTHORITY: this source self-identifies as UNAFFILIATED/UNOFFICIAL "
                f'("{disclaimer}") — do NOT describe it as an official source or '
                f"attribute statements to an official platform based on it."
            )
        lines.append(f"Title: {ranked.title or '(no title)'}")
        lines.append("")
        body = source.extracted_text or "(no content available — snippet only)"
        lines.append(body)
        lines.append("")
        lines.append("---")
        lines.append("")

    return HumanMessage(content="\n".join(lines))


def _domain_from_url(url: str) -> str:
    """Best-effort registered-domain extraction for prompt formatting."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except (TypeError, ValueError):
        return "(unknown)"
    if host.startswith("www."):
        host = host[4:]
    return host or "(unknown)"


# ── Single attempt against a model ───────────────────────────────────


async def _attempt(llm: Any, human_prompt: Any, deadline_s: float) -> tuple[str | None, str | None]:
    """Run one synthesis attempt against *llm*.

    Returns ``(text, None)`` on success or ``(None, reason)`` on
    failure. ``reason`` is one of:

    * ``"timeout"`` — the LLM call exceeded *deadline_s*.
    * ``"exception:<TypeName>"`` — an unhandled exception bubbled out
      of the LLM call.
    * ``"empty-response"`` — the LLM call completed within budget but
      returned empty / whitespace-only content (or
      ``generate_summary``'s internal safety net fired).

    The pre-refactor version conflated all three into ``None`` and
    the caller labelled everything ``"empty-response"`` regardless of
    cause — see Bug I in the cogtrix41c investigation report.
    """
    import asyncio

    def _call() -> str | None:
        # generate_summary has its own internal ``timeout_seconds`` guard
        # that returns None on timeout. We set it slightly above our
        # outer asyncio deadline so the outer ``wait_for`` catches the
        # timeout case first and we can classify it explicitly. If
        # asyncio.wait_for fires the inner thread is abandoned, but
        # bounded by generate_summary's own timeout a couple of seconds
        # later — wasteful but capped.
        return generate_summary(
            llm,
            [human_prompt],
            existing_summary=None,
            purpose="web_search_synthesis",
            timeout_seconds=int(deadline_s) + 2,
        )

    try:
        text = await asyncio.wait_for(asyncio.to_thread(_call), timeout=deadline_s)
    except TimeoutError:
        log.info("synthesis attempt timed out after %.1fs", deadline_s)
        return None, "timeout"
    except Exception as exc:  # noqa: BLE001
        log.warning("synthesis attempt raised: %s", exc)
        return None, f"exception:{type(exc).__name__}"

    if text is None or not text.strip():
        # ``generate_summary`` returned None/empty within our budget —
        # either the LLM actually returned empty content, or its
        # internal safety nets (timeout / exception) fired. The latter
        # are already logged by ``generate_summary`` itself; we use a
        # generic label here because we can't reliably distinguish at
        # this layer without parsing logs.
        return None, "empty-response"
    return text, None


# ── Validation pipeline ──────────────────────────────────────────────


def _validate(text: str | None) -> tuple[str | None, str | None]:
    """Run the four post-call validators on *text*.

    Returns ``(processed_text, None)`` on success or ``(None, reason)``
    when the synthesis must be dropped (caller invokes fallback or
    falls through to Sources-only).
    """
    if text is None or not text.strip():
        return None, "empty-response"

    # 1) URL line-drop. Drop any line containing :// or www.
    cleaned = _drop_url_lines(text)

    # 2) Citation-presence regex on Key-findings statement lines. Drop
    #    uncited lines; if >50% of statement lines fail, drop the
    #    whole synthesis.
    cleaned, citation_status = _enforce_citation_presence(cleaned)
    if citation_status == "majority-uncited":
        return None, "citation-majority-uncited"

    # 3) Schema check — must have either Key findings (normal case) or
    #    only Gaps (Rule-8 case). Section order must be Key findings →
    #    Disagreements → Gaps when multiple sections present.
    schema_ok = _check_schema(cleaned)
    if not schema_ok:
        return None, "schema-invalid"

    # 4) Length cap — truncate at the soft cap if hard cap exceeded.
    cleaned = _truncate_overlong(cleaned)

    return cleaned, None


def _drop_url_lines(text: str) -> str:
    """Drop every line containing a URL marker. Whole-line drop, not
    substring strip — ADR-0056 calls this out explicitly to avoid
    grammatically broken sentences.
    """
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if _URL_MARKER_RE.search(line):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        log.info("synthesis URL-line drop: %d line(s) removed", dropped)
    return "\n".join(kept)


_CITATION_DROP_THRESHOLD = 0.8
"""Fraction of Key-findings statement lines that may lack inline
citations before the synthesis is rejected as untrustworthy.

Set to 0.8 (was 0.5) after cogtrix37 testing showed qwen3-coder and
similar small open-weight models reliably produce coherent prose
synthesis but emit inline ``[①]`` brackets on roughly 30-50% of lines
rather than ~90%+. The 50% threshold rejected almost every
small-model synthesis, even when the actual content was sourced
correctly and the Sources block was present below.

The guard is still meaningful at 0.8: a synthesis with *zero*
citations is almost certainly hallucinated content and should be
dropped. A synthesis with even one in five lines citing a source
is connected enough to the input extracts that the agent can
verify-by-reading-Sources if needed.
"""


def _enforce_citation_presence(text: str) -> tuple[str, str]:
    """Drop Key-findings statement lines that lack a citation.

    Returns ``(processed_text, status)`` where status is one of:
      * ``"ok"`` — no issues OR drops within tolerance.
      * ``"majority-uncited"`` — more than
        ``_CITATION_DROP_THRESHOLD`` of Key-findings statement lines
        lacked citations; caller treats as synthesis-failure.
    """
    lines = text.splitlines()
    in_key_findings = False
    in_subtopic = False
    total_statements = 0
    dropped_statements = 0
    out: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        # Section header transitions.
        if stripped.startswith("## "):
            in_key_findings = stripped == _SECTION_KEY_FINDINGS
            in_subtopic = False
            out.append(line)
            continue
        if stripped.startswith("### "):
            in_subtopic = in_key_findings  # only within Key findings
            out.append(line)
            continue

        # Outside Key-findings sub-topics we don't enforce citation
        # presence (Disagreements / Gaps have their own structure).
        if not in_subtopic or not stripped:
            out.append(line)
            continue

        total_statements += 1
        if _CITATION_RE.search(stripped):
            out.append(line)
        else:
            dropped_statements += 1
            log.info("synthesis citation-drop: %r", stripped[:80])

    if total_statements > 0:
        drop_ratio = dropped_statements / total_statements
        if drop_ratio > _CITATION_DROP_THRESHOLD:
            return text, "majority-uncited"

    return "\n".join(out), "ok"


def _check_schema(text: str) -> bool:
    """Validate the section layout per the synthesis-prompt doc Rule 5.

    Accepted shapes:
      * Has ``## Key findings`` (optionally followed by Disagreements
        and/or Gaps in that order).
      * Rule-8 case: only ``## Gaps`` present.

    Disagreements without Key findings is invalid.
    """
    has_kf = _SECTION_KEY_FINDINGS in text
    has_disagreements = _SECTION_DISAGREEMENTS in text
    has_gaps = _SECTION_GAPS in text

    if not has_kf:
        # Rule-8 case: only Gaps allowed.
        if has_disagreements:
            return False
        return has_gaps

    # Validate order when multiple sections present.
    kf_idx = text.index(_SECTION_KEY_FINDINGS)
    if has_disagreements:
        disagree_idx = text.index(_SECTION_DISAGREEMENTS)
        if disagree_idx <= kf_idx:
            return False
        if has_gaps:
            gaps_idx = text.index(_SECTION_GAPS)
            if gaps_idx <= disagree_idx:
                return False
    elif has_gaps:
        gaps_idx = text.index(_SECTION_GAPS)
        if gaps_idx <= kf_idx:
            return False

    return True


def _truncate_overlong(text: str) -> str:
    """Truncate at the soft cap if the hard cap is exceeded.

    Word count is whitespace-token-based (no stop-word stripping). The
    truncation marker tells downstream readers something was cut.
    """
    words = text.split()
    if len(words) <= _LENGTH_HARD_WORD_CAP:
        return text
    truncated = " ".join(words[:_LENGTH_WORD_CAP])
    log.info(
        "synthesis truncated %d → %d words",
        len(words),
        _LENGTH_WORD_CAP,
    )
    return f"{truncated} … [synthesis truncated]"


# ── Public helper for callers ────────────────────────────────────────


def get_fallback_model_name() -> str:
    """Resolve the smaller-model name for the synthesis retry path.

    Returns ``$SUMMARIZER_FALLBACK_MODEL`` or the default
    ``claude-haiku-4-5``. The synthesiser doesn't construct the
    LangChain ``ChatModel`` itself — the orchestrator (PR-E) owns
    that and passes the constructed model in via ``llm_fallback``.
    """
    return os.getenv("SUMMARIZER_FALLBACK_MODEL", "claude-haiku-4-5")


__all__ = [
    "SynthesisResult",
    "get_fallback_model_name",
    "synthesise",
]
