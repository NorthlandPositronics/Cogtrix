"""Response-content detectors for the orchestration router.

Extracted from ``cogtrix_core/orchestration/graph.py`` as the second step of the
graph.py 5-module split proposed by the /forge audit
(architect finding A1.2, 2026-05-23). All six detectors operate on a
single AIMessage's content (plus, in two cases, the surrounding
``messages`` list) and return ``bool``. They have no graph-build /
langgraph-runtime dependency.

Detectors (in order of original definition):

* :func:`_is_action_intent` — sentence-local intent-lead + tool-verb
  match. Used to catch "I'll create the file" with no tool call.
* :func:`_has_incompleteness_signal` — narrows intent detection to
  multi-step sentences ("first", "to start", "step 1").
* :func:`_is_hallucinated_completion` — past-tense list-item verbs
  whose stem matches an *available* but never-called tool name.
* :func:`_looks_like_phantom_tool_markup` — XML or JSON-array tool-call
  markup emitted as final text.
* :func:`_looks_like_markdown_phantom_report` — fabricated structured
  markdown report (table + numbered section + claim-of-action signal).
* :func:`_looks_like_fabricated_success_after_tool_errors` — explicit
  success claim immediately after a contiguous block of tool errors.

Also re-exported: the regex constants each detector consumes, the
``_unwrap_code_fence`` helper, and ``_stuck_detection_headline``
(used both inside this module and by other graph.py code paths —
graph.py re-imports it).

Leading underscores on names are preserved from the original graph.py
module for back-compat with existing call sites in the orchestration
nodes and the test suite.
"""

from __future__ import annotations

import re
import typing
from typing import Any

from cogtrix_core.orchestration.tool_message_kinds import is_resolution_failure_message

# ── _is_action_intent regex set ───────────────────────────────────────

# Phrases that introduce a planned future action
_INTENT_LEAD_RE = re.compile(
    r"\b(?:"
    # First-person modal / future forms
    r"I(?:"
    r"'ll\b"  # I'll
    r"|\s+will\b"  # I will
    r"|\s+am\s+going\s+to\b"  # I am going to
    r"|'m\s+going\s+to\b"  # I'm going to
    r"|'m\s+about\s+to\b"  # I'm about to
    r"|\s+need\s+to\b"  # I need to
    r"|\s+have\s+to\b"  # I have to
    r"|\s+should\b"  # I should
    r"|\s+must\b"  # I must
    r"|\s+will\s+now\b"  # I will now
    r"|'ll\s+now\b"  # I'll now
    r"|'m\s+now\b"  # I'm now [verb-ing]
    r"|'ll\s+(?:proceed|go\s+ahead|start|begin|continue)\b"  # I'll proceed/...
    r")"
    # Imperative / collaborative
    r"|Let(?:'s|\s+(?:me|us))\b"  # Let me / Let's / Let us
    # Temporal modifiers + intent
    r"|Now\s+(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # Now I'll / Now let me
    r"|(?:First|Next|Then|Finally|Additionally|Also)(?:\s*,)?\s+"
    r"(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # First/Next/Then ... I'll
    # Implicit future
    r"|(?:Going|About)\s+to\b"  # Going to / About to
    r"|Time\s+to\b"  # Time to
    r"|I\s+can\s+now\b"  # I can now
    r")",
    re.IGNORECASE,
)

# Verb stems that indicate a tool-requiring operation.
# Long stems (5+ chars) use \w{0,8} for inflection coverage.
# Short/ambiguous stems use explicit endings to prevent overmatch.
_TOOL_VERB_RE = re.compile(
    r"\b(?:"
    # File / content operations — long stems
    r"creat\w{0,8}|generat\w{0,8}|overwrite\w{0,8}|append\w{0,8}"
    r"|delet\w{0,8}|remov\w{0,8}|replac\w{0,8}|insert\w{0,8}|rename\w{0,8}"
    r"|modif\w{0,8}|patch\w{0,8}|refactor\w{0,8}"
    # Read / fetch
    r"|fetch\w{0,8}|retriev\w{0,8}|download\w{0,8}" r"|search\w{0,8}|crawl\w{0,8}|scrape\w{0,8}"
    # Build / execution
    r"|build\w{0,8}|compil\w{0,8}|execut\w{0,8}|launch\w{0,8}"
    r"|install\w{0,8}|deploy\w{0,8}|configur\w{0,8}|initializ\w{0,8}"
    r"|bootstrap\w{0,8}|provision\w{0,8}|scaffold\w{0,8}"
    # Code / dev
    r"|implement\w{0,8}|develop\w{0,8}|debug\w{0,8}|resolv\w{0,8}"
    # Network / API
    r"|upload\w{0,8}|commit\w{0,8}|invok\w{0,8}|submit\w{0,8}|request\w{0,8}"
    r"|publish\w{0,8}|broadcast\w{0,8}"
    # Data processing
    r"|transform\w{0,8}|extract\w{0,8}|analyz\w{0,8}|analys\w{0,8}"
    r"|comput\w{0,8}|calculat\w{0,8}|process\w{0,8}|pars\w{0,8}"
    r"|export\w{0,8}|migrat\w{0,8}|convert\w{0,8}|classif\w{0,8}"
    # Infra
    r"|register\w{0,8}|connect\w{0,8}|verif\w{0,8}|inspect\w{0,8}|examin\w{0,8}" r"|clone\w{0,8}"
    # Short stems — explicit inflections only
    r"|writ(?:e|es|ing|ten)\b"
    r"|read(?:s|ing)?\b"
    r"|open(?:s|ing|ed)?\b"
    r"|load(?:s|ing|ed)?\b"
    r"|run(?:s|ning)?\b"
    r"|start(?:s|ing|ed)?\b"
    r"|send(?:s|ing)?\b|sent\b"
    r"|post(?:s|ing|ed)?\b"
    r"|call(?:s|ing|ed)?\b"
    r"|pull(?:s|ing|ed)?\b"
    r"|push(?:es|ing|ed)?\b"
    r"|fix(?:es|ing|ed)?\b"
    r"|test(?:s|ing|ed)?\b"
    r"|edit(?:s|ing|ed)?\b"
    r"|update(?:s|d|ing)?\b"
    r"|sav(?:e|es|ing|ed)\b"
    r"|stor(?:e|es|ing|ed)\b"
    r"|add(?:s|ing|ed)?\b"
    r"|cod(?:e|es|ing|ed)\b"
    r"|defin(?:e|es|ing|ed)\b"
    r"|list(?:s|ing|ed)?\b"
    r"|output(?:s|ting|ted)?\b"
    r"|check(?:s|ing|ed)?\b"
    # Multi-word constructions
    r"|set\s+up\b" r"|spin\s+up\b" r"|look\s+up\b" r"|wire\s+up\b" r"|stand\s+up\b" r")\b",
    re.IGNORECASE,
)


# Phrases that LOOK like intent leads but are actually conversational.
# "let me know" is the most common false positive.
_INTENT_FALSE_POSITIVE_RE = re.compile(
    r"""
    \b(?:
        let\s+me\s+know\b                   # "let me know if..."
      | let\s+me\s+(?:explain|clarify|describe|summarize|outline|walk\s+you\s+through)\b
      | I\s+(?:want\s+to|need\s+to|would\s+like\s+to)\s+(?:mention|note|clarify|explain|point\s+out)\b
      | I\s+should\s+(?:mention|note|clarify|point\s+out)\b
      | please\s+(?:note|be\s+aware|keep\s+in\s+mind)\b
      | (?:^|(?:please|just|do)\s+)note\s+that\b    # "note that" / "please note that"
      | (?:^|(?:please|just)\s+)keep\s+in\s+mind\b  # sentence-initial or after polite prefix
      | (?:it(?:'s|\s+is)\s+)?worth\s+(?:noting|mentioning)\b
      | it(?:'s|\s+is)\s+(?:worth|important\s+to)\s+(?:note|mention)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_action_intent(message: Any) -> bool:
    """Return True when the model describes a planned action but emits no tool calls.

    Checks for an intent-lead phrase (``I'll``, ``Let me``, ``Going to``, etc.)
    paired with a tool-action verb (``create``, ``run``, ``fetch``, etc.)
    **in the same sentence**.  The sentence-locality requirement prevents false
    positives where the lead phrase appears in a closing remark (e.g. "Feel
    free to let me know") and the verb appears elsewhere (e.g. "reading" in a
    weather table).
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False

    # Split into sentences (on . ! ? or newlines) and check each independently.
    # This prevents a verb in one sentence from pairing with an intent lead
    # in a completely different sentence.
    sentences = re.split(r"[.!?\n]", text)
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        has_intent = bool(_INTENT_LEAD_RE.search(s))
        has_verb = bool(_TOOL_VERB_RE.search(s))
        if not (has_intent and has_verb):
            continue
        # A real intent+verb pair exists in this sentence.  Only suppress it
        # when the FP phrase is the SOLE intent-shaped construct — i.e. it
        # appears but there is no additional "I'll / Let me / Going to" lead-in
        # that could pair with the verb.  This prevents "please note that I'll
        # run the build" from being silently dropped.  The original "let me
        # know" case still suppresses correctly because "let me know" fires the
        # FP regex AND does not typically combine with an action verb in the
        # same sentence.
        fp_match = _INTENT_FALSE_POSITIVE_RE.search(s)
        if fp_match:
            # Re-check: is the intent lead-in ONLY the FP phrase itself, or is
            # there a genuine action lead-in beyond the FP match?
            text_after_fp = s[fp_match.end() :]
            text_before_fp = s[: fp_match.start()]
            if not (
                _INTENT_LEAD_RE.search(text_after_fp) or _INTENT_LEAD_RE.search(text_before_fp)
            ):
                continue  # FP phrase is the only lead-in — suppress
        return True
    return False


# ── _is_refusal ───────────────────────────────────────────────────────
#
# A deliberate decline-to-act is NOT "intention without action" — it is a
# considered non-action. The action-intent recovery must never nudge the
# model to "proceed and call the tool(s)" on top of a refusal: doing so
# converts an honest refusal (e.g. declining a system-prompt-forbidden
# action such as an unauthorized payment) into the forbidden action.
# See #1851. Patterns target first-person declines and authorization
# blockers, and exclude soft hedges ("I cannot guarantee …") so a refusal
# verb inside an otherwise-actionable answer does not over-trip.

_REFUSAL_RE = re.compile(
    r"(?im)(?:"
    # First-person decline to act — but NOT a soft "cannot guarantee/
    # ensure/promise/confirm" hedge (those qualify an answer, they don't
    # decline the action).
    r"\bi\s+(?:can(?:no|')?t|cannot|won'?t|will\s+not|must\s+not|"
    r"should\s+not|shouldn'?t|mustn'?t|(?:'m|am)\s+not\s+able\s+to|"
    r"(?:'m|am)\s+unable\s+to|(?:'m|am)\s+not\s+going\s+to)\s+"
    r"(?!guarantee|ensure|promise|be\s+sure|be\s+certain|confirm\b)\w+"
    r"|\bi\s+(?:have\s+to|need\s+to|will\s+have\s+to|must)\s+decline\b"
    r"|\b(?:i\s+am|i'm)\s+not\s+authori[sz]ed\b"
    r"|\bnot\s+authori[sz]ed\s+to\b"
    r"|\bwithout\s+(?:proper\s+|prior\s+)?(?:approval|authori[sz]ation)\b"
    r"|\b(?:require[sd]?)\s+(?:prior\s+|proper\s+)?(?:approval|authori[sz]ation)\b"
    r"|\bapproval\s+is\s+required\b"
    r")"
)


# ── _is_sycophantic_prefix (#1713) ────────────────────────────────────
#
# Anchors on ``^\s*`` because the prefix MUST be at the very start of the
# response — embedded "you're right" further in the text (e.g. quoting
# the user, discussing rights, etc.) is legitimate. The trailing class
# ``[\s\-—–,.!:;]+`` consumes the punctuation/separator the model uses
# to glue the prefix to the actual content ("You're right - let me",
# "You're right, let me", "You're right. Let me", "I apologize — I'll").
#
# We deliberately do NOT consume an "I apologize for X" extension here.
# Earlier prototype matched ``i apologize(?:\s+for[^.!?]{0,80})?`` which
# greedily ate the next clause when the model wrote "I apologize for the
# inconvenience but ..." — the strip would remove the entire 80-char
# span and leave a malformed remainder that broke downstream scoring on
# shard D × kimi-k2-5 (PR #1731 first iteration). The conservative form
# below requires the verb-phrase prefix to be immediately followed by a
# separator (whitespace + dash, comma, period, etc.). Apology clauses
# with an inline "for X" clause stay intact; only the bare-verb-then-
# separator pattern matches.
_SYCOPHANTIC_PREFIX_RE = re.compile(
    r"^\s*(?:"
    # ── Multi-word openers (unambiguous: any whitespace separator is OK) ──
    # Existing variants — kept identical to preserve current behaviour.
    r"(?:"
    r"you're absolutely right"
    r"|you are absolutely right"
    r"|you're right"
    r"|you are right"
    r"|you're raising an important point"
    r"|you're raising a (?:good|valid|fair) point"
    r"|i sincerely apologize"
    r"|i apologize"
    r"|my apologies"
    # #1866 additions — variants surfaced in the Q3 holistic-test exchange
    # against cogtrix:release-next @ 2bb52c7. All multi-word; each
    # performs the same validate-the-user-before-responding role as the
    # original set.
    r"|you're correct"
    r"|you are correct"
    r"|you make a (?:good|valid|fair) point"
    r"|that's a (?:good|valid|fair) point"
    r"|good point"
    r"|fair enough"
    r")"
    r"[\s\-—–,.!:;]+"
    r"|"
    # ── Bare-word openers (#1866) ─────────────────────────────────────────
    # ``Correct`` / ``Indeed`` / ``Absolutely`` standing alone are
    # substantive in many contexts (``Correct configuration requires …``,
    # ``Indeed an interesting question, but the answer is …``,
    # ``Absolutely amazing — let me explain``). Restrict the bare-word
    # form to cases where it is IMMEDIATELY followed by a punctuation
    # separator (``.``, ``,``, ``—``, ``-``, ``!``, ``:``, ``;``) —
    # optionally preceded by whitespace. This catches the validation
    # opener (``Correct. The path is …``, ``Indeed, this is the file …``,
    # ``Absolutely! I'll do it.``) without false-firing on substantive
    # uses that continue with another word.
    r"(?:correct|indeed|absolutely)\s*[,.\-—–!:;]+" r")",
    re.IGNORECASE,
)


def _is_sycophantic_prefix(message: Any) -> bool:
    """Return True when the response opens with a sycophantic validation
    phrase (``You're absolutely right``, ``I apologize``, …).

    The system-prompt rule in ``cogtrix_core/agent/core.py:build_system_prompt``
    forbids these prefixes, but RLHF-tuned chat models bypass the rule
    under user pushback and emit them anyway (Bug G #1713). A response
    that carries a tool call is never sycophancy — the prefix is a
    final-answer artifact.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    return bool(_SYCOPHANTIC_PREFIX_RE.match(content))


def _is_refusal(message: Any) -> bool:
    """Return True when the response is a deliberate decline/refusal to act.

    Used to suppress the action-intent recovery nudge: a refusal is a
    *considered* non-action, not a forgotten one. Nudging the model to
    "proceed, call the tool(s)" on top of it can turn an honest refusal
    (e.g. declining a forbidden / unauthorized action) into that action
    (#1851). A response that carries a tool call is never a refusal here.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    return bool(_REFUSAL_RE.search(content))


def text_is_refusal(text: str) -> bool:
    """String-form companion to :func:`_is_refusal`.

    Returns True when *text* contains a deliberate decline/refusal
    pattern (``I cannot pay``, ``I am not authorized``, ``approval is
    required``, etc.).  Used by the response-content detectors in
    ``cogtrix_core/orchestration/verification.py`` to short-circuit on refusals
    — see #1960.  A correctly-formed safety refusal is NOT a
    fabrication / unverified-entity / unsupported-attribution case;
    the recovery layer must let it pass.
    """
    if not text or not isinstance(text, str):
        return False
    return bool(_REFUSAL_RE.search(text))


# ── _has_incompleteness_signal ────────────────────────────────────────

# Phrases that signal a multi-step task is incomplete — the model used
# sequential language ("first", "to start") but stopped before finishing.
# Only fires when paired with an action-intent detection (intent lead +
# tool verb in the same sentence), which guards against false positives
# from purely conversational uses of these words.
_INCOMPLETENESS_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"first\b"  # "create the PO first" → there's a second step
    r"|to\s+start\b"  # "to start, let me..."
    r"|to\s+begin\b"  # "to begin with..."
    r"|initially\b"  # "initially create the PO"
    r"|step\s*1\b"  # "step 1: create the PO"
    r")",
    re.IGNORECASE,
)


def _has_incompleteness_signal(text: str) -> bool:
    """Return True when the text signals an incomplete multi-step operation.

    Detects language that implies the model planned more steps but
    stopped early — e.g. "first", "to start", "step 1".  Only relevant
    when ``_is_action_intent`` has already identified an intent-lead
    + tool-verb pair in the same sentence; this function narrows the
    nudge from generic to specific.
    """
    if not text or not isinstance(text, str):
        return False
    # Restrict to the sentence containing the incompleteness signal
    # so a "first" in a generic intro doesn't contaminate a later sentence.
    sentences = re.split(r"[.!?\n]", text)
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if _INCOMPLETENESS_SIGNAL_RE.search(s) and _INTENT_LEAD_RE.search(s):
            return True
    return False


# ── _is_hallucinated_completion ───────────────────────────────────────

# Past-tense action verbs at the start of numbered or bulleted list items.
# Used to detect hallucinated-completion summaries where the model claims
# "1. Notified the VP..." without actually calling notify_approver.
_PAST_TENSE_LIST_VERB_RE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•])\s+([A-Z][a-z]{2,}(?:ied|ed))\b",
    re.MULTILINE,
)


def _is_hallucinated_completion(
    message: Any,
    messages: typing.Sequence[Any],
    available_tool_names: list[str],
) -> bool:
    """Return True when the final response claims past-tense completion
    of an action whose corresponding tool was never actually called.

    Pattern observed on gpt-oss-20b for finance_invoice_approval_workflow:

        1. Classified as "medium" tier (amount $12,500).
        2. Routed to the VP-level approval queue.
        3. Notified the VP for review.   ← notify_approver never called

    The standard ``_is_action_intent`` detector matches only future-tense
    intent leads ("I'll notify", "Let me..."); past-tense passive claims
    slip past it.  A weaker/cheaper model that confabulates a completion
    summary without executing every required step would otherwise reach
    the user, fail scenario assertions, and burn the run budget with no
    recovery cycle.

    Heuristic:
    1. Find past-tense verbs at the start of numbered/bulleted list items
       in the final text response.
    2. For each verb, derive a stem by stripping ``ed`` / ``ied``.
    3. If the stem appears in an *available* tool name, AND that tool was
       not called in the conversation, treat as hallucinated completion.

    Stem-matching is intentionally permissive ("notif" matches
    ``notify_approver``); the false-positive ceiling is bounded by the
    "in the available tool list" check — a verb that doesn't correspond
    to any tool the agent could have called never triggers.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False

    verbs = _PAST_TENSE_LIST_VERB_RE.findall(content)
    if not verbs:
        return False

    called_names: set[str] = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else None
            if name:
                called_names.add(name.lower())

    available_lower = [t.lower() for t in available_tool_names]

    for verb in verbs:
        stem = re.sub(r"(?:ied|ed)$", "", verb.lower())
        if len(stem) < 4:
            # Too short to disambiguate ("Read"-ed → "read" is fine but
            # "Set"-? doesn't end in ed/ied so won't match anyway).
            continue
        for tool_name in available_lower:
            if stem in tool_name and tool_name not in called_names:
                return True
    return False


# ── _looks_like_phantom_tool_markup ───────────────────────────────────

_PHANTOM_TOOL_MARKUP_RE = re.compile(
    r"<\s*(?:function_calls|invoke|Call|antml:function_calls)\b",
    re.IGNORECASE,
)

# DSML-style tool-call markup leaked into final text (#1862). Some open-
# weights model tokenizers (deepseek-v4, certain Qwen variants) wrap their
# control tokens with the fullwidth vertical bar U+FF5C ('｜') — e.g.
# ``<｜｜DSML｜｜tool_calls>…<｜｜DSML｜｜invoke name="http_get">…``
# or the simpler ``<｜tool_call｜>…``. The fullwidth bar is extremely
# rare in legitimate prose, so anchoring on ``<`` + ``｜`` + a tool-call
# keyword stays high-precision. Without this detection the response was
# silently treated as a final answer and `handle_phantom` was bypassed.
_PHANTOM_DSML_MARKUP_RE = re.compile(
    r"<\s*｜+\s*(?:DSML\s*｜+\s*)?"
    r"(?:tool_calls?|invoke|parameter|function_call|fnctl|tool_use|tool_outputs?)\b"
    r"|<\s*｜\s*tool_call\s*｜\s*>",
    re.IGNORECASE,
)

# JSON array of {"tool": "...", "arguments": {...}} objects emitted as literal text
# instead of structured tool_calls (e.g. model hallucinates tool calls as JSON).
_PHANTOM_JSON_TOOL_RE = re.compile(
    r'"tool"\s*:\s*"[^"\n]{1,120}"',
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_+-]+)?\s*\n(?P<body>.*)\n```\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_code_fence(content: str) -> str:
    """Return the body of a fenced code block, or the original content."""
    match = _CODE_FENCE_RE.match(content)
    if match:
        return match.group("body")
    return content


def _looks_like_phantom_tool_markup(message: Any) -> bool:
    """Return True when plain text resembles raw tool-call markup.

    Catches two families:
    - XML-style:  ``<function_calls>``, ``<invoke>``, ``<Call>``, etc.
    - JSON-style: ``[{"tool": "...", "arguments": {...}}]`` arrays in text

    Both should be treated as phantom tool-call responses so the graph can
    recover instead of accepting hallucinated markup as final prose.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    candidate = _unwrap_code_fence(content)
    if _PHANTOM_TOOL_MARKUP_RE.search(candidate):
        return True
    # #1862: DSML / open-weights tokenizer-control variants
    # (<｜｜DSML｜｜tool_calls>…, <｜tool_call｜>…). The fullwidth bar is
    # rare enough in prose that anchoring on it is high-precision.
    if _PHANTOM_DSML_MARKUP_RE.search(candidate):
        return True
    # JSON phantom: require the message to look like raw JSON at the start so
    # prose that merely mentions {"tool": ...} or "arguments" does not trip it.
    stripped = candidate.lstrip()
    if not stripped or stripped[0] not in "[{":
        return False
    return bool(_PHANTOM_JSON_TOOL_RE.search(candidate) and '"arguments"' in candidate)


# ── _looks_like_markdown_phantom_report + success/error helpers ───────

_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|.+\|", re.MULTILINE)
_NUMBERED_SECTION_RE = re.compile(r"^#{1,4}\s+\d+\.", re.MULTILINE)

# Fake-tool-output content signals. The original
# ``_looks_like_markdown_phantom_report`` fired on *any* response with a
# markdown table + numbered headings, which caught every well-structured
# educational answer (cogtrix45.log turn 3 — Bug K). The fix is to also
# require a "claim-of-action" content signal so the detector matches
# fabricated tool reports without tripping on legitimate prose that
# happens to be well structured.
#
# Each alternative below corresponds to a distinct fabrication pattern:
#
# 1. Bullet-point past-tense claim with a retrieval verb:
#    ``- Retrieved last 8 messages``
# 2. First-person past-tense action with a typical object phrase:
#    ``I retrieved the 8 most recent messages``
# 3. Report-style section headers:
#    ``Sources:``, ``References:``, ``Tool Results:``, ``Search Results:``
# 4. "According to" tool/search/results claim:
#    ``According to my search, …``
# 5. Results/search/data acting as the subject of a reporting verb:
#    ``The results show X``, ``Search returned Y``
#
# What this deliberately does NOT match (the false-positive corpus):
# - Tutorials describing a *technique* in imperative or infinitive form
#   ("Before each turn, retrieve top-3 chunks") — verb not at bullet
#   start and not in first-person past tense.
# - Reflective prose ("Here are the core strategies I use") — "I use"
#   is not in the past-tense verb list.
# - Apologetic prefixes ("I'll address it directly") — same reason.
_FAKE_TOOL_OUTPUT_SIGNAL_RE = re.compile(
    r"(?im)"
    # 1. Bullet-point past-tense claim of action.
    r"^\s*[-*]\s+(?:Retrieved|Fetched|Found|Got|Pulled|Searched|Queried|Returned|Listed)\s+\S"
    r"|"
    # 2. First-person past-tense + typical object phrase ("I retrieved
    #    the/a/N/all/recent …"). The trailing alternation prevents
    #    matching educational phrasing like "I find that …".
    r"\bI\s+(?:retrieved|fetched|queried|searched|found|looked\s+up|checked|ran|executed|pulled)"
    r"\s+(?:the|a|an|\d+|all|some|recent|last|top)\b"
    r"|"
    # 3. Report-style section headers (line on its own).
    r"^\s*(?:Sources?|References?|Search Results?|Tool Results?|Output)\s*:\s*$"
    r"|"
    # 4. "According to my/the search/results/findings".
    r"\baccording\s+to\s+(?:my|the)\s+(?:search|query|results?|findings?)\b"
    r"|"
    # 5. Reporting-verb structure with results/search/data as subject.
    r"\b(?:results?|search|query|data)\s+(?:show(?:s|ed)?|indicate(?:s|d)?|return(?:ed|s)?)\b"
)

_SUCCESS_CLAIM_RE = re.compile(
    r"(?:✅|:white_check_mark:|\bsuccess(?:ful|fully)?\b|\b(?:created|completed|finished)\s+successfully\b)",
    re.IGNORECASE,
)
_NEGATED_SUCCESS_RE = re.compile(
    r"\b(?:not|no|failed|unable|cannot|can't|couldn't|didn't)\b.{0,24}"
    r"(?:success(?:ful|fully)?|created|completed|finished)\b",
    re.IGNORECASE,
)
_TOOL_ERROR_INDICATORS = (
    "error:",
    "failed",
    "http error",
    "timed out",
    "permission denied",
    "access denied",
    "not found",
    "tool not loaded",
    "path outside allowed",
    "cannot",
)

# ── #1869: fabricated-action-success-without-tool-call detector ────────
#
# Sibling to :data:`_SUCCESS_CLAIM_RE` (which only matches the literal
# `successful` / `created` / `completed` / `finished` lexicon). This
# regex captures definite-tense side-effecting-action completion claims
# in flowing prose — the Q9/Q10 holistic-test reproducers
# (cogtrix:release-next @ 2bb52c7):
#
#   Q9:  "The file /workspace/...verification.py has been deleted from
#         the codebase as requested."
#   Q10: "The file /workspace/...text.py already contains the safe_divide
#         function based on the successful write operations in this
#         session."
#
# The verb stems below cover the destructive / mutating operations that
# Cogtrix's catalog can perform (write/patch/append/...) plus broader
# action-verbs that the LLM tends to claim completion for. Each
# alternative requires a tense-marking auxiliary (`has been`, `is now`,
# `I have`, `I've`, etc.) before the verb so we don't false-positive on
# habitual / future / conditional uses ("I will delete X", "we delete
# files daily", "if I create the file...").
_SIDE_EFFECT_VERB_BASE = (
    r"(?:"
    r"delet\w*|remov\w*|creat\w*|writ\w*|written|wrote|"
    r"sav\w*|install\w*|sent|publish\w*|"
    r"commit\w*|push\w*|"
    r"mov\w*|renam\w*|copi\w*|copy|copied|copying|"
    r"add\w*|updat\w*|modif\w*|"
    r"patch\w*|append\w*|"
    r"overwrote|overwritten|overwrit\w*|"
    r"wip\w*|clear\w*|truncat\w*|drop\w*|"
    r"ran|run\w*|execut\w*|"
    r"chang\w*|edit\w*|"
    r"complet\w*|finish\w*|don\w*|"
    r"merged?"
    r")"
)

_ACTION_COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:"
    # (Has|Have|Had) been [adverb] <verb>  →  "has been deleted"
    rf"(?:has|have|had)\s+been\s+(?:successfully\s+|just\s+|already\s+)?{_SIDE_EFFECT_VERB_BASE}"
    r"|"
    # (Is|Are|Was|Were) [now] [successfully] <verb>  →  "is removed", "was overwritten"
    rf"(?:is|are|was|were)\s+(?:now\s+)?(?:successfully\s+)?{_SIDE_EFFECT_VERB_BASE}" r"|"
    # I/We (have|'ve|just|already|finally|successfully|now) <verb>  →  "I have deleted"
    rf"(?:i|we)\s+(?:have|'ve|already|just|finally|successfully|now)"
    rf"\s+(?:successfully\s+|just\s+|already\s+)?{_SIDE_EFFECT_VERB_BASE}"
    r"|"
    # I've/We've <verb>  →  "I've added"
    rf"(?:i've|we've)\s+(?:successfully\s+|just\s+|already\s+)?{_SIDE_EFFECT_VERB_BASE}" r"|"
    # The <subject> (has been|is|got|was) <verb>  →  Q9 reproducer
    r"(?:the\s+(?:file|files|directory|folder|change|fix|update|function|"
    r"class|method|line|lines|code|content|repo|repository|commit|branch|"
    rf"module|package|symlink|record|entry|config|setting))\s+"
    rf"(?:has\s+been|have\s+been|had\s+been|is|are|got|was|were)"
    rf"\s+(?:successfully\s+)?(?:now\s+)?{_SIDE_EFFECT_VERB_BASE}"
    r"|"
    # Successfully <verb>  →  "Successfully committed"
    rf"successfully\s+{_SIDE_EFFECT_VERB_BASE}" r"|"
    # Q10 smoking-gun: "based on [the] [prior] successful X operation(s)".
    # Allows 1–3 modifier words (the/my/prior/previous/earlier/recent/past)
    # so we catch both "the successful patch operations" and "the prior
    # successful write operations".
    r"based\s+on\s+(?:(?:the|my|prior|previous|earlier|recent|past)\s+){1,3}"
    r"successful\s+\w+\s+operations?"
    r")\b",
    re.IGNORECASE,
)

# Negation guard: must catch "has not been deleted", "I couldn't write",
# "the file isn't created", etc. Wider verb coverage than
# :data:`_NEGATED_SUCCESS_RE` because Q9/Q10's prose uses broader
# side-effecting verbs.
_NEGATED_ACTION_CLAIM_RE = re.compile(
    r"\b(?:not|never|no|cannot|can't|couldn't|wasn't|weren't|hasn't|"
    r"haven't|hadn't|didn't|isn't|aren't|won't|wouldn't|unable\s+to|"
    r"failed\s+to|refuse\s+to)\b"
    rf".{{0,32}}?{_SIDE_EFFECT_VERB_BASE}",
    re.IGNORECASE,
)


def _looks_like_markdown_phantom_report(message: Any) -> bool:
    """Return True when the response is a fabricated structured markdown report.

    The "markdown phantom" variant generates a plausible-looking tool-output
    report (numbered sections, tables) from training-data memory without calling
    any tools.  Unlike XML/JSON phantoms it contains no markup syntax — just
    normal prose and tables — so ``_looks_like_phantom_tool_markup`` misses it.

    Detection signal — all three required:

    1. A markdown table row, and
    2. A numbered section header, and
    3. A claim-of-action content signal (see
       ``_FAKE_TOOL_OUTPUT_SIGNAL_RE``).

    The third condition was added after Bug K (cogtrix45.log turn 3): the
    original two-signal heuristic misclassified well-structured *educational*
    answers as fabricated tool reports, causing the agent to topic-drift back
    to stale conversation history. Fabricated reports almost always contain
    a past-tense claim of retrieval ("Retrieved last 8 messages") or a
    report-style header ("Sources:") — the new signal captures that intent
    without tripping on tutorials that compare techniques in tables.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or len(content) < 80:
        return False
    return bool(
        _MARKDOWN_TABLE_ROW_RE.search(content)
        and _NUMBERED_SECTION_RE.search(content)
        and _FAKE_TOOL_OUTPUT_SIGNAL_RE.search(content)
    )


def _looks_like_fabricated_success_after_tool_errors(
    messages: typing.Sequence[Any],
    last_message: Any,
) -> bool:
    """Return True when final success text contradicts immediately prior tool errors.

    Guard scope is intentionally narrow:
    - only inspects the contiguous ToolMessage block immediately before ``last_message``
    - only fires when *all* of those tool results look like errors
    - only fires when ``last_message`` contains an explicit success claim
    """
    if getattr(last_message, "tool_calls", None):
        return False
    content = getattr(last_message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    if not _SUCCESS_CLAIM_RE.search(content):
        return False
    if _NEGATED_SUCCESS_RE.search(content):
        return False

    i = len(messages) - 2
    recent_tool_messages: list[Any] = []
    while i >= 0 and hasattr(messages[i], "tool_call_id"):
        recent_tool_messages.append(messages[i])
        i -= 1
    if not recent_tool_messages:
        return False

    for tool_msg in recent_tool_messages:
        # #1921 (preferred path): a dispatcher-synthesised ToolMessage carries
        # a ``cogtrix.kind`` marker for any unresolvable / disabled / not-
        # loaded outcome. Substring matching on the prose drifts over time
        # (the "is in the catalog but not loaded" message did not start
        # with the legacy "tool not loaded" indicator and silently bypassed
        # this guard — the #1919 test5 reproducer). The kind check is
        # immune to phrasing changes inside the dispatcher.
        if is_resolution_failure_message(tool_msg):
            continue
        # Substring fallback: real tools that raise / return error strings
        # cannot carry the kind marker without per-tool changes. The
        # legacy indicator allowlist still covers those.
        tool_content = getattr(tool_msg, "content", "")
        if not isinstance(tool_content, str):
            return False
        headline = _stuck_detection_headline(tool_content).lower()
        headline = headline.lstrip()
        if not any(headline.startswith(ind) for ind in _TOOL_ERROR_INDICATORS):
            return False

    return True


# ── #1871: fabricated-tool-error-quote detector ────────────────────────
#
# Polarity-flipped sibling of
# :func:`_looks_like_fabricated_action_success_without_tool_call`. That
# detector catches confident *success* claims with no tool call; this one
# catches confident *error* attributions — the model quotes a verbatim
# error string ("Read-only file system", "Tool not loaded", etc.) and
# presents it as observed tool output, but the quoted span does not appear
# in any ``ToolMessage`` in the current turn. Q13 / Q14 / Q15 of the
# #1869 holistic-test battery produced three different, mutually
# contradictory fabricated error strings across three consecutive turns.
#
# Detection flow:
#   1. Find a lead-in match (:data:`_TOOL_ERROR_QUOTE_LEAD_RE`) — phrases
#      that frame what follows as a quoted error/output from a tool/system.
#   2. Extract the nearest quoted span (:func:`_extract_quoted_span`) in
#      the 200-char window after the lead-in. Quote pairs supported:
#      ASCII double / single / smart double / smart single / backticks.
#   3. Walk back through ``messages`` to collect ``ToolMessage.content``
#      values in the current turn (since the most recent ``HumanMessage``).
#   4. If the quoted span (case-insensitive) appears as a substring of any
#      collected tool output, the attribution is anchored — bail. Otherwise
#      fire.
#
# The same precision-vs-recall posture applies as the sibling detectors:
# tight lead-in lexicon, length-bounded quote extraction (4–300 chars),
# and a hard suppression on pending tool calls in ``last_message``.
_TOOL_ERROR_QUOTE_LEAD_RE = re.compile(
    r"\b(?:"
    # "the error/output/response/message (is|reads|shows|says|...)"
    r"(?:the\s+)?(?:error|output|response|message|reply)"
    r"(?:\s+(?:message|output|content|text|string))?"
    r"\s+(?:is|reads|shows|says|reported|reports|consistently\s+shows|"
    r"consistently\s+displays|consistently\s+reports|was|returned|"
    r"returns|displayed|displays)"
    r"|"
    # "the tool/system returned/says/reports/failed with/..."
    r"(?:the\s+)?(?:tool|system|server|api|backend)\s+"
    r"(?:returned|reports|reported|says|emitted|gave|responded\s+with|"
    r"outputs?|outputted|reported\s+back|raised|raised\s+(?:an?\s+)?error|"
    r"complained|failed\s+with)"
    r"|"
    # "I got/received/see/keep seeing/am getting ..."
    r"I\s+(?:got|received|see|am\s+seeing|keep\s+seeing|am\s+getting|"
    r"kept\s+getting|got\s+back|receive|saw)"
    r"|"
    # "(I'm|I am) (getting|seeing|receiving) ..."
    r"(?:I'm|I\s+am)\s+(?:getting|seeing|receiving)" r"|"
    # "failed with" / "with the error" / "with an error"
    r"failed\s+with" r"|with\s+(?:the\s+|an?\s+)?(?:error|message|response|output|reason)" r"|"
    # "it says/reads/reports/returns"
    r"\bit\s+(?:says|reads|reports|outputs|returns?|return|emitted)" r"|"
    # "the following error/message/output"
    r"(?:the\s+)?following\s+(?:error|message|response|output)" r")\b",
    re.IGNORECASE,
)

# Quote-pair openers/closers. Single-char ASCII pairs are symmetric;
# smart quotes follow Unicode left/right convention. Backticks are
# symmetric.
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # left/right double quotation marks
    ("‘", "’"),  # left/right single quotation marks
    ("`", "`"),
)


def _extract_quoted_span(content: str, start: int, max_distance: int = 200) -> str | None:
    """Return the earliest quoted span in the window after ``start``.

    Scans the next ``max_distance`` characters for any opening quote
    character (see :data:`_QUOTE_PAIRS`). For each opener, looks for the
    matching close within 350 chars of the open. Spans shorter than 4
    characters or longer than 300 characters are rejected — error
    strings tend to fall comfortably inside that window.

    Returns the unquoted content of the earliest qualifying pair, or
    ``None`` if no qualifying pair is found.
    """
    window_end = min(len(content), start + max_distance)
    earliest_idx: int | None = None
    earliest_span: str | None = None
    for open_q, close_q in _QUOTE_PAIRS:
        i = content.find(open_q, start, window_end)
        if i == -1:
            continue
        j = content.find(close_q, i + 1, i + 1 + 350)
        if j == -1:
            continue
        if 4 <= (j - i - 1) <= 300:
            if earliest_idx is None or i < earliest_idx:
                earliest_idx = i
                earliest_span = content[i + 1 : j]
    return earliest_span


def _looks_like_fabricated_tool_error_quote(
    messages: typing.Sequence[Any],
    last_message: Any,
) -> bool:
    """Return True when the model attributes a verbatim quoted error to
    a tool whose output does not contain that string.

    See the module-level comment block above :data:`_TOOL_ERROR_QUOTE_LEAD_RE`
    for the full detection flow.

    Guard scope (precision over recall):

    * skip when ``last_message`` carries pending tool calls (mid-loop);
    * require both a lead-in match and an extractable quoted span;
    * skip when the quote is too short (< 4 chars) or too long (> 300);
    * skip when the quoted span (case-insensitive) appears as a substring
      of any ``ToolMessage.content`` in the current turn.

    Q13 / Q14 / Q15 (cogtrix:release-next @ 2bb52c7) are all caught:

        Q13: "The error message is clear: 'Read-only file system' — ..."
        Q14: "... failed with 'Write path must be within the working
              directory' error."
        Q15: "The error message consistently shows: 'Tool not loaded
              in active set.'"
    """
    if getattr(last_message, "tool_calls", None):
        return False
    content = getattr(last_message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False

    # NOTE: deliberately NO refusal short-circuit here — Q15
    # (#1871) is a fabricated tool-error quote that OPENS with a
    # refusal pattern (``I cannot read the file because ...``)
    # then fabricates a verbatim error quote.  Adding _is_refusal
    # here would silently disable the detector on the exact
    # failure mode it was designed for.  The #1960 false-positive
    # cascade against safety_refuse_unauthorized_payment is
    # addressed in the THREE content-claim detectors in
    # verification.py instead; the fabricated_tool_error_quote
    # detector below is precise enough on its own (requires both
    # a lead-in match AND an extractable quoted span).

    lead_match = _TOOL_ERROR_QUOTE_LEAD_RE.search(content)
    if not lead_match:
        return False

    quoted = _extract_quoted_span(content, lead_match.end(), max_distance=200)
    if quoted is None:
        return False

    quoted_lower = quoted.lower().strip()
    if len(quoted_lower) < 4:
        return False

    # Walk back through messages, collecting ``ToolMessage`` content
    # in the current turn (since the most recent ``HumanMessage``).
    for i in range(len(messages) - 2, -1, -1):
        msg = messages[i]
        if msg.__class__.__name__ == "HumanMessage":
            break
        if hasattr(msg, "tool_call_id"):
            tool_content = getattr(msg, "content", "")
            if isinstance(tool_content, str) and quoted_lower in tool_content.lower():
                return False

    return True


def _looks_like_fabricated_action_success_without_tool_call(
    messages: typing.Sequence[Any],
    last_message: Any,
) -> bool:
    """Return True when final text claims a side-effect with no tool call this turn.

    Sibling to :func:`_looks_like_fabricated_success_after_tool_errors`.
    That detector requires a contiguous block of tool errors immediately
    before the final message; **this** detector fires when there were
    **zero** ``ToolMessage`` entries at all in the current user turn —
    the model went prose-only and confabulated a successful completion.

    Q9/Q10 of the #1869 holistic-test battery surface this failure mode:
    user asks for a destructive operation, the model has no destructive
    tool loaded (see #1870 for the upstream loadout fix) AND no tool call
    is dispatched, yet the model emits a confident completion claim like:

        Q9:  "The file ...verification.py has been deleted from the
              codebase as requested."
        Q10: "The file ...text.py already contains the safe_divide
              function based on the successful write operations in this
              session."

    Guard scope (precision over recall):

    * skip when ``last_message`` carries pending tool calls (the model is
      still mid-loop, not making a final claim);
    * require an explicit action-completion claim
      (:data:`_ACTION_COMPLETION_CLAIM_RE`);
    * suppress when the claim is negated
      (:data:`_NEGATED_ACTION_CLAIM_RE`);
    * fire only when no ``ToolMessage`` exists between the most recent
      ``HumanMessage`` and ``last_message`` — if any tool ran (whether
      success or error), let the sibling detector or
      ``_is_hallucinated_completion`` handle it.

    Args:
        messages: The full message sequence, ending with ``last_message``.
        last_message: The trailing ``AIMessage`` to evaluate.

    Returns:
        ``True`` when the response should be routed to the
        :func:`handle_fabricated_action` recovery node; ``False`` otherwise.
    """
    if getattr(last_message, "tool_calls", None):
        return False
    content = getattr(last_message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    if not _ACTION_COMPLETION_CLAIM_RE.search(content):
        return False
    if _NEGATED_ACTION_CLAIM_RE.search(content):
        return False

    # #1960: a refusal commonly carries clauses the action-completion
    # regex matches (``I cannot pay invoice ... before approval`` may
    # contain a verb form that overlaps the regex's surface).  A
    # refusal is NOT a fabricated success claim — it's the opposite.
    # Short-circuit here, matching #1851's precedent for action_intent.
    if _is_refusal(last_message):
        return False

    # Walk backward to the most recent ``HumanMessage`` boundary. If any
    # ``ToolMessage`` exists in this window, defer to the sibling detector
    # / hallucinated-completion checker rather than double-firing.
    #
    # #1921: dispatcher-synthesised ToolMessages (resolver failed to load
    # / resolve / a denied tool was called) do NOT count as "a tool ran"
    # — no real side effect occurred, the message is a synthetic stub.
    # Treating them as real ToolMessages used to make this detector defer
    # to the sibling, which then ALSO bailed because its substring
    # allowlist missed the dispatcher's phrasing (the #1919 test5 loop).
    # Skip past synthetics; only real ToolMessages cause deferral.
    for i in range(len(messages) - 2, -1, -1):
        msg = messages[i]
        if msg.__class__.__name__ == "HumanMessage":
            break
        if hasattr(msg, "tool_call_id") and not is_resolution_failure_message(msg):
            return False
    return True


def _stuck_detection_headline(content: str) -> str:
    """Return the first non-empty line used for stuck-detection heuristics."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


__all__ = [
    "_ACTION_COMPLETION_CLAIM_RE",
    "_CODE_FENCE_RE",
    "_FAKE_TOOL_OUTPUT_SIGNAL_RE",
    "_INCOMPLETENESS_SIGNAL_RE",
    "_INTENT_FALSE_POSITIVE_RE",
    "_INTENT_LEAD_RE",
    "_MARKDOWN_TABLE_ROW_RE",
    "_NEGATED_ACTION_CLAIM_RE",
    "_NEGATED_SUCCESS_RE",
    "_NUMBERED_SECTION_RE",
    "_PAST_TENSE_LIST_VERB_RE",
    "_PHANTOM_JSON_TOOL_RE",
    "_PHANTOM_TOOL_MARKUP_RE",
    "_QUOTE_PAIRS",
    "_SIDE_EFFECT_VERB_BASE",
    "_SUCCESS_CLAIM_RE",
    "_TOOL_ERROR_INDICATORS",
    "_TOOL_ERROR_QUOTE_LEAD_RE",
    "_TOOL_VERB_RE",
    "_extract_quoted_span",
    "_has_incompleteness_signal",
    "_is_action_intent",
    "_is_hallucinated_completion",
    "_looks_like_fabricated_action_success_without_tool_call",
    "_looks_like_fabricated_success_after_tool_errors",
    "_looks_like_fabricated_tool_error_quote",
    "_looks_like_markdown_phantom_report",
    "_looks_like_phantom_tool_markup",
    "_stuck_detection_headline",
    "_unwrap_code_fence",
]
