"""Unverified-claim safety guard.

The agent sometimes answers with a categorical claim about external
state — *"The latest release is X"*, *"The file says Y"*, *"The
USD/GBP rate is Z"* — without calling the verification tool that
would actually fetch ground truth. This module routes such
responses back to a recovery node that nudges the model to verify
before answering.

Why ``date_claim`` is NOT here
============================================================
A prior iteration of this module included a ``date_claim`` rule
that fired when the response named a calendar date without a
preceding ``get_current_datetime`` call. We removed it: the
orchestration already injects today's date into the system prompt
(``src/agent/core.py:build_system_prompt``) AND prefixes every
``HumanMessage`` with ``[YYYY-MM-DD HH:MM:SS UTC]``
(``src/memory/manager.BaseMemoryManager._inject_timestamps``).
A response that names today's date is repeating ground truth that
the agent was just handed — calling ``get_current_datetime`` would
return the same value. The tool call is pure friction.

Rules in this module target categories where the agent does NOT
have an injected ground-truth source:

* **weather_claim** — current/forecast weather, requires
  ``get_weather`` (preferred) or ``web_search``.
* **exchange_rate_claim** — FX conversion rate, requires
  ``web_search``.
* **latest_version_claim** — "latest version of X", requires
  ``web_search``.
* **file_content_claim** — assertions about a specific file's
  contents, require ``read_file`` (or a sibling reader).

Design properties
============================================================

* **Rule registry.** Each ``VerificationRule`` declares a regex
  for the claim shape, the required tool name (or alternatives),
  and the recovery nudge text. Adding a new rule is a single
  registry entry — no orchestration changes needed.
* **Conservative detection.** Each rule's regex requires both a
  *claim phrase* and an *evidence token* (a real number, version
  string, filename, etc.) so legitimate prose paraphrase doesn't
  trip.
* **Bounded retries.** One revision attempt; after that the
  agent's answer ships as-is — we'd rather surface a possibly-
  stale answer than loop forever.

Out of scope here (deliberate)
============================================================

* No content rewriting. We never modify the model's text directly
  — only re-prompt and let the model produce a new answer.
* No semantic verification. We don't check whether the *value* the
  agent claimed matches the *value* the tool would return; we only
  check whether the tool was called. The tool's job is to provide
  ground truth; the agent's job is to use it. Mismatch between the
  agent's claim and the tool's response is a separate bug class
  (fabricated-success-after-tool-error, which has its own
  detector).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# ── Shared helpers ────────────────────────────────────────────────────

# Currency / FX context cues that distinguish a real exchange-rate
# claim from prose that happens to mention numbers. Money-symbol
# tokens (``$``, ``€``, ``£``, ``¥``, ``₹``) and 3-letter ISO codes
# cover the common shapes search results emit.
_CURRENCY_SYMBOL = r"[$€£¥₹]"
_CURRENCY_CODE = (
    r"USD|EUR|GBP|NZD|AUD|CAD|JPY|CNY|INR|CHF|SGD|HKD|MXN|BRL|"
    r"ZAR|SEK|NOK|DKK|PLN|CZK|TRY|RUB|KRW"
)

# Semantic-version token: 1.2.3 / 1.2 / v1.2.3 / v3.0.0-rc1 — anchored
# enough that ordinary numeric prose ("about 1.2 million") can't pass.
_VERSION_TOKEN = r"\bv?\d+\.\d+(?:\.\d+)?(?:-(?:rc|beta|alpha|dev|pre)\.?\d*)?\b"


# ── weather_claim ─────────────────────────────────────────────────────
# Fires on shapes like "It is sunny in London", "Forecast: 22°C
# tomorrow", "Light rain expected". Requires a real meteorological
# token (temperature reading, weather condition word) — generic prose
# mentioning "weather" without specifics doesn't trip.

_WEATHER_CONDITION = (
    r"(?:sunny|cloudy|overcast|rainy?|raining|drizzl(?:e|ing)|"
    r"snow(?:y|ing)?|fog(?:gy)?|mist(?:y)?|haz(?:y|e)|"
    r"thunderstorms?|stormy?|wind(?:y)?|gust(?:y|s)|hail|sleet|"
    r"clear(?:\s+skies)?|partly\s+cloud(?:y|less)|"
    r"humid(?:ity)?|dry|chilly|cold|warm|hot|mild|freezing)"
)
# Forecaster idiom — "high of 22", "low of 11". Domain-specific enough
# that it stays a standalone weather signal without a sibling context
# word. ("Low of 22" can appear outside weather — golf scores, etc. —
# but in those domains the surrounding prose virtually always names
# the domain, and the false-positive cost is bounded.)
_FORECAST_IDIOM = r"(?:high|low)\s+of\s+\d{1,3}(?:\s*°?\s*[CF]?)?\b"

# Raw degree reading — "22°C", "72°F", "-5°C". Bare temperatures must
# NOT be a standalone weather signal: the same shape appears in
# cooking recipes ("internal temp 165°F"), oven temps ("350°F"),
# medical ("fever of 38°C"), engineering ("engine at 80°C"). We
# only treat them as weather when a weather-context word is nearby
# (see _WEATHER_CONTEXT below). This avoids the cogtrix55 regression
# where a recipe answer tripped weather_claim and the orchestrator
# dropped the response.
_BARE_TEMPERATURE = r"(?:[-+]?\d{1,3}\s*°\s*[CF]\b|\d{1,3}\s*°\b)"

# Weather-specific context words. Deliberately NOT including the
# singular "temperature" (which is mostly used in non-weather senses:
# body temperature, internal temperature, engine temperature). The
# plural "temperatures" overwhelmingly refers to weather/climate, so
# it IS included.
_WEATHER_CONTEXT = (
    r"(?:weather|forecast|outside|outdoor(?:s)?|"
    r"today|tomorrow|tonight|this\s+(?:morning|afternoon|evening)|"
    r"currently|right\s+now|temperatures|"
    r"sunrise|sunset|humidity|precipitation|"
    r"wind\s+speed|chance\s+of\s+(?:rain|snow))"
)

_WEATHER_CLAIM_RE = re.compile(
    r"(?im)"
    # Direct condition assertions
    rf"\b(?:it|today|tomorrow|forecast|weather)\b.{{0,50}}?{_WEATHER_CONDITION}"
    r"|"
    rf"{_WEATHER_CONDITION}.{{0,50}}?\b(?:today|tomorrow|currently|now)\b"
    r"|"
    # Forecaster idiom ("high of 22", "low of 11") — standalone
    rf"{_FORECAST_IDIOM}"
    r"|"
    # Bare temperature reading with weather context preceding
    rf"\b{_WEATHER_CONTEXT}\b.{{0,80}}?{_BARE_TEMPERATURE}"
    r"|"
    # Bare temperature reading with weather context following
    rf"{_BARE_TEMPERATURE}.{{0,30}}?\b{_WEATHER_CONTEXT}\b"
)


# ── exchange_rate_claim ───────────────────────────────────────────────
# Fires on shapes like "1 USD = 0.74 GBP", "100 NZD ≈ €50",
# "the exchange rate is 1.23". Requires both a currency context AND
# a numeric value, so prose like "exchange rates fluctuate daily"
# doesn't match.

_EXCHANGE_RATE_CLAIM_RE = re.compile(
    r"(?im)"
    # "1 USD = 0.74 GBP" / "100 NZD = €50" — explicit conversion form
    rf"\b\d+(?:[.,]\d+)?\s*(?:{_CURRENCY_CODE}|{_CURRENCY_SYMBOL})\s*"
    rf"(?:=|≈|to|→|->|/)\s*"
    rf"(?:{_CURRENCY_SYMBOL}\s*)?\d+(?:[.,]\d+)?\s*"
    rf"(?:{_CURRENCY_CODE})?\b"
    r"|"
    # "exchange rate is 0.74"
    rf"\b(?:exchange|conversion|fx|forex|spot)\s+rate\s+"
    rf"(?:is|of|=|≈)\s*[~]?\s*\d+(?:[.,]\d+)?"
    r"|"
    # "USD/GBP at 0.74" / "GBP-USD rate 1.23"
    rf"\b(?:{_CURRENCY_CODE})[/\-](?:{_CURRENCY_CODE})\s+"
    rf"(?:rate\s+|at\s+)?\d+(?:[.,]\d+)?"
)


# ── latest_version_claim ──────────────────────────────────────────────
# Fires on shapes like "The latest version of Python is 3.13",
# "Released version 2.1.0", "current release is v4.2". The version
# token is required so prose like "we use the latest version" (no
# specific version named) doesn't trip — that's a vague reference,
# not a verifiable claim.

_LATEST_VERSION_CLAIM_RE = re.compile(
    rf"(?im)"
    rf"\b(?:the\s+)?(?:latest|current|newest|stable|most\s+recent)\s+"
    rf"(?:released?\s+)?(?:version|release|build)\s+"
    rf"(?:of\s+\S+\s+)?(?:is|=|:)?\s*{_VERSION_TOKEN}"
    r"|"
    rf"\b(?:released?|published?|shipped)\s+(?:as\s+)?{_VERSION_TOKEN}"
    r"|"
    rf"\b{_VERSION_TOKEN}\s+(?:is|was)\s+(?:the\s+)?"
    rf"(?:latest|current|newest|stable|most\s+recent)\b"
)


# ── file_content_claim ────────────────────────────────────────────────
# Fires on shapes like "The file config.yaml contains X",
# "src/foo.py defines bar", "the README says Y". Requires a file-path
# shape (extension OR path separator) so prose like "the file is
# small" doesn't match.
#
# Required tool: read_file (or a sibling reader — see required_tools).

_FILE_PATH_TOKEN = (
    # path/to/file.ext (any common ext) OR just a filename with a
    # known extension. Backticks tolerated.
    r"`?(?:[\w/.\-]+/)?[\w\-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|"
    r"toml|md|txt|cfg|ini|sh|bash|zsh|fish|rs|go|rb|php|java|c|cpp|h|hpp|"
    r"html?|css|scss|less|sql|xml|csv|tsv|log|env|lock|gitignore|"
    r"dockerfile|makefile|cmake)`?"
)
_FILE_CONTENT_CLAIM_RE = re.compile(
    r"(?im)"
    rf"\b{_FILE_PATH_TOKEN}\s+(?:contains?|defines?|declares?|"
    rf"says?|reads?|sets?|specifies|exports?|imports?|includes?|has)\b"
    r"|"
    rf"\b(?:the\s+|inside\s+|in\s+)?{_FILE_PATH_TOKEN}\s*[,:]"
    rf"\s+(?:the|it)\s+(?:contains?|defines?|says?|reads?|sets?|"
    rf"declares?|specifies|exports?|has)\b"
    r"|"
    rf"\baccording\s+to\s+{_FILE_PATH_TOKEN}\b"
)


@dataclass(frozen=True)
class VerificationRule:
    """One rule in the verification registry.

    Adding a new claim category (time, exchange rate, version, file
    content, etc.) means adding a ``VerificationRule`` to
    ``VERIFICATION_RULES`` — nothing else needs to change.

    Attributes
    ----------
    name:
        Short identifier surfaced in log messages and recovery hints.
    claim_re:
        Compiled regex matching the *claim shape* in the response
        text. Must require an actual evidence token (number, version,
        filename, etc.), not just a verb phrase, to avoid false
        positives on legitimate prose.
    required_tools:
        Tuple of tool names that satisfy the verification — ANY ONE
        of these called this turn marks the claim as verified. A
        single-tool rule passes ``("only_acceptable_tool",)``.
        Multi-tool rules let the agent satisfy the guard via
        sibling readers (e.g. ``read_file`` or ``view_file``).
    nudge_template:
        The recovery prompt injected when the rule fires. Should
        explain *what* claim was caught and *which* tool to call.
        The phrase MUST end with explicit instruction so the model
        doesn't repeat the unverified answer.
    """

    name: str
    claim_re: re.Pattern[str]
    required_tools: tuple[str, ...]
    nudge_template: str

    @property
    def required_tool(self) -> str:
        """Back-compat accessor for callers that read a single tool name.

        Returns the first acceptable tool. Existing code paths that
        log or render ``rule.required_tool`` keep working without
        change; new code should prefer ``rule.required_tools``.
        """
        return self.required_tools[0]


VERIFICATION_RULES: tuple[VerificationRule, ...] = (
    VerificationRule(
        name="weather_claim",
        claim_re=_WEATHER_CLAIM_RE,
        # ``get_weather`` is the preferred verifier (structured,
        # current). ``web_search`` is the broad fallback for places
        # the dedicated tool can't reach (e.g. micro-regions).
        required_tools=("get_weather", "web_search"),
        nudge_template=(
            "Your response made a weather assertion (current conditions, "
            "forecast, or temperature) without calling `get_weather` or "
            "`web_search`. Weather data is time-sensitive — your training "
            "snapshot is not a reliable source. Call `get_weather` (or "
            "`web_search` as a fallback) now, then revise your answer "
            "using only the data the tool returns. If neither tool yields "
            "useful data, say so honestly rather than guessing."
        ),
    ),
    VerificationRule(
        name="exchange_rate_claim",
        claim_re=_EXCHANGE_RATE_CLAIM_RE,
        required_tools=("web_search",),
        nudge_template=(
            "Your response quoted a specific exchange / FX rate "
            "(e.g. ``1 USD = 0.74 GBP``) without calling `web_search` "
            "to fetch a current quote. FX rates change continuously — "
            "your training data is months or years out of date. Call "
            "`web_search` now to look up the rate, then revise your "
            "answer with the value the search returns. Cite the source "
            "and the time the rate was retrieved."
        ),
    ),
    VerificationRule(
        name="latest_version_claim",
        claim_re=_LATEST_VERSION_CLAIM_RE,
        required_tools=("web_search",),
        nudge_template=(
            "Your response asserted a specific latest / current version "
            "number without calling `web_search` to verify it. Releases "
            "ship constantly — the version you remember from training is "
            "almost certainly behind by now. Call `web_search` for the "
            "official release announcement (project changelog or release "
            "page), then revise the answer using the verified value."
        ),
    ),
    VerificationRule(
        name="file_content_claim",
        claim_re=_FILE_CONTENT_CLAIM_RE,
        # ``read_file`` is canonical. ``patch_file`` and similar readers
        # don't satisfy the guard — they don't return content. A future
        # ``view_file`` (read-only mirror) would be added here.
        required_tools=("read_file",),
        nudge_template=(
            "Your response claimed something about the contents of a "
            "specific file without calling `read_file` (or a sibling "
            "reader) to inspect it. You cannot describe what a file says "
            "from training data — files in this workspace are arbitrary "
            "and unknown to your training. Call `read_file` now on the "
            "exact path you cited, then revise the answer to quote or "
            "summarise what the tool actually returned."
        ),
    ),
)


def detect_unverified_claim(
    response_content: str,
    tool_names_called_this_turn: Iterable[str],
) -> VerificationRule | None:
    """Return the first ``VerificationRule`` that matched the response
    without its required tool having been called this turn, or ``None``.

    Parameters
    ----------
    response_content:
        The final AIMessage content text (not tool-call args, not
        prior turns — only the current response).
    tool_names_called_this_turn:
        Iterable of tool names invoked during the current turn. The
        guard checks this against each rule's ``required_tool``.

    Returns
    -------
    The first matching rule, or ``None`` if no unverified claim is
    present. We return the rule itself (not just the name) so the
    caller can read ``nudge_template`` for the recovery message.

    A rule is considered SATISFIED if ANY of its ``required_tools``
    was called this turn. Multi-tool rules let the agent satisfy
    the guard via sibling readers — e.g. ``weather_claim`` accepts
    either ``get_weather`` or ``web_search``.
    """
    if not response_content or not response_content.strip():
        return None

    called = set(tool_names_called_this_turn)
    for rule in VERIFICATION_RULES:
        if any(t in called for t in rule.required_tools):
            continue
        if rule.claim_re.search(response_content):
            return rule
    return None


def collect_tool_names_this_turn(messages: list[Any], turn_start_idx: int) -> list[str]:
    """Walk forward from ``turn_start_idx`` and collect the names of
    every tool that was called this turn.

    Parameters
    ----------
    messages:
        The full conversation message list (langchain-style).
    turn_start_idx:
        Index of the first message belonging to the current turn —
        typically the index of the user's HumanMessage that opened
        this turn. Messages strictly before this index belong to
        prior turns and are ignored.

    Returns
    -------
    Tool names in invocation order. A tool called twice is listed
    twice; deduplication is the caller's choice via ``set()``.
    """
    if turn_start_idx >= len(messages):
        return []

    names: list[str] = []
    for msg in messages[turn_start_idx:]:
        # Two shapes of tool-call evidence:
        # 1. AIMessage with tool_calls=[{"name": ..., "args": ...}, ...]
        # 2. ToolMessage with .name set (the response side; pair to a call)
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    n = tc.get("name")
                    if isinstance(n, str) and n:
                        names.append(n)
        else:
            n = getattr(msg, "name", None)
            if isinstance(n, str) and n and hasattr(msg, "tool_call_id"):
                # ToolMessage path — the response side of a tool call.
                # We could double-count, but the call side is the
                # canonical signal so we prefer that. Only pick this
                # up if no tool_calls were captured upstream
                # (defensive — shouldn't happen in well-formed runs).
                if n not in names:
                    names.append(n)
    return names


# ── Unverified-entity detection (cogtrix47 Issues 5 + 6) ─────────────
#
# When the user supplies a high-specificity identifier in their prompt
# (a SKU, a specific retailer name, a multi-word product/service name)
# and the agent repeats it in the final response WITHOUT any tool
# result confirming the entity exists, the response is operating on an
# unverified premise. The cogtrix47 run did this with two such
# identifiers:
#
#   * ``Soudal Fix All Silirub 1GH-EJ4`` — SKU that appears in zero
#     search results; only ``Fix All Crystal`` / ``Sanitary Silicone``
#     came back.
#   * ``PowerTool shop by Praterstern`` — store name never confirmed
#     by any tool result.
#
# Three pattern categories anchor the detector. Each must be specific
# enough that ordinary prose (city names, single-word brands like
# ``Soudal`` or ``Vienna``) cannot trip:
#
#   1. SKU-shape — alphanumeric token with at least one digit AND one
#      hyphen, length 4-30 (catches ``1GH-EJ4``, ``ISO-9001``).
#   2. Qualifier phrase — ``<TitleCase> {shop|store|retailer|...}``
#      (catches ``PowerTool shop``).
#   3. Three or more consecutive TitleCase words (catches
#      ``Fix All Silirub``).

_SKU_RE = re.compile(
    # Lookahead: token must contain ≥1 digit AND ≥1 hyphen so we
    # don't match plain ALLCAPS words like "USA" or "EUR".
    r"\b(?=[A-Z0-9\-]*\d)(?=[A-Z0-9\-]*-)[A-Z0-9\-]{4,30}\b"
)

_STORE_QUALIFIER_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)"
    r"\s+(?:shop|store|retailer|outlet|warehouse|depot|"
    r"branch|location|office|showroom|boutique|kiosk|stall)\b"
)

_MULTI_TITLE_RE = re.compile(
    # 3+ consecutive TitleCase words. Each word starts with capital,
    # contains 2+ letters total. Trailing-word boundary anchored.
    r"\b(?:[A-Z][a-z]{1,}\s+){2,}[A-Z][a-z]{1,}\b"
)


def _extract_specific_entities(text: str) -> list[str]:
    """Return the set of high-specificity identifiers in *text*.

    Three pattern categories — SKU, store-qualifier phrase, 3+ TitleCase
    words. The categories are intentionally narrow so ordinary
    descriptive prose ("Vienna's hardware stores") doesn't surface as
    a candidate. Duplicates collapse; ordering preserves first
    appearance for downstream readability.
    """
    seen: dict[str, None] = {}
    for ent in _SKU_RE.findall(text):
        seen.setdefault(ent, None)
    for ent in _STORE_QUALIFIER_RE.findall(text):
        seen.setdefault(ent, None)
    for ent in _MULTI_TITLE_RE.findall(text):
        seen.setdefault(ent, None)
    return list(seen.keys())


def detect_unverified_entities(
    response_content: str,
    user_prompt: str,
    tool_message_contents: Iterable[str],
    *,
    max_returned: int = 3,
) -> list[str]:
    """Return user-supplied specific entities the agent repeated in its
    response without any tool result confirming their existence.

    Parameters
    ----------
    response_content:
        The final AIMessage content text.
    user_prompt:
        The user's most recent ``HumanMessage`` content — source of
        the entity candidates.
    tool_message_contents:
        Iterable of ToolMessage content strings from the current
        turn. An entity is considered verified if it appears (case-
        insensitive substring match) in any one of these.
    max_returned:
        Cap on the returned list to avoid swamping the recovery nudge
        when the user introduced many specific identifiers.

    Returns
    -------
    A list of entity strings in first-appearance order. Empty list
    means no unverified entities — the response is internally
    grounded for whatever identifiers it cites.
    """
    if not response_content or not response_content.strip():
        return []
    if not user_prompt or not user_prompt.strip():
        return []

    candidates = _extract_specific_entities(user_prompt)
    if not candidates:
        return []

    response_lower = response_content.lower()
    tool_blob = "\n".join(c for c in tool_message_contents if isinstance(c, str)).lower()

    unverified: list[str] = []
    for ent in candidates:
        ent_lower = ent.lower()
        if ent_lower not in response_lower:
            # Agent didn't repeat this identifier — nothing to verify.
            continue
        if ent_lower in tool_blob:
            # A tool result confirmed (or at least mentioned) it —
            # the agent has grounds.
            continue
        unverified.append(ent)
        if len(unverified) >= max_returned:
            break
    return unverified


def collect_tool_message_contents(messages: list[Any], turn_start_idx: int) -> list[str]:
    """Walk forward from ``turn_start_idx`` and collect the content of
    every ToolMessage in the current turn.

    Pairs with ``detect_unverified_entities`` — the entity detector
    needs to scan tool results for verifying mentions.
    """
    if turn_start_idx >= len(messages):
        return []
    out: list[str] = []
    for msg in messages[turn_start_idx:]:
        if not hasattr(msg, "tool_call_id"):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            out.append(content)
    return out


_UNVERIFIED_ENTITY_NUDGE = (
    "Your response repeats {n_word} specific identifier{plural} from "
    "the user's prompt that NO tool result this turn has confirmed: "
    "{entities}. The agent is treating user-supplied names as ground "
    "truth without grounding them in evidence — exactly the failure "
    "mode that produces confident-but-fabricated answers about "
    "products / shops / SKUs that do not exist.\n\n"
    "Revise your response to either:\n"
    "  (a) cite the tool result that confirms each identifier "
    "(by URL or extract); or\n"
    "  (b) state plainly that you could not verify the identifier and "
    'suggest the user clarify it (e.g. "I could not find any product '
    "matching the SKU 'X' — could you confirm the exact code?\"); or\n"
    "  (c) drop the unverified identifier and answer with the closest "
    "verified alternative the search did return.\n\n"
    "Do NOT fabricate a confirmation. Do NOT silently substitute a "
    "different identifier and continue as if the user's name were "
    "correct."
)


def format_unverified_entity_nudge(entities: list[str]) -> str:
    """Render the recovery nudge for one or more unverified entities."""
    n = len(entities)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    quoted = ", ".join(f"'{e}'" for e in entities)
    return _UNVERIFIED_ENTITY_NUDGE.format(n_word=n_word, plural=plural, entities=quoted)


# ── Output-fidelity guard (#1841) ─────────────────────────────────────
#
# The rule registry and the entity detector both check *research effort*
# (was a tool called? is a user-named identifier present in a tool
# result?). Neither checks *research fidelity*: that a verbatim quote or
# explicitly-attributed statement the model puts in its answer actually
# appears in a tool result this turn. The next67 trial surfaced the gap —
# the model fabricated a blockquote ("`kimi-k2.6` was officially
# discontinued …") that inverted the fetched source (which said the
# deprecated `kimi-k2` *series* was discontinued and to USE `kimi-k2.6`).
#
# This detector targets QUOTED spans (double-quoted text and `>`
# blockquotes) only — explicit verbatim claims of "the source says
# exactly this." Free paraphrase is deliberately out of scope
# (paraphrase-fidelity is a separate, false-positive-prone problem).

# Markdown emphasis / inline-code marks to drop before comparison so the
# model's reformatting (``**bold**``, `` `code` ``) doesn't cause a
# spurious mismatch against the raw tool text.
_FIDELITY_MD_STRIP_RE = re.compile(r"[`*_]+")
_FIDELITY_WS_RE = re.compile(r"\s+")
_QUOTE_GLYPHS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

# Quoted-span extractors.
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$", re.MULTILINE)
_DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]+)"')
_ELLIPSIS_SPLIT_RE = re.compile(r"\.{3,}|…")

# A quoted span shorter than this (in significant words) is not treated
# as a substantive claim — quoting "OK" or a single product token must
# not trip the guard.
_MIN_QUOTE_WORDS = 6


def _normalize_for_fidelity(text: str) -> str:
    """Casefold + strip markdown marks + collapse whitespace so a faithful
    quote matches the raw tool text despite cosmetic reformatting."""
    t = text.translate(_QUOTE_GLYPHS).lower()
    t = _FIDELITY_MD_STRIP_RE.sub("", t)
    return _FIDELITY_WS_RE.sub(" ", t).strip()


def _strip_quote_wrapping(span: str) -> str:
    """Remove surrounding whitespace and a single layer of wrapping quote
    characters from an extracted span."""
    s = span.strip()
    if len(s) >= 2 and s[0] in "\"'“‘" and s[-1] in "\"'”’":
        s = s[1:-1].strip()
    return s


def _extract_quoted_spans(text: str) -> list[str]:
    """Return the substantive quoted spans in *text* — `>` blockquote
    lines and double-quoted substrings, de-wrapped of quote characters."""
    spans: list[str] = []
    for m in _BLOCKQUOTE_RE.finditer(text):
        spans.append(_strip_quote_wrapping(m.group(1)))
    for m in _DOUBLE_QUOTE_RE.finditer(text):
        spans.append(_strip_quote_wrapping(m.group(1)))
    return [s for s in spans if s]


def _quote_is_grounded(norm_span: str, grounded_blob: str) -> bool:
    """A normalized quote is grounded if it (or, for elided quotes, each
    of its non-trivial segments) is a substring of the grounded blob."""
    if not norm_span:
        return True
    if norm_span in grounded_blob:
        return True
    segments = [seg.strip() for seg in _ELLIPSIS_SPLIT_RE.split(norm_span)]
    segments = [seg for seg in segments if len(seg.split()) >= 3]
    if len(segments) >= 2:
        return all(seg in grounded_blob for seg in segments)
    return False


def detect_unsupported_quote(
    response_content: str,
    tool_message_contents: Iterable[str],
    user_prompt: str = "",
    *,
    min_quote_words: int = _MIN_QUOTE_WORDS,
    max_returned: int = 3,
) -> list[str]:
    """Return substantive quoted spans in the response that appear in no
    tool result this turn (and aren't quotes of the user's own prompt).

    Targets fabricated verbatim quotes / fabricated source citations —
    the output-fidelity gap (#1841). Quoted spans only; paraphrase is out
    of scope by design.

    Parameters
    ----------
    response_content: the final AIMessage content text.
    tool_message_contents: ToolMessage content strings from this turn —
        the ground-truth corpus a quote must be traceable to.
    user_prompt: the user's most recent message; quoting the user's own
        words back is legitimate, so it counts as grounding.
    """
    if not response_content or not response_content.strip():
        return []

    spans = _extract_quoted_spans(response_content)
    if not spans:
        return []

    grounded_sources = [c for c in tool_message_contents if isinstance(c, str)]
    if user_prompt:
        grounded_sources.append(user_prompt)
    grounded_blob = _normalize_for_fidelity("\n".join(grounded_sources))

    unsupported: list[str] = []
    seen: set[str] = set()
    for span in spans:
        norm = _normalize_for_fidelity(span)
        if len(norm.split()) < min_quote_words:
            continue  # too short to be a substantive claim
        if norm in seen:
            continue
        seen.add(norm)
        if not grounded_blob or not _quote_is_grounded(norm, grounded_blob):
            unsupported.append(span)
            if len(unsupported) >= max_returned:
                break
    return unsupported


_UNSUPPORTED_QUOTE_NUDGE = (
    "Your response presents {n_word} quoted/attributed statement{plural} "
    "that appear in NO tool result from this turn: {quotes}. Presenting a "
    "verbatim quote or a source attribution that you cannot trace to "
    "fetched content is fabrication — even when the surrounding topic was "
    "researched. This is the exact failure mode where a model invents an "
    'authoritative-sounding quote (e.g. "the official platform states …") '
    "that the source never said, or inverts what it did say.\n\n"
    "Revise your response to either:\n"
    "  (a) quote VERBATIM from an actual tool result (copy the exact text, "
    "and only attribute it to the source that actually contains it); or\n"
    "  (b) drop the quotation marks / attribution and state the point as "
    "your own summary, clearly grounded in what the tools returned; or\n"
    "  (c) if the tools did not establish the point, say so plainly rather "
    "than manufacturing a citation.\n\n"
    "Do NOT fabricate a quote. Do NOT attribute a statement to a source "
    "that does not contain it. Do NOT change the subject of a real quote "
    "(e.g. attributing a statement about one model/version to a different "
    "one)."
)


def format_unsupported_quote_nudge(quotes: list[str]) -> str:
    """Render the recovery nudge for one or more unsupported quotes."""
    n = len(quotes)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))

    def _clip(q: str) -> str:
        q = q.strip()
        return q if len(q) <= 120 else q[:117] + "..."

    quoted = "; ".join(f'"{_clip(q)}"' for q in quotes)
    return _UNSUPPORTED_QUOTE_NUDGE.format(n_word=n_word, plural=plural, quotes=quoted)


# ── Version-scope-collapse guard (#1843) ──────────────────────────────
#
# The fidelity guard (#1841) catches a *fabricated quote* — text the model
# attributes to a source that the source never said. The version-scope
# guard catches a subtler, adjacent failure: the model takes a status that
# the evidence genuinely scopes to one identifier and *re-scopes* it onto a
# more specific (child / newer) identifier. In the next67 trial the fetched
# docs said the deprecated ``kimi-k2`` *series* was discontinued and to USE
# ``kimi-k2.6``; the model reported that ``kimi-k2.6`` itself was
# discontinued, and under challenge invented that ``kimi-k2.5`` was *also*
# discontinued. Both are children of the ``kimi-k2`` parent the source
# actually scoped the status to. Textually ``kimi-k2`` ⊂ ``kimi-k2.6`` /
# ``kimi-k2.5``; semantically they are opposite — the classic
# series→version substring confusion.
#
# Why effort/fidelity guards miss it: ``detect_unverified_claim`` only
# checks the verifier was called (it was); ``detect_unsupported_quote``
# only fires on *quoted* spans (a status stated as prose or in a table cell
# is not a quote); and every presence-based check is satisfied because
# ``kimi-k2`` *is* present in the tool blob and is a substring of the
# child ID. Scope requires modelling *which* identifier a status attaches
# to — not merely whether a token appears.
#
# Detection is deliberately conservative to satisfy the "no false positives
# on legitimate version-specific claims" requirement:
#
#   * **Sentence-scoped, nearest-ID attribution.** A status attaches to the
#     nearest model-ID *within its own sentence / table row*. This handles
#     the contrastive form ("``kimi-k2.6`` is current, unlike the
#     discontinued ``kimi-k2`` series") — there the status's nearest ID is
#     the parent, so the child is never marked claimed.
#   * **Negation aware.** "is not discontinued" / "isn't deprecated" do not
#     count as a status assertion.
#   * **Canonical IDs only.** We match canonical hyphen/dotted identifiers
#     (``kimi-k2.6``, ``gpt-4``) — the form tool outputs and the model's
#     own citations use. Display variants ("Kimi K2.6") are out of scope.
#   * **Flag only true scope-collapse.** A child claim is flagged only when
#     (a) the source does NOT support that status for the child's exact ID,
#     AND (b) the source scopes that status to a prefix-*parent* of the
#     child. A genuinely version-specific claim the source supports, or a
#     pure fabrication with no parent in the evidence, is left to the other
#     guards.
#
# Lifecycle-status only (discontinued / deprecated / retired / sunset /
# EOL): these are the high-harm misattributions (telling a user a current
# model is dead). Release/launch status is intentionally excluded — version
# release dates legitimately differ and the false-positive cost is higher.

_NEG_STATUS_RE = re.compile(
    r"\b(?:discontinued|deprecated|sunset(?:ted|ting)?|retired|"
    r"decommissioned|end[\s-]of[\s-]life|eol|"
    r"no\s+longer\s+(?:available|supported|maintained|offered))\b",
    re.IGNORECASE,
)

# Negation tokens that, when they immediately precede a status word, mean
# the model is NOT asserting that status ("kimi-k2.6 is not discontinued").
_PRE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn[’']t|wasn[’']t|aren[’']t|weren[’']t|n[’']t)\b",
    re.IGNORECASE,
)

# Canonical model/version identifier: an alpha stem of ≥2 letters, at least
# one digit somewhere, and at least one ``-``/``.`` separator group. Matches
# ``kimi-k2``, ``kimi-k2.6``, ``gpt-4``, ``claude-opus-4-7``. The greedy
# separator group means ``kimi-k2.6`` is captured whole (not just
# ``kimi-k2``), so the child and parent are distinct tokens. The leading
# ``[a-z]{2,}`` and the digit lookahead exclude bare versions (``1.2.3``),
# abbreviations (``e.g``), Slack/object IDs (``u0b115kt39d``), and
# dotted-but-digitless names (``node.js``).
_MODEL_ID_RE = re.compile(r"\b(?=[a-z0-9.\-]*\d)[a-z]{2,}[a-z0-9]*(?:[-.][a-z0-9]+)+\b")

# Split into sentences / table rows WITHOUT breaking version tokens: a
# sentence ends on .!? followed by whitespace (so the ``.`` inside
# ``kimi-k2.6`` — followed by a digit — never splits), or on a newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\n\r]+")

# A status attached to an ID more than this many characters away (within a
# single long sentence) is too distant to be a reliable attribution.
_SCOPE_WINDOW = 100


@dataclass(frozen=True)
class VersionScopeMismatch:
    """One version-scope-collapse finding.

    ``claimed_id`` is the specific identifier the response attaches the
    status to; ``scoped_to_id`` is the prefix-parent the *evidence*
    actually scopes that status to; ``status`` is the matched lifecycle
    word (``discontinued`` etc.).
    """

    claimed_id: str
    status: str
    scoped_to_id: str


def _nearest_id(
    ids: list[tuple[str, int, int]], s_start: int, s_end: int, *, window: int = _SCOPE_WINDOW
) -> str | None:
    """Return the model-ID token nearest to a status span, or ``None`` if
    the closest ID is farther than *window* characters."""
    best: str | None = None
    best_gap: int | None = None
    for tok, i_start, i_end in ids:
        if i_end <= s_start:
            gap = s_start - i_end
        elif i_start >= s_end:
            gap = i_start - s_end
        else:
            gap = 0
        if gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = tok, gap
    return best


def _nonnegated_status_spans(lowered_segment: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, word)`` for each non-negated lifecycle-status
    mention in a (lowercased) sentence/row."""
    out: list[tuple[int, int, str]] = []
    for sm in _NEG_STATUS_RE.finditer(lowered_segment):
        pre = lowered_segment[max(0, sm.start() - 24) : sm.start()]
        if _PRE_NEGATION_RE.search(pre):
            continue
        out.append((sm.start(), sm.end(), sm.group(0)))
    return out


def _response_status_claims(text: str) -> list[tuple[str, str]]:
    """Return ``(model_id, status_word)`` pairs the response asserts —
    status attached to its nearest model-ID within each sentence/row."""
    claims: list[tuple[str, str]] = []
    for seg in _SENTENCE_SPLIT_RE.split(text):
        lowered = seg.lower()
        ids = [(m.group(0), m.start(), m.end()) for m in _MODEL_ID_RE.finditer(lowered)]
        if not ids:
            continue
        for s_start, s_end, word in _nonnegated_status_spans(lowered):
            nearest = _nearest_id(ids, s_start, s_end)
            if nearest is not None:
                claims.append((nearest, word))
    return claims


def _source_status_scope(tool_blob: str) -> tuple[set[str], set[str]]:
    """Return ``(supported_ids, primary_ids)`` from the evidence.

    ``supported_ids`` is every ID that shares a sentence/row with a
    non-negated lifecycle status — the generous set used to decide whether
    a claimed ID's status is genuinely backed (avoids false positives when
    the source really does scope the status to that exact ID).
    ``primary_ids`` is the nearest ID to each status — the precise set used
    to find the parent a status is actually scoped to.
    """
    supported: set[str] = set()
    primary: set[str] = set()
    for seg in _SENTENCE_SPLIT_RE.split(tool_blob):
        lowered = seg.lower()
        ids = [(m.group(0), m.start(), m.end()) for m in _MODEL_ID_RE.finditer(lowered)]
        if not ids:
            continue
        spans = _nonnegated_status_spans(lowered)
        if not spans:
            continue
        supported.update(item[0] for item in ids)
        for span in spans:
            nearest = _nearest_id(ids, span[0], span[1])
            if nearest is not None:
                primary.add(nearest)
    return supported, primary


def _best_parent(child_id: str, parent_candidates: set[str]) -> str | None:
    """Return the longest ID in *parent_candidates* that is a prefix-parent
    of *child_id* (proper prefix ending on a ``.``/``-`` version boundary),
    or ``None``. ``kimi-k2`` is a parent of ``kimi-k2.6``; ``kimi-k2`` is
    NOT a parent of ``kimi-k20`` (boundary char ``0`` is not a separator)."""
    parents = [
        y
        for y in parent_candidates
        if y != child_id and child_id.startswith(y) and child_id[len(y)] in ".-"
    ]
    return max(parents, key=len) if parents else None


def detect_version_scope_mismatch(
    response_content: str,
    tool_message_contents: Iterable[str],
    *,
    max_returned: int = 3,
) -> list[VersionScopeMismatch]:
    """Return version-scope-collapse findings (#1843).

    Flags each case where the response attaches a lifecycle status to a
    specific model-ID while the evidence scopes that status only to a
    prefix-*parent* of that ID (and does not back it for the exact ID).
    Returns ``[]`` when nothing is found — including when the response makes
    no status claim, when no tool content is available to check against, or
    when every claim is either source-supported or a pure fabrication with
    no parent in the evidence (the latter is the other guards' concern).
    """
    if not response_content or not response_content.strip():
        return []
    claims = _response_status_claims(response_content)
    if not claims:
        return []
    blob = "\n".join(c for c in tool_message_contents if isinstance(c, str))
    if not blob.strip():
        return []
    supported, primary = _source_status_scope(blob)
    if not primary:
        return []

    results: list[VersionScopeMismatch] = []
    seen: set[tuple[str, str]] = set()
    for claimed_id, status in claims:
        if claimed_id in supported:
            continue  # evidence backs this exact ID — a true claim
        parent = _best_parent(claimed_id, primary)
        if parent is None:
            continue  # no parent scope in evidence — not a scope collapse
        key = (claimed_id, parent)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            VersionScopeMismatch(claimed_id=claimed_id, status=status, scoped_to_id=parent)
        )
        if len(results) >= max_returned:
            break
    return results


_VERSION_SCOPE_NUDGE = (
    "Your response attaches a lifecycle status to {n_word} specific "
    "model/version identifier{plural} that the fetched sources scope to a "
    "DIFFERENT (broader / parent) identifier: {pairs}. This is version-scope "
    "collapse — a statement about a deprecated *series* or a different "
    "version is being reattributed to a more specific version. Textually "
    "`{parent}` is a substring of `{child}`, but they are not the same "
    "thing, and the sources may actually list `{child}` as current or "
    "available.\n\n"
    "Re-read the tool results and revise your answer to:\n"
    '  (a) attach the "{status}" status to the EXACT identifier the '
    "source scopes it to (`{parent}`), not to `{child}`; and\n"
    "  (b) state the actual status of `{child}` using ONLY what the sources "
    "say about that exact identifier — do not assume it inherits the "
    "parent's status.\n\n"
    "Do NOT carry a parent/series status down onto a specific version. Do "
    "NOT invent a status for a version the sources do not mention."
)


def format_version_scope_nudge(mismatches: list[VersionScopeMismatch]) -> str:
    """Render the recovery nudge for one or more version-scope mismatches."""
    n = len(mismatches)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    pairs = "; ".join(
        f'`{m.claimed_id}` (sources scope "{m.status}" to `{m.scoped_to_id}`)' for m in mismatches
    )
    example = mismatches[0]
    return _VERSION_SCOPE_NUDGE.format(
        n_word=n_word,
        plural=plural,
        pairs=pairs,
        child=example.claimed_id,
        parent=example.scoped_to_id,
        status=example.status,
    )


__all__ = [
    "VERIFICATION_RULES",
    "VerificationRule",
    "VersionScopeMismatch",
    "collect_tool_message_contents",
    "collect_tool_names_this_turn",
    "detect_unsupported_quote",
    "detect_unverified_claim",
    "detect_unverified_entities",
    "detect_version_scope_mismatch",
    "format_unsupported_quote_nudge",
    "format_unverified_entity_nudge",
    "format_version_scope_nudge",
]
