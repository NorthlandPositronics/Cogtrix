"""Heuristic-based memory mode selector.

Classifies user prompts as one of three memory modes: ``conversation``,
``code``, or ``reasoning``.  The classification is entirely regex-based —
no LLM calls are made — so it runs in well under 1 ms.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Reasoning signals
# ---------------------------------------------------------------------------

_REASONING_WORDS = re.compile(
    r"\b(?:"
    r"analy[sz]e|design|plan|architect|investigate|research|"
    r"compare|evaluate|strategy|tradeoffs?|"
    r"explain|reason|reasoning|"
    r"implications?|consequences?|"
    r"assess|critique|brainstorm|blueprint|outline"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Code signals
# ---------------------------------------------------------------------------

# ``` fenced code block
_CODE_BLOCK = re.compile(r"`{3}")

# 4-space indented line (common code indentation)
_CODE_INDENT = re.compile(r"^    \S", re.MULTILINE)

# Source file extensions (must follow a word character to avoid false matches)
_CODE_EXTENSIONS = re.compile(
    r"\b\w+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|cc|c\b|h\b|rb|php|cs|swift|kt|"
    r"sh|bash|yaml|yml|json|toml|sql|html|css|scss|vue|svelte|dart|lua|ex|exs)\b",
    re.IGNORECASE,
)

# Keywords that strongly indicate programming context
_CODE_KEYWORDS = re.compile(
    r"\b(?:"
    # Syntax forms that are unambiguous with a following identifier
    r"def\s+\w|class\s+\w|import\s+\w|"
    # Unambiguous programming terms
    r"refactor|debug|compile|traceback|stacktrace|breakpoint|"
    # Python built-in exception names
    r"NameError|TypeError|ValueError|IndexError|KeyError|AttributeError|"
    r"RuntimeError|ImportError|SyntaxError|StopIteration|OSError|"
    # Java / C# exceptions
    r"NullPointerException|RuntimeException|NullReferenceException|"
    # Test frameworks
    r"pytest|unittest|jest|mocha|rspec|jasmine|vitest|"
    # Package managers / build tools
    r"npm\b|pip\b|cargo\b|gradle\b|webpack\b|vite\b|dockerfile\b|"
    # Programming language names
    r"python|javascript|typescript|golang|kotlin|swift|haskell|scala|"
    r"nodejs|fastapi|django|flask|express|"
    # Common code-context nouns / verbs
    r"function|method|algorithm|codebase|repository|"
    r"variable|parameter|endpoint|api\b|"
    r"implement|lint|deploy"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_code_signals(text: str) -> bool:
    """Return ``True`` if the prompt contains code-specific patterns."""
    return bool(
        _CODE_BLOCK.search(text)
        or _CODE_INDENT.search(text)
        or _CODE_EXTENSIONS.search(text)
        or _CODE_KEYWORDS.search(text)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_memory_mode(prompt: str) -> str:
    """Classify a prompt and return the most appropriate memory mode name.

    Returns one of: ``"conversation"``, ``"code"``, ``"reasoning"``.

    Rules (in priority order):

    1. **reasoning** — prompt contains planning/analysis keywords (analyze,
       design, plan, architect, investigate, research, compare, evaluate,
       strategy, tradeoff, explain, reason, …), *or* the prompt is longer
       than 500 characters and contains no code patterns.
    2. **code** — prompt contains a fenced code block (```), a 4-space
       indented block, a source-file extension (.py, .js, …), or
       programming keywords (function, debug, refactor, NameError, …).
    3. **conversation** — default for casual chat, short questions, and
       anything not matched above.

    This is a pure heuristic — no LLM calls are made.  Expected latency is
    well under 1 ms.

    Args:
        prompt: Raw user prompt text.

    Returns:
        One of ``"conversation"``, ``"code"``, or ``"reasoning"``.
    """
    if not prompt or not prompt.strip():
        return "conversation"

    has_code = _has_code_signals(prompt)

    # Rule 1a: explicit reasoning/analysis keywords
    if _REASONING_WORDS.search(prompt):
        return "reasoning"

    # Rule 1b: long analytical prompt with no code signals
    if len(prompt) > 500 and not has_code:
        return "reasoning"

    # Rule 2: code signals
    if has_code:
        return "code"

    # Rule 3: conversation (default)
    return "conversation"


def should_switch_mode(current_mode: str, recent_prompts: list[str]) -> str | None:
    """Suggest a memory mode switch based on recent conversation patterns.

    Examines the last three prompts and returns a suggested mode if at
    least two of them independently classify to a mode that differs from
    ``current_mode``.  This 2-of-3 threshold prevents thrashing when the
    user asks a one-off question in a different domain.

    Args:
        current_mode: The currently active memory mode name.
        recent_prompts: List of recent user prompts (only the last 3 are
            examined).  Must contain at least 2 prompts for any suggestion
            to be made.

    Returns:
        The suggested mode name (``"code"``, ``"reasoning"``, or
        ``"conversation"``), or ``None`` if no switch is recommended.
    """
    window = recent_prompts[-3:]
    if len(window) < 2:
        return None

    modes = [classify_memory_mode(p) for p in window]

    # Check in a fixed order so ties are broken deterministically
    for candidate in ("code", "reasoning", "conversation"):
        if candidate != current_mode and modes.count(candidate) >= 2:
            return candidate

    return None
