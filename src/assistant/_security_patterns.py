"""
Shared security-pattern constants for the assistant subsystem.

Centralizes confusable-character maps, injection-detection regexes, and
the skeleton normalization function so that any update to homoglyph
coverage or injection detection requires a single change.

Used by:
- src/assistant/guardrails.py
- src/assistant/campaign.py
"""

from __future__ import annotations

import re
import unicodedata

# ── Confusable / homoglyph map ──────────────────────────────────────

CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic -> Latin
    "\u0430": "a",
    "\u0410": "A",  # а/А
    "\u0441": "c",
    "\u0421": "C",  # с/С
    "\u0435": "e",
    "\u0415": "E",  # е/Е
    "\u043d": "h",
    "\u041d": "H",  # н/Н
    "\u0456": "i",
    "\u0406": "I",  # і/І
    "\u0458": "j",  # ј
    "\u043e": "o",
    "\u041e": "O",  # о/О
    "\u0440": "p",
    "\u0420": "P",  # р/Р
    "\u0455": "s",  # ѕ
    "\u0443": "y",  # у
    "\u0445": "x",
    "\u0425": "X",  # х/Х
    "\u0412": "B",  # В
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u0422": "T",  # Т
    # Greek -> Latin
    "\u03b1": "a",
    "\u0391": "A",  # α/Α
    "\u03b5": "e",
    "\u0395": "E",  # ε/Ε
    "\u03bf": "o",
    "\u039f": "O",  # ο/Ο
    "\u0392": "B",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03a5": "Y",
    "\u0396": "Z",
}

CONFUSABLE_TRANS: dict[int, str] = str.maketrans(CONFUSABLE_MAP)

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"(system\s+prompt|system\s+message)\s+is",
        r"you\s+are\s+now\s+(a|an|the)\b",
        r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)",
        r"pretend\s+(you\s+are|to\s+be|you're)\b",
        r"act\s+as\s+(if\s+)?(you\s+are|a|an)",
        r"(new\s+)?instructions?:\s",
        r"override\s+(previous|all|your)\b",
        r"forget\s+(everything|all|previous|your)\b",
        r"(drop|clear|reset|erase|wipe)\s+(all|everything|previous|prior|your|the)\b",
        # "clear/reset/drop the context" — use `clear` only with an explicit determiner so
        # adjective uses like "no clear context" or "a clear professional context" don't match.
        r"\bclear\s+(all|the|your|my|previous|prior|entire)\s+.{0,50}\b(context|history|memory|instructions?|rules?|prompts?)\b",
        r"\b(drop|reset|erase|wipe)\s+.{0,100}\b(context|history|memory|instructions?|rules?|prompts?)\b",
        r"now\s+you\s+are\s+(a|an|the|my)\b",
        r"from\s+now\s+on\s+you\s+(are|will|should|must)\b",
        r"stop\s+being\s+(a|an|the)\b",
        r"\bDAN\b.{0,200}\bmode\b",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"\[system\]",
        r"<\|?(system|im_start|im_end)\|?>",
        r"```\s*(system|prompt)",
    ]
]


def skeleton(text: str) -> str:
    """Reduce text to a Latin skeleton for confusable-resistant matching."""
    return unicodedata.normalize("NFKC", text).translate(CONFUSABLE_TRANS)
