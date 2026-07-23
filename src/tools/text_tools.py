"""
Text processing tools - Various utilities for text manipulation and analysis.
"""

import re

from pydantic import BaseModel, Field


class WordCountInput(BaseModel):
    """Input schema for word count."""

    text: str = Field(description="The text to analyze")


class FindReplaceInput(BaseModel):
    """Input schema for find and replace."""

    text: str = Field(description="The text to process")
    find: str = Field(description="The text or pattern to find")
    replace: str = Field(description="The replacement text")
    use_regex: bool = Field(default=False, description="Whether to treat 'find' as a regex pattern")
    case_sensitive: bool = Field(default=True, description="Whether the search is case-sensitive")


class ExtractUrlsInput(BaseModel):
    """Input schema for URL extraction."""

    text: str = Field(description="The text to extract URLs from")


class ExtractEmailsInput(BaseModel):
    """Input schema for email extraction."""

    text: str = Field(description="The text to extract email addresses from")


class TextCompareInput(BaseModel):
    """Input schema for text comparison."""

    text1: str = Field(description="First text")
    text2: str = Field(description="Second text")


class SplitTextInput(BaseModel):
    """Input schema for splitting text."""

    text: str = Field(description="The text to split")
    delimiter: str = Field(default="\n", description="Delimiter to split on")
    max_parts: int | None = Field(default=None, description="Maximum number of parts")


class TrimTextInput(BaseModel):
    """Input schema for trimming text."""

    text: str = Field(description="The text to trim")
    max_length: int = Field(description="Maximum length of output")
    add_ellipsis: bool = Field(default=True, description="Whether to add '...' when truncated")


def word_count(text: str) -> str:
    """
    Count words, characters, lines, and other statistics in text.

    Args:
        text: The text to analyze

    Returns:
        Statistics about the text
    """
    if not text:
        return "Empty text provided"

    lines = text.split("\n")
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

    # Count sentences (rough estimate)
    sentences = len(re.findall(r"[.!?]+", text))

    # Count paragraphs (blocks separated by blank lines)
    paragraphs = len(re.split(r"\n\s*\n", text.strip()))

    # Average word length (guard against zero words)
    avg_word_len = f"{chars_no_spaces / len(words):.1f}" if words else "0.0"

    return f"""Text Statistics:
- Characters: {chars:,} (with spaces)
- Characters: {chars_no_spaces:,} (without spaces)
- Words: {len(words):,}
- Lines: {len(lines):,}
- Sentences: {sentences:,} (approximate)
- Paragraphs: {paragraphs:,}
- Average word length: {avg_word_len} characters"""


def find_replace(
    text: str,
    find: str,
    replace: str,
    use_regex: bool = False,
    case_sensitive: bool = True,
) -> str:
    """
    Find and replace text, optionally using regex.

    Args:
        text: The text to process
        find: The text or pattern to find
        replace: The replacement text
        use_regex: Whether to treat 'find' as a regex pattern
        case_sensitive: Whether the search is case-sensitive

    Returns:
        Processed text with replacements
    """
    if not text:
        return "Empty text provided"

    if not find:
        return "Error: Nothing to find"

    try:
        flags = 0 if case_sensitive else re.IGNORECASE

        if use_regex:
            result = re.sub(find, replace, text, flags=flags)
            count = len(re.findall(find, text, flags=flags))
        else:
            if case_sensitive:
                count = text.count(find)
                result = text.replace(find, replace)
            else:
                # Case-insensitive replace without regex
                pattern = re.escape(find)
                count = len(re.findall(pattern, text, flags=re.IGNORECASE))
                result = re.sub(pattern, replace, text, flags=re.IGNORECASE)

        return f"Made {count} replacement(s):\n\n{result}"

    except re.error as e:
        return f"Regex error: {e}"
    except Exception as e:
        return f"Error: {e}"


def extract_urls(text: str) -> str:
    """
    Extract all URLs from text.

    Args:
        text: The text to extract URLs from

    Returns:
        List of found URLs or message if none found
    """
    if not text:
        return "Empty text provided"

    # URL pattern
    url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    urls = re.findall(url_pattern, text)

    # Clean up URLs (remove trailing punctuation that's likely not part of URL)
    cleaned = []
    for url in urls:
        # Remove trailing punctuation
        while url and url[-1] in ".,;:!?)\"'":
            url = url[:-1]
        if url:
            cleaned.append(url)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for url in cleaned:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    if not unique:
        return "No URLs found in text"

    return f"Found {len(unique)} URL(s):\n" + "\n".join(f"- {url}" for url in unique)


def extract_emails(text: str) -> str:
    """
    Extract all email addresses from text.

    Args:
        text: The text to extract email addresses from

    Returns:
        List of found email addresses or message if none found
    """
    if not text:
        return "Empty text provided"

    # Email pattern
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    emails = re.findall(email_pattern, text)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for email in emails:
        email_lower = email.lower()
        if email_lower not in seen:
            seen.add(email_lower)
            unique.append(email)

    if not unique:
        return "No email addresses found in text"

    return f"Found {len(unique)} email address(es):\n" + "\n".join(f"- {email}" for email in unique)


def text_compare(text1: str, text2: str) -> str:
    """
    Compare two texts and show differences.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Comparison results
    """
    if text1 == text2:
        return "The texts are identical."

    results = []

    # Length comparison
    len1, len2 = len(text1), len(text2)
    results.append(f"Length: {len1} vs {len2} characters ({abs(len1 - len2)} difference)")

    # Word count comparison
    words1, words2 = len(text1.split()), len(text2.split())
    results.append(f"Words: {words1} vs {words2} ({abs(words1 - words2)} difference)")

    # Line count comparison
    lines1, lines2 = len(text1.splitlines()), len(text2.splitlines())
    results.append(f"Lines: {lines1} vs {lines2} ({abs(lines1 - lines2)} difference)")

    # Find first difference position
    min_len = min(len1, len2)
    first_diff = None
    for i in range(min_len):
        if text1[i] != text2[i]:
            first_diff = i
            break

    if first_diff is None and len1 != len2:
        first_diff = min_len

    if first_diff is not None:
        context_start = max(0, first_diff - 20)
        context_end = min(min_len, first_diff + 20)

        results.append(f"\nFirst difference at position {first_diff}:")
        results.append(f"  Text 1: ...{repr(text1[context_start:context_end])}...")
        results.append(f"  Text 2: ...{repr(text2[context_start:context_end])}...")

    return "\n".join(results)


def split_text(text: str, delimiter: str = "\n", max_parts: int | None = None) -> str:
    """
    Split text by a delimiter.

    Args:
        text: The text to split
        delimiter: Delimiter to split on
        max_parts: Maximum number of parts

    Returns:
        Numbered list of parts
    """
    if not text:
        return "Empty text provided"

    # Handle special delimiters
    if delimiter == "\\n":
        delimiter = "\n"
    elif delimiter == "\\t":
        delimiter = "\t"

    if max_parts:
        parts = text.split(delimiter, max_parts - 1)
    else:
        parts = text.split(delimiter)

    # Filter empty parts
    parts = [p for p in parts if p.strip()]

    if not parts:
        return "No parts after splitting (text may not contain the delimiter)"

    result = [f"Split into {len(parts)} part(s):"]
    for i, part in enumerate(parts, 1):
        preview = part[:100] + "..." if len(part) > 100 else part
        preview = preview.replace("\n", "\\n")
        result.append(f"{i}. {preview}")

    return "\n".join(result)


def trim_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Trim text to a maximum length.

    Args:
        text: The text to trim
        max_length: Maximum length of output
        add_ellipsis: Whether to add '...' when truncated

    Returns:
        Trimmed text
    """
    if not text:
        return "Empty text provided"

    if max_length < 1:
        return "Error: max_length must be at least 1"

    if len(text) <= max_length:
        return text

    if add_ellipsis:
        if max_length < 4:
            return text[:max_length]
        return text[: max_length - 3] + "..."
    else:
        return text[:max_length]


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "word_count",
        "description": "Count words, characters, lines, sentences, and paragraphs in text.",
        "input_schema": WordCountInput,
        "requires_confirmation": False,
        "function": word_count,
    },
    {
        "name": "find_replace",
        "description": (
            "Find and replace text. Supports regex patterns and case-insensitive matching."
        ),
        "input_schema": FindReplaceInput,
        "requires_confirmation": False,
        "function": find_replace,
    },
    {
        "name": "extract_urls",
        "description": "Extract all URLs (http/https links) from text.",
        "input_schema": ExtractUrlsInput,
        "requires_confirmation": False,
        "function": extract_urls,
    },
    {
        "name": "extract_emails",
        "description": "Extract all email addresses from text.",
        "input_schema": ExtractEmailsInput,
        "requires_confirmation": False,
        "function": extract_emails,
    },
    {
        "name": "text_compare",
        "description": "Compare two texts and show differences (length, words, first difference).",
        "input_schema": TextCompareInput,
        "requires_confirmation": False,
        "function": text_compare,
    },
    {
        "name": "split_text",
        "description": "Split text by a delimiter into parts.",
        "input_schema": SplitTextInput,
        "requires_confirmation": False,
        "function": split_text,
    },
    {
        "name": "trim_text",
        "description": "Trim text to a maximum length, optionally adding ellipsis.",
        "input_schema": TrimTextInput,
        "requires_confirmation": False,
        "function": trim_text,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "word_count",
    "find_replace",
    "extract_urls",
    "extract_emails",
    "text_compare",
    "split_text",
    "trim_text",
    "WordCountInput",
    "FindReplaceInput",
    "ExtractUrlsInput",
    "ExtractEmailsInput",
    "TextCompareInput",
    "SplitTextInput",
    "TrimTextInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
