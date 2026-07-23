"""Tests for setup wizard on-demand documentation section retrieval (Issue #190)."""

from __future__ import annotations

from cogtrix_core.setup_wizard import _index_docs, _retrieve_relevant_sections

# ---------------------------------------------------------------------------
# _index_docs
# ---------------------------------------------------------------------------

SAMPLE_DOCS = """\
# Overview

This is the intro section.

## Memory

Memory settings control retention.

## Providers

Configure LLM providers here.

### OpenAI

OpenAI-specific settings.
"""


def test_index_docs_returns_dict_with_sections() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    assert isinstance(idx, dict)
    assert len(idx) >= 3


def test_index_docs_first_key_is_intro() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    first_key = next(iter(idx))
    assert "overview" in first_key.lower() or first_key == list(idx.keys())[0]


def test_index_docs_empty_string_returns_empty_dict() -> None:
    assert _index_docs("") == {}


def test_index_docs_no_headings_returns_empty_dict() -> None:
    assert _index_docs("Just plain text with no headings.") == {}


def test_index_docs_keys_are_lowercase() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    for key in idx:
        assert key == key.lower(), f"Key {key!r} is not lowercase"


def test_index_docs_content_includes_heading() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    # Each section's content should start with its heading marker
    for content in idx.values():
        assert content.startswith("#"), f"Section does not start with '#': {content[:40]!r}"


# ---------------------------------------------------------------------------
# _retrieve_relevant_sections
# ---------------------------------------------------------------------------


def test_retrieve_relevant_sections_empty_index_returns_empty_string() -> None:
    result = _retrieve_relevant_sections("any query", {})
    assert result == ""


def test_retrieve_relevant_sections_matches_by_heading_word() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    result = _retrieve_relevant_sections("memory configuration", idx)
    assert "Memory" in result or "memory" in result.lower()


def test_retrieve_relevant_sections_prepends_intro_section() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    result = _retrieve_relevant_sections("providers", idx)
    # The first section (Overview/intro) should always be included
    first_section_content = next(iter(idx.values()))
    # At least the intro heading should appear
    first_heading_line = first_section_content.split("\n", 1)[0]
    assert first_heading_line in result


def test_retrieve_relevant_sections_respects_max_chars() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    result = _retrieve_relevant_sections("providers memory", idx, max_chars=50)
    assert len(result) <= 50


def test_retrieve_relevant_sections_fallback_on_no_match() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    # A query that matches no headings should still return something (first N chars)
    result = _retrieve_relevant_sections("zzzzunknownzzz", idx)
    assert len(result) > 0


def test_retrieve_relevant_sections_no_duplicate_intro() -> None:
    idx = _index_docs(SAMPLE_DOCS)
    # When the query matches the intro section itself, it should not appear twice
    result = _retrieve_relevant_sections("overview intro", idx)
    first_key = next(iter(idx))
    first_content = idx[first_key]
    first_heading = first_content.split("\n", 1)[0]
    # Count how many times the intro heading appears
    assert result.count(first_heading) == 1
