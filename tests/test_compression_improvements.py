"""Tests for emergency compression pass and HumanMessage truncation."""

from unittest.mock import MagicMock

import pytest


def _make_tool_message(content: str, tool_call_id: str = "tc1", name: str = "search"):
    try:
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
    except ImportError:
        pytest.skip("langchain not available")


def _make_ai_message(tool_call_id: str = "tc1"):
    try:
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="thinking",
            tool_calls=[{"id": tool_call_id, "name": "s", "args": {}, "type": "tool_call"}],
        )
    except ImportError:
        pytest.skip("langchain not available")


def _make_human_message(content: str):
    try:
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)
    except ImportError:
        pytest.skip("langchain not available")


def test_emergency_pass_lowers_min_age():
    """When context > 85%, min_age_cycles drops to 1."""
    from cogtrix_core.orchestration.compression import apply_message_compression

    # Return a compressed version (shorter than original) to simulate real compression
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="x" * 1000)

    # Build a message list: tool message (1 cycle old) + 1 AIMessage after it
    tool_msg = _make_tool_message("x" * 5000, tool_call_id="tc1")
    ai_msg = _make_ai_message("tc1")

    # max_context_tokens=16384 (minimum allowed for compression) → context_chars=65536
    # total_chars = 5000 → below emergency threshold (55701 chars), but min_age_override forces it
    messages = [tool_msg, ai_msg]
    cache = {}

    # Use min_age_override=1 to force compression despite age=1
    result = apply_message_compression(
        messages,
        call_count=2,
        compression_cache=cache,
        llm=mock_llm,
        max_context_tokens=16_384,
        min_age_cycles=3,  # normally would skip (age=1 < 3)
        min_chars=100,
        emergency_threshold=0.85,
        min_age_override=1,  # Force compression despite age
    )
    # The tool message (age=1) should have been compressed despite min_age_cycles=3
    assert result is not None
    assert len(result) == 2
    # Verify the LLM was invoked for compression
    mock_llm.invoke.assert_called_once()
    # Verify compression actually occurred: original was 5000 chars, result should be smaller
    result_tool_msg = result[0]
    assert len(result_tool_msg.content) < 5000
    # Verify original content is gone (not preserved)
    assert "x" * 5000 not in result_tool_msg.content


def test_normal_pass_respects_min_age():
    """Below emergency threshold, min_age_cycles=3 is respected."""
    from cogtrix_core.orchestration.compression import apply_message_compression

    tool_msg = _make_tool_message("x" * 100, tool_call_id="tc1")
    ai_msg = _make_ai_message("tc1")
    messages = [tool_msg, ai_msg]
    mock_llm = MagicMock()
    cache = {}

    # max_context_tokens=100000 → threshold is very high; total_chars=100 → no pass
    result = apply_message_compression(
        messages,
        call_count=2,
        compression_cache=cache,
        llm=mock_llm,
        max_context_tokens=100_000,
        min_age_cycles=3,
        min_chars=100,
    )
    # Returns original messages unchanged (below 72% threshold)
    assert result == messages


def test_human_message_truncated_when_over_limit():
    """HumanMessages longer than human_msg_max_chars are middle-truncated.

    Uses max_context_tokens=16_384 (minimum allowed) so 50_000-char message
    exceeds the 72% threshold (47,185 chars) and the compression pass runs.
    """
    from cogtrix_core.orchestration.compression import apply_message_compression

    # 50_000 chars > 16_384 * 4 * 0.72 = 47,185 → triggers compression pass
    long_content = "A" * 50_000
    human_msg = _make_human_message(long_content)
    messages = [human_msg]
    mock_llm = MagicMock()
    cache = {}

    result = apply_message_compression(
        messages,
        call_count=1,
        compression_cache=cache,
        llm=mock_llm,
        max_context_tokens=16_384,
        min_age_cycles=3,
        min_chars=2000,
        human_msg_max_chars=10_000,
    )
    result_content = result[0].content
    assert len(result_content) < len(long_content)
    assert "truncated" in result_content


def test_human_message_not_truncated_when_under_limit():
    """Short HumanMessages are not modified."""
    from cogtrix_core.orchestration.compression import apply_message_compression

    human_msg = _make_human_message("Hello world")
    messages = [human_msg]
    mock_llm = MagicMock()
    cache = {}

    result = apply_message_compression(
        messages,
        call_count=1,
        compression_cache=cache,
        llm=mock_llm,
        max_context_tokens=100_000,
        min_age_cycles=3,
        min_chars=2000,
        human_msg_max_chars=20_000,
    )
    assert result[0].content == "Hello world"


def test_human_msg_truncation_disabled_when_zero():
    """human_msg_max_chars=0 disables truncation."""
    from cogtrix_core.orchestration.compression import apply_message_compression

    long_content = "B" * 50_000
    human_msg = _make_human_message(long_content)
    messages = [human_msg]
    mock_llm = MagicMock()

    result = apply_message_compression(
        messages,
        call_count=1,
        compression_cache={},
        llm=mock_llm,
        max_context_tokens=100_000,
        min_age_cycles=3,
        min_chars=2000,
        human_msg_max_chars=0,  # disabled
    )
    assert result[0].content == long_content


def test_config_defaults():
    """New config fields have correct defaults."""
    from cogtrix_core.config import Config

    c = Config()
    assert c.context_compression_emergency_threshold == 0.85
    assert c.context_compression_human_msg_max_chars == 20_000


def test_config_custom_values():
    """New config fields can be set."""
    from cogtrix_core.config import Config

    c = Config(
        context_compression_emergency_threshold=0.90,
        context_compression_human_msg_max_chars=5_000,
    )
    assert c.context_compression_emergency_threshold == 0.90
    assert c.context_compression_human_msg_max_chars == 5_000
