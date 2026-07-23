"""Unit tests for JSON phantom tool-call detection."""

from unittest.mock import MagicMock

from cogtrix_core.orchestration.graph import _looks_like_phantom_tool_markup


def _message(content: str, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [] if tool_calls is None else tool_calls
    return msg


def test_detects_inline_json_array_phantom():
    assert _looks_like_phantom_tool_markup(_message('[{"tool": "search", "arguments": {}}]'))


def test_detects_code_block_json_phantom():
    assert _looks_like_phantom_tool_markup(
        _message('```json\n[{"tool": "search", "arguments": {}}]\n```')
    )


def test_detects_multiple_tool_json_phantom():
    assert _looks_like_phantom_tool_markup(
        _message(
            '[{"tool": "github_list_issues", "arguments": {"state": "open"}}, '
            '{"tool": "github_list_pull_requests", "arguments": {"state": "open"}}, '
            '{"tool": "github_search_issues", "arguments": {"query": "bug"}}, '
            '{"tool": "github_list_commits", "arguments": {"sha": "next"}}, '
            '{"tool": "slack_get_channel_history", "arguments": {"channel_id": "C0AVAHW6HJS"}}]'
        )
    )


def test_returns_false_for_real_tool_calls():
    assert not _looks_like_phantom_tool_markup(
        _message(
            '[{"tool": "search", "arguments": {}}]',
            tool_calls=[{"name": "search", "args": {}, "id": "tc1"}],
        )
    )


def test_returns_false_for_prose_with_tool_word_without_arguments():
    assert not _looks_like_phantom_tool_markup(
        _message('Use the "tool": hammer for this repair, not the search tool.')
    )


def test_xml_phantom_still_detected():
    assert _looks_like_phantom_tool_markup(
        _message('<function_calls><invoke name="list_issues"></invoke></function_calls>')
    )


def test_call_phantom_still_detected():
    assert _looks_like_phantom_tool_markup(
        _message("<Call><Function>list_issues</Function></Call>")
    )
