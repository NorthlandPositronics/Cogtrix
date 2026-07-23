"""Tests for src/tools/slack_tools.py"""

from __future__ import annotations

import pytest

from src.tools import slack_tools as st


class TestMarkdownToSlackConverter:
    """Unit tests for the markdown-to-Slack-mrkdwn converter."""

    def test_convert_bold_double_asterisk(self) -> None:
        assert st._convert_bold("**hello**") == "*hello*"
        assert st._convert_bold("**hello world**") == "*hello world*"

    def test_convert_bold_preserve_single_asterisk(self) -> None:
        assert st._convert_bold("*hello*") == "*hello*"

    def test_convert_bold_multiple(self) -> None:
        text = "**bold1** and **bold2**"
        assert st._convert_bold(text) == "*bold1* and *bold2*"

    def test_convert_heading_h1(self) -> None:
        assert st._convert_heading("# Title") == "*Title*"

    def test_convert_heading_h2(self) -> None:
        assert st._convert_heading("## Subtitle") == "*Subtitle*"

    def test_convert_heading_h3(self) -> None:
        assert st._convert_heading("### Section") == "_Section_"

    def test_convert_heading_not_a_heading(self) -> None:
        assert st._convert_heading("Not a heading") == "Not a heading"
        assert st._convert_heading("#not-a-heading") == "#not-a-heading"

    def test_is_table_row_yes(self) -> None:
        assert st._is_table_row("| a | b |")
        assert st._is_table_row("| Dev | Issues |")

    def test_is_table_row_no(self) -> None:
        assert not st._is_table_row("normal text")
        assert not st._is_table_row("|single|")

    def test_is_table_separator_yes(self) -> None:
        assert st._is_table_separator("|---|---|")
        assert st._is_table_separator("| --- | --- |")
        assert st._is_table_separator("|:---:|:---|")

    def test_is_table_separator_no(self) -> None:
        assert not st._is_table_separator("| a | b |")
        assert not st._is_table_separator("normal")

    def test_convert_tables_simple(self) -> None:
        md = "| Dev | Issues |\n" "|-----|--------|\n" "| C0derR0cks | #333 |\n" "| Flexo | #476 |"
        result = st._convert_tables(md)
        assert "```" in result
        assert "| Dev | Issues |" in result
        assert "| C0derR0cks | #333 |" in result
        assert "|-----|--------|" not in result  # separator removed

    def test_convert_tables_with_regular_text(self) -> None:
        md = "Header\n" "| A | B |\n" "|---|---|\n" "| 1 | 2 |\n" "Footer"
        result = st._convert_tables(md)
        assert "Header" in result
        assert "Footer" in result
        assert "```" in result
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result

    def test_convert_markdown_full(self) -> None:
        md = (
            "# Sprint Status\n"
            "**bold text**\n"
            "| Dev | Task |\n"
            "|-----|------|\n"
            "| C0derR0cks | #333 |\n"
            "Regular _italic_ text."
        )
        result = st._convert_markdown_to_slack(md)
        assert "*Sprint Status*" in result
        assert "*bold text*" in result
        assert "```" in result
        assert "Regular _italic_ text." in result

    def test_code_block_preserved(self) -> None:
        md = "```python\n" "**not_bold**\n" "# not a heading\n" "```"
        result = st._convert_markdown_to_slack(md)
        assert "**not_bold**" in result
        assert "# not a heading" in result


class TestSlackToolConfigured:
    """Integration-ish tests for cogtrix_slack_post_message when unconfigured."""

    def test_post_unconfigured_returns_error(self) -> None:
        # Ensure the module is in an unconfigured state for this test
        original_client = st._client
        original_has = st._HAS_SLACK
        st._client = None
        st._HAS_SLACK = True
        try:
            result = st.cogtrix_slack_post_message("C123", "hello")
            assert "Error" in result
            assert "bot token" in result.lower()
        finally:
            st._client = original_client
            st._HAS_SLACK = original_has

    def test_tool_config_metadata(self) -> None:
        assert len(st.TOOL_CONFIGS) == 1
        cfg = st.TOOL_CONFIGS[0]
        assert cfg["name"] == "cogtrix_slack_post_message"
        assert "markdown" in cfg["description"].lower()


class TestSlackPostMessageInput:
    """Validation tests for the Pydantic input schema."""

    def test_valid_input(self) -> None:
        inp = st.SlackPostMessageInput(channel_id="C0AVAHW6HJS", text="hello")
        assert inp.channel_id == "C0AVAHW6HJS"
        assert inp.text == "hello"

    def test_missing_channel_id_raises(self) -> None:
        with pytest.raises(ValueError):  # pydantic.ValidationError
            st.SlackPostMessageInput(text="hello")

    def test_missing_text_raises(self) -> None:
        with pytest.raises(ValueError):  # pydantic.ValidationError
            st.SlackPostMessageInput(channel_id="C0AVAHW6HJS")
