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


class TestSlackBotTokenEnvVar:
    """Issue #913: ``COGTRIX_SLACK_BOT_TOKEN`` was documented in
    ``CONFIGURATION.md`` but no code read it.  Users who followed the
    docs hit a silent failure: the env var was ignored, the bot token
    stayed empty, and Slack tools were absent from the registry with
    no warning logged.

    The fix mirrors the WhatsApp / Telegram pattern: env var is the
    highest-priority override on top of the config-file value.  These
    tests pin both branches.
    """

    def _isolate_module_state(self):
        """Snapshot and restore module-level state for clean test isolation."""
        return (st._client, dict(st._slack_config), st._HAS_SLACK)

    def _restore(self, snapshot) -> None:
        st._client, slack_cfg, has = snapshot
        st._slack_config = dict(slack_cfg)
        st._HAS_SLACK = has

    def _force_slack_available(self, monkeypatch) -> None:
        """Force the slack-sdk-available branch with a stub WebClient
        so the tests run even when slack-sdk isn't installed in the
        dev environment.
        """
        st._HAS_SLACK = True
        st._client = None
        monkeypatch.setattr(st, "WebClient", lambda token: object())

    def test_env_var_token_used_when_config_token_absent(self, monkeypatch) -> None:
        """An empty config + ``COGTRIX_SLACK_BOT_TOKEN`` env var must
        propagate the env-var token into ``_slack_config`` so
        ``is_configured()`` reports True.  This is the docs-promised
        behaviour that previously did nothing.
        """
        snap = self._isolate_module_state()
        try:
            self._force_slack_available(monkeypatch)
            monkeypatch.setenv("COGTRIX_SLACK_BOT_TOKEN", "xoxb-from-env-913")

            # Empty config dict — relying entirely on the env var.
            st.configure_slack_tools({})

            assert (
                st._slack_config.get("bot_token") == "xoxb-from-env-913"
            ), "env-var token must be propagated into _slack_config (issue #913)"
            assert st.is_configured() is True, (
                "COGTRIX_SLACK_BOT_TOKEN env var was set but is_configured() "
                "still reports False — the docs-promised env var fallback "
                "is missing"
            )
        finally:
            self._restore(snap)
            monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

    def test_env_var_overrides_config_token(self, monkeypatch) -> None:
        """When BOTH the config file AND the env var supply a token,
        env var wins — matches WhatsApp / Telegram precedence and the
        usual "env var = highest priority" convention.
        """
        snap = self._isolate_module_state()
        try:
            self._force_slack_available(monkeypatch)
            monkeypatch.setenv("COGTRIX_SLACK_BOT_TOKEN", "xoxb-env-wins")

            st.configure_slack_tools({"bot_token": "xoxb-config-loses"})

            assert st._slack_config.get("bot_token") == "xoxb-env-wins"
        finally:
            self._restore(snap)
            monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

    def test_no_env_var_keeps_config_token(self, monkeypatch) -> None:
        """When the env var is not set, the config-file token is kept
        verbatim — unchanged behaviour for users who never used the
        env var.
        """
        snap = self._isolate_module_state()
        try:
            self._force_slack_available(monkeypatch)
            monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)

            st.configure_slack_tools({"bot_token": "xoxb-config-only"})

            assert st._slack_config.get("bot_token") == "xoxb-config-only"
        finally:
            self._restore(snap)

    def test_empty_env_var_does_not_override(self, monkeypatch) -> None:
        """An empty-string env var (e.g. ``export COGTRIX_SLACK_BOT_TOKEN=``)
        must NOT clobber a real token from the config file.  Treating
        empty string as "unset" prevents accidental misconfiguration.
        """
        snap = self._isolate_module_state()
        try:
            self._force_slack_available(monkeypatch)
            monkeypatch.setenv("COGTRIX_SLACK_BOT_TOKEN", "")

            st.configure_slack_tools({"bot_token": "xoxb-config-real"})

            assert st._slack_config.get("bot_token") == "xoxb-config-real"
        finally:
            self._restore(snap)
            monkeypatch.delenv("COGTRIX_SLACK_BOT_TOKEN", raising=False)
