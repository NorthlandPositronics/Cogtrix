"""Slack tools — post formatted messages to Slack with markdown-to-mrkdwn conversion.

Tools:
    cogtrix_slack_post_message  — post a message to a Slack channel, converting
                                   markdown syntax to Slack mrkdwn/blocks.

Configuration (services.slack in .cogtrix.yaml):
    bot_token:  "xoxb-..."   # Slack Bot User OAuth Token (required)

Environment overrides (highest priority):
    COGTRIX_SLACK_BOT_TOKEN    Slack Bot User OAuth Token; overrides
                               ``services.slack.bot_token`` from the config
                               file.  An empty-string value is ignored so
                               ``export COGTRIX_SLACK_BOT_TOKEN=`` does not
                               accidentally clear the config-file value.

This tool exists because the MCP ``slack_post_message`` tool passes raw text
through Slack's mrkdwn parser, which does **not** support markdown tables or
``**bold**`` (double-asterisk) syntax.  The wrapper converts tables to code
blocks and ``**text**`` to ``*text*`` before posting with ``mrkdwn=True``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from cogtrix_core.tools.delegate import register_tool_categories

if TYPE_CHECKING:
    from cogtrix_core.config import Config

from cogtrix_core.tools.error_sanitizer import sanitize_error

try:
    from slack_sdk import WebClient  # type: ignore[import-untyped,import-not-found]
    from slack_sdk.errors import SlackApiError  # type: ignore[import-untyped,import-not-found]

    _HAS_SLACK = True
except ImportError:  # pragma: no cover
    _HAS_SLACK = False
    WebClient = None  # type: ignore[misc,assignment]
    SlackApiError = Exception  # type: ignore[misc,assignment]

log = logging.getLogger("cogtrix.tools.slack")

# ── Module-level state (set by configure_slack_tools) ─────────────────────────

_client: Any = None  # WebClient or None when unconfigured
_slack_config: dict[str, Any] = {}


# ── Configuration ─────────────────────────────────────────────────────────────


def configure_slack_tools(slack: dict[str, Any]) -> None:
    """Set runtime configuration from the services.slack dict.

    Mirrors the WhatsApp / Telegram precedence: the
    ``COGTRIX_SLACK_BOT_TOKEN`` environment variable, when set to a
    non-empty value, overrides the ``bot_token`` from the config file
    (issue #913).  An empty-string env var is treated as unset so it
    cannot accidentally clear a valid config-file token.
    """
    global _client, _slack_config
    _slack_config = {**slack}

    # Env-var override (highest priority).  Empty strings are ignored
    # so ``export COGTRIX_SLACK_BOT_TOKEN=`` does not silently disable
    # a working config-file token.
    env_token = os.environ.get("COGTRIX_SLACK_BOT_TOKEN", "").strip()
    if env_token:
        _slack_config["bot_token"] = env_token

    if not _HAS_SLACK:
        return

    token = _slack_config.get("bot_token", "")
    if token:
        _client = WebClient(token=token)  # type: ignore[operator]
    else:
        _client = None


def TOOL_SETUP(config: Config) -> None:  # type: ignore[name-defined]
    """Called automatically by ToolRegistry after loading this module."""
    svc = getattr(config, "services", {}) or {}
    slack = svc.get("slack", {}) or {}
    configure_slack_tools(slack)


def is_configured() -> bool:
    """Return True when slack-sdk is installed and a bot token is configured."""
    if not _HAS_SLACK:
        return False
    return bool(_slack_config.get("bot_token"))


# ── Markdown → Slack mrkdwn converter ─────────────────────────────────────────


def _convert_bold(text: str) -> str:
    """Replace markdown ``**bold**`` with Slack mrkdwn ``*bold*``."""
    # Use a negative lookbehind/lookahead to avoid matching single asterisks
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)


def _convert_heading(line: str) -> str:
    """Replace markdown headings with bold text."""
    match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if match:
        level = len(match.group(1))
        content = match.group(2).strip()
        # Level 1-2 → bold; 3-6 → italic
        if level <= 2:
            return f"*{content}*"
        return f"_{content}_"
    return line


def _is_table_row(line: str) -> bool:
    """Return True if *line* looks like a markdown table row."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    parts = stripped.split("|")
    # Need at least 2 content cells (e.g. | A | B | → ['', ' A ', ' B ', ''])
    cells = [p.strip() for p in parts if p.strip()]
    return len(cells) >= 2


def _is_table_separator(line: str) -> bool:
    """Return True if *line* is a markdown table separator like ``|---|---|``."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    inner = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
    return all(c in "-|: \t" for c in inner)


def _convert_tables(text: str) -> str:
    """Replace markdown tables with fenced code blocks.

    Slack's mrkdwn parser does not support ``|cell|`` syntax, so we preserve
    visual alignment by wrapping the table in triple back-ticks.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _is_table_row(lines[i]):
            table_lines: list[str] = []
            while i < len(lines) and (_is_table_row(lines[i]) or _is_table_separator(lines[i])):
                table_lines.append(lines[i])
                i += 1
            # Remove separator rows from the output
            display_lines = [ln for ln in table_lines if not _is_table_separator(ln)]
            out.append("```")
            out.extend(display_lines)
            out.append("```")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _convert_markdown_to_slack(text: str) -> str:
    """Convert common markdown syntax to Slack mrkdwn-compatible text.

    Supported conversions:
    - ``**bold**`` → ``*bold*``
    - ``# heading`` → ``*heading*``
    - Tables → fenced code blocks
    - Everything else passes through unchanged (Slack mrkdwn already handles
      ``_italic_``, `` `code` ``, ``>quote``, ``- list``, ``:emoji:``).
    """
    text = _convert_tables(text)
    # Process bold and headings line-by-line after table conversion so we
    # don't accidentally match inside code blocks.
    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            result.append(line)
            continue
        if in_code:
            result.append(line)
            continue
        line = _convert_bold(line)
        line = _convert_heading(line)
        result.append(line)
    return "\n".join(result)


# ── Input schema ──────────────────────────────────────────────────────────────


class SlackPostMessageInput(BaseModel):
    channel_id: str = Field(
        ...,
        description="Slack channel ID (e.g. 'C0AVAHW6HJS') to post the message to.",
    )
    text: str = Field(
        ...,
        description=(
            "Message text in markdown syntax.  Tables and ``**bold**`` will be "
            "converted automatically to Slack-compatible formatting."
        ),
    )


# ── Tool definition ───────────────────────────────────────────────────────────


def cogtrix_slack_post_message(channel_id: str, text: str) -> str:
    """Post *text* to the Slack channel *channel_id*.

    Markdown syntax (``**bold**``, tables, headings) is converted automatically
    to Slack mrkdwn before the message is sent.  Requires ``services.slack.bot_token``
    to be configured in ``.cogtrix.yaml``.

    Returns the message timestamp on success, or an error string on failure.
    """
    if not _HAS_SLACK:
        return "Error: slack-sdk is not installed. Run: uv add 'cogtrix[slack]'"
    if _client is None:
        return (
            "Error: Slack bot token is not configured. "
            "Add services.slack.bot_token to .cogtrix.yaml"
        )

    converted = _convert_markdown_to_slack(text)

    try:
        resp = _client.chat_postMessage(
            channel=channel_id,
            text=converted,
            mrkdwn=True,
        )
        ts: str | None = resp.get("ts")
        return f"Message posted successfully (ts={ts})."
    except SlackApiError as exc:
        err = exc.response.get("error", "unknown")  # type: ignore[union-attr]
        log.error("Slack post failed: %s", err)
        return f"Error: Slack API returned '{err}'."
    except Exception as exc:
        log.error("Slack post failed: %s", exc)
        return f"Error: {sanitize_error(exc)}"


# ── Tool metadata ─────────────────────────────────────────────────────────────


TOOL_CONFIGS = [
    {
        "name": "cogtrix_slack_post_message",
        "description": (
            "Post a message to a Slack channel with automatic markdown-to-mrkdwn "
            "conversion. Supports **bold**, tables, and headings. "
            "Requires services.slack.bot_token in .cogtrix.yaml."
        ),
        "input": SlackPostMessageInput,
        "category": "messaging",
    }
]


register_tool_categories({"cogtrix_slack_post_message": "messaging"})
