"""
Slack channel for Cogtrix assistant mode.

Polls Slack conversations via the Web API using bot token authentication.
Tracks per-channel last-seen timestamps to avoid replaying messages across
poll cycles.  Cold-start: seeds watermarks from the most recent message in
each joined conversation so the pre-process backlog is not replayed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.assistant.channel import Channel, IncomingMessage, SendResult, parse_duration

log = logging.getLogger("cogtrix")

try:
    from slack_sdk import WebClient  # type: ignore[import-untyped,import-not-found]
    from slack_sdk.errors import SlackApiError  # type: ignore[import-untyped,import-not-found]

    _HAS_SLACK = True
except ImportError:
    WebClient = None  # type: ignore[assignment,misc]
    SlackApiError = Exception  # type: ignore[assignment,misc]
    _HAS_SLACK = False

_VALID_FILTER_MODES = frozenset({"none", "allow", "ignore", "blacklist"})


# ── Helpers ────────────────────────────────────────────────────────────────────


def _normalize_filter_mode(mode: str) -> str:
    """Map legacy filter_mode values to current names."""
    mode = mode.strip().lower()
    if mode == "whitelist":
        return "allow"
    if mode not in _VALID_FILTER_MODES:
        log.warning("Unrecognized Slack filter_mode %r — defaulting to 'none'", mode)
        return "none"
    return mode


def _resolve_contact(name_or_id: str, phonebook: dict[str, str]) -> str:
    """Resolve a phonebook display name to a Slack user ID, or return the raw value."""
    key = name_or_id.strip()
    for k, v in phonebook.items():
        if k.lower() == key.lower():
            return v
    return key


def _check_receive_contact(
    sender_id: str,
    filter_mode: str,
    contacts: list[str],
    phonebook: dict[str, str],
) -> bool:
    """Return True if the sender passes the contact filter."""
    if filter_mode == "none":
        return True
    normalized = {_resolve_contact(c, phonebook) for c in contacts}
    if filter_mode == "allow":
        return sender_id in normalized
    if filter_mode in ("ignore", "blacklist"):
        return sender_id not in normalized
    return True


# ── Channel ────────────────────────────────────────────────────────────────────


class SlackChannel(Channel):
    """Slack channel backed by a bot token, using REST-based polling.

    Configuration (under ``services.slack`` in .cogtrix.yaml)::

        services:
          slack:
            bot_token: "xoxb-..."        # Bot token (required)
            app_token: ""                # App-level token (Socket Mode — optional)
            poll_interval: 3.0           # seconds between poll cycles
            filter_mode: "none"          # none | allow | ignore | blacklist
            contacts: []                 # Slack user IDs or display names
            phonebook: {}                # {display_name: user_id}
            allowed_channels: []         # only respond in these channel IDs
            ignore_older_than: ""        # e.g. "24h"
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if not _HAS_SLACK:
            raise ImportError("slack-sdk not installed: pip install 'cogtrix[slack]'")

        self._config = config
        self._bot_token: str = config.get("bot_token", "")
        self._filter_mode: str = _normalize_filter_mode(config.get("filter_mode", "none"))
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._allowed_channels: list[str] = [str(c) for c in config.get("allowed_channels", [])]
        self._ignore_older_than: float | None = parse_duration(config.get("ignore_older_than"))

        # channel_id → last-seen Slack timestamp string (watermark)
        self._last_ts: dict[str, str] = {}
        # Ordered list of joined channel IDs (populated at first poll)
        self._joined_channels: list[str] = []
        self._seeded: bool = False

        # user_id → display name (lazy-populated to avoid rate limits)
        self._user_cache: dict[str, str] = {}

        self._client: Any = WebClient(token=self._bot_token) if self._bot_token else None

        if self._filter_mode != "none":
            log.info(
                "Slack filter_mode=%s with %d contacts",
                self._filter_mode,
                len(self._contacts),
            )

    # ── Channel interface ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "slack"

    def poll(self) -> list[IncomingMessage]:
        """Return new inbound messages since the last call.

        First call seeds per-channel watermarks and returns an empty list
        (cold-start protection — mirrors Discord/Telegram behaviour).
        """
        if self._client is None:
            return []

        if not self._seeded:
            self._joined_channels = self._discover_joined_channels()
            for ch_id in self._joined_channels:
                try:
                    resp = self._client.conversations_history(channel=ch_id, limit=1)
                    msgs = resp.get("messages", [])
                    if msgs:
                        self._last_ts[ch_id] = msgs[0]["ts"]
                except Exception as exc:
                    log.debug("Slack cold-start seed failed for %s: %s", ch_id, exc)
            self._seeded = True
            log.debug("Slack cold-start: seeded %d channel(s)", len(self._joined_channels))
            return []

        result: list[IncomingMessage] = []
        for ch_id in list(self._joined_channels):
            if self._allowed_channels and ch_id not in self._allowed_channels:
                continue

            oldest = self._last_ts.get(ch_id)
            try:
                kwargs: dict[str, Any] = {"channel": ch_id, "limit": 100}
                if oldest:
                    kwargs["oldest"] = oldest
                resp = self._client.conversations_history(**kwargs)
                raw: list[dict[str, Any]] = resp.get("messages", [])
            except Exception as exc:
                log.warning("Slack: failed to fetch messages from %s: %s", ch_id, exc)
                continue

            if not raw:
                continue

            # Slack returns messages newest-first; reverse to chronological order
            raw = list(reversed(raw))

            new_last: str | None = None
            for msg in raw:
                ts: str = msg.get("ts", "")
                if not ts:
                    continue
                new_last = ts

                # Skip bot-authored messages
                if msg.get("bot_id") or msg.get("subtype") == "bot_message":
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                try:
                    timestamp = float(ts)
                except ValueError:
                    timestamp = time.time()

                # Age filter
                if self._ignore_older_than is not None:
                    age = time.time() - timestamp
                    if age > self._ignore_older_than:
                        log.debug(
                            "Slack: skipping %s — too old (%.0fs > %.0fs)",
                            ts,
                            age,
                            self._ignore_older_than,
                        )
                        continue

                sender_id = msg.get("user", "")
                sender_name = self._resolve_user_name(sender_id)

                if not _check_receive_contact(
                    sender_id, self._filter_mode, self._contacts, self._phonebook
                ):
                    continue

                result.append(
                    IncomingMessage(
                        channel=self.name,
                        chat_id=ch_id,
                        message_id=ts,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        text=text,
                        timestamp=timestamp,
                    )
                )

            if new_last:
                self._last_ts[ch_id] = new_last

        return result

    def send(self, chat_id: str, text: str) -> SendResult:
        """Send *text* to Slack conversation *chat_id*.

        Slack allows up to 40,000 characters per message, so no splitting
        is needed for normal assistant responses.
        """
        if self._client is None:
            return SendResult(ok=False, error="Slack client not initialized")
        try:
            resp = self._client.chat_postMessage(channel=chat_id, text=text)
            message_id: str | None = resp.get("ts") or None
            return SendResult(ok=True, message_id=message_id)
        except Exception as exc:
            log.error("Slack: send failed to %s: %s", chat_id, exc)
            return SendResult(ok=False, error=str(exc))

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        if self._client is None:
            return SendResult(ok=False, error="Slack client not initialized")
        try:
            self._client.chat_update(channel=chat_id, ts=message_id, text=text)
            return SendResult(ok=True, message_id=message_id)
        except Exception as exc:
            log.error("Slack: edit failed for message %s: %s", message_id, exc)
            return SendResult(ok=False, error=str(exc))

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._client is None:
            return False
        try:
            self._client.chat_delete(channel=chat_id, ts=message_id)
            return True
        except Exception as exc:
            log.debug("Slack: delete failed for message %s: %s", message_id, exc)
            return False

    def is_ready(self) -> bool:
        return _HAS_SLACK and bool(self._bot_token)

    @classmethod
    def is_configured(cls, config: dict[str, Any]) -> bool:
        """Return True if a non-empty bot_token is present in *config*."""
        return bool(config.get("bot_token"))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _discover_joined_channels(self) -> list[str]:
        """Return IDs of all conversations the bot has joined (channels + DMs)."""
        if self._client is None:
            return []

        channels: list[str] = []
        try:
            cursor: str | None = None
            while True:
                kwargs: dict[str, Any] = {
                    "types": "public_channel,private_channel,im,mpim",
                    "exclude_archived": True,
                    "limit": 200,
                }
                if cursor:
                    kwargs["cursor"] = cursor
                resp = self._client.conversations_list(**kwargs)
                for ch in resp.get("channels", []):
                    ch_id = ch.get("id")
                    if not ch_id:
                        continue
                    # DMs (is_im/is_mpim) are always accessible to the bot;
                    # public/private channels require membership.
                    if ch.get("is_im") or ch.get("is_mpim") or ch.get("is_member"):
                        channels.append(ch_id)
                cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
                if not cursor:
                    break
        except Exception as exc:
            log.warning("Slack: failed to discover channels: %s", exc)

        return channels

    def _resolve_user_name(self, user_id: str) -> str | None:
        """Return display name for *user_id*, using a local cache."""
        if not user_id:
            return None
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        if self._client is None:
            return user_id
        try:
            resp = self._client.users_info(user=user_id)
            profile = (resp.get("user") or {}).get("profile") or {}
            name: str = profile.get("display_name") or profile.get("real_name") or user_id
            self._user_cache[user_id] = name
            return name
        except Exception:
            return user_id
