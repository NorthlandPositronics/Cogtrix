"""
Discord channel for Cogtrix assistant mode.

Polls Discord text channels via the REST API using bot token authentication.
Tracks per-channel message watermarks (snowflake IDs) to avoid replaying
messages across poll cycles.  Cold-start: seeds watermarks from the most
recent message in each channel so the pre-process backlog is not replayed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.assistant.channel import Channel, IncomingMessage, SendResult, parse_duration

log = logging.getLogger("cogtrix")

try:
    import discord  # type: ignore[import-untyped,import-not-found]  # noqa: F401

    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False

_DISCORD_API_BASE = "https://discord.com/api/v10"
_DISCORD_EPOCH_MS = 1420070400000  # 2015-01-01T00:00:00Z in milliseconds
_MAX_MESSAGE_LEN = 2000

# Discord channel types that carry text messages
_TEXT_CHANNEL_TYPES = frozenset({0, 5})  # GUILD_TEXT, GUILD_NEWS

_VALID_FILTER_MODES = frozenset({"none", "allow", "ignore", "blacklist"})


# ── Helpers ────────────────────────────────────────────────────────────────────


def _normalize_filter_mode(mode: str) -> str:
    """Map legacy filter_mode values to current names."""
    mode = mode.strip().lower()
    if mode == "whitelist":
        return "allow"
    if mode not in _VALID_FILTER_MODES:
        log.warning("Unrecognized Discord filter_mode %r — defaulting to 'none'", mode)
        return "none"
    return mode


def _resolve_contact(name_or_id: str, phonebook: dict[str, str]) -> str:
    """Resolve a phonebook nickname or return the raw ID string."""
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


def _snowflake_to_timestamp(snowflake_id: str) -> float:
    """Convert a Discord snowflake ID to a Unix timestamp (seconds)."""
    return ((int(snowflake_id) >> 22) + _DISCORD_EPOCH_MS) / 1000.0


# ── REST client ────────────────────────────────────────────────────────────────


class _DiscordRestClient:
    """Thin synchronous wrapper around Discord REST API v10."""

    def __init__(self, bot_token: str) -> None:
        self._headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, **params: Any) -> Any:
        url = f"{_DISCORD_API_BASE}{path}"
        resp = _requests.get(url, headers=self._headers, params=params or None, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{_DISCORD_API_BASE}{path}"
        resp = _requests.post(url, headers=self._headers, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{_DISCORD_API_BASE}{path}"
        resp = _requests.patch(url, headers=self._headers, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> bool:
        url = f"{_DISCORD_API_BASE}{path}"
        resp = _requests.delete(url, headers=self._headers, timeout=10)
        return resp.status_code == 204

    def get_guilds(self) -> list[dict[str, Any]]:
        return self._get("/users/@me/guilds")  # type: ignore[return-value]

    def get_guild_channels(self, guild_id: str) -> list[dict[str, Any]]:
        return self._get(f"/guilds/{guild_id}/channels")  # type: ignore[return-value]

    def get_messages(
        self,
        channel_id: str,
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if after:
            params["after"] = after
        return self._get(f"/channels/{channel_id}/messages", **params)  # type: ignore[return-value]

    def send_message(self, channel_id: str, content: str) -> dict[str, Any]:
        return self._post(f"/channels/{channel_id}/messages", {"content": content})  # type: ignore[return-value]

    def edit_message(self, channel_id: str, message_id: str, content: str) -> dict[str, Any]:
        return self._patch(  # type: ignore[return-value]
            f"/channels/{channel_id}/messages/{message_id}",
            {"content": content},
        )

    def delete_message(self, channel_id: str, message_id: str) -> bool:
        return self._delete(f"/channels/{channel_id}/messages/{message_id}")


# ── Channel ────────────────────────────────────────────────────────────────────


class DiscordChannel(Channel):
    """Discord channel backed by a bot token, using REST-based polling.

    Configuration (under ``services.discord`` in .cogtrix.yaml)::

        services:
          discord:
            bot_token: "YOUR_BOT_TOKEN"
            poll_interval: 2.0
            filter_mode: "none"       # none | allow | ignore | blacklist
            contacts: []              # user IDs or display names
            phonebook: {}             # {name: user_id}
            allowed_guilds: []        # restrict to these guild IDs
            ignore_older_than: ""     # e.g. "24h"
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if not _HAS_DISCORD:
            raise ImportError("discord.py not installed: pip install 'cogtrix[discord]'")
        self._config = config
        self._bot_token: str = config.get("bot_token", "")
        self._filter_mode: str = _normalize_filter_mode(config.get("filter_mode", "none"))
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._allowed_guilds: list[str] = [str(g) for g in config.get("allowed_guilds", [])]
        self._ignore_older_than: float | None = parse_duration(config.get("ignore_older_than"))

        # channel_id → last-seen snowflake ID (watermark)
        self._last_seen: dict[str, str] = {}
        # channel_id → guild_id (populated at first poll)
        self._channel_guilds: dict[str, str] = {}
        self._seeded: bool = False

        self._client: _DiscordRestClient | None = (
            _DiscordRestClient(self._bot_token) if self._bot_token and _REQUESTS_AVAILABLE else None
        )

        if self._filter_mode != "none":
            log.info(
                "Discord filter_mode=%s with %d contacts",
                self._filter_mode,
                len(self._contacts),
            )

    # ── Channel interface ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "discord"

    def poll(self) -> list[IncomingMessage]:
        """Return new inbound messages since the last call.

        First call seeds per-channel watermarks and returns an empty list
        (cold-start protection mirrors the Telegram behaviour).
        """
        if self._client is None:
            return []

        if not self._seeded:
            self._channel_guilds = self._discover_text_channels()
            for ch_id in self._channel_guilds:
                try:
                    msgs = self._client.get_messages(ch_id, limit=1)
                    if msgs:
                        self._last_seen[ch_id] = str(msgs[0]["id"])
                except Exception as exc:
                    log.debug("Discord cold-start seed failed for %s: %s", ch_id, exc)
            self._seeded = True
            log.debug("Discord cold-start: seeded %d channel(s)", len(self._channel_guilds))
            return []

        result: list[IncomingMessage] = []
        for ch_id in list(self._channel_guilds):
            after = self._last_seen.get(ch_id)
            try:
                raw = self._client.get_messages(ch_id, after=after)
            except Exception as exc:
                log.warning("Discord: failed to fetch messages from %s: %s", ch_id, exc)
                continue

            if not raw:
                continue

            # Discord returns messages newest-first; reverse to chronological order
            raw = list(reversed(raw))

            new_last: str | None = None
            for msg in raw:
                msg_id = str(msg.get("id", ""))
                if not msg_id:
                    continue
                new_last = msg_id

                # Skip bot-authored messages
                author = msg.get("author") or {}
                if author.get("bot"):
                    continue

                text = (msg.get("content") or "").strip()
                if not text:
                    continue

                ts = _snowflake_to_timestamp(msg_id)

                # Age filter
                if self._ignore_older_than is not None:
                    age = time.time() - ts
                    if age > self._ignore_older_than:
                        log.debug(
                            "Discord: skipping %s — too old (%.0fs > %.0fs)",
                            msg_id,
                            age,
                            self._ignore_older_than,
                        )
                        continue

                sender_id = str(author.get("id", ""))
                sender_name = author.get("global_name") or author.get("username") or sender_id

                if not _check_receive_contact(
                    sender_id, self._filter_mode, self._contacts, self._phonebook
                ):
                    continue

                result.append(
                    IncomingMessage(
                        channel=self.name,
                        chat_id=ch_id,
                        message_id=msg_id,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        text=text,
                        timestamp=ts,
                    )
                )

            if new_last:
                self._last_seen[ch_id] = new_last

        return result

    def send(self, chat_id: str, text: str) -> SendResult:
        """Send *text* to Discord channel *chat_id*.

        Messages longer than 2000 characters are split into multiple sends;
        only the last message_id is returned.
        """
        if self._client is None:
            return SendResult(ok=False, error="Discord client not initialized")

        chunks = [text[i : i + _MAX_MESSAGE_LEN] for i in range(0, len(text), _MAX_MESSAGE_LEN)]
        last_message_id: str | None = None
        for chunk in chunks:
            try:
                resp = self._client.send_message(chat_id, chunk)
                last_message_id = str(resp.get("id", "")) or None
            except Exception as exc:
                log.error("Discord: send failed to %s: %s", chat_id, exc)
                return SendResult(ok=False, error=str(exc))

        return SendResult(ok=True, message_id=last_message_id)

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        if self._client is None:
            return SendResult(ok=False, error="Discord client not initialized")
        try:
            resp = self._client.edit_message(chat_id, message_id, text)
            return SendResult(ok=True, message_id=str(resp.get("id", message_id)))
        except Exception as exc:
            log.error("Discord: edit failed for message %s: %s", message_id, exc)
            return SendResult(ok=False, error=str(exc))

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._client is None:
            return False
        try:
            return self._client.delete_message(chat_id, message_id)
        except Exception as exc:
            log.debug("Discord: delete failed for message %s: %s", message_id, exc)
            return False

    def is_ready(self) -> bool:
        if not _HAS_DISCORD or not _REQUESTS_AVAILABLE:
            return False
        return bool(self._bot_token)

    @classmethod
    def is_configured(cls, config: dict[str, Any]) -> bool:
        """Return True if a non-empty bot_token is present in *config*."""
        return bool(config.get("bot_token"))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _discover_text_channels(self) -> dict[str, str]:
        """Return ``{channel_id: guild_id}`` for all accessible text channels.

        Respects *allowed_guilds* if configured.
        """
        if self._client is None:
            return {}

        channels: dict[str, str] = {}
        try:
            guilds = self._client.get_guilds()
        except Exception as exc:
            log.warning("Discord: failed to fetch guilds: %s", exc)
            return {}

        for guild in guilds:
            guild_id = str(guild.get("id", ""))
            if not guild_id:
                continue
            if self._allowed_guilds and guild_id not in self._allowed_guilds:
                continue
            try:
                guild_channels = self._client.get_guild_channels(guild_id)
            except Exception as exc:
                log.debug("Discord: failed to fetch channels for guild %s: %s", guild_id, exc)
                continue
            for ch in guild_channels:
                if ch.get("type") in _TEXT_CHANNEL_TYPES:
                    channels[str(ch["id"])] = guild_id

        return channels
