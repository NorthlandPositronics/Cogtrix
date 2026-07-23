"""
Thin HTTP client for the Telegram Bot API.

Telegram bots communicate through the official Bot API at
``https://api.telegram.org/bot<token>/``.  This module handles only HTTP
transport — all business logic (contact filtering, rate limiting,
configuration) lives in ``telegram.py``.

Reference: https://core.telegram.org/bots/api
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import requests  # type: ignore[import-untyped]

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes returned by the client
# ---------------------------------------------------------------------------


@dataclass
class SendResult:
    """Result of a send operation."""

    ok: bool
    message_id: int | None = None
    error: str | None = None


@dataclass
class TelegramMessage:
    """A single Telegram message."""

    message_id: int
    date: int
    chat_id: int
    chat_title: str | None = None
    from_id: int | None = None
    from_username: str | None = None
    from_first_name: str | None = None
    text: str = ""
    is_outgoing: bool = False
    has_photo: bool = False
    has_document: bool = False
    update_id: int = 0


@dataclass
class BotInfo:
    """Basic information about the bot."""

    id: int
    username: str
    first_name: str
    can_read_messages: bool = False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TelegramBotClient:
    """Low-level HTTP client for the Telegram Bot API.

    Args:
        token:   Bot token from @BotFather (e.g. ``123456:ABC-DEF...``).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        token: str,
        timeout: int = 15,
    ) -> None:
        self.token = token
        self.timeout = timeout
        self._base_url = f"https://api.telegram.org/bot{token}"

    # -- helpers -----------------------------------------------------------

    def _url(self, method: str) -> str:
        return f"{self._base_url}/{method}"

    def _post(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Make a POST request to the Bot API and return the JSON response."""
        resp = requests.post(
            self._url(method),
            json=kwargs,
            timeout=self.timeout,
        )
        return resp.json()  # type: ignore[no-any-return]

    def _get(self, method: str, **params: Any) -> dict[str, Any]:
        """Make a GET request to the Bot API and return the JSON response."""
        clean_params = {k: v for k, v in params.items() if v is not None}
        resp = requests.get(
            self._url(method),
            params=clean_params,
            timeout=self.timeout,
        )
        return resp.json()  # type: ignore[no-any-return]

    # -- bot info ----------------------------------------------------------

    def get_me(self) -> BotInfo:
        """Return basic information about the bot (``getMe``)."""
        data = self._get("getMe")
        if not data.get("ok"):
            raise RuntimeError(data.get("description", "getMe failed"))
        result = data["result"]
        return BotInfo(
            id=result["id"],
            username=result.get("username", ""),
            first_name=result.get("first_name", ""),
            can_read_messages=not result.get("is_bot", True),
        )

    # -- send --------------------------------------------------------------

    def send_message(self, chat_id: int | str, text: str) -> SendResult:
        """Send a text message via ``sendMessage``."""
        try:
            data = self._post(
                "sendMessage",
                chat_id=chat_id,
                text=text,
            )
            if data.get("ok"):
                return SendResult(
                    ok=True,
                    message_id=data["result"].get("message_id"),
                )
            return SendResult(ok=False, error=data.get("description", "Unknown error"))
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Telegram API")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Telegram API request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    def send_photo(
        self,
        chat_id: int | str,
        photo_url: str,
        caption: str | None = None,
    ) -> SendResult:
        """Send a photo via ``sendPhoto`` using a URL."""
        try:
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "photo": photo_url,
            }
            if caption:
                kwargs["caption"] = caption
            data = self._post("sendPhoto", **kwargs)
            if data.get("ok"):
                return SendResult(
                    ok=True,
                    message_id=data["result"].get("message_id"),
                )
            return SendResult(ok=False, error=data.get("description", "Unknown error"))
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Telegram API")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Telegram API request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    # -- receive -----------------------------------------------------------

    def get_updates(
        self,
        offset: int | None = None,
        limit: int = 20,
        timeout: int = 0,
    ) -> list[TelegramMessage]:
        """Fetch recent messages via ``getUpdates``.

        Args:
            offset: Only return updates with ID >= offset (used for pagination).
            limit:  Max number of updates (1-100).
            timeout: Long-polling timeout in seconds (0 = no long poll).
        """
        try:
            data = self._get(
                "getUpdates",
                offset=offset,
                limit=limit,
                timeout=timeout,
            )
            if not data.get("ok"):
                return []
        except Exception:
            return []

        messages: list[TelegramMessage] = []
        for update in data.get("result", []):
            msg = update.get("message") or update.get("channel_post")
            if not msg:
                continue
            chat = msg.get("chat", {})
            from_user = msg.get("from", {})
            messages.append(
                TelegramMessage(
                    message_id=msg.get("message_id", 0),
                    date=msg.get("date", 0),
                    chat_id=chat.get("id", 0),
                    chat_title=chat.get("title") or chat.get("first_name"),
                    from_id=from_user.get("id"),
                    from_username=from_user.get("username"),
                    from_first_name=from_user.get("first_name"),
                    text=msg.get("text", ""),
                    is_outgoing=False,
                    has_photo=bool(msg.get("photo")),
                    has_document=bool(msg.get("document")),
                    update_id=update.get("update_id", 0),
                )
            )
        return messages

    # -- health check ------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if the bot token is valid and API is reachable."""
        try:
            self.get_me()
            return True
        except Exception:
            return False
