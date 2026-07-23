"""
Thin HTTP client for the Waha WhatsApp API.

Waha is a self-hosted Docker container that wraps WhatsApp Web behind a
clean REST API.  This module handles only HTTP transport — all business
logic (contact filtering, rate limiting, configuration) lives in
``whatsapp.py``.

Reference: https://waha.devlike.pro/docs/overview/introduction/
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    message_id: str | None = None
    error: str | None = None


@dataclass
class Message:
    """A single WhatsApp message."""

    id: str
    timestamp: int
    from_number: str
    to: str | None = None
    body: str = ""
    from_me: bool = False
    has_media: bool = False
    media_url: str | None = None


@dataclass
class SessionInfo:
    """Basic session status from Waha."""

    name: str
    status: str
    me: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class WahaClient:
    """Low-level HTTP client for a single Waha instance.

    Args:
        base_url: Waha server URL, e.g. ``http://localhost:3000``.
        api_key:  Optional ``X-Api-Key`` header value.
        session:  Waha session name (default ``"default"``).
        timeout:  HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: str | None = None,
        session: str = "default",
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = session
        self.timeout = timeout

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # -- session -----------------------------------------------------------

    def get_session(self) -> SessionInfo:
        """Return the current Waha session status."""
        resp = requests.get(
            self._url(f"/api/sessions/{self.session}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return SessionInfo(
            name=data.get("name", self.session),
            status=data.get("status", "UNKNOWN"),
            me=data.get("me") or {},
        )

    # -- send --------------------------------------------------------------

    def send_text(self, chat_id: str, text: str) -> SendResult:
        """Send a text message via ``POST /api/sendText``."""
        payload = {
            "session": self.session,
            "chatId": chat_id,
            "text": text,
        }
        try:
            resp = requests.post(
                self._url("/api/sendText"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return SendResult(ok=False, error=f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            return SendResult(ok=True, message_id=data.get("id"))
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Waha server")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Waha request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str | None = None,
        filename: str | None = None,
    ) -> SendResult:
        """Send an image via ``POST /api/sendImage``."""
        file_obj: dict[str, str] = {
            "mimetype": "image/jpeg",
            "url": image_url,
        }
        if filename:
            file_obj["filename"] = filename

        payload: dict[str, Any] = {
            "session": self.session,
            "chatId": chat_id,
            "file": file_obj,
        }
        if caption:
            payload["caption"] = caption

        try:
            resp = requests.post(
                self._url("/api/sendImage"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return SendResult(ok=False, error=f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            return SendResult(ok=True, message_id=data.get("id"))
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Waha server")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Waha request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    def send_file(
        self,
        chat_id: str,
        file_url: str,
        filename: str | None = None,
        caption: str | None = None,
        mimetype: str = "application/octet-stream",
    ) -> SendResult:
        """Send a file/document via ``POST /api/sendFile``."""
        file_obj: dict[str, str] = {
            "mimetype": mimetype,
            "url": file_url,
        }
        if filename:
            file_obj["filename"] = filename

        payload: dict[str, Any] = {
            "session": self.session,
            "chatId": chat_id,
            "file": file_obj,
        }
        if caption:
            payload["caption"] = caption

        try:
            resp = requests.post(
                self._url("/api/sendFile"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return SendResult(ok=False, error=f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            return SendResult(ok=True, message_id=data.get("id"))
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Waha server")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Waha request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    # -- receive -----------------------------------------------------------

    def get_messages(
        self,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """Fetch recent messages via ``GET /api/messages``.

        Args:
            chat_id: Restrict to a specific chat (``None`` = all chats).
            limit:   Maximum number of messages to return.
        """
        params: dict[str, str | int] = {
            "session": self.session,
            "limit": limit,
        }
        if chat_id:
            params["chatId"] = chat_id

        try:
            resp = requests.get(
                self._url("/api/messages"),
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_messages: list[dict[str, Any]] = resp.json()
        except Exception:
            return []

        messages: list[Message] = []
        for raw in raw_messages:
            messages.append(
                Message(
                    id=raw.get("id", ""),
                    timestamp=raw.get("timestamp", 0),
                    from_number=raw.get("from", ""),
                    to=raw.get("to"),
                    body=raw.get("body", ""),
                    from_me=raw.get("fromMe", False),
                    has_media=raw.get("hasMedia", False),
                    media_url=(raw.get("media") or {}).get("url"),
                )
            )
        return messages

    # -- health check ------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if the Waha session is in WORKING state."""
        try:
            info = self.get_session()
            return info.status == "WORKING"
        except Exception:
            return False
