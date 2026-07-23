"""
Thin HTTP client for the Waha WhatsApp API.

Waha is a self-hosted Docker container that wraps WhatsApp Web behind a
clean REST API.  This module handles only HTTP transport — all business
logic (contact filtering, rate limiting, configuration) lives in
``whatsapp.py``.

Reference: https://waha.devlike.pro/docs/overview/introduction/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cogtrix")

# After this many consecutive failures of the same poll operation, escalate from
# DEBUG to a single WARNING so a silently-dead WhatsApp poll is visible at normal
# operator verbosity (#2229).
_POLL_FAILURE_ESCALATION_THRESHOLD = 3

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
    # WAHA delivery ack (#2413): -1 ERROR, 0 PENDING, 1 SERVER, 2 DEVICE
    # (delivered to recipient), 3 READ, 4 PLAYED. ``ack < 2`` = never delivered.
    # ``None`` when the field is absent from the WAHA payload.
    ack: int | None = None


@dataclass
class SessionInfo:
    """Basic session status from Waha."""

    name: str
    status: str
    me: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatOverview:
    """Summary of a chat from the chats overview endpoint."""

    id: str
    name: str | None = None
    last_message: Message | None = None
    archived: bool = False


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
        # Per-operation count of consecutive poll failures, so a persistently
        # failing poll (#2229) gets escalated from DEBUG to a single WARNING with
        # the actionable HTTP body — instead of looping forever at DEBUG while the
        # assistant silently receives nothing.
        self._consecutive_poll_failures: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _describe_poll_error(self, exc: Exception) -> str:
        """Build an actionable description of a poll failure.

        Surfaces the HTTP response body (the *reason*, e.g. WAHA's "Enable NOWEB
        store …" 400) which ``str(exc)`` alone drops, plus a hint for the common
        NOWEB-store misconfiguration (#2229)."""
        parts: list[str] = [str(exc)]
        resp = getattr(exc, "response", None)
        body = ""
        if resp is not None:
            try:
                body = (resp.text or "").strip()
            except Exception:
                body = ""
        if body:
            parts.append(f"response body: {body[:500]}")
            low = body.lower()
            if "noweb" in low or ("store" in low and "enable" in low):
                parts.append(
                    "hint: enable the NOWEB store on this WAHA session "
                    "(start the session with the NOWEB engine store enabled)."
                )
        return " | ".join(parts)

    def _record_poll_failure(self, operation: str, exc: Exception) -> None:
        """Track a consecutive poll failure and escalate once it is persistent.

        Logs at DEBUG normally; emits a single WARNING when the failure count
        reaches ``_POLL_FAILURE_ESCALATION_THRESHOLD`` (not every cycle), so a
        dead poll is visible at normal verbosity without spamming the log.
        """
        n = self._consecutive_poll_failures.get(operation, 0) + 1
        self._consecutive_poll_failures[operation] = n
        detail = self._describe_poll_error(exc)
        if n == _POLL_FAILURE_ESCALATION_THRESHOLD:
            log.warning(
                "WhatsApp poll '%s' has failed %d times in a row — the assistant "
                "is receiving no messages on this channel. %s",
                operation,
                n,
                detail,
            )
        else:
            log.debug("Failed to %s (consecutive failure #%d): %s", operation, n, detail)

    def _record_poll_success(self, operation: str) -> None:
        """Reset the failure counter; log recovery if it had escalated."""
        if self._consecutive_poll_failures.get(operation, 0) >= _POLL_FAILURE_ESCALATION_THRESHOLD:
            log.info(
                "WhatsApp poll '%s' recovered after %d consecutive failures",
                operation,
                self._consecutive_poll_failures[operation],
            )
        self._consecutive_poll_failures[operation] = 0

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

    def start_session(self) -> bool:
        """Start the Waha session via ``POST /api/sessions/{name}/start``.

        Returns ``True`` if the session was started (or was already running).
        """
        try:
            resp = requests.post(
                self._url(f"/api/sessions/{self.session}/start"),
                json={},
                headers=self._headers(),
                timeout=self.timeout,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def resolve_lid(self, lid: str) -> str | None:
        """Resolve a ``@lid`` identifier to a phone number via the Lids API.

        Returns the phone number in ``@c.us`` format, or ``None`` if the LID
        cannot be resolved (not in contacts or not a group admin).
        """
        try:
            resp = requests.get(
                self._url(f"/api/{self.session}/lids/{lid}"),
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return None
            data = resp.json()
            return data.get("pn") or None
        except Exception:
            return None

    def check_exists(self, phone: str) -> bool | None:
        """Check whether a number is registered on WhatsApp (#2413).

        ``GET /api/contacts/check-exists?phone=…`` → WAHA's ``numberExists``.
        Accepts a bare number or a ``@c.us`` chatId (the ``@`` suffix and a
        leading ``+`` are stripped). Returns ``True``/``False`` per WAHA, or
        ``None`` when the check itself cannot be completed (network error,
        non-2xx, malformed body) so callers can **fail open** — never block a
        legitimate send on a transient WAHA hiccup, only on an explicit
        ``numberExists=false``.
        """
        number = phone.split("@", 1)[0].lstrip("+").strip()
        if not number:
            return None
        try:
            resp = requests.get(
                self._url("/api/contacts/check-exists"),
                params={"phone": number, "session": self.session},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return None
            val = resp.json().get("numberExists")
            return bool(val) if val is not None else None
        except Exception:
            return None

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

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously sent message via Waha API."""
        try:
            resp = requests.put(
                self._url(f"/api/{self.session}/chats/{chat_id}/messages/{message_id}"),
                json={"text": text},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return SendResult(ok=False, error=f"HTTP {resp.status_code}: {resp.text}")
            return SendResult(ok=True, message_id=message_id)
        except requests.exceptions.ConnectionError:
            return SendResult(ok=False, error="Cannot connect to Waha server")
        except requests.exceptions.Timeout:
            return SendResult(ok=False, error="Waha request timed out")
        except Exception as exc:
            return SendResult(ok=False, error=str(exc))

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message via ``DELETE /api/{session}/chats/{chatId}/messages/{messageId}``."""
        try:
            resp = requests.delete(
                self._url(f"/api/{self.session}/chats/{chat_id}/messages/{message_id}"),
                headers=self._headers(),
                timeout=self.timeout,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def archive_chat(self, chat_id: str) -> bool:
        """Archive a chat via ``POST /api/{session}/chats/{chatId}/archive``."""
        try:
            resp = requests.post(
                self._url(f"/api/{self.session}/chats/{chat_id}/archive"),
                json={},
                headers=self._headers(),
                timeout=self.timeout,
            )
            return resp.status_code < 400
        except Exception:
            return False

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
        except Exception as exc:
            self._record_poll_failure("fetch messages", exc)
            return []
        self._record_poll_success("fetch messages")

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
                    ack=raw.get("ack"),
                )
            )
        return messages

    def get_chat_messages(
        self,
        chat_id: str,
        limit: int = 20,
        *,
        download_media: bool = False,
        filter_from_me: bool | None = None,
        filter_timestamp_gte: int | None = None,
    ) -> list[Message]:
        """Fetch messages from a specific chat via
        ``GET /api/{session}/chats/{chatId}/messages``.

        Server-side ``filter.fromMe`` and ``filter.timestamp.gte`` are applied
        client-side to work around a WAHA WEBJS engine bug where the evaluate
        crashes with ``TypeError: Cannot read properties of undefined (reading 't')``
        when these filters are passed as query parameters.

        Args:
            chat_id: The chat to fetch messages from.
            limit:   Maximum number of messages to return.
            download_media: Whether to include media download URLs.
            filter_from_me: Client-side filter by sender (``True``/``False``/``None``).
            filter_timestamp_gte: Only return messages at or after this timestamp.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "downloadMedia": str(download_media).lower(),
        }

        resp = requests.get(
            self._url(f"/api/{self.session}/chats/{chat_id}/messages"),
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw_messages: list[dict[str, Any]] = resp.json()

        messages: list[Message] = []
        for raw in raw_messages:
            from_me = raw.get("fromMe", False)
            if filter_from_me is not None and from_me != filter_from_me:
                continue
            ts = raw.get("timestamp", 0)
            if filter_timestamp_gte is not None and ts < filter_timestamp_gte:
                continue
            messages.append(
                Message(
                    id=raw.get("id", ""),
                    timestamp=ts,
                    from_number=raw.get("from", ""),
                    to=raw.get("to"),
                    body=raw.get("body", ""),
                    from_me=from_me,
                    has_media=raw.get("hasMedia", False),
                    media_url=(raw.get("media") or {}).get("url"),
                    ack=raw.get("ack"),
                )
            )
        return messages

    def get_message_ack(self, chat_id: str, message_id: str) -> int | None:
        """Return the delivery ack of a previously-sent message, or ``None`` when
        it can't be determined (#2413).

        WAHA ack scale (see ``Message.ack``): ``ack < 2`` = never reached the
        recipient's device (undelivered). Fetches our own recent outbound messages
        for the chat and matches by id. Returns ``None`` — not a guess — when the
        message isn't found or the fetch fails, so callers can fail open and never
        stop a campaign on an unverifiable delivery status.
        """
        try:
            msgs = self.get_chat_messages(
                chat_id, limit=50, download_media=False, filter_from_me=True
            )
        except Exception:
            return None
        for m in msgs:
            if m.id == message_id:
                return m.ack
        return None

    def get_chats_overview(self, limit: int = 50) -> list[ChatOverview]:
        """Fetch chat summaries via ``GET /api/{session}/chats/overview``."""
        try:
            resp = requests.get(
                self._url(f"/api/{self.session}/chats/overview"),
                params={"limit": limit, "offset": 0},
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_chats: list[dict[str, Any]] = resp.json()
        except Exception as exc:
            self._record_poll_failure("fetch chats overview", exc)
            return []
        self._record_poll_success("fetch chats overview")

        result: list[ChatOverview] = []
        for chat in raw_chats:
            last_msg_raw = chat.get("lastMessage")
            last_msg: Message | None = None
            if last_msg_raw:
                last_msg = Message(
                    id=last_msg_raw.get("id", ""),
                    timestamp=last_msg_raw.get("timestamp", 0),
                    from_number=last_msg_raw.get("from", ""),
                    to=last_msg_raw.get("to"),
                    body=last_msg_raw.get("body", ""),
                    from_me=last_msg_raw.get("fromMe", False),
                    has_media=last_msg_raw.get("hasMedia", False),
                    media_url=(last_msg_raw.get("media") or {}).get("url"),
                )
            if "archive" in chat:
                archived = bool(chat["archive"])
            else:
                archived = bool((chat.get("_chat") or {}).get("archive", False))
            result.append(
                ChatOverview(
                    id=chat.get("id", ""),
                    name=chat.get("name"),
                    last_message=last_msg,
                    archived=archived,
                )
            )
        return result

    def download_media(
        self, media_url: str, *, max_bytes: int = 8 * 1024 * 1024
    ) -> tuple[bytes, str] | None:
        """GET media bytes + mimetype from a Waha media URL (X-Api-Key auth).

        Returns ``(data, mimetype)`` for ``image/*`` only; ``None`` on error,
        non-image content type, or oversize payload.

        The media URL is already absolute (from the Waha payload) so it is used
        directly without ``self._url()``.  A JSON Content-Type header would be
        wrong for a binary GET, so only the auth header is sent.
        """
        auth_headers: dict[str, str] = {}
        if self.api_key:
            auth_headers["X-Api-Key"] = self.api_key
        try:
            resp = requests.get(
                media_url,
                headers=auth_headers,
                timeout=self.timeout,
                stream=True,
            )
            if resp.status_code >= 400:
                log.debug("download_media HTTP %d for %s", resp.status_code, media_url)
                return None
            content_type: str = resp.headers.get("Content-Type", "")
            mimetype = content_type.split(";")[0].strip()
            if not mimetype.startswith("image/"):
                log.debug("download_media skipped non-image content-type %r", mimetype)
                return None
            content_length_hdr = resp.headers.get("Content-Length")
            if content_length_hdr is not None:
                try:
                    if int(content_length_hdr) > max_bytes:
                        log.warning(
                            "download_media skipped oversize image: Content-Length %s > %d",
                            content_length_hdr,
                            max_bytes,
                        )
                        return None
                except ValueError:
                    pass
            data = resp.raw.read(max_bytes + 1)
            if len(data) > max_bytes:
                log.warning(
                    "download_media skipped oversize image: read %d bytes > %d",
                    len(data),
                    max_bytes,
                )
                return None
            return data, mimetype
        except requests.exceptions.ConnectionError:
            log.debug("download_media connection error for %s", media_url)
            return None
        except requests.exceptions.Timeout:
            log.debug("download_media timed out for %s", media_url)
            return None
        except Exception as exc:
            log.warning("download_media unexpected error for %s: %s", media_url, exc)
            return None

    # -- health check ------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if the Waha session is in WORKING state."""
        try:
            info = self.get_session()
            return info.status == "WORKING"
        except Exception:
            return False
