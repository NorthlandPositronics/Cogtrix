"""
WhatsApp channel for Cogtrix assistant mode.

Wraps WahaClient to provide polling and sending via the Waha self-hosted API.
Contact filtering mirrors the logic in src/tools/whatsapp.py.
"""

from __future__ import annotations

import collections
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.assistant.channel import Channel, IncomingMessage, SendResult, parse_duration
from src.tools._whatsapp_client import REQUESTS_AVAILABLE, ChatOverview, Message, WahaClient

log = logging.getLogger("cogtrix")

_REACTIVATION_LOOKBACK: float = 300.0  # seconds; limits replay after watermark eviction


def _normalize_number(number: str) -> str:
    """Normalize a phone number or Waha chatId to E.164 form.

    Handles ``@c.us``, ``@s.whatsapp.net``, and ``@lid`` (Linked ID) formats.
    LID-format identifiers are returned as-is since they can't be mapped to
    phone numbers.
    """
    key = number.strip()
    if "@lid" in key:
        return key
    key = key.replace("@c.us", "").replace("@s.whatsapp.net", "")
    if key.isdigit():
        key = f"+{key}"
    return key


def _extract_phone_from_name(name: str | None) -> str | None:
    """Try to extract an E.164 phone number from a chat display name.

    Waha often sets ``chat.name`` to a formatted phone number for contacts
    that aren't saved in the phone book (e.g. ``+971 50 406 9790``).
    Returns the normalized number or ``None`` if the name doesn't look like one.
    """
    if not name:
        return None
    digits = re.sub(r"[^0-9+]", "", name)
    if not digits:
        return None
    if digits.startswith("+") and len(digits) >= 8:
        return digits
    if digits.isdigit() and len(digits) >= 7:
        return f"+{digits}"
    return None


_VALID_FILTER_MODES = frozenset({"none", "allow", "ignore", "blacklist"})


def _normalize_filter_mode(mode: str) -> str:
    """Map legacy filter_mode values to current names."""
    mode = mode.strip().lower()
    if mode == "whitelist":
        return "allow"
    if mode not in _VALID_FILTER_MODES:
        log.warning("Unrecognized filter_mode %r — defaulting to 'none'", mode)
        return "none"
    return mode


def _check_receive_contact(
    from_field: str,
    filter_mode: str,
    contacts: list[str],
    chat_id: str = "",
    chat_name: str | None = None,
    phonebook: dict[str, str] | None = None,
) -> bool:
    """Return True if the inbound sender passes the contact filter.

    Builds a set of candidate identifiers from the message metadata and checks
    them against the contacts list.  For ``@lid`` senders whose phone number
    is not available, the phonebook is consulted: if any phonebook *value*
    (a phone number in the contacts list) is mapped from a *key* that
    case-insensitively matches the ``chat_name``, the sender is considered
    to match that contact entry.
    """
    if filter_mode == "none":
        return True

    candidates = {_normalize_number(from_field)}
    if chat_id and chat_id != from_field:
        candidates.add(_normalize_number(chat_id))
    phone_from_name = _extract_phone_from_name(chat_name)
    if phone_from_name:
        candidates.add(phone_from_name)

    # Phonebook reverse-lookup: match chat display name to a phonebook key
    # so that "@lid" contacts whose name matches a phonebook entry (e.g.
    # phonebook "me" → "+971503308667" and chat name "D A") can be resolved.
    # This doesn't help directly — but if the user adds an entry like
    # "d a" → "+971503308667", it will match.
    #
    # More usefully: allow the contacts list to contain raw LID identifiers
    # (e.g. "178774490505455@lid") alongside phone numbers.
    if phonebook and chat_name:
        name_lower = chat_name.strip().lower()
        for pb_key, pb_value in phonebook.items():
            if pb_key.strip().lower() == name_lower:
                candidates.add(_normalize_number(pb_value))

    resolved_contacts: list[str] = []
    for c in contacts:
        if phonebook and c in phonebook:
            resolved_contacts.append(phonebook[c])
        else:
            resolved_contacts.append(c)
    normalized_contacts = {_normalize_number(c) for c in resolved_contacts}

    if filter_mode == "allow":
        return bool(candidates & normalized_contacts)
    if filter_mode in ("ignore", "blacklist"):
        return not bool(candidates & normalized_contacts)
    return True


class WhatsAppChannel(Channel):
    """WhatsApp channel backed by a Waha instance."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._client = WahaClient(
            base_url=config.get("waha_url", "http://localhost:3000"),
            api_key=config.get("api_key"),
            session=config.get("session", "default"),
        )
        self._filter_mode: str = _normalize_filter_mode(config.get("filter_mode", "none"))
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._LID_CACHE_MAX: int = 1024
        self._LID_NEGATIVE_TTL: float = float(config.get("lid_negative_ttl", 300.0))
        self._lid_cache: collections.OrderedDict[str, tuple[str | None, float]] = (
            collections.OrderedDict()
        )
        self._lid_cache_lock = threading.Lock()
        self._chat_watermarks: dict[str, int] = {}
        self._watermark_timestamps: dict[str, float] = {}
        self._overview_snapshot: dict[str, str] = {}
        self._snapshot_timestamps: dict[str, float] = {}
        self._SNAPSHOT_TTL: float = 3600.0
        self._WATERMARK_TTL: float = 604800.0
        self._seen_ids: dict[str, float] = {}
        self._SEEN_TTL: float = 600.0
        self._overview_limit: int = int(config.get("overview_limit", 50))
        self._ignore_archived: bool = bool(config.get("ignore_archived", True))
        self._ignore_older_than: float | None = parse_duration(config.get("ignore_older_than"))
        self._locally_archived: set[str] = set()
        self._archived_snapshot: set[str] = set()
        self._chat_errors: dict[str, tuple[int, float]] = {}
        self._message_fetch_limit: int = int(config.get("message_fetch_limit", 50))
        self._FETCH_ERROR_BASE: float = 30.0
        self._FETCH_ERROR_MAX: float = 300.0
        self._session_check_interval: float = 60.0
        self._last_session_check: float = 0.0
        if self._filter_mode != "none":
            log.info(
                "WhatsApp filter_mode=%s with %d contacts",
                self._filter_mode,
                len(self._contacts),
            )

    @property
    def name(self) -> str:
        return "whatsapp"

    def _resolve_lid(self, lid: str) -> str | None:
        """Resolve a ``@lid`` identifier to a phone number, with LRU/TTL caching."""
        now = time.monotonic()
        with self._lid_cache_lock:
            entry = self._lid_cache.get(lid)
            if entry is not None:
                phone, expires_at = entry
                if now < expires_at:
                    self._lid_cache.move_to_end(lid)
                    return phone
                del self._lid_cache[lid]

        pn = self._client.resolve_lid(lid)

        with self._lid_cache_lock:
            expires_at = float("inf") if pn else now + self._LID_NEGATIVE_TTL
            self._lid_cache[lid] = (pn, expires_at)
            self._lid_cache.move_to_end(lid)
            while len(self._lid_cache) > self._LID_CACHE_MAX:
                self._lid_cache.popitem(last=False)
        if pn:
            log.debug("Resolved LID %s → %s", lid, pn)
        return pn

    def _ensure_session(self) -> None:
        """Check if the Waha session is active and start it if not.

        Throttled to run at most once per ``_session_check_interval`` seconds
        to avoid hammering the Waha API on every poll cycle.
        """
        now = time.monotonic()
        if now - self._last_session_check < self._session_check_interval:
            return
        self._last_session_check = now
        try:
            info = self._client.get_session()
            if info.status == "WORKING":
                return
            log.warning("Waha session '%s' is %s — attempting restart", info.name, info.status)
            if self._client.start_session():
                log.info("Waha session '%s' start request sent", self._client.session)
            else:
                log.error("Failed to start Waha session '%s'", self._client.session)
        except Exception as exc:
            log.debug("Waha session check failed: %s", exc)

    def _can_skip_chat(self, chat: ChatOverview) -> bool:
        """Return True when the chat can be skipped without fetching messages.

        Conservative: only skips when a definitive no-match decision is reachable
        from chat.id and chat.name alone.
        """
        if self._filter_mode == "none":
            return False

        if "@g.us" in chat.id:
            return False

        candidates: set[str] = set()
        if chat.id:
            candidates.add(_normalize_number(chat.id))
        phone_from_name = _extract_phone_from_name(chat.name)
        if phone_from_name:
            candidates.add(phone_from_name)
        if self._phonebook and chat.name:
            name_lower = chat.name.strip().lower()
            for pb_key, pb_value in self._phonebook.items():
                if pb_key.strip().lower() == name_lower:
                    candidates.add(_normalize_number(pb_value))

        resolved_contacts: list[str] = []
        for c in self._contacts:
            if self._phonebook and c in self._phonebook:
                resolved_contacts.append(self._phonebook[c])
            else:
                resolved_contacts.append(c)
        normalized_contacts = {_normalize_number(c) for c in resolved_contacts}

        match = bool(candidates & normalized_contacts)

        if self._filter_mode == "allow":
            return bool(candidates) and not match

        if self._filter_mode == "ignore":
            return match

        # blacklist: never skip — delete+archive side effects require message fetch
        return False

    def _prefetch_lids(self, messages: list[Message]) -> None:
        """Resolve all uncached @lid identifiers in parallel before processing."""
        now = time.monotonic()
        uncached: set[str] = set()
        with self._lid_cache_lock:
            for msg in messages:
                if "@lid" in msg.from_number:
                    entry = self._lid_cache.get(msg.from_number)
                    if entry is None:
                        uncached.add(msg.from_number)
                    else:
                        _, expires_at = entry
                        if now >= expires_at:
                            uncached.add(msg.from_number)

        if not uncached:
            return

        if len(uncached) == 1:
            self._resolve_lid(next(iter(uncached)))
            return

        # Use explicit ThreadPoolExecutor (not `with`) so shutdown(wait=False)
        # can be used on timeout — `__exit__` calls shutdown(wait=True) which
        # blocks on hung threads.
        pool = ThreadPoolExecutor(max_workers=min(len(uncached), 8))
        try:
            futures = [pool.submit(self._resolve_lid, number) for number in uncached]
            for future in futures:
                try:
                    future.result(timeout=10)
                except TimeoutError:
                    future.cancel()
                    log.warning("LID resolution timed out after 10s — skipping one number")
        finally:
            pool.shutdown(wait=False)

    def poll(self) -> list[IncomingMessage]:
        self._ensure_session()

        now = time.monotonic()
        self._evict_stale_snapshots(now)

        chats = self._client.get_chats_overview(limit=self._overview_limit)
        if len(chats) >= self._overview_limit:
            log.warning(
                "Chat overview returned %d chats (limit %d) — some chats may be missed",
                len(chats),
                self._overview_limit,
            )
        result: list[IncomingMessage] = []

        changed_chats: list[ChatOverview] = []
        new_archived_snapshot: set[str] = set()
        for chat in chats:
            if chat.id in self._locally_archived:
                if not chat.archived:
                    try:
                        self._client.archive_chat(chat.id)
                    except Exception as exc:
                        log.debug("Failed to archive chat %s: %s", chat.id, exc)
                continue
            if self._ignore_archived and chat.archived:
                new_archived_snapshot.add(chat.id)
                # Record the last_message so we can distinguish auto-unarchive
                # (new message) from manual unarchive (same message) later.
                if chat.last_message is not None:
                    self._overview_snapshot[chat.id] = chat.last_message.id
                    self._snapshot_timestamps[chat.id] = now
                continue
            # Auto-unarchived by an incoming message: the chat was archived on
            # the previous poll but now appears non-archived AND has a new
            # last_message.  Re-archive it and skip — the user archived it
            # intentionally; WhatsApp auto-unarchived on the incoming message.
            # If last_message is unchanged, the user manually unarchived the
            # chat (deliberate) so we let it through.
            if self._ignore_archived and chat.id in self._archived_snapshot:
                prev_last = self._overview_snapshot.get(chat.id)
                cur_last = chat.last_message.id if chat.last_message else None
                if cur_last is not None and cur_last != prev_last:
                    log.debug(
                        "Chat %s was auto-unarchived by incoming message — re-archiving",
                        chat.id,
                    )
                    try:
                        self._client.archive_chat(chat.id)
                    except Exception as exc:
                        log.debug("Failed to re-archive chat %s: %s", chat.id, exc)
                    # Update snapshot so we don't keep re-archiving the same message
                    self._overview_snapshot[chat.id] = cur_last
                    self._snapshot_timestamps[chat.id] = now
                    new_archived_snapshot.add(chat.id)
                    continue
            msg = chat.last_message
            if msg is None:
                continue
            if self._overview_snapshot.get(chat.id) == msg.id:
                continue
            if self._can_skip_chat(chat):
                self._overview_snapshot[chat.id] = msg.id
                self._snapshot_timestamps[chat.id] = now
                continue
            changed_chats.append(chat)
        self._archived_snapshot = new_archived_snapshot

        all_fetched: list[tuple[Message, ChatOverview]] = []
        for chat in changed_chats:
            err_entry = self._chat_errors.get(chat.id)
            if err_entry is not None:
                _, retry_after = err_entry
                if now < retry_after:
                    continue
            try:
                messages = self._fetch_new_messages(chat)
            except Exception as exc:
                prev_count = err_entry[0] if err_entry else 0
                new_count = prev_count + 1
                backoff = min(
                    self._FETCH_ERROR_BASE * (2 ** (new_count - 1)),
                    self._FETCH_ERROR_MAX,
                )
                self._chat_errors[chat.id] = (new_count, now + backoff)
                log.warning(
                    "Failed to fetch messages for chat %s (attempt %d, retry in %.0fs): %s",
                    chat.id,
                    new_count,
                    backoff,
                    exc,
                )
                continue
            self._chat_errors.pop(chat.id, None)
            for msg in messages:
                all_fetched.append((msg, chat))
            if chat.last_message is not None:
                self._overview_snapshot[chat.id] = chat.last_message.id
                self._snapshot_timestamps[chat.id] = now

        self._prefetch_lids([m for m, _ in all_fetched])

        for msg, chat in all_fetched:
            incoming = self._process_message(msg, chat, now)
            if incoming is not None:
                result.append(incoming)

        return result

    def _fetch_new_messages(self, chat: ChatOverview) -> list[Message]:
        """Fetch unseen user messages for a chat using per-chat watermark."""
        watermark = self._chat_watermarks.get(chat.id, 0)
        if watermark > 0:
            filter_ts = watermark
        else:
            # No watermark: new chat or evicted watermark — look back a short window only
            # to prevent re-processing old messages after watermark eviction (BUG-055).
            filter_ts = time.time() - _REACTIVATION_LOOKBACK

        messages = self._client.get_chat_messages(
            chat_id=chat.id,
            limit=self._message_fetch_limit,
            filter_from_me=False,
            filter_timestamp_gte=filter_ts,
        )
        messages.sort(key=lambda m: m.timestamp)
        return messages

    def _process_message(
        self, msg: Message, chat: ChatOverview, now: float
    ) -> IncomingMessage | None:
        """Process a single message. Returns IncomingMessage or None if filtered."""
        if msg.timestamp > self._chat_watermarks.get(chat.id, 0):
            self._chat_watermarks[chat.id] = msg.timestamp
            self._watermark_timestamps[chat.id] = now

        if msg.from_me:
            return None

        if self._ignore_older_than is not None:
            age = time.time() - msg.timestamp
            if age > self._ignore_older_than:
                log.debug(
                    "Skipping message %s from %s — too old (%.0fs > %.0fs)",
                    msg.id,
                    chat.id,
                    age,
                    self._ignore_older_than,
                )
                return None

        if len(self._seen_ids) > 500:
            cutoff = now - self._SEEN_TTL
            self._seen_ids = {k: v for k, v in self._seen_ids.items() if v > cutoff}
        if msg.id in self._seen_ids:
            return None

        resolved_from = msg.from_number
        resolved_phone: str | None = None
        if "@lid" in msg.from_number:
            pn = self._resolve_lid(msg.from_number)
            if pn:
                resolved_from = pn
                resolved_phone = pn
        resolved_chat_id = chat.id
        if "@lid" in chat.id and chat.id != msg.from_number:
            pn = self._resolve_lid(chat.id)
            if pn:
                resolved_chat_id = pn

        if not _check_receive_contact(
            resolved_from,
            self._filter_mode,
            self._contacts,
            chat_id=resolved_chat_id,
            chat_name=chat.name,
            phonebook=self._phonebook,
        ):
            if self._filter_mode == "blacklist":
                try:
                    self._client.delete_message(chat.id, msg.id)
                    self._client.archive_chat(chat.id)
                except Exception as exc:
                    log.warning("Blacklist action failed for chat %s: %s", chat.id, exc)
                self._locally_archived.add(chat.id)
                log.info("Blacklisted: deleted message and archived chat %s", chat.id)
            else:
                log.debug(
                    "Filtered out message from chat_id=%s name=%r (from=%s)",
                    chat.id,
                    chat.name,
                    msg.from_number,
                )
            return None

        if not msg.body.strip():
            return None

        self._seen_ids[msg.id] = now

        sender = msg.from_number
        for suffix in ("@c.us", "@s.whatsapp.net", "@lid"):
            sender = sender.replace(suffix, "")

        return IncomingMessage(
            channel=self.name,
            chat_id=chat.id,
            message_id=msg.id,
            sender_id=sender,
            sender_name=chat.name,
            text=msg.body,
            timestamp=float(msg.timestamp),
            resolved_phone=resolved_phone,
        )

    def _evict_stale_snapshots(self, now: float) -> None:
        """Remove snapshot and watermark entries for chats inactive past their TTLs."""
        if self._snapshot_timestamps:
            snapshot_cutoff = now - self._SNAPSHOT_TTL
            stale_snapshots = [
                cid for cid, ts in self._snapshot_timestamps.items() if ts < snapshot_cutoff
            ]
            for cid in stale_snapshots:
                self._overview_snapshot.pop(cid, None)
                self._snapshot_timestamps.pop(cid, None)

        if self._watermark_timestamps:
            watermark_cutoff = now - self._WATERMARK_TTL
            stale_watermarks = [
                cid for cid, ts in self._watermark_timestamps.items() if ts < watermark_cutoff
            ]
            for cid in stale_watermarks:
                self._chat_watermarks.pop(cid, None)
                self._watermark_timestamps.pop(cid, None)

    def send(self, chat_id: str, text: str) -> SendResult:
        result = self._client.send_text(chat_id, text)
        if not result.ok:
            log.error("WhatsApp send failed to %s: %s", chat_id, result.error)
        return SendResult(ok=result.ok, message_id=result.message_id, error=result.error)

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        result = self._client.edit_message(chat_id, message_id, text)
        if not result.ok:
            log.error("WhatsApp edit failed for %s/%s: %s", chat_id, message_id, result.error)
        return SendResult(
            ok=result.ok,
            message_id=message_id if result.ok else None,
            error=result.error,
        )

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        return self._client.delete_message(chat_id, message_id)

    def archive_chat(self, chat_id: str) -> bool:
        return self._client.archive_chat(chat_id)

    def unarchive_locally(self, chat_id: str) -> bool:
        """Remove *chat_id* from the local archive suppression set.

        Called when a chat is un-blacklisted so it can be processed again.
        Returns True if the chat was in the set.
        """
        was_present = chat_id in self._locally_archived
        self._locally_archived.discard(chat_id)
        return was_present

    def is_ready(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        allow_receive = self._config.get("allow_receive", True)
        if not allow_receive:
            return False
        try:
            return self._client.is_ready()
        except Exception:
            return False
