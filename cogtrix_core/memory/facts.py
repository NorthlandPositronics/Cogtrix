"""Persistent distilled facts for long-lived memory sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cogtrix_core.utils.atomic_write import atomic_write_json
from cogtrix_core.utils.path_safety import _sanitize_session_id

_FACTS_FILE_SUFFIX = "_facts.json"
_DEFAULT_TTL_DAYS = 7
_MAX_FACTS = 15


@dataclass(slots=True)
class FactsSnapshot:
    facts: list[str]
    created_at: datetime
    ttl_days: int


class PersistentFactsStore:
    """Persist distilled facts per session as a small JSON document."""

    def __init__(self, session_id: str, storage_dir: str = "data/memory/facts"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        safe_id = _sanitize_session_id(session_id)
        self._facts_path = (self.storage_dir / f"{safe_id}{_FACTS_FILE_SUFFIX}").resolve()
        try:
            self._facts_path.relative_to(self.storage_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected in session_id: {session_id!r}") from None

    def _read_payload(self) -> dict[str, Any] | None:
        if not self._facts_path.exists():
            return None
        try:
            payload = json.loads(self._facts_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _payload_to_snapshot(self, payload: dict[str, Any]) -> FactsSnapshot | None:
        facts_raw = payload.get("facts", [])
        if not isinstance(facts_raw, list):
            return None
        facts = [str(f).strip() for f in facts_raw if str(f).strip()]
        if not facts:
            return None

        created_raw = payload.get("created_at")
        ttl_raw = payload.get("ttl_days", _DEFAULT_TTL_DAYS)
        try:
            created_at = datetime.fromisoformat(str(created_raw))
        except Exception:
            return None
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        try:
            ttl_days = int(ttl_raw)
        except Exception:
            ttl_days = _DEFAULT_TTL_DAYS
        return FactsSnapshot(facts=facts[:_MAX_FACTS], created_at=created_at, ttl_days=ttl_days)

    def _is_snapshot_expired(self, snapshot: FactsSnapshot) -> bool:
        if snapshot.ttl_days <= 0:
            return True
        age = datetime.now(UTC) - snapshot.created_at
        return age > timedelta(days=snapshot.ttl_days)

    def save(self, facts: list[str], ttl_days: int = _DEFAULT_TTL_DAYS) -> None:
        cleaned = [str(f).strip() for f in facts if str(f).strip()]
        if not cleaned:
            self.clear()
            return
        payload = {
            "facts": cleaned[:_MAX_FACTS],
            "created_at": datetime.now(UTC).isoformat(),
            "ttl_days": int(ttl_days),
        }
        with atomic_write_json(self._facts_path) as handle:
            json.dump(payload, handle)

    def load(self) -> FactsSnapshot | None:
        payload = self._read_payload()
        if payload is None:
            return None
        snapshot = self._payload_to_snapshot(payload)
        if snapshot is None:
            self.clear()
            return None
        if self._is_snapshot_expired(snapshot):
            self.clear()
            return None
        return snapshot

    def is_expired(self) -> bool:
        payload = self._read_payload()
        if payload is None:
            return False
        snapshot = self._payload_to_snapshot(payload)
        return bool(snapshot and self._is_snapshot_expired(snapshot))

    def clear(self) -> None:
        self._facts_path.unlink(missing_ok=True)
