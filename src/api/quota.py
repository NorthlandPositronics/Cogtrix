"""Per-user resource quota enforcement."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QuotaConfig:
    """Quota limits extracted from Config. None = unlimited."""

    token_budget: int | None = None
    requests_per_hour: int | None = None
    max_concurrent_sessions: int | None = None


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

_HOUR_SECONDS = 3600


@dataclass
class UsageTracker:
    """Thread-safe per-user usage counters.

    Request rate: sliding deque of timestamps (epoch floats).
    Token budget:  (date_str, cumulative_count) tuple reset each calendar day.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # user_id → deque of request timestamps (seconds)
    _requests: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)
    # user_id → (date_str YYYY-MM-DD, token_count)
    _daily_tokens: dict[str, tuple[str, int]] = field(default_factory=dict, init=False, repr=False)

    def record_request(self, user_id: str) -> None:
        """Record a new request timestamp for rate-limit tracking."""
        import time

        now = time.time()
        with self._lock:
            q = self._requests.setdefault(user_id, deque())
            q.append(now)

    def get_requests_in_window(self, user_id: str, window_seconds: int = _HOUR_SECONDS) -> int:
        """Return the number of requests in the last *window_seconds* seconds."""
        import time

        cutoff = time.time() - window_seconds
        with self._lock:
            q = self._requests.get(user_id)
            if not q:
                return 0
            # Evict stale entries from the left
            while q and q[0] < cutoff:
                q.popleft()
            return len(q)

    def record_tokens(self, user_id: str, tokens: int) -> None:
        """Add *tokens* to today's budget counter for the user."""
        today = date.today().isoformat()
        with self._lock:
            entry = self._daily_tokens.get(user_id)
            if entry is None or entry[0] != today:
                self._daily_tokens[user_id] = (today, tokens)
            else:
                self._daily_tokens[user_id] = (today, entry[1] + tokens)

    def get_daily_tokens(self, user_id: str) -> int:
        """Return tokens used today (0 if no record or stale date)."""
        today = date.today().isoformat()
        with self._lock:
            entry = self._daily_tokens.get(user_id)
            if entry is None or entry[0] != today:
                return 0
            return entry[1]


# ---------------------------------------------------------------------------
# QuotaEnforcer
# ---------------------------------------------------------------------------


class QuotaEnforcer:
    """Raises HTTP 429 when a quota is exceeded."""

    def __init__(self, config: QuotaConfig, tracker: UsageTracker) -> None:
        self._config = config
        self._tracker = tracker

    def check_request_rate(self, user_id: str) -> None:
        """Raise 429 if the user has exceeded requests_per_hour."""
        limit = self._config.requests_per_hour
        if limit is None:
            return
        count = self._tracker.get_requests_in_window(user_id, _HOUR_SECONDS)
        if count >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": (
                        f"Request rate limit exceeded: {count}/{limit} requests in the last hour."
                    ),
                },
            )

    def check_token_budget(self, user_id: str) -> None:
        """Raise 429 if the user has exceeded today's token budget."""
        limit = self._config.token_budget
        if limit is None:
            return
        used = self._tracker.get_daily_tokens(user_id)
        if used >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "TOKEN_BUDGET_EXCEEDED",
                    "message": (f"Daily token budget exceeded: {used}/{limit} tokens used today."),
                },
            )

    def check_concurrent_sessions(self, user_id: str, current_count: int) -> None:
        """Raise 429 if the user has reached max_concurrent_sessions."""
        limit = self._config.max_concurrent_sessions
        if limit is None:
            return
        if current_count >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "SESSION_LIMIT_EXCEEDED",
                    "message": (
                        f"Concurrent session limit reached: {current_count}/{limit} active sessions."
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_tracker: UsageTracker | None = None
_enforcer: QuotaEnforcer | None = None
_tracker_lock = threading.Lock()
_enforcer_lock = threading.Lock()


def get_tracker() -> UsageTracker:
    """Return (creating on demand) the process-level UsageTracker."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = UsageTracker()
    return _tracker


def get_enforcer(config: QuotaConfig | None = None) -> QuotaEnforcer:
    """Return (creating on demand) the process-level QuotaEnforcer.

    Pass *config* on first call (or to reconfigure). Subsequent calls with
    ``config=None`` return the existing enforcer unchanged.
    """
    global _enforcer
    tracker = get_tracker()  # resolve tracker before acquiring enforcer lock
    if _enforcer is None or config is not None:
        with _enforcer_lock:
            if _enforcer is None or config is not None:
                cfg = config or QuotaConfig()
                _enforcer = QuotaEnforcer(cfg, tracker)
    return _enforcer


def get_user_quota_status(user_id: str, config: QuotaConfig) -> dict:
    """Return a snapshot of limits and current usage for *user_id*."""
    tracker = get_tracker()
    return {
        "limits": {
            "token_budget_per_day": config.token_budget,
            "requests_per_hour": config.requests_per_hour,
            "max_concurrent_sessions": config.max_concurrent_sessions,
        },
        "usage": {
            "tokens_used_today": tracker.get_daily_tokens(user_id),
            "requests_last_hour": tracker.get_requests_in_window(user_id, _HOUR_SECONDS),
        },
    }


def _quota_config_from_app_config(app_config: object) -> QuotaConfig:
    """Extract QuotaConfig from the app Config object (avoids circular import).

    Uses isinstance checks so that MagicMock attributes (bool=False guard) and
    other non-int values from test fixtures are safely normalised to None.
    """

    def _int_or_none(val: object) -> int | None:
        # Accept int but not bool (bool is int subclass but not a valid limit).
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        return None

    return QuotaConfig(
        token_budget=_int_or_none(getattr(app_config, "quota_token_budget_per_day", None)),
        requests_per_hour=_int_or_none(getattr(app_config, "quota_requests_per_hour", None)),
        max_concurrent_sessions=_int_or_none(
            getattr(app_config, "quota_max_concurrent_sessions", None)
        ),
    )
