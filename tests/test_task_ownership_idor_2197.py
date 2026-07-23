"""#2197 — task-ownership gate must deny-by-default (IDOR closure).

`_assert_task_ownership` previously did ``if task_user_id and task_user_id !=
current_user.user_id`` — so a task with an empty/legacy ``user_id`` (agent- and
CLI-spawned background tasks default ``user_id=''``) skipped the per-user check
entirely, leaving only the org check. Combined with an org-less caller +
org-less task that made any other authenticated user able to read/cancel/inspect
those tasks (cross-user disclosure). The gate now denies unless the caller owns
the task or is an admin.

These are direct unit tests of the gate (the HTTP routes additionally gate on
`require_org_context`, which masks the org-less path, so the security boundary
is best asserted at the function level).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from cogtrix_core.api.auth import TokenData  # noqa: E402
from cogtrix_core.api.org_context import OrgContext  # noqa: E402
from cogtrix_core.api.routes.tasks import _assert_task_ownership  # noqa: E402


def _user(uid: str, role: str = "user") -> TokenData:
    return TokenData(user_id=uid, role=role, raw_claims={})


def _ctx(uid: str, role: str = "user", org_id: str | None = None) -> OrgContext:
    return OrgContext(user_id=uid, role=role, org_id=org_id)


def _task(user_id: str = "", org_id: str | None = None):
    return SimpleNamespace(user_id=user_id, org_id=org_id)


def test_unowned_task_denied_to_other_user() -> None:
    """The IDOR: an empty-user_id task must NOT be accessible to another user."""
    with pytest.raises(HTTPException) as exc:
        _assert_task_ownership(_task(user_id="", org_id=None), _user("A"), _ctx("A"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "TASK_ACCESS_DENIED"


def test_cross_user_owned_task_denied() -> None:
    with pytest.raises(HTTPException) as exc:
        _assert_task_ownership(_task(user_id="B", org_id=None), _user("A"), _ctx("A"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "TASK_ACCESS_DENIED"


def test_owner_allowed() -> None:
    # Same user + same (None) org → no raise.
    _assert_task_ownership(_task(user_id="A", org_id=None), _user("A"), _ctx("A"))


def test_owner_allowed_with_org() -> None:
    _assert_task_ownership(_task(user_id="A", org_id="org1"), _user("A"), _ctx("A", org_id="org1"))


def test_admin_may_access_unowned_task() -> None:
    # Admins can reach legacy/unowned tasks (user check bypassed; org check
    # admin-bypassed). This is what restores operator access to orphaned tasks.
    _assert_task_ownership(
        _task(user_id="", org_id=None), _user("admin1", role="admin"), _ctx("admin1", role="admin")
    )


def test_admin_may_access_other_users_task() -> None:
    _assert_task_ownership(
        _task(user_id="B", org_id="org2"),
        _user("admin1", role="admin"),
        _ctx("admin1", role="admin", org_id="org1"),
    )
