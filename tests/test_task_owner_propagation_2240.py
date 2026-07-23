"""#2240 — agent/CLI-spawned background tasks must inherit the turn's owner.

After #2197 made the task-ownership gate deny-by-default, tasks created with an
empty ``user_id`` (the agent's background tool) became admin-only — so a
non-admin creator could no longer retrieve their own task. ``submit_task`` now
inherits the per-turn owner from a ContextVar (set in the API turn path) when no
explicit owner is passed, so the creator regains access. Cross-user access stays
denied by the #2197 gate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cogtrix_core.tasks.queue import reset_task_owner, set_task_owner, submit_task


def _mock_queue():
    q = MagicMock()
    q.submit.return_value = "task-123"
    return q


class TestTaskOwnerPropagation:
    def test_submit_inherits_context_owner(self) -> None:
        q = _mock_queue()
        with patch("cogtrix_core.tasks.queue.get_task_queue", return_value=q):
            token = set_task_owner("user-A")
            try:
                submit_task("researcher", "do x")  # no explicit owner — agent path
            finally:
                reset_task_owner(token)
        assert q.submit.call_args.kwargs["user_id"] == "user-A"

    def test_explicit_owner_is_not_overridden(self) -> None:
        q = _mock_queue()
        with patch("cogtrix_core.tasks.queue.get_task_queue", return_value=q):
            token = set_task_owner("user-A")
            try:
                submit_task("researcher", "do x", user_id="user-B")
            finally:
                reset_task_owner(token)
        assert q.submit.call_args.kwargs["user_id"] == "user-B"

    def test_no_owner_set_stays_empty(self) -> None:
        # CLI / non-API spawn: no ContextVar set → empty owner (admin-only).
        q = _mock_queue()
        with patch("cogtrix_core.tasks.queue.get_task_queue", return_value=q):
            submit_task("researcher", "do x")
        assert q.submit.call_args.kwargs["user_id"] == ""

    def test_reset_clears_owner_no_leak(self) -> None:
        # After reset, a subsequent spawn must NOT inherit the prior turn's owner.
        q = _mock_queue()
        with patch("cogtrix_core.tasks.queue.get_task_queue", return_value=q):
            token = set_task_owner("user-A")
            reset_task_owner(token)
            submit_task("researcher", "do x")
        assert q.submit.call_args.kwargs["user_id"] == ""
