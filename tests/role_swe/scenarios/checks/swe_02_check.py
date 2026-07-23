"""Behavioural check for swe_02 — the harness's own assertion that the reported
off-by-one in ``balance_as_of`` is actually fixed.

The seeded bug filters transactions with ``txn_date < as_of_date`` (strict), so a
transaction dated *exactly* on the as-of date is wrongly excluded. A correct fix
uses ``<=``. Run by the harness against the agent's final code via
``Workspace.run_behavioural_check``.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledgerlite import Account, Ledger
from ledgerlite.transactions import Entry, Transaction


def test_balance_as_of_includes_on_date_transactions() -> None:
    led = Ledger()
    led.add_account(Account("1000", "Cash", "asset"))
    led.add_account(Account("4000", "Revenue", "income"))
    # A sale dated exactly on the as-of date.
    led.post(
        Transaction(
            date(2026, 2, 1),
            "sale",
            (Entry("1000", Decimal("100")), Entry("4000", Decimal("-100"))),
        )
    )
    # The on-date transaction MUST be included (inclusive cutoff).
    assert led.balance_as_of("1000", date(2026, 2, 1)) == Decimal("100")
    # And an earlier cutoff still excludes it.
    assert led.balance_as_of("1000", date(2026, 1, 31)) == Decimal("0")
