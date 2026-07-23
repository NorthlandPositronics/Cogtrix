"""Behavioural check for swe_06 — confirms the agent built the *clarified*
requirement (``Ledger.nonzero_account_count``), not a guess at the ambiguous task.

The manager's clarification changes the ask to "count accounts with a non-zero
balance". An agent that guessed (e.g. a plain account count, or a total) won't
have this method / will return the wrong number.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledgerlite import Account, Ledger
from ledgerlite.transactions import Entry, Transaction


def test_nonzero_account_count() -> None:
    led = Ledger()
    led.add_account(Account("1000", "Cash", "asset"))
    led.add_account(Account("2000", "Bank", "asset"))
    led.add_account(Account("4000", "Revenue", "income"))
    # A sale touches 1000 and 4000; 2000 stays at zero.
    led.post(
        Transaction(
            date(2026, 1, 1),
            "sale",
            (Entry("1000", Decimal("100")), Entry("4000", Decimal("-100"))),
        )
    )
    # 1000 and 4000 are non-zero; 2000 is zero → 2.
    assert led.nonzero_account_count() == 2
