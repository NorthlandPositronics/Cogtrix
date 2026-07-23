"""Baseline tests for ledgerlite — the suite the agent must keep green."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ledgerlite import (
    Account,
    Entry,
    Ledger,
    Transaction,
    UnbalancedTransactionErr,
    UnknownAccountErr,
)


def _cash_and_income() -> Ledger:
    ledger = Ledger()
    ledger.add_account(Account("1000", "Cash", "asset"))
    ledger.add_account(Account("4000", "Sales", "income"))
    return ledger


def test_balanced_transaction_posts_and_updates_balances() -> None:
    ledger = _cash_and_income()
    ledger.post(
        Transaction(
            date(2026, 1, 1),
            "First sale",
            (Entry("1000", Decimal("100")), Entry("4000", Decimal("-100"))),
        )
    )
    assert ledger.balance("1000") == Decimal("100")
    assert ledger.balance("4000") == Decimal("-100")


def test_unbalanced_transaction_is_rejected() -> None:
    with pytest.raises(UnbalancedTransactionErr):
        Transaction(date(2026, 1, 1), "bad", (Entry("1000", Decimal("100")),))


def test_post_unknown_account_raises() -> None:
    ledger = _cash_and_income()
    txn = Transaction(
        date(2026, 1, 1),
        "to nowhere",
        (Entry("1000", Decimal("5")), Entry("9999", Decimal("-5"))),
    )
    with pytest.raises(UnknownAccountErr):
        ledger.post(txn)


def test_balance_of_unknown_account_raises() -> None:
    ledger = _cash_and_income()
    with pytest.raises(UnknownAccountErr):
        ledger.balance("9999")
