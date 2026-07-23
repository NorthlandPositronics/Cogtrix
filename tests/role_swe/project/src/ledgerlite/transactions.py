"""Transactions and their entries.

A :class:`Transaction` is a dated set of :class:`Entry` lines that MUST balance
to ``Decimal("0")`` (the double-entry invariant). Amounts are always ``Decimal``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Entry:
    """A single line of a transaction.

    Args:
        account_code: The account this line posts to.
        amount: Positive for a debit, negative for a credit. Always ``Decimal``.
    """

    account_code: str
    amount: Decimal


@dataclass(frozen=True)
class Transaction:
    """A balanced, dated set of entries.

    Args:
        txn_date: The date the transaction is effective.
        description: Free-text description.
        entries: The lines; their amounts must sum to ``Decimal("0")``.

    Raises:
        UnbalancedTransactionErr: If the entry amounts do not sum to zero.
    """

    txn_date: date
    description: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        total = sum((e.amount for e in self.entries), Decimal("0"))
        if total != Decimal("0"):
            from ledgerlite.errors import UnbalancedTransactionErr

            raise UnbalancedTransactionErr(f"transaction entries must sum to 0, got {total}")
