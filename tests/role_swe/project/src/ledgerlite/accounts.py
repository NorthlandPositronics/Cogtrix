"""Accounts in the ledger.

An :class:`Account` is an immutable identity (code + name + type). Balances are
computed from transactions by the :class:`~ledgerlite.ledger.Ledger`, never stored
on the account itself.
"""

from __future__ import annotations

from dataclasses import dataclass

ACCOUNT_TYPES = frozenset({"asset", "liability", "equity", "income", "expense"})


@dataclass(frozen=True)
class Account:
    """A named account in the chart of accounts.

    Args:
        code: Short unique account code (e.g. ``"1000"``).
        name: Human-readable account name (e.g. ``"Cash"``).
        type: One of :data:`ACCOUNT_TYPES`.
    """

    code: str
    name: str
    type: str

    def __post_init__(self) -> None:
        if self.type not in ACCOUNT_TYPES:
            from ledgerlite.errors import LedgerErr

            raise LedgerErr(f"unknown account type {self.type!r}; expected one of {ACCOUNT_TYPES}")
