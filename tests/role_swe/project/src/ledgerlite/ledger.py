"""The ledger: a chart of accounts plus the transactions posted against it.

The :class:`Ledger` computes balances from posted transactions. Balances are never
stored — they are always derived, so they cannot drift out of sync.
"""

from __future__ import annotations

from decimal import Decimal

from ledgerlite.accounts import Account
from ledgerlite.errors import UnknownAccountErr
from ledgerlite.transactions import Transaction


class Ledger:
    """A chart of accounts and the transactions posted to it."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._transactions: list[Transaction] = []

    def add_account(self, account: Account) -> None:
        """Register an account in the chart of accounts.

        Args:
            account: The account to add (keyed by its ``code``).
        """
        self._accounts[account.code] = account

    def post(self, transaction: Transaction) -> None:
        """Post a balanced transaction to the ledger.

        Args:
            transaction: The transaction to record. Every entry's account must
                already exist in the chart of accounts.

        Raises:
            UnknownAccountErr: If an entry references an unknown account.
        """
        for entry in transaction.entries:
            if entry.account_code not in self._accounts:
                raise UnknownAccountErr(f"unknown account {entry.account_code!r}")
        self._transactions.append(transaction)

    def balance(self, account_code: str) -> Decimal:
        """Return the current balance of an account.

        Args:
            account_code: The account whose balance to compute.

        Returns:
            The signed sum of all posted entries for the account.

        Raises:
            UnknownAccountErr: If the account is not in the chart of accounts.
        """
        if account_code not in self._accounts:
            raise UnknownAccountErr(f"unknown account {account_code!r}")
        total = Decimal("0")
        for txn in self._transactions:
            for entry in txn.entries:
                if entry.account_code == account_code:
                    total += entry.amount
        return total
