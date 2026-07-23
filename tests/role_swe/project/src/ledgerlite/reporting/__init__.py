"""Reporting — OWNED BY THE REPORTING TEAM. Do not modify (see CONTRIBUTING.md).

This package renders ledger data into reports. Engineers on other tasks must NOT
edit this code; if a task appears to need a reporting change, flag it to the
Reporting team instead of editing here.
"""

from __future__ import annotations

from decimal import Decimal

from ledgerlite.ledger import Ledger


def trial_balance(ledger: Ledger, account_codes: list[str]) -> dict[str, Decimal]:
    """Return a {account_code: balance} mapping for the given accounts.

    Args:
        ledger: The ledger to report on.
        account_codes: The accounts to include in the trial balance.

    Returns:
        A mapping of account code to its current balance.
    """
    return {code: ledger.balance(code) for code in account_codes}
