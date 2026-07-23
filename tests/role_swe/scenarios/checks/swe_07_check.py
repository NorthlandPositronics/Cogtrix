"""Behavioural check for swe_07 — the harness's own assertion that the agent's
``Ledger.transfer`` preserves the double-entry invariant.

Run by the harness (not the main suite) against the agent's final code via
``Workspace.run_behavioural_check``. A naive transfer that posts unbalanced
entries either raises ``UnbalancedTransactionErr`` or leaves balances that don't
net to zero — both make this fail.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledgerlite import Account, Ledger


def test_transfer_preserves_double_entry_invariant() -> None:
    led = Ledger()
    led.add_account(Account("1000", "Cash", "asset"))
    led.add_account(Account("2000", "Bank", "asset"))

    led.transfer("1000", "2000", Decimal("50"), date(2026, 1, 1))

    # Double-entry: every posting nets to zero across all accounts.
    assert led.balance("1000") + led.balance("2000") == Decimal("0")
    # And the money actually moved: 50 out of cash, 50 into bank.
    assert led.balance("1000") == Decimal("-50")
    assert led.balance("2000") == Decimal("50")
