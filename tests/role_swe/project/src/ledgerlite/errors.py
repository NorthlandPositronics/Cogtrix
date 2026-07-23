"""Exception hierarchy for ledgerlite.

Per CONTRIBUTING: all custom exceptions end in ``Err`` and subclass
:class:`LedgerErr`. The public API never raises bare builtins.
"""

from __future__ import annotations


class LedgerErr(Exception):
    """Base class for all ledgerlite errors."""


class UnbalancedTransactionErr(LedgerErr):
    """Raised when a transaction's entries do not sum to zero."""


class UnknownAccountErr(LedgerErr):
    """Raised when an entry references an account not in the ledger."""
