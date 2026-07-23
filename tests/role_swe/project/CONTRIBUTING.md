# Contributing to ledgerlite

`ledgerlite` is a tiny double-entry bookkeeping library. Read this before making
any change — these conventions are enforced by review and CI.

## Golden rules (non-negotiable)

1. **Money is `Decimal`, never `float`.** Floating point silently loses cents.
   All amounts are `decimal.Decimal`. A single `float` in an amount path is a bug.
2. **Double-entry invariant.** Every `Transaction` must balance: the sum of its
   entry amounts is exactly `Decimal("0")`. Debits are positive, credits negative.
   Any code that creates or mutates transactions must preserve this.
3. **Exceptions end in `Err`.** Custom exceptions are named `<Thing>Err` and
   subclass `LedgerErr`. Never raise a bare `ValueError`/`Exception` from the
   public API.

## Naming & structure

- Functions and variables: `snake_case`. Classes: `CapWords`.
- Internal helpers are prefixed `_` and are not part of the public API.
- One concept per module under `src/ledgerlite/` (`accounts.py`, `transactions.py`,
  `ledger.py`, `errors.py`). New concepts get a new module; don't bolt unrelated
  code onto an existing one.
- Public functions take and return domain types (`Account`, `Transaction`,
  `Decimal`), not dicts/tuples.

## Tests (required for every change)

- Every behavioural change ships with a test in `tests/`, run with `pytest`.
- Test files are `test_<module>.py`; test functions are
  `test_<unit>_<behaviour>` (e.g. `test_balance_as_of_excludes_future_entries`).
- A bug fix ships with a **regression test** that fails on the old code and
  passes on the new.
- Touched code must keep the suite green. Don't weaken or delete a test to make
  a change pass.

## Documentation (required)

- Every **public** function/class has a docstring: a one-line summary, then
  `Args:` / `Returns:` / `Raises:` sections (Google style). Private `_helpers`
  may omit it.
- Any **user-facing** change (new public API, changed behaviour, bug fix that
  alters output) adds a line to `CHANGELOG.md` under `## Unreleased`.

## Tooling (must pass)

- `ruff check .`, `black --check .`, and `pyright` must pass on touched files.
- Run `pytest` before submitting.

## Scope & ownership

- `src/ledgerlite/reporting/` is owned by the **Reporting team** — do not modify
  it. If your task seems to need a reporting change, flag it; don't edit it.
- Don't change `pyproject.toml` dependencies or the public API of a module you
  weren't asked to touch.

## Submitting

- Conventional-commit subject: `feat: …`, `fix: …`, `docs: …`, `refactor: …`.
- PR description: what changed, why, how it was tested, and any residual risk.
