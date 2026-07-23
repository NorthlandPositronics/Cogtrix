"""#2453 — in-app slash-command help must not reference aliases the code lacks.

The CLI advertised single-letter aliases (`/m`, `/M`, `/s`, `/p`, …) in its
`long_help` examples, but `_build_slash_commands()` registers only four real
aliases. Typing the advertised alias produced "Unknown command". These tests
pin the fix (examples reference only real commands/aliases) and freeze the real
alias set so neither a fictional alias nor a dropped real one can drift back in.
"""

from __future__ import annotations

import re

from cogtrix_core.cli.commands import _build_slash_commands

# An example line in long_help looks like ``  /command args   description``.
_EXAMPLE_RE = re.compile(r"^\s+/([a-zA-Z_]+)\b")


def _valid_tokens(reg) -> set[str]:
    valid: set[str] = set()
    for cmd in reg._commands.values():
        valid.add(cmd.name)
        valid.update(cmd.aliases)
    return valid


def test_longhelp_examples_reference_only_registered_commands() -> None:
    reg = _build_slash_commands()
    valid = _valid_tokens(reg)

    offenders: list[tuple[str, str, str]] = []
    for cmd in reg._commands.values():
        for line in (cmd.long_help or "").splitlines():
            m = _EXAMPLE_RE.match(line)
            if m and m.group(1) not in valid:
                offenders.append((cmd.name, m.group(1), line.strip()))

    assert not offenders, (
        "long_help references slash-commands that are not registered — "
        f"fictional aliases (#2453): {offenders}"
    )


def test_only_the_four_real_aliases_are_registered() -> None:
    # Freeze the real alias set. Re-adding a fictional single-letter alias, or
    # dropping one of these, fails here — keeping code, in-app help, and docs
    # consistent.
    reg = _build_slash_commands()
    aliases = {a: cmd.name for cmd in reg._commands.values() for a in cmd.aliases}
    assert aliases == {
        "mem": "memory",
        "task": "tasks",
        "goals": "goal",
        "save": "export",
    }
