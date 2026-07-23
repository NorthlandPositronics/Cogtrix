"""Doc-drift guards for #2063.

The migration count documented in ARCHITECTURE.md must match the number of
Alembic version scripts on disk. This is the lightweight CI check the issue
asks for: it fails loudly the next time a migration is added without updating
the docs, instead of letting onboarding/incident docs silently rot.

(The tool-count and ADR-index items from #2063 are not guarded here: the tool
catalogue size depends on which provider keys are configured, and the ADR index
lives in the separate docs/optional submodule.)
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic_revision_count() -> int:
    versions = _REPO_ROOT / "alembic" / "versions"
    return len([p for p in versions.glob("*.py") if p.name != "__init__.py"])


def test_architecture_migration_count_matches_alembic() -> None:
    arch = (_REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    m = re.search(r"(\d+)\s+migrations ship with the project", arch)
    assert (
        m is not None
    ), "ARCHITECTURE.md no longer states the migration count in the expected form"
    documented = int(m.group(1))
    actual = _alembic_revision_count()
    assert documented == actual, (
        f"ARCHITECTURE.md documents {documented} migrations but "
        f"alembic/versions has {actual}. Update docs/ARCHITECTURE.md (#2063)."
    )


def test_architecture_migration_range_upper_bound_matches() -> None:
    """The documented (0001–NNNN) range upper bound should match the count too."""
    arch = (_REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    m = re.search(r"\(0001[–-](\d{4})\)", arch)
    assert m is not None, "ARCHITECTURE.md no longer states the migration range (0001–NNNN)"
    assert int(m.group(1)) == _alembic_revision_count()
