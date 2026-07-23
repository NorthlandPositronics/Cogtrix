"""Scenario-integrity tests: every scenario YAML is well-formed and self-consistent.

No Docker, no LLM — guards against a scenario referencing a missing check or seed
script, a malformed assignment, or a break-fix scenario without its fault seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.role_sysadmin.run import Scenario, load_scenario

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_ALL = sorted(_SCENARIOS_DIR.glob("sa_*.yaml"))


def test_scenarios_exist() -> None:
    assert _ALL, "no scenario YAMLs found"


@pytest.mark.parametrize("path", _ALL, ids=lambda p: p.stem)
def test_scenario_is_well_formed(path: Path) -> None:
    scenario = load_scenario(path)
    assert isinstance(scenario, Scenario)
    assert scenario.id and scenario.id.startswith("role_sa_")
    assert scenario.area in {"provisioning", "security", "ops", "breakfix"}
    assert len(scenario.assignment) > 40, "assignment looks empty/too short"

    # The on-box check must exist and be a shell script.
    check = scenario.check_path
    assert check.exists(), f"missing check script: {check}"
    assert check.read_text(encoding="utf-8").startswith("#!"), "check is not a script"

    # Break-fix scenarios must ship a fault seed; others must not claim one.
    if scenario.requires_rootcause:
        assert scenario.seed_setup is not None, "break-fix scenario has no seed"
        assert scenario.seed_setup.exists(), f"missing seed: {scenario.seed_setup}"
    if scenario.seed_setup is not None:
        assert scenario.seed_setup.exists(), f"missing seed: {scenario.seed_setup}"


def test_breakfix_scenarios_have_seeds() -> None:
    breakfix = [load_scenario(p) for p in _ALL]
    seeded = [s.id for s in breakfix if s.seed_setup is not None]
    rootcause = [s.id for s in breakfix if s.requires_rootcause]
    assert sorted(seeded) == sorted(rootcause), "seed and requires_rootcause sets must match"
