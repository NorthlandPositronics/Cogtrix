"""Pytest entry point for the quality harness CI gate."""

from __future__ import annotations

import pytest

from tests.quality.harness import run_scenario
from tests.quality.metrics import compute_all
from tests.quality.scenario import load_all_scenarios

ALL_SCENARIOS = load_all_scenarios()
CRITICAL_SCENARIOS = [scenario for scenario in ALL_SCENARIOS if scenario.is_critical]
TIER1_SCENARIOS = [scenario for scenario in ALL_SCENARIOS if scenario.is_tier1]
TIER2_SCENARIOS = [scenario for scenario in ALL_SCENARIOS if scenario.is_tier2]


@pytest.mark.quality
@pytest.mark.critical
@pytest.mark.parametrize("scenario", CRITICAL_SCENARIOS, ids=[s.id for s in CRITICAL_SCENARIOS])
def test_critical_tier1(scenario):
    """Block CI if any critical Tier 1 metric fails."""
    result = run_scenario(scenario)
    report = compute_all(result)
    assert report.tier1_pass, f"Tier 1 failure in scenario '{scenario.id}':\n" + "\n".join(
        f"  {warning}" for warning in report.tier2_warnings
    )


@pytest.mark.quality
@pytest.mark.tier1
@pytest.mark.parametrize("scenario", TIER1_SCENARIOS, ids=[s.id for s in TIER1_SCENARIOS])
def test_all_scenarios_tier1(scenario):
    """Run all Tier 1 scenarios and fail on Tier 1 violations."""
    result = run_scenario(scenario)
    report = compute_all(result)
    assert report.tier1_pass, f"Tier 1 failure in '{scenario.id}': {report.tier2_warnings}"


@pytest.mark.quality
@pytest.mark.tier2
@pytest.mark.parametrize("scenario", TIER2_SCENARIOS, ids=[s.id for s in TIER2_SCENARIOS])
def test_all_scenarios_tier2(scenario):
    """Run all Tier 2 scenarios and compute warning gates."""
    result = run_scenario(scenario)
    report = compute_all(result)
    assert report.scenario_id == scenario.id
