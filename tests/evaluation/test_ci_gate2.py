"""Tests for Gate 2 CI smoke runner."""

from __future__ import annotations

from pathlib import Path

from tests.evaluation.ci_gate2 import run_gate2_smoke, score_result
from tests.evaluation.runner import EvalResult, EvalScenario, ModelConfig


def _scenario() -> EvalScenario:
    return EvalScenario(
        id="procurement_po_approval_basic",
        domain="procurement",
        title="PO approval",
        description="Create a PO from a supplier quote.",
        user_prompt="Create a PO.",
        system_prompt="You are Cogtrix.",
        tools_required=["create_po"],
        expected_outcome="PO created.",
        success_criteria=["contains:PO created"],
    )


def _model() -> ModelConfig:
    return ModelConfig(
        id="mock",
        provider="anthropic",
        display_name="Mock",
        tier="smoke",
        smoke=True,
        env_key="ANTHROPIC_API_KEY",
        model_id="mock-model",
        openrouter_model_id="anthropic/mock-model",
    )


def _result(
    passed: bool = True,
    error: str | None = None,
    *,
    task_completion: bool = True,
) -> EvalResult:
    """Build a mock EvalResult with task_completion mirroring tool_calls.

    The default is task_completion=True because tool_calls_made covers
    tool_calls_required.  Tests that need to exercise the partial-
    completion branch (issue #1268) pass task_completion=False
    explicitly.
    """
    return EvalResult(
        scenario_id="procurement_po_approval_basic",
        model_id="mock",
        model_display_name="Mock",
        passed=passed,
        tool_calls_made=["create_po"],
        tool_calls_required=["create_po"],
        turns_used=2,
        elapsed_seconds=0.1,
        final_response="PO created.",
        error=error,
        task_completion=task_completion,
        tool_selection_rate=100.0 if task_completion else 0.0,
    )


def test_score_result_uses_dict_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_score_scenario(payload, judge_model="claude-sonnet-4-6"):
        captured["payload"] = payload
        captured["judge_model"] = judge_model
        return 0.75

    monkeypatch.setattr("tests.evaluation.ci_gate2.score_scenario", fake_score_scenario)

    assert score_result(_scenario(), _result(), judge_model="custom-judge") == 0.75
    assert captured["judge_model"] == "custom-judge"
    assert captured["payload"]["scenario"]["id"] == "procurement_po_approval_basic"
    assert captured["payload"]["scenario"]["tools_required"] == ["create_po"]


def test_run_gate2_smoke_logs_scores_and_failure(monkeypatch) -> None:
    # Isolate from real environment keys — only set the one we want.
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")

    logs: list[str] = []
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.run_scenario",
        lambda scenario, model, active_key=None: _result(False),
    )
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 0.25,
    )

    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert exit_code == 1
    score_lines = [line for line in logs if "score=" in line]
    assert score_lines, f"No score line in logs{logs}"
    assert "score=0.25" in score_lines[0]
    assert "passed=False" in score_lines[0]


def test_run_gate2_smoke_judge_rescues_substring_mismatch(monkeypatch) -> None:
    """Substring success-criteria are too brittle for natural-language
    variation ("VP" vs "Vice President"); when the agent called every
    required tool AND the judge approves, a substring mismatch alone
    does not fail the run.

    Strict-gate behaviour (issue #1268): structural ``task_completion``
    is the floor, judge ≥ 0.5 is the ceiling — the substring check
    inside ``EvalResult.passed`` is allowed to flip False without
    failing the gate.  This test pins that allowance.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-value")

    logs: list[str] = []
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.run_scenario",
        # passed=False (substring mismatch) but task_completion=True
        # (all required tools were actually called).
        lambda scenario, model, active_key=None: _result(passed=False, task_completion=True),
    )
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 1.0,
    )

    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert exit_code == 0, f"Judge score 1.0 should pass; logs={logs}"
    score_lines = [line for line in logs if "score=" in line]
    assert score_lines and "passed=True" in score_lines[0]


def test_run_gate2_smoke_judge_cannot_rescue_partial_completion(monkeypatch) -> None:
    """Issue #1268: a run that called only some required tools must fail
    even when the judge LLM gives a passing score on the prose response.

    DeepSeek-V3 once finished ``finance_invoice_approval_workflow`` after
    calling ``classify_invoice`` only — skipping ``route_for_approval``
    and ``notify_approver`` — yet the judge rated the summary at 0.50
    and the gate let it through.  The strict gate must instead require
    structural completion as a hard floor that the judge cannot lift.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-value")

    logs: list[str] = []
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.run_scenario",
        # task_completion=False simulates the partial-completion path.
        lambda scenario, model, active_key=None: _result(passed=False, task_completion=False),
    )
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        # Judge generously gives 1.0 on prose alone.
        lambda scenario, result, judge_model="claude-sonnet-4-6": 1.0,
    )

    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert exit_code == 1, (
        "Partial completion must fail the gate even with a perfect judge score; " f"logs={logs}"
    )
    # The PARTIAL_COMPLETION diagnostic line must be emitted so CI logs are
    # actionable when the gate flips from previously-passing to failing.
    assert any(
        "PARTIAL_COMPLETION" in line for line in logs
    ), f"Missing PARTIAL_COMPLETION diagnostic in logs: {logs}"


def test_run_gate2_smoke_retries_on_empty_response_flake(monkeypatch) -> None:
    """DeepSeek-V3 via OpenRouter occasionally returns an empty response
    (tools=0, content='', no error) on certain prompts.  This is the
    same class of transient flake as a network timeout — the agent
    produced nothing usable through no fault of the test.

    The smoke runner must retry once before flunking the run, matching
    the existing transient-error retry behaviour.  Without retry, every
    smoke run has a non-trivial chance of failing on whichever DeepSeek
    scenario hit the empty-response path.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-value")

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the empty-response flake: no tools, no content,
            # no error.  Today this is reported as a clean run.
            return EvalResult(
                scenario_id=scenario.id,
                model_id=model.id,
                model_display_name=model.display_name,
                passed=False,
                tool_calls_made=[],
                tool_calls_required=list(scenario.tools_required),
                turns_used=1,
                elapsed_seconds=0.5,
                final_response="",
                error=None,
                task_completion=False,
                tool_selection_rate=0.0,
            )
        # Second attempt succeeds with full completion.
        return _result(passed=True, task_completion=True)

    logs: list[str] = []
    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 1.0,
    )

    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert call_count["n"] == 2, (
        f"Expected one retry after empty-response flake; got n={call_count['n']}. " f"Logs: {logs}"
    )
    assert exit_code == 0, f"Empty-response flake should retry and pass; logs={logs}"
    assert any(
        "RETRY" in line and "empty_response" in line.lower() for line in logs
    ), f"Expected RETRY ... empty_response log; logs={logs}"


def test_run_gate2_smoke_does_not_retry_genuine_partial_completion(monkeypatch) -> None:
    """Empty-response retry must NOT mask genuine partial completion —
    a run that called SOME tools but not all required ones is a real
    failure (issue #1268), not a flake.  Retrying it would defeat the
    strict gate's purpose.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-value")

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        # Genuine partial completion: ONE tool was called and a final
        # response was generated, but other required tools were skipped.
        # This is the #1268 failure mode and must be reported as fail.
        return EvalResult(
            scenario_id=scenario.id,
            model_id=model.id,
            model_display_name=model.display_name,
            passed=False,
            tool_calls_made=["create_po"],
            tool_calls_required=["create_po", "route_approval_request"],
            turns_used=2,
            elapsed_seconds=1.0,
            final_response="PO created.",
            error=None,
            task_completion=False,
            tool_selection_rate=50.0,
        )

    logs: list[str] = []
    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 1.0,
    )

    multi_tool_scenario = EvalScenario(
        id="procurement_po_approval_basic",
        domain="procurement",
        title="PO approval",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=["create_po", "route_approval_request"],
        expected_outcome="",
        success_criteria=[],
    )

    exit_code = run_gate2_smoke(
        scenarios=[multi_tool_scenario],
        models=[_model()],
        emit=logs.append,
    )

    assert call_count["n"] == 1, (
        "Genuine partial completion (some tools called, some missing) must NOT retry; "
        f"got n={call_count['n']}. Logs: {logs}"
    )
    assert exit_code == 1, f"Genuine partial completion must fail; logs={logs}"


def test_run_gate2_smoke_falls_back_on_captured_auth_error(monkeypatch) -> None:
    """When ``run_scenario`` catches a 402/credits error and stuffs it into
    ``EvalResult.error``, ``_try_run_with_key`` must still recognize it and
    fall back to the next priority key — otherwise the entire scenario fails
    on the first key when a later key would have worked."""
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-broke-value")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-good-value")

    # First call (OpenRouter) returns a 402-credits error in EvalResult.error.
    # Second call (the next eligible key, OpenAI) returns a clean pass.
    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _result(
                passed=False,
                error=(
                    "Error code: 402 - {'error': {'message': "
                    "'This request requires more credits...'}}"
                ),
            )
        return _result(passed=True)

    logs: list[str] = []
    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 1.0,
    )

    # Use an OpenAI-keyed model so OPENAI_API_KEY is eligible as the second key.
    openai_model = ModelConfig(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        tier="A",
        smoke=True,
        env_key="OPENAI_API_KEY",
        model_id="gpt-4o",
        openrouter_model_id="openai/gpt-4o",
    )

    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[openai_model],
        emit=logs.append,
    )

    assert (
        call_count["n"] == 2
    ), f"Expected fallback to second key (n=2 calls), got n={call_count['n']}. Logs: {logs}"
    assert exit_code == 0, f"Fallback path should succeed; logs={logs}"
    assert any(
        "KEY_FAIL" in line and "OPENROUTER_API_KEY" in line for line in logs
    ), f"Expected KEY_FAIL log for OpenRouter; logs={logs}"


def test_try_run_with_key_retries_on_transient_error(monkeypatch) -> None:
    """Transient errors (timeout, connection, rate-limit) should trigger one
    retry before giving up — this reduces flakiness from temporary provider
    issues (see issue #1124)."""
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("Connection timed out after 30s")
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model(),
        ("ANTHROPIC_API_KEY", "sk-test"),
        emit=logs.append,
    )

    assert call_count["n"] == 2, f"Expected 2 calls (1 retry), got {call_count['n']}"
    assert result is not None
    assert result.passed is True
    assert any("RETRY" in line for line in logs), f"Expected RETRY log; logs={logs}"


def test_try_run_with_key_retries_on_transient_error_in_result(monkeypatch) -> None:
    """When ``run_scenario`` catches a transient error and stuffs it into
    ``EvalResult.error``, the retry logic must recognise it and retry."""
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _result(
                passed=False,
                error="Rate limit exceeded: 429 Too Many Requests",
            )
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model(),
        ("ANTHROPIC_API_KEY", "sk-test"),
        emit=logs.append,
    )

    assert call_count["n"] == 2, f"Expected 2 calls (1 retry), got {call_count['n']}"
    assert result is not None
    assert result.passed is True
    assert any("RETRY" in line for line in logs), f"Expected RETRY log; logs={logs}"


# ── Workflow guard: Gate 2 must block merge ──────────────────────────────────


def test_ci_summary_gates_on_gate2_result() -> None:
    """The CI Summary job (.github/workflows/ci.yml) must check
    ``needs.gate2.result`` before allowing the workflow to pass.

    Without this check, a Gate 2 failure surfaces as
    ``conclusion=cancelled`` (because the gate2 job actively runs
    ``gh run cancel`` to free CI capacity on failure) and the CI
    Summary job can complete green — letting branch protection
    auto-merge the PR despite a broken LLM smoke run.

    This test guards the fix for the merge-block hole that allowed
    PR #1276 to land with a known Gate 2 failure.  A future edit
    that drops gate2 from the loop will make this test fail.
    """
    ci_yml = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    text = ci_yml.read_text()

    # The summary job must reference gate2's result inside the
    # required-checks loop.  The loop body fails on either ``failure``
    # or ``cancelled`` — both are the relevant signals for Gate 2.
    assert "needs.gate2.result" in text, "ci.yml is missing any reference to needs.gate2.result"

    # Locate the loop and assert gate2 is one of the iterated values.
    # The exact pattern is a `for result in \` block with one
    # "${{ needs.<name>.result }}" entry per line, ending in a check
    # for failure/cancelled.
    loop_start = text.find("for result in")
    assert loop_start != -1, "CI Summary required-checks loop not found in ci.yml"
    loop_end = text.find("done", loop_start)
    assert loop_end != -1, "CI Summary loop has no terminator"
    loop_body = text[loop_start:loop_end]
    assert "needs.gate2.result" in loop_body, (
        "Gate 2 result must be inside the required-checks loop, not just "
        "the summary table.  Without this, a cancelled Gate 2 lets the PR "
        "auto-merge.  See PR #1276 — the bug this guard prevents."
    )
    # Belt-and-braces: the failure/cancelled check must still be present.
    assert (
        "failure" in loop_body and "cancelled" in loop_body
    ), "CI Summary loop must fail on either 'failure' or 'cancelled' status"


def test_gate2_job_does_not_self_cancel_workflow() -> None:
    """The gate2 job must NOT call ``gh run cancel`` on failure.

    Layer B of the merge-block defenses: a self-cancel turns Gate 2's
    own status into ``conclusion=cancelled`` instead of ``failure``.
    GitHub treats those two inconsistently across branch protection
    and CI Summary aggregation paths, which is the surface PR #1276
    exploited to merge silently.  Letting the job exit with a clean
    ``failure`` makes the merge-block uniform: both branch protection
    (Layer A) and CI Summary (the loop test above) handle it.

    Sibling fail-fast is still preserved through the ``needs:`` chain
    on test-api (which lists gate2 as a dependency) — that's passive,
    consistent, and doesn't require the actions:write permission.

    A future edit that re-introduces ``gh run cancel`` on the gate2
    job will fail this test and prompt a re-evaluation.
    """
    import yaml

    ci_yml = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(ci_yml.read_text())

    gate2 = workflow.get("jobs", {}).get("gate2")
    assert gate2 is not None, "ci.yml has no jobs.gate2 — workflow refactor broke this test"

    # No step in the gate2 job may invoke `gh run cancel`.
    for step in gate2.get("steps", []):
        run = step.get("run", "") or ""
        assert "gh run cancel" not in run, (
            f"gate2 step {step.get('name')!r} runs `gh run cancel`, which "
            "would surface a Gate 2 failure as conclusion=cancelled instead "
            "of failure — the same status confusion that allowed PR #1276 "
            "to silently merge.  Remove the active cancel; passive "
            "fail-fast via test-api's needs: chain is sufficient."
        )

    # The actions:write permission was added solely for the cancel call;
    # without that step, gate2 should not need it.  Detecting it here
    # locks in the principle of least privilege for this job.
    permissions = gate2.get("permissions", {})
    assert permissions.get("actions") != "write", (
        "gate2 has actions:write permission but no step that needs it — "
        "the cancel-on-failure step that justified it has been removed.  "
        "Drop the permission to stay at least-privilege."
    )


def test_test_job_depends_on_gate2() -> None:
    """The Python unit-test job must wait for Gate 2 to succeed.

    Without this dependency, ``test`` runs in parallel with ``gate2``
    and burns up to 15 min of CI minutes on every failed smoke run.
    Making ``test`` depend on ``gate2`` causes test to be skipped
    (never started) when Gate 2 fails — saving the wasted minutes
    at the cost of ~5–6 min slower happy path.

    A future edit that drops gate2 from test's needs (returning to
    parallel execution) will fail this guard and prompt a re-eval.
    """
    import yaml

    ci_yml = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(ci_yml.read_text())

    test_job = workflow.get("jobs", {}).get("test")
    assert test_job is not None, "ci.yml has no jobs.test — workflow refactor broke this test"

    needs = test_job.get("needs", [])
    # ``needs`` may be a single string or a list — normalise to list.
    if isinstance(needs, str):
        needs = [needs]
    assert "gate2" in needs, (
        f"jobs.test.needs is {needs!r} — gate2 must be in this list so "
        "the unit suite is skipped when Gate 2 fails.  See PR #1277 / the "
        "May 2026 silent merge for why parallel execution was abandoned."
    )


def test_run_gate2_smoke_default_emit_flushes_each_line() -> None:
    """``run_gate2_smoke`` must emit each ``[gate2] ...`` line with
    ``flush=True`` so progress is visible when stdout is block-buffered
    (nohup, CI log redirection, ``run_in_background``).  Without this,
    a multi-minute Gate 2 run looks like a hang until completion — a
    pattern we burned engineer time on diagnosing as a runner deadlock.

    Verifies the *default* emit is wired through ``flush=True``.  Users
    overriding ``emit=`` are responsible for their own flushing.
    """
    from functools import partial

    from tests.evaluation.ci_gate2 import _flushing_print, run_gate2_smoke

    # _flushing_print is the wrapper exposed by the module; it must be a
    # callable that, when invoked, ends up flushing the underlying stream.
    assert callable(_flushing_print)
    # Detect: it is partial(print, flush=True) — keyword captured by partial.
    assert isinstance(_flushing_print, partial)
    assert _flushing_print.func is print
    assert _flushing_print.keywords.get("flush") is True, (
        "Default emit must be wired with flush=True so per-scenario "
        "progress is visible in block-buffered stdout contexts."
    )

    # Belt-and-braces: the function's default emit parameter must be the
    # flushing wrapper, not the raw print builtin.
    import inspect

    sig = inspect.signature(run_gate2_smoke)
    default_emit = sig.parameters["emit"].default
    assert (
        default_emit is _flushing_print
    ), "run_gate2_smoke.emit default has drifted away from _flushing_print"
