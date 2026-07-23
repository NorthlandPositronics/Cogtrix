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

    assert (
        exit_code == 1
    ), f"Partial completion must fail the gate even with a perfect judge score; logs={logs}"
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

    assert (
        call_count["n"] == 2
    ), f"Expected one retry after empty-response flake; got n={call_count['n']}. Logs: {logs}"
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
    on the first key when a later key would have worked.

    Native-key-first ordering interacts with this test: because the model
    under test (gpt-4o) has ``env_key=OPENAI_API_KEY``, the OpenAI key is
    tried before OpenRouter.  When OpenAI returns the captured 402, the
    fallback path picks OpenRouter as the second key.  The fallback
    *behaviour* the test pins is identical to before; only the key
    ordering swaps.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-good-value")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-broke-value")

    # First call (OpenAI, the model's native key) returns a 402-credits
    # error in EvalResult.error.  Second call (OpenRouter, the fallback)
    # returns a clean pass.
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

    # gpt-4o has env_key=OPENAI_API_KEY, so native-key-first routing tries
    # OPENAI_API_KEY first.  OpenRouter is the fallback via its wildcard
    # cover (the model has an openrouter_model_id).
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
        "KEY_FAIL" in line and "OPENAI_API_KEY" in line for line in logs
    ), f"Expected KEY_FAIL log for native OPENAI_API_KEY; logs={logs}"


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


# ── Issue #1994: per-model retry backoff ─────────────────────────────────────


def _model_with_backoff(seconds: int) -> ModelConfig:
    """Mock model that opts in to a non-zero ``retry_backoff_seconds``.

    Used to exercise the issue #1994 backoff path without depending on
    kimi-k2-5's specific YAML row.
    """
    return ModelConfig(
        id="mock-backoff",
        provider="anthropic",
        display_name="Mock (backoff)",
        tier="smoke",
        smoke=True,
        env_key="ANTHROPIC_API_KEY",
        model_id="mock-backoff-model",
        openrouter_model_id="anthropic/mock-backoff-model",
        retry_backoff_seconds=seconds,
    )


def test_backoff_sleeps_between_attempts_on_empty_response(monkeypatch) -> None:
    """Issue #1994: when a model opts in to ``retry_backoff_seconds``, an
    empty-response retry must pause for that many seconds before the
    second attempt so the upstream capacity window has time to clear.

    Without the pause, the immediate retry hits the same closed window
    every time and burns the retry budget without ever recovering.
    """
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Empty-response flake: no tools, no content, no error.
            return EvalResult(
                scenario_id=scenario.id,
                model_id=model.id,
                model_display_name=model.display_name,
                passed=False,
                tool_calls_made=[],
                tool_calls_required=list(scenario.tools_required),
                turns_used=1,
                elapsed_seconds=0.3,
                final_response="",
                error=None,
                task_completion=False,
                tool_selection_rate=0.0,
            )
        return _result(passed=True, task_completion=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    sleeps: list[float] = []
    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model_with_backoff(60),
        ("OPENROUTER_API_KEY", "or-test"),
        emit=logs.append,
        sleep=sleeps.append,
    )

    assert call_count["n"] == 2, f"Expected 2 attempts, got {call_count['n']}"
    assert sleeps == [
        60
    ], f"Expected one 60s backoff before retry; got sleeps={sleeps}. Logs={logs}"
    assert result is not None and result.passed is True
    assert any(
        "BACKOFF" in line and "mock-backoff" in line and "empty_response" in line for line in logs
    ), f"Expected BACKOFF diagnostic in logs: {logs}"


def test_backoff_sleeps_between_attempts_on_transient_error(monkeypatch) -> None:
    """Backoff must apply on transient-error retries too — a 429 from the
    same OpenRouter route benefits from the same wait as an empty-response
    flake, since both indicate the upstream window is closed.
    """
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _result(
                passed=False,
                error="429 Too Many Requests",
            )
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    sleeps: list[float] = []
    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model_with_backoff(45),
        ("OPENROUTER_API_KEY", "or-test"),
        emit=logs.append,
        sleep=sleeps.append,
    )

    assert call_count["n"] == 2
    assert sleeps == [45], f"Expected one 45s backoff; got {sleeps}. Logs={logs}"
    assert result is not None and result.passed is True
    assert any(
        "BACKOFF" in line and "transient_error" in line for line in logs
    ), f"Expected transient_error BACKOFF diagnostic: {logs}"


def test_backoff_sleeps_between_attempts_on_transient_exception(monkeypatch) -> None:
    """The exception path (run_scenario raises rather than returning an
    error-stuffed result) must also honour the configured backoff.
    """
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("Connection timed out after 30s")
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    sleeps: list[float] = []
    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model_with_backoff(30),
        ("OPENROUTER_API_KEY", "or-test"),
        emit=logs.append,
        sleep=sleeps.append,
    )

    assert call_count["n"] == 2
    assert sleeps == [30], f"Expected one 30s backoff; got {sleeps}. Logs={logs}"
    assert result is not None and result.passed is True
    assert any(
        "BACKOFF" in line and "transient_exception" in line for line in logs
    ), f"Expected transient_exception BACKOFF diagnostic: {logs}"


def test_no_backoff_when_retry_backoff_seconds_is_zero(monkeypatch) -> None:
    """Default behaviour: models without an opt-in retry_backoff_seconds
    must continue to retry immediately.  This preserves the historical
    behaviour for every model in the registry that has not been
    explicitly tuned for a capacity-window upstream.
    """
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _result(
                passed=False,
                error="503 Service Unavailable",
            )
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    sleeps: list[float] = []
    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model(),  # default retry_backoff_seconds == 0
        ("OPENROUTER_API_KEY", "or-test"),
        emit=logs.append,
        sleep=sleeps.append,
    )

    assert call_count["n"] == 2
    assert sleeps == [], f"Expected no backoff when retry_backoff_seconds=0; got {sleeps}"
    assert result is not None and result.passed is True
    assert not any("BACKOFF" in line for line in logs), f"BACKOFF must not log when opt-out: {logs}"


def test_no_backoff_when_first_attempt_succeeds(monkeypatch) -> None:
    """The backoff must only fire when an actual retry is taken — a
    first-attempt success on a model with retry_backoff_seconds set must
    NOT pause for 60 seconds before returning.
    """
    from tests.evaluation.ci_gate2 import _try_run_with_key

    call_count = {"n": 0}

    def fake_run_scenario(scenario, model, active_key=None):
        call_count["n"] += 1
        return _result(passed=True)

    monkeypatch.setattr("tests.evaluation.ci_gate2.run_scenario", fake_run_scenario)

    sleeps: list[float] = []
    logs: list[str] = []
    result = _try_run_with_key(
        _scenario(),
        _model_with_backoff(60),
        ("OPENROUTER_API_KEY", "or-test"),
        emit=logs.append,
        sleep=sleeps.append,
    )

    assert call_count["n"] == 1
    assert sleeps == [], f"No retry happened, so no backoff expected; got {sleeps}"
    assert result is not None and result.passed is True


def test_kimi_k2_5_models_yaml_opts_in_to_backoff() -> None:
    """Pin issue #1994's config side: ``kimi-k2-5`` in models.yaml must
    carry a non-zero ``retry_backoff_seconds``.  Reverting that value to
    zero is a regression that breaks the documented mitigation.

    This is a contract test, not a behaviour test — it guards the YAML
    knob so a future edit cannot silently strip the opt-in without a
    test flag.
    """
    from tests.evaluation.runner import get_model

    kimi = get_model("kimi-k2-5")
    assert kimi.retry_backoff_seconds > 0, (
        "kimi-k2-5 must opt in to retry_backoff_seconds per issue #1994; "
        f"got retry_backoff_seconds={kimi.retry_backoff_seconds}"
    )


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
    """jobs.test no longer waits for gate2 — the trade-off has been re-evaluated.

    **Original rationale (PR #1277):** Making ``test`` depend on ``gate2``
    prevented wasted CI capacity on the 1-in-50 PRs where Gate 2 fails,
    at a cost of ~5–6 min slower happy path.

    **Current rationale (PR #1377):** Gate 2 failures are rare (~2 % of PRs).
    The guaranteed 9-min tax on every successful run is more expensive
    than the occasional 3-min unit-test burn on the failure path.
    ``test-api`` still serialises on ``gate2``, protecting the expensive
    28-min API suite.  ``test`` (now ~1 min with xdist) runs in parallel
    with ``gate2`` so the critical path is: gate2 → test-api.
    """
    import yaml

    ci_yml = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(ci_yml.read_text())

    test_job = workflow.get("jobs", {}).get("unit-tests")
    assert test_job is not None, "ci.yml has no jobs.unit-tests — workflow refactor broke this test"

    needs = test_job.get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    assert "gate2" not in needs, (
        f"jobs.unit-tests.needs is {needs!r} — unit-tests was intentionally decoupled from "
        "gate2 in PR #1377.  See the re-evaluated trade-off comment above.  "
        "If you are restoring this dependency, update this test to match."
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


# ── Key-routing: native API first, OpenRouter fallback ──────────────────────


def test_candidate_keys_for_model_puts_native_first() -> None:
    """A model's native env_key must be tried before OpenRouter.

    Concrete case that motivated this routing: DeepSeek-V4-Flash uses
    thinking mode and requires ``reasoning_content`` to be threaded
    back on subsequent turns.  OpenRouter's pass-through does not
    always preserve that field, so a multi-turn scenario gets a 400
    from DeepSeek on turn 2.  Hitting the native DeepSeek API first
    side-steps the regression and only falls back to OpenRouter if
    DEEPSEEK_API_KEY is absent or rejected.
    """
    from tests.evaluation.ci_gate2 import _candidate_keys_for_model
    from tests.evaluation.runner import ModelConfig

    deepseek_model = ModelConfig(
        id="deepseek-v4-flash",
        provider="deepseek",
        display_name="DeepSeek-V4-Flash",
        tier="A",
        smoke=True,
        env_key="DEEPSEEK_API_KEY",
        model_id="deepseek-chat",
        openrouter_model_id="deepseek/deepseek-v4-flash",
    )
    full_keys = [
        ("OPENROUTER_API_KEY", "or-x"),
        ("CEREBRAS_API_KEY", "cb-x"),
        ("DEEPSEEK_API_KEY", "ds-x"),
        ("OPENAI_API_KEY", "oa-x"),
        ("ANTHROPIC_API_KEY", "an-x"),
    ]
    ordered = _candidate_keys_for_model(full_keys, deepseek_model)
    assert ordered[0][0] == "DEEPSEEK_API_KEY", (
        "DeepSeek models must try DEEPSEEK_API_KEY before OpenRouter; "
        f"got order {[k[0] for k in ordered]}"
    )
    # All other keys must still be present as fallbacks, in their original
    # priority order (OpenRouter, Cerebras, OpenAI, Anthropic).
    assert [k[0] for k in ordered[1:]] == [
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]


def test_is_model_unavailable_error_recognises_common_shapes() -> None:
    """``_is_model_unavailable_error`` is the fall-back predicate that
    rescues a Gate 2 cell when a provider has retired a model_id behind
    our back (the Cerebras llama-3.3-70b → gpt-oss-120b transition in
    May 2026 was the motivating case).

    The check is keyword-based against ``str(exc).lower()``, so any
    shape that includes one of the recognised phrases must trip.  This
    test pins the keyword set so a future tightening (e.g. requiring
    an HTTP status code in the string) cannot regress the predicate
    silently.
    """
    from tests.evaluation.ci_gate2 import _is_model_unavailable_error

    positives = [
        "Error code: 400 - {'error': {'message': 'model not found', 'code': 'model_not_found'}}",
        "Model llama-3.3-70b does not exist on this endpoint.",
        "unknown model: gpt-oss-120b",
        "model is not available in your region",
        "no such model registered",
    ]
    for msg in positives:
        assert _is_model_unavailable_error(Exception(msg)), msg

    # Negatives — common errors that should NOT trip the predicate, so
    # they continue to follow their normal handling paths
    # (auth/quota / transient / strict-gate failure).
    negatives = [
        "401 Unauthorized",
        "Connection reset by peer",
        "Recursion limit of 25 reached",
        "Scenario timed out after 90s",
    ]
    for msg in negatives:
        assert not _is_model_unavailable_error(Exception(msg)), msg


def test_candidate_keys_for_model_respects_prefer_openrouter_flag() -> None:
    """When ``model.prefer_openrouter`` is True the native-key promotion
    is skipped — OpenRouter (the first entry in the configured
    ``_KEY_PRIORITY``) stays first.

    Used as a per-model escape hatch.  Currently ``deepseek-v4-flash``
    sets the flag while issue #1391 (reasoning_content threading) is
    open: DeepSeek's native API 400s on every multi-turn scenario but
    OpenRouter's pass-through lets some complete.  The flag exists
    *exactly* to let one model fall through to OpenRouter without
    forcing every other model back onto the pre-routing path.
    """
    from tests.evaluation.ci_gate2 import _candidate_keys_for_model
    from tests.evaluation.runner import ModelConfig

    deepseek_via_openrouter = ModelConfig(
        id="deepseek-v4-flash",
        provider="deepseek",
        display_name="DeepSeek-V4-Flash",
        tier="B",
        smoke=True,
        env_key="DEEPSEEK_API_KEY",
        model_id="deepseek-v4-flash",
        openrouter_model_id="deepseek/deepseek-v4-flash",
        prefer_openrouter=True,  # the workaround for #1391
    )
    full_keys = [
        ("OPENROUTER_API_KEY", "or-x"),
        ("DEEPSEEK_API_KEY", "ds-x"),
        ("OPENAI_API_KEY", "oa-x"),
    ]
    ordered = _candidate_keys_for_model(full_keys, deepseek_via_openrouter)
    # The flag must keep the original priority order — OpenRouter first.
    assert ordered == full_keys, (
        "prefer_openrouter=True must short-circuit the native-promotion "
        f"step; got {[k[0] for k in ordered]}, expected {[k[0] for k in full_keys]}"
    )


def test_candidate_keys_for_model_falls_back_when_native_absent() -> None:
    """When the native key is not in the environment, ordering is unchanged.

    Preserves the "one OpenRouter key gets everything to work" property
    for solo developers who only set ``OPENROUTER_API_KEY`` locally.
    """
    from tests.evaluation.ci_gate2 import _candidate_keys_for_model
    from tests.evaluation.runner import ModelConfig

    deepseek_model = ModelConfig(
        id="deepseek-v4-flash",
        provider="deepseek",
        display_name="DeepSeek-V4-Flash",
        tier="A",
        smoke=True,
        env_key="DEEPSEEK_API_KEY",
        model_id="deepseek-chat",
        openrouter_model_id="deepseek/deepseek-v4-flash",
    )
    # No DEEPSEEK_API_KEY in the candidate set — OpenRouter must remain
    # the first choice via the wildcard cover.
    only_openrouter = [("OPENROUTER_API_KEY", "or-x")]
    assert _candidate_keys_for_model(only_openrouter, deepseek_model) == only_openrouter


# ── Workflow guard: api-tests must be in the CI Summary failure loop ─────────


def test_ci_summary_gates_on_api_tests_result() -> None:
    """The CI Summary required-checks loop must include ``api-tests``.

    Without this, an api-tests failure leaves CI Summary green and a
    PR can auto-merge with a broken API integration suite — the same
    shape as the PR #1276 incident that brought ``gate2`` into the
    loop.  Adding api-tests to the loop closes the symmetric hole.
    """
    ci_yml = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    text = ci_yml.read_text()

    loop_start = text.find("for result in")
    assert loop_start != -1, "CI Summary required-checks loop not found in ci.yml"
    loop_end = text.find("done", loop_start)
    assert loop_end != -1, "CI Summary loop has no terminator"
    loop_body = text[loop_start:loop_end]
    assert "needs.api-tests.result" in loop_body, (
        "api-tests must be inside the CI Summary required-checks loop, not "
        "just the summary table.  Without this, a cancelled or failed "
        "api-tests shard lets the PR auto-merge — the same merge-block "
        "hole that PR #1276 exploited for Gate 2."
    )


# ── Workflow guard: shard file lists must match the filesystem ──────────────


def _extract_test_paths_from_job(ci_yml_text: str, job_name: str) -> set[str]:
    """Return every ``tests/...py`` path mentioned in a job's run block.

    Used by the shard-coverage guards to compare CI's static file lists
    against what is actually present in the repository.  The shard
    case blocks use plain ``tests/foo.py`` paths (no globs, no helpers)
    so a regex sweep over the job's YAML slice is sufficient — no need
    to parse the embedded shell.
    """
    import re

    m = re.search(rf"^  {re.escape(job_name)}:\n", ci_yml_text, re.MULTILINE)
    assert m, f"ci.yml has no jobs.{job_name}"
    start = m.start()
    rest = ci_yml_text[start + 1 :]
    next_job = re.search(r"^  [a-z][a-z0-9_-]*:\n", rest, re.MULTILINE)
    end = (start + 1 + next_job.start()) if next_job else len(ci_yml_text)
    job_block = ci_yml_text[start:end]
    return set(re.findall(r"(tests/[A-Za-z0-9_/]+\.py)", job_block))


def test_api_tests_shard_files_exist() -> None:
    """Every test file listed in an api-tests A-D shard must exist on disk.

    With shard E added as a runtime-computed catch-all (find +
    comm -23), the previous "every API test file on disk must be in
    some shard" assertion is no longer needed — any file not in A-D
    automatically lands in E.  The remaining guard is stale-entry
    detection: a typo or rename that leaves a deleted file referenced
    in A-D would cause pytest to error out at collection time on the
    affected shard.  Catch that earlier with a clearer message.
    """
    repo = Path(__file__).parents[2]
    ci_yml = repo / ".github" / "workflows" / "ci.yml"

    referenced = _extract_test_paths_from_job(ci_yml.read_text(), "api-tests")
    missing = sorted(p for p in referenced if not (repo / p).exists())
    assert not missing, (
        f"api-tests A-D case block references files that no longer exist: {missing}.  "
        f"Remove them from .github/workflows/ci.yml (jobs.api-tests).  "
        f"(Note: shard E would still pick up any on-disk api test file not in A-D — "
        f"this guard only catches stale paths pointing to deleted files.)"
    )


def test_unit_tests_shard_files_exist() -> None:
    """Every test file listed in a unit-tests A-D shard must exist on disk.

    With shard E added as a runtime-computed catch-all (find +
    comm -23), the previous "new files silently skipped" concern is
    handled at CI time: anything not in A-D and not under
    api/regression/test_api_* lands in shard E automatically.  This
    guard catches the remaining failure mode — A-D listing a path
    that no longer exists on disk, which would error out at pytest
    collection time on the affected shard.
    """
    repo = Path(__file__).parents[2]
    ci_yml = repo / ".github" / "workflows" / "ci.yml"

    referenced = _extract_test_paths_from_job(ci_yml.read_text(), "unit-tests")
    missing = sorted(p for p in referenced if not (repo / p).exists())
    assert not missing, (
        f"unit-tests case block references files that do not exist: {missing}.  "
        f"Remove them from .github/workflows/ci.yml (jobs.unit-tests) or "
        f"restore the missing test file."
    )


# ── Bug L follow-up (2026-05-20): tool errors and SKIP must fail ──────────


def test_final_passed_vetoes_when_unrecovered_tool_errors_present(monkeypatch) -> None:
    """A run with passing structural + judge metrics must still fail when
    ``tool_errors_unrecovered`` is non-empty.

    Motivating case: gpt-oss-20b emitted ``http_get`` with dict-shaped
    headers, the schema raised a pydantic ValidationError, and the
    model fell back to an honest "could not find" answer that satisfied
    the success_criteria. Before this veto Gate 2 reported pass=True
    despite the tool failure.
    """
    from tests.evaluation.ci_gate2 import _final_passed

    scenario = _scenario()
    result_with_errors = EvalResult(
        scenario_id=scenario.id,
        model_id="mock",
        model_display_name="Mock",
        passed=True,
        tool_calls_made=["create_po"],
        tool_calls_required=["create_po"],
        turns_used=2,
        elapsed_seconds=0.1,
        final_response="PO created.",
        task_completion=True,
        tool_selection_rate=100.0,
        tool_errors=["http_get: Error executing http_get: 1 validation error"],
        tool_errors_unrecovered=["http_get: Error executing http_get: 1 validation error"],
    )

    assert _final_passed(scenario, result_with_errors, score=1.0) is False


def test_final_passed_allows_recovered_tool_errors(monkeypatch) -> None:
    """Issue #1787: a tool error that the model recovered from on a
    successful retry of the same tool must NOT veto the pass gate.

    Motivating case: kimi-k2-5 hallucinates ``supplier_tier`` on
    ``route_for_approval``, gets a pydantic ``extra_forbidden`` error,
    retries with the correct ``tier`` field, and produces a correct
    final response.  ``tool_errors`` still contains the diagnostic
    line (operator can see the model misfired), but
    ``tool_errors_unrecovered`` is empty so the gate no longer fails.
    """
    from tests.evaluation.ci_gate2 import _final_passed

    scenario = _scenario()
    result_recovered = EvalResult(
        scenario_id=scenario.id,
        model_id="mock",
        model_display_name="Mock",
        passed=True,
        tool_calls_made=["create_po", "create_po"],
        tool_calls_required=["create_po"],
        turns_used=3,
        elapsed_seconds=0.1,
        final_response="PO created.",
        task_completion=True,
        tool_selection_rate=100.0,
        tool_errors=["create_po: Error executing create_po: extra_forbidden"],
        tool_errors_unrecovered=[],
    )

    assert _final_passed(scenario, result_recovered, score=1.0) is True


def test_final_passed_passes_when_no_tool_errors() -> None:
    """Sanity baseline: with structural+judge metrics passing and no
    tool errors, the gate returns True."""
    from tests.evaluation.ci_gate2 import _final_passed

    scenario = _scenario()
    result_clean = _result(passed=True, task_completion=True)
    assert result_clean.tool_errors == []
    assert result_clean.tool_errors_unrecovered == []
    assert _final_passed(scenario, result_clean, score=1.0) is True


def test_run_gate2_smoke_fails_when_keys_exhausted(monkeypatch) -> None:
    """An exhausted-keys SKIP must surface as a failure.

    Before this fix, when every priority key returned 402 or was
    ineligible the scenario fell through with `continue` and the gate
    returned 0 (success). A real-world OpenRouter credit drain on
    kimi-k2-5 reached exactly that state.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    # Provide a key so the early "no key configured" return-0 path
    # is not taken — this is the case where credits got drained mid-run.
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-value")

    # Simulate _try_run_with_key returning None (every key rejected).
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2._try_run_with_key",
        lambda scenario, model, key, emit, max_retries=1: None,
    )

    logs: list[str] = []
    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert exit_code == 1, f"Exhausted keys must fail Gate 2; logs={logs}"
    fail_lines = [line for line in logs if line.startswith("[gate2] FAIL")]
    assert fail_lines, (
        "Expected an explicit FAIL line for the exhausted scenario in logs; " f"got logs={logs}"
    )
    assert "all keys exhausted" in fail_lines[0]


def test_run_gate2_smoke_no_keys_set_still_returns_zero(monkeypatch) -> None:
    """Sanity baseline: a CI environment with no priority keys at all
    must still return 0 (advisory skip) — this is the legitimate
    unconfigured-CI exit, not a credit-drain.
    """
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)

    logs: list[str] = []
    exit_code = run_gate2_smoke(
        scenarios=[_scenario()],
        models=[_model()],
        emit=logs.append,
    )

    assert exit_code == 0
    assert any("no API key set" in line for line in logs)
