# SWE Role-Test Harness (`tests/role_swe`)

A holistic role test for **team software-development competency** — the sibling of
[`tests/role_pm`](../role_pm). Where PM grades a single retrieval-and-format
answer, SWE grades a whole **pull-request lifecycle**: the agent must comprehend a
real project, follow its conventions, make accurate self-tested patches,
collaborate with a simulated manager / reviewer / QA, adapt to feedback, keep
in-scope, and report honestly.

**Design & methodology:** `docs/optional/testing/swe-role-test-harness.md` (in the
`cogtrix.docs` submodule) — read it first; it holds the rationale, the scenario
taxonomy, the scorecard, and the sign-off decisions.

## Layout

| Path | Status | Contents |
|---|---|---|
| `project/` | ✅ built | The **ledgerlite** SUT — a real, runnable double-entry library with strong, *discoverable* conventions (`CONTRIBUTING.md`). A fixture, **excluded** from the main Cogtrix pytest/ruff/black (see `conftest.py` + root `pyproject.toml`). |
| `conventions.py` | ✅ built | Canary-rule checks (Decimal-not-float, `Err`-suffix, docstrings, CHANGELOG, test-added/naming, off-limits boundary) → deterministic `conventions_respected` signal. |
| `test_conventions.py` | ✅ built | Self-tests for the checker (run in the main suite; no LLM). |
| `scenarios/` | ✅ all 7 built | One YAML per scenario: `swe_01` feature-add · `swe_02` bug-fix (seeded defect) · `swe_03` review change-request · `swe_04` spurious QA defect (push-back) · `swe_05` out-of-scope boundary · `swe_06` ambiguous + requirement-change · `swe_07` invariant break. Plus `scenarios/checks/` (executable behavioural checks) and `scenarios/seeds/` (pre-baseline defect overlays). |
| `workspace.py` | ✅ built | Per-run **git-isolated clone** of `project/`; `changed_files` / `diff` / `run_tests` / `run_lint`; `seed_dir` overlay (plant a defect pre-baseline) + `run_behavioural_check` (out-of-tree harness assertion). All deterministic. |
| `personas.py` | ✅ built | `PersonaChannel` + manager/reviewer/QA deterministic state machines (reviewer runs the canaries + a scripted first-pass change-request, QA gates on the suite + optional spurious defect). Backs the `message_teammate` tool. |
| `scorecard.py` | ✅ built | Measurable signal aggregation → `clean_pass` + `bug_count`: conventions/canaries, suite, boundary, `reached_done`, `review_iterations`, `escalated` (swe_05), `behavioural_ok` (swe_02/04/06/07), `pushed_back` (swe_04), `asked_manager` (swe_06). Quality/LLM-judge fields reserved. |
| `test_workspace_personas.py`, `test_scorecard.py` | ✅ built | End-to-end self-tests of the loop (real git + pytest against the SUT; no LLM). |
| `message_teammate.py` | ✅ built | The agent-facing `message_teammate(role, message)` tool, backed by `PersonaChannel`. |
| `system_prompt.md` | ✅ built | The engineer-role system prompt. |
| `run.py` | ✅ built | `run_scenario` (load → seed → isolate → agent seam → score → JSON report) + `run_repeated` / `aggregate_scorecards` (N repeats → pass-rate). The live `cogtrix_agent_fn` wires the canonical Cogtrix file/shell tools + `message_teammate` over the workspace (sandboxed via process `cwd`). A crashing/looping agent run is scored as a failed run, not a lost cell (#2314). |
| `test_run.py` | ✅ built | Drives `run_scenario` / `run_repeated` with scripted mock agents (one per competency + crash/flaky cases) — proves the whole pipeline without a model. |

### Running a live cycle

```
# one scenario, one model
python -m tests.role_swe.run --scenario 04 --model deepseek-v4-pro --report-dir /tmp/swe

# N repeats → pass-rate (a single run is statistically meaningless for a
# stochastic agent; the harness reports clean_passes/repeats + failure-mode
# frequencies in <id>_summary.json)
python -m tests.role_swe.run --scenario 04 --model deepseek-v4-pro --repeats 5 --report-dir /tmp/swe
```

This exercises `cogtrix_agent_fn` against a real model — it needs the subject-model
keys and spends credits, so it's the explicit live step. Subjects: `qwen3-coder`
(local Spark — needs `VLLM_LOCAL_API_KEY`), `deepseek-v4-pro`, `kimi-k2-6` (both
via `OPENROUTER_API_KEY`). The SUT is excluded from the main suite, so the harness
**never runs in CI** (local-only).

### First cross-model cycle (2026-06-28, N=5)

| scenario | deepseek-v4-pro | kimi-k2-6 | qwen3-coder |
|---|---|---|---|
| 01 feature-add | 100% | 80% | 20% |
| 02 bug-fix | 100% | 100% | 60% |
| 03 review-cycle | 80% | 60% | 0% |
| 04 spurious-defect | 40% | 40% | 60% |
| 05 boundary | 100% | 100% | 0% |
| 06 ambiguous | 60% | 80% | 60% |
| 07 invariant | 80% | 80% | 60% |
| **overall** | **80%** | **77%** | **37%** |

Headline signals: the adversarial scenarios carry the discrimination (the capable
models score 100% on feature-add / bug-fix / boundary). **swe_04 push-back is the
top-tier blind spot** (deepseek & kimi both cave 60% of the time). qwen3-coder
loops to the recursion cap on the collaboration-heavy scenarios (0% on review &
boundary).

## Sign-off decisions (see the design doc)

- **SUT:** synthetic `ledgerlite` (forces genuine comprehension).
- **Persona channel:** a `message_teammate(role, message)` tool, backed by
  deterministic persona state machines (reproducible grading; real-channel /
  GitHub-PR backends are a level-2/3 roadmap).
- **Models:** subject = qwen3-coder / kimi-k2.6 / deepseek-v4-pro; result
  *validation* (LLM-judge, defect-reality verification) = **SOTA only**, never the
  subject grading itself.
- **Grading:** rubric-first (deterministic) for v1; SOTA-LLM quality judge later.

## Why the SUT is excluded from the main suite

`project/` is a **fixture**, not Cogtrix source: it imports the uninstalled
`ledgerlite` package and follows *its own* conventions. The harness runs *its*
tests against a per-scenario workspace copy. The root `pyproject.toml`
force-excludes `tests/role_swe/project` from ruff/black, and `conftest.py`
`collect_ignore`s it from pytest.

## Roadmap

- **SOTA LLM quality judge** — populate the reserved `comprehension` /
  `collaboration_tone` / `honest_reporting` scorecard fields (rubric-first v1 leaves
  them `None`).
- **Mid-task requirement change** delivered *after* the agent starts (swe_06 v1
  surfaces the change on the first clarifying question).
- Real-channel / GitHub-PR persona backends (design-doc levels 2/3).
