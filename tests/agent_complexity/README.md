# Agent Complexity Test Fleet

Multi-container Docker-based stress tests for the Cogtrix agent. Runs N
parallel containers, each driving a distinct task that exercises a
different combination of tool tiers, task length, and recovery surface.
Surfaces orchestration-level defects (resolver loops, dispatcher
guidance drift, retry storms) that unit tests can't catch because they
require real LLM + tool dispatch under load.

Originated as the manual shell recipe in personal memory; codified
here under [#1930](https://github.com/NorthlandPositronics/Cogtrix/issues/1930)
after [.agent-test-1918](https://github.com/NorthlandPositronics/Cogtrix/issues/1919)
showed the recipe had silently drifted out of sync with the config layout.

---

## Quickstart

```bash
# Build a fresh image from the current source tree and run all 5 scenarios:
python -m tests.agent_complexity.runner --build

# Use an existing image:
python -m tests.agent_complexity.runner --image-tag cogtrix:latest

# Run a subset:
python -m tests.agent_complexity.runner --scenarios gas,sec

# Custom output directory + longer per-task budget for slow models:
python -m tests.agent_complexity.runner \
    --output-dir /tmp/fleet-$(date +%s) \
    --task-timeout 1200
```

The runner exits **0** when every scenario completed within budget and
without tool failures; **1** otherwise. Per-task log files land in
`--output-dir` (default `.agent-fleet-logs/`).

---

## What gets run

`scenarios.py` defines 5 default tasks spanning the complexity matrix:

| Slug | Complexity tier | What it exercises |
|---|---|---|
| `gas` | COMPLEX_ACTION | File writes (Google Apps Script web app + README) |
| `pyda` | MODERATE | Shell + tests (Python data analysis script + unittest) |
| `sec` | MODERATE | Reasoning + file write (security audit + hardened script) |
| `wasi` | COMPLEX_RESEARCH | Search + fetch (2000+ word WASI research report) |
| `jq` | MODERATE | Shell + tests + recovery (jq-lite CLI + test suite) |

Each scenario is a `Scenario(slug, complexity, prompt, expected_tools)`
dataclass. The runner records which expected tools the agent did NOT
invoke — useful signal when the agent took an unusual recovery path.

---

## Adding a scenario

Edit `scenarios.py` and append to `DEFAULT_SCENARIOS`:

```python
Scenario(
    slug="myscenario",
    complexity="MODERATE",
    prompt="Do the thing that exercises Y …",
    expected_tools=("write_file", "execute_shell_command"),
),
```

The slug becomes part of the container name and log filename — keep it
ASCII alphanumeric + hyphen, ≤ 32 chars.

---

## CLI reference

| Flag | Default | Notes |
|---|---|---|
| `--image-tag` | `cogtrix:latest` | Image to run when not building |
| `--build` | off | Build `cogtrix:<--build-tag>` from `docker/Dockerfile` first |
| `--build-tag` | `fleet-runner` | Tag for the freshly-built image |
| `--config-path` | resolved | Override the cogtrix config bind-mount source |
| `--output-dir` | `.agent-fleet-logs` | Per-task log directory (created if missing) |
| `--task-timeout` | `720` | Per-task wall-clock budget (seconds) |
| `--verbosity` | `3` | Cogtrix `--verbosity` flag inside each container |
| `--container-prefix` | `fleet-` | Container name prefix (run multiple fleets concurrently with distinct prefixes) |
| `--scenarios` | all | Comma-separated slugs (e.g. `gas,sec`) |
| `--log-level` | `INFO` | Runner log level |

---

## Config resolution

The runner resolves the cogtrix config to bind-mount into each
container in this order:

1. `--config-path` argument.
2. `src.config.find_config_file()` — the canonical resolver
   (`./.cogtrix.*` → `~/.cogtrix.*` → `~/.config/cogtrix/cogtrix.*`).
3. Legacy fallback: `~/.cogtrix/config/cogtrix.yaml`.

If nothing is found, the runner exits with code **2** and prints
every path it tried.

**Why not just hardcode `~/.cogtrix.yaml`?** That's exactly what the
original shell recipe did — and when the config layout migrated to
`~/.cogtrix/config/cogtrix.yaml` the recipe broke silently. Docker
auto-created an empty root-owned directory at `~/.cogtrix.yaml`
which then took manual `sudo` to clean up. The runner uses the
canonical resolver so a config-path change won't silently break the
recipe again.

---

## Output

Per-task fields in the summary scorecard:

```
═══════════════════════════════════════════════════════
  Fleet summary
═══════════════════════════════════════════════════════
  ✓ gas      245s  turns= 16  tool_calls= 13  failures=0  errors=0  warnings=2  cps=2
           top tools: write_file×5, list_directory×3, read_file×2
  ✓ pyda     312s  turns= 28  tool_calls= 25  failures=0  errors=0  warnings=4  cps=3
           top tools: execute_shell_command×16, write_file×4, list_directory×3
  ✓ sec       58s  turns=  4  tool_calls=  4  failures=0  errors=0  warnings=0  cps=0
           top tools: write_file×2, read_file×2
  ✓ wasi     412s  turns=  5  tool_calls= 18  failures=0  errors=0  warnings=8  cps=0
           top tools: web_search×9, http_get×9
  ✓ jq       290s  turns= 18  tool_calls= 17  failures=0  errors=0  warnings=2  cps=2
           top tools: execute_shell_command×10, write_file×3, run_tests×2
═══════════════════════════════════════════════════════
```

The check mark next to each slug = `completed and tool_failures == 0`.

Full per-call traces live in the per-task log file (one per scenario,
`test<N>-<slug>.log` under `--output-dir`).

---

## Troubleshooting

**`Container fleet-1-gas already running.`** A prior run left containers
alive. `docker ps --filter name=fleet-` to find them, `docker stop`
them, or pass `--container-prefix fleet2-` to avoid the collision.

**`--config-path /home/user/.cogtrix.yaml is a directory`.** Docker's
bind-mount default auto-creates the source path as a directory when
the file doesn't exist. Likely a stale artifact from an older shell
recipe. `sudo rmdir /home/user/.cogtrix.yaml` then re-run.

**Scenario timed out.** The COMPLEX_RESEARCH tier (e.g. `wasi`) can
exceed the default 720s budget on slower providers or under contention.
Raise `--task-timeout` or run that scenario alone.

**Different slug subset.** `--scenarios gas` runs only T1; useful for
iterating on a single failure mode without paying the full ~10 min for
all 5.
