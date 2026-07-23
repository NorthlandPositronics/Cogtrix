"""Agent complexity test fleet runner.

CLI entry point. Run with::

    python -m tests.agent_complexity.runner --help

Replaces the manual shell recipe documented in ``CLAUDE.md`` (#1930)
with a first-class runner that:

  1. Resolves the cogtrix config path via the standard
     :func:`src.config.find_config_file` machinery (so a config layout
     change doesn't silently break the recipe — the issue that
     originally surfaced in ``.agent-test-1918``).
  2. Pre-creates per-task log files so Docker bind-mount-as-file works
     reliably (no empty-directory trap at the source path).
  3. Launches N parallel containers, one per scenario.
  4. Polls until all exit (default 12-min timeout per task).
  5. Parses each log for tool calls, errors, completion signals;
     prints a per-task scorecard + overall exit code.

NOT pytest-collected — filename intentionally not ``test_*.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

# Standard fallback paths consulted when ``find_config_file()`` returns
# None.  The order mirrors the cogtrix install conventions documented
# in cogtrix_core/config.py + the legacy ``~/.cogtrix/config/cogtrix.yaml``
# layout some installs use (e.g. when ``cogtrix --setup`` historically
# wrote there).  None of these paths are auto-created by the runner.
_LEGACY_CONFIG_FALLBACKS: tuple[Path, ...] = (
    Path.home() / ".cogtrix" / "config" / "cogtrix.yaml",
    Path.home() / ".cogtrix" / "config" / "cogtrix.yml",
)

# Default per-task wall-clock budget in seconds. Tuned for the
# COMPLEX_RESEARCH tier (~10 min observed on next73); MODERATE tasks
# typically complete in 3–5 min.
_DEFAULT_TASK_TIMEOUT_S = 720

# How often the runner polls Docker for container status (seconds).
_POLL_INTERVAL_S = 15

# Per-scenario log line patterns used by the parser.
_RE_LLM_CHAT_START = re.compile(r"\bLLM_CHAT_START\b")
_RE_LLM_TOOL_CALL = re.compile(r"\bLLM_TOOL_CALL:\s*(\S+)\s+args=")
_RE_TOOL_FAILED = re.compile(r"\bTool failed:\s*(\S+)")
_RE_ERROR_LEVEL = re.compile(r"\[ERROR\]")
_RE_WARNING_LEVEL = re.compile(r"\[WARNING\]")
_RE_DUPLICATE_CACHE = re.compile(r"Duplicate call")
_RE_CHECKPOINT = re.compile(r"Tool:\s*checkpoint\b")
_RE_AGENT_RESPONSE = re.compile(r"\bAgent response\b")


log = logging.getLogger("agent_complexity.runner")


# ── Config resolution ─────────────────────────────────────────────────


def resolve_config_path(override: Path | None = None) -> Path:
    """Return an existing cogtrix YAML/JSON config file.

    Priority:
      1. ``override`` argument (e.g. ``--config-path``).
      2. ``src.config.find_config_file()`` — the canonical resolver.
      3. The legacy fallbacks in :data:`_LEGACY_CONFIG_FALLBACKS`.

    Raises:
        FileNotFoundError: nothing found.  The caller should surface
            the message verbatim — it lists every path we tried so
            the operator can pick where to put the config.
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--config-path {override} does not exist.")
        if override.is_dir():
            raise FileNotFoundError(
                f"--config-path {override} is a directory (likely a stale "
                "Docker bind-mount artefact — see #1919 / agent-test-1918 notes)."
            )
        return override

    # Canonical resolver — late import keeps the runner usable even
    # when ``src.config`` import fails at collection time (e.g. when
    # the runner is invoked outside the venv).
    try:
        from cogtrix_core.config import find_config_file

        found = find_config_file()
    except Exception as exc:  # noqa: BLE001
        log.debug("find_config_file raised: %s — falling back to legacy paths", exc)
        found = None

    if found is not None and found.is_file():
        return found

    for candidate in _LEGACY_CONFIG_FALLBACKS:
        if candidate.is_file():
            return candidate

    tried = [
        "(canonical search via src.config.find_config_file)",
        *(str(p) for p in _LEGACY_CONFIG_FALLBACKS),
    ]
    raise FileNotFoundError(
        "No cogtrix config file found. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nPass --config-path /path/to/cogtrix.yaml to override."
    )


def _resolve_env_file(override: Path | None, config_path: Path) -> Path | None:
    """Resolve the secrets ``--env-file`` to inject into each container (#2219).

    Priority:
      1. explicit ``override`` (``--env-file``) — returned if it exists.
      2. a ``.env`` sibling of *config_path* (e.g. ``tests/comprehensive/.env``
         beside ``cogtrix.comprehensive.yaml``), if present.
      3. ``None`` — no secrets file; the caller warns.

    An explicit-but-missing override resolves to ``None`` (skip), so passing a
    non-existent path or ``-`` is a clean opt-out rather than an error.
    """
    if override is not None:
        return override if override.is_file() else None
    sibling = config_path.parent / ".env"
    return sibling if sibling.is_file() else None


# ── Docker helpers ────────────────────────────────────────────────────


def _docker_inspect_running(name: str) -> bool:
    """True iff ``docker ps`` lists a container with this name."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return name in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _build_image(tag: str, *, repo_root: Path) -> None:
    """Build ``cogtrix:<tag>`` from the project root.

    Streams output to stderr so the operator sees progress.  Raises
    on non-zero exit — a build failure should abort the fleet.
    """
    dockerfile = repo_root / "docker" / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"Dockerfile not found at {dockerfile}")

    log.info("Building cogtrix:%s from %s …", tag, dockerfile)
    result = subprocess.run(
        ["docker", "build", "-t", f"cogtrix:{tag}", "-f", str(dockerfile), str(repo_root)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker build failed (exit {result.returncode})")
    log.info("Built cogtrix:%s", tag)


def _build_run_cmd(
    *,
    name: str,
    image: str,
    config_path: Path,
    log_path: Path,
    prompt: str,
    verbosity: int,
    env_file: Path | None = None,
) -> list[str]:
    """Build the ``docker run`` argv for one scenario (pure; unit-testable).

    When *env_file* is given, its ``KEY=VALUE`` lines are injected into the
    container via ``--env-file`` so Cogtrix's ``_apply_env_vars`` resolves keyed
    providers/tools (e.g. ``COGTRIX_PROVIDER_SPARK_API_KEY``, ``TAVILY_API_KEY``)
    from secrets that live outside the (secret-free) mounted config (#2219).
    """
    cmd = ["docker", "run", "-d", "--rm", "--name", name]
    if env_file is not None:
        cmd += ["--env-file", str(env_file)]
    cmd += [
        "-v",
        f"{config_path}:/app/.cogtrix.yaml:ro",
        "-v",
        f"{log_path}:/tmp/cogtrix.log",
        image,
        "--verbosity",
        str(verbosity),
        "--debug",
        "--log",
        "/tmp/cogtrix.log",
        "-y",
        "--prompt",
        prompt,
    ]
    return cmd


def _launch_container(
    *,
    name: str,
    image: str,
    config_path: Path,
    log_path: Path,
    prompt: str,
    verbosity: int,
    env_file: Path | None = None,
) -> str:
    """Start a detached container for one scenario, return its ID."""
    cmd = _build_run_cmd(
        name=name,
        image=image,
        config_path=config_path,
        log_path=log_path,
        prompt=prompt,
        verbosity=verbosity,
        env_file=env_file,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker run failed for {name} (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


# ── Log parsing ───────────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class ScenarioResult:
    slug: str
    container_name: str
    container_id: str
    log_path: Path
    elapsed_s: float
    timed_out: bool
    turns: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    errors: int = 0
    warnings: int = 0
    duplicate_cache_hits: int = 0
    checkpoints: int = 0
    completed: bool = False
    top_tools: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    missing_expected_tools: list[str] = dataclasses.field(default_factory=list)


def _parse_log(
    log_path: Path,
    *,
    expected_tools: Sequence[str] | None,
) -> dict:
    """Aggregate counters + tool histogram from a single scenario's log."""
    turns = 0
    tool_calls = 0
    tool_failures = 0
    errors = 0
    warnings = 0
    dup_cache = 0
    checkpoints = 0
    completed = False
    tool_counter: Counter[str] = Counter()

    if not log_path.is_file():
        return {
            "turns": 0,
            "tool_calls": 0,
            "tool_failures": 0,
            "errors": 0,
            "warnings": 0,
            "duplicate_cache_hits": 0,
            "checkpoints": 0,
            "completed": False,
            "top_tools": [],
            "missing_expected_tools": list(expected_tools or ()),
        }

    with log_path.open(encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if _RE_LLM_CHAT_START.search(line):
                turns += 1
            m = _RE_LLM_TOOL_CALL.search(line)
            if m:
                tool_calls += 1
                tool_counter[m.group(1)] += 1
            if _RE_TOOL_FAILED.search(line):
                tool_failures += 1
            if _RE_ERROR_LEVEL.search(line):
                errors += 1
            if _RE_WARNING_LEVEL.search(line):
                warnings += 1
            if _RE_DUPLICATE_CACHE.search(line):
                dup_cache += 1
            if _RE_CHECKPOINT.search(line):
                checkpoints += 1
            if _RE_AGENT_RESPONSE.search(line):
                completed = True

    invoked = set(tool_counter)
    missing = [t for t in (expected_tools or ()) if t not in invoked]

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "tool_failures": tool_failures,
        "errors": errors,
        "warnings": warnings,
        "duplicate_cache_hits": dup_cache,
        "checkpoints": checkpoints,
        "completed": completed,
        "top_tools": tool_counter.most_common(3),
        "missing_expected_tools": missing,
    }


# ── Orchestration ─────────────────────────────────────────────────────


def _format_summary(results: Sequence[ScenarioResult]) -> str:
    """Build the human-readable end-of-fleet report."""
    lines = ["", "═══════════════════════════════════════════════════════", "  Fleet summary"]
    lines.append("═══════════════════════════════════════════════════════")
    for r in results:
        status = "✓" if r.completed and r.tool_failures == 0 else "✗"
        timeout_note = "  [TIMEOUT]" if r.timed_out else ""
        lines.append(
            f"  {status} {r.slug:6s}  {r.elapsed_s:5.0f}s  "
            f"turns={r.turns:3d}  tool_calls={r.tool_calls:3d}  "
            f"failures={r.tool_failures}  errors={r.errors}  "
            f"warnings={r.warnings}  cps={r.checkpoints}{timeout_note}"
        )
        if r.top_tools:
            top_str = ", ".join(f"{name}×{n}" for name, n in r.top_tools)
            lines.append(f"           top tools: {top_str}")
        if r.missing_expected_tools:
            lines.append(
                f"           missing expected tools: " f"{', '.join(r.missing_expected_tools)}"
            )
    lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(lines) + "\n"


def run_fleet(
    *,
    image: str,
    config_path: Path,
    output_dir: Path,
    scenarios,
    task_timeout_s: int,
    verbosity: int,
    container_prefix: str,
    env_file: Path | None = None,
) -> list[ScenarioResult]:
    """Run all *scenarios* in parallel; return parsed results.

    Caller is responsible for creating *output_dir* before invoking.
    Per-task log files are pre-created here so Docker bind-mount
    treats them as files rather than auto-creating directories
    (the trap surfaced in agent-test-1918).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    launched: list[ScenarioResult] = []
    start_wall = time.monotonic()

    for i, scenario in enumerate(scenarios, start=1):
        log_path = output_dir / f"test{i}-{scenario.slug}.log"
        log_path.touch()  # pre-create so Docker mounts as file
        container_name = f"{container_prefix}{i}-{scenario.slug}"

        # If a prior run's container still exists, refuse to clobber.
        if _docker_inspect_running(container_name):
            raise RuntimeError(
                f"Container {container_name} already running. "
                f"Stop it (`docker stop {container_name}`) or pass a "
                f"different --container-prefix."
            )

        log.info(
            "Launching %s (scenario=%s, complexity=%s) → %s",
            container_name,
            scenario.slug,
            scenario.complexity,
            log_path,
        )
        cid = _launch_container(
            name=container_name,
            image=image,
            config_path=config_path,
            log_path=log_path,
            prompt=scenario.prompt,
            verbosity=verbosity,
            env_file=env_file,
        )
        launched.append(
            ScenarioResult(
                slug=scenario.slug,
                container_name=container_name,
                container_id=cid,
                log_path=log_path,
                elapsed_s=0.0,
                timed_out=False,
            )
        )

    # Poll until all containers exit OR the per-task budget elapses.
    deadline = start_wall + task_timeout_s
    pending = {r.container_name: r for r in launched}
    while pending and time.monotonic() < deadline:
        still_running = {name: r for name, r in pending.items() if _docker_inspect_running(name)}
        completed_this_tick = set(pending) - set(still_running)
        for name in completed_this_tick:
            pending[name].elapsed_s = time.monotonic() - start_wall
            log.info("  ✓ %s exited (%.0fs)", name, pending[name].elapsed_s)
        pending = still_running
        if pending:
            time.sleep(_POLL_INTERVAL_S)

    # Any still-pending containers timed out — stop them and mark.
    for name, r in pending.items():
        log.warning("  ✗ %s timed out after %.0fs — stopping", name, task_timeout_s)
        subprocess.run(
            ["docker", "stop", "--time", "5", name],
            capture_output=True,
            check=False,
            timeout=30,
        )
        r.timed_out = True
        r.elapsed_s = task_timeout_s

    # Parse each scenario's log.
    scenario_by_slug = {s.slug: s for s in scenarios}
    for r in launched:
        parsed = _parse_log(
            r.log_path,
            expected_tools=scenario_by_slug[r.slug].expected_tools,
        )
        for key, value in parsed.items():
            setattr(r, key, value)

    return launched


# ── CLI ───────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.agent_complexity.runner",
        description=(
            "Run the agent complexity test fleet against a cogtrix Docker "
            "image. Launches N parallel containers (one per scenario), "
            "tails their logs, prints a per-task scorecard. See "
            "tests/agent_complexity/README.md for the full design."
        ),
    )
    parser.add_argument(
        "--image-tag",
        default="cogtrix:latest",
        help=(
            "Docker image to run. Default: cogtrix:latest. Use --build "
            "to build a fresh image from the current source tree."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Build a fresh cogtrix:<--build-tag> from docker/Dockerfile "
            "before launching. Overrides --image-tag."
        ),
    )
    parser.add_argument(
        "--build-tag",
        default="fleet-runner",
        help="Tag for the freshly-built image when --build is set.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help=(
            "Path to cogtrix.yaml to bind-mount into each container. "
            "Default: resolved via src.config.find_config_file() with "
            "fallback to ~/.cogtrix/config/cogtrix.yaml."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Path to a KEY=VALUE secrets file injected into each container via "
            "docker --env-file, so keyed providers/tools resolve from a "
            "secret-free config (#2219). Default: auto-detect a '.env' sibling "
            "of the resolved config; pass '-' (or a missing path) to skip."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".agent-fleet-logs"),
        help="Directory for per-task log files. Created if missing.",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=_DEFAULT_TASK_TIMEOUT_S,
        help="Per-task wall-clock budget in seconds (default: 720).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=3,
        choices=(0, 1, 2, 3, 4),
        help="Cogtrix --verbosity flag passed to each container (default: 3).",
    )
    parser.add_argument(
        "--container-prefix",
        default="fleet-",
        help=(
            "Prefix for container names (slug appended). Default: 'fleet-'. "
            "Change this when running multiple fleets concurrently to "
            "avoid name collisions."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Comma-separated scenario slugs to run (e.g. 'gas,sec'). "
            "Default: all DEFAULT_SCENARIOS."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Runner log level (DEBUG, INFO, WARNING). Default: INFO.",
    )
    return parser


def _select_scenarios(slugs_arg: str):
    from tests.agent_complexity.scenarios import DEFAULT_SCENARIOS

    if not slugs_arg.strip():
        return list(DEFAULT_SCENARIOS)
    wanted = [s.strip() for s in slugs_arg.split(",") if s.strip()]
    by_slug = {s.slug: s for s in DEFAULT_SCENARIOS}
    selected = []
    for slug in wanted:
        if slug not in by_slug:
            raise SystemExit(
                f"Unknown scenario slug '{slug}'. Available: " f"{', '.join(sorted(by_slug))}"
            )
        selected.append(by_slug[slug])
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent.parent

    try:
        config_path = resolve_config_path(args.config_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    log.info("Using config: %s", config_path)

    # Resolve the secrets file: explicit --env-file, else a '.env' sibling of
    # the config (e.g. tests/comprehensive/.env next to the comprehensive
    # config). Inject it into each container so keyed providers/tools work from
    # a secret-free config (#2219).
    env_file = _resolve_env_file(args.env_file, config_path)
    if env_file is not None:
        log.info("Injecting secrets via --env-file: %s", env_file)
    else:
        log.warning(
            "No secrets --env-file found (looked for a '.env' beside %s). "
            "Keyed providers/tools (spark, tavily, ...) will be unauthenticated "
            "in-container unless the mounted config carries inline keys.",
            config_path,
        )

    if args.build:
        try:
            _build_image(args.build_tag, repo_root=repo_root)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"Build failed: {exc}", file=sys.stderr)
            return 3
        image = f"cogtrix:{args.build_tag}"
    else:
        image = args.image_tag
    log.info("Using image: %s", image)

    scenarios = _select_scenarios(args.scenarios)
    log.info("Running %d scenarios: %s", len(scenarios), ", ".join(s.slug for s in scenarios))

    try:
        results = run_fleet(
            image=image,
            config_path=config_path,
            output_dir=args.output_dir,
            scenarios=scenarios,
            task_timeout_s=args.task_timeout,
            verbosity=args.verbosity,
            container_prefix=args.container_prefix,
            env_file=env_file,
        )
    except RuntimeError as exc:
        print(f"Fleet failed: {exc}", file=sys.stderr)
        return 4

    summary = _format_summary(results)
    print(summary)

    # Exit non-zero when any scenario failed (timed out OR exited with
    # tool_failures OR didn't complete).
    failed = [r for r in results if r.timed_out or r.tool_failures > 0 or not r.completed]
    if failed:
        log.warning("%d scenario(s) failed: %s", len(failed), ", ".join(r.slug for r in failed))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
