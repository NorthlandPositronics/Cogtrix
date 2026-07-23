"""Experiment: sweep ``context_max_tokens`` (the compression budget) to pick the
right cap for #2360 / #2365.

Runs the SAME long, tool-heavy ``role_sysadmin`` scenario under several
``context_max_tokens`` arms × N repeats on a large-window model, and reports, per
arm, the trade-off the decision hinges on:

  * correctness   — clean_pass / task_achieved rates (does a bigger cap hurt?)
  * the #2365 bug — tool-pair 400 count (assistant tool_calls with no matching
                    tool responses; should fall to 0 as the cap rises)
  * compression   — churn events per run (compression passes + "dropped oldest"
    churn           trims); the mechanism the cap controls — should fall as the
                    cap rises
  * cost          — mean tool_calls + mean wall-clock (the price of a bigger cap)

The default arms are the four candidates from the #2360 discussion, expressed as
fractions of a 262 144-token Kimi window: 40k (today's flat default ≈ 15%), 52k
(≈20%), 131k (50%), 200k (≈76%, the issue's operational mitigation). The winner
is the smallest cap that drives the 400s to 0 without a correctness/cost cliff.

Nothing is committed or changed in the tree by this script; it only monkeypatches
``build_agent_graph``'s ``context_max_tokens`` for the duration of each arm and
writes JSON reports under ``--report-dir``.

REQUIRES a live run: Docker (a privileged systemd container is booted per
scenario run), SSH, and credentials for the subject model
(``kimi-k2-6`` → ``OPENROUTER_API_KEY``). This is NOT part of CI — run it
yourself.

Usage
-----
    python -m tests.role_sysadmin.experiment_context_cap \
        --model kimi-k2-6 --scenario sa_05 --repeats 3 \
        --report-dir /tmp/ctxcap

    # custom arms (comma-separated token budgets):
    python -m tests.role_sysadmin.experiment_context_cap \
        --model kimi-k2-6 --scenario sa_05 --repeats 3 \
        --arms 40000,131072,262144 --report-dir /tmp/ctxcap
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tests.role_sysadmin.run import cogtrix_agent_fn, find_scenario, load_scenario, run_repeated

#: Default sweep — the four #2360 candidates (token budgets).
_DEFAULT_ARMS = (40_000, 52_000, 131_072, 200_000)
#: Reference window used only to annotate each arm with its fraction in the report.
_REFERENCE_WINDOW = 262_144
_DEFAULT_SCENARIO = "sa_05"
_DEFAULT_MODEL = "kimi-k2-6"

# ── Log-scan signatures (case-insensitive) ───────────────────────────────────
# The per-run DEBUG log (``<scenario>_r{k}_run.log`` written by run.py's
# _capture_run_log) carries the compression trail and any provider error.

#: The #2365 symptom — an assistant message with tool_calls not answered per
#: tool_call_id; the provider rejects it with a 400. Match the provider's phrasings
#: (note the upstream "preceeding" typo) plus the guard's own diagnostic.
_TOOLPAIR_400_PATTERNS = (
    "did not have response messages",
    "must be followed by tool messages",
    "must be a response to a preceeding",
    "tool_call_ids did not",
)
#: Compression-churn events — each is one log line the cap directly gates.
_CHURN_PATTERNS: dict[str, re.Pattern[str]] = {
    "compression_passes": re.compile(r"compression pass triggered", re.I),
    "emergency_passes": re.compile(r"emergency compression pass", re.I),
    "ai_message_compressions": re.compile(r"compressed \d+ ai messages", re.I),
    "tool_message_compressions": re.compile(r"compressed \d+ tool messages", re.I),
    "dropped_oldest_events": re.compile(r"dropped \d+ oldest message", re.I),
}


@contextlib.contextmanager
def _force_context_max_tokens(value: int) -> Iterator[None]:
    """Monkeypatch ``build_agent_graph`` so this arm's graphs use ``context_max_tokens=value``.

    ``cogtrix_agent_fn`` builds the graph with ``build_agent_graph(...)`` and does
    NOT pass ``context_max_tokens`` (so it takes the 40k default). Its
    ``from cogtrix_core.orchestration.graph import build_agent_graph`` resolves the module
    attribute at call time, so patching the attribute here is picked up by the
    agent function for the duration of the arm. Restored on exit.
    """
    import cogtrix_core.orchestration.graph as _graph_mod

    _orig = _graph_mod.build_agent_graph

    def _patched(*args: Any, **kwargs: Any) -> Any:
        kwargs["context_max_tokens"] = value
        kwargs.setdefault("context_compression", True)
        return _orig(*args, **kwargs)

    _graph_mod.build_agent_graph = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        _graph_mod.build_agent_graph = _orig  # type: ignore[assignment]


def _scan_run_logs(arm_dir: Path) -> dict[str, int]:
    """Count the #2365 tool-pair 400s and compression-churn events across an arm's run logs."""
    counts: dict[str, int] = {"tool_pair_400s": 0}
    counts.update({k: 0 for k in _CHURN_PATTERNS})
    for log_path in sorted(arm_dir.glob("*_run.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lowered = text.lower()
        counts["tool_pair_400s"] += sum(lowered.count(p) for p in _TOOLPAIR_400_PATTERNS)
        for key, pat in _CHURN_PATTERNS.items():
            counts[key] += len(pat.findall(text))
    return counts


def run_arm(
    *,
    scenario: Any,
    model: str,
    repeats: int,
    arm_value: int,
    arm_dir: Path,
    judge: Any,
    dod_gate: bool,
) -> dict[str, Any]:
    """Run one ``context_max_tokens`` arm (N repeats) and return its combined metrics."""
    arm_dir.mkdir(parents=True, exist_ok=True)
    with _force_context_max_tokens(arm_value):
        agent_fn = cogtrix_agent_fn(model, dod_gate=dod_gate)
        summary = run_repeated(scenario, agent_fn, repeats=repeats, report_dir=arm_dir, judge=judge)
    churn = _scan_run_logs(arm_dir)
    return {
        "context_max_tokens": arm_value,
        "fraction_of_reference_window": round(arm_value / _REFERENCE_WINDOW, 3),
        "clean_passes": summary["clean_passes"],
        "repeats": summary["repeats"],
        "pass_rate": summary["pass_rate"],
        "task_achieved_rate": summary["task_achieved_rate"],
        "recovered_from_step_limit_count": summary["recovered_from_step_limit_count"],
        "mean_tool_calls": summary["mean_tool_calls"],
        "mean_elapsed_seconds": summary["mean_elapsed_seconds"],
        # The #2365 symptom + the churn mechanism the cap controls.
        "tool_pair_400s": churn["tool_pair_400s"],
        "churn": {k: churn[k] for k in _CHURN_PATTERNS},
        "bug_frequency": summary["bug_frequency"],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the per-arm comparison the decision reads off."""
    hdr = (
        f"{'cap':>8} {'frac':>5} {'pass':>7} {'task':>6} {'400s':>5} "
        f"{'churn':>6} {'drops':>6} {'tools':>6} {'sec':>7}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        churn_total = r["churn"]["compression_passes"] + r["churn"]["emergency_passes"]
        print(
            f"{r['context_max_tokens']:>8} "
            f"{r['fraction_of_reference_window'] * 100:>4.0f}% "
            f"{r['clean_passes']:>3}/{r['repeats']:<3} "
            f"{r['task_achieved_rate'] * 100:>5.0f}% "
            f"{r['tool_pair_400s']:>5} "
            f"{churn_total:>6} "
            f"{r['churn']['dropped_oldest_events']:>6} "
            f"{r['mean_tool_calls']:>6.1f} "
            f"{r['mean_elapsed_seconds']:>7.0f}"
        )
    print(
        "\nRead: '400s' is the #2365 symptom (want 0); 'churn' = compression passes, "
        "'drops' = hard trims (both should fall as the cap rises); 'tools'/'sec' are "
        "the cost of a bigger cap. Pick the smallest cap with 400s=0 and no "
        "pass/task regression."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep context_max_tokens on a role_sysadmin scenario (#2360/#2365)."
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="subject model id (large window)")
    parser.add_argument("--scenario", default=_DEFAULT_SCENARIO, help="scenario id / prefix")
    parser.add_argument("--repeats", type=int, default=3, help="runs per arm (default 3)")
    parser.add_argument(
        "--arms",
        default=",".join(str(a) for a in _DEFAULT_ARMS),
        help="comma-separated context_max_tokens budgets",
    )
    parser.add_argument("--report-dir", required=True, help="root dir for per-arm JSON reports")
    parser.add_argument(
        "--judge", default=None, help="SOTA judge model id (honesty/root-cause); optional"
    )
    parser.add_argument(
        "--no-dod-gate", action="store_true", help="disable the verify/hand-off gate"
    )
    args = parser.parse_args(argv)

    try:
        arms = [int(x) for x in args.arms.split(",") if x.strip()]
    except ValueError:
        parser.error("--arms must be comma-separated integers, e.g. 40000,131072,200000")
    if not arms:
        parser.error("--arms produced no values")

    scenario = load_scenario(find_scenario(args.scenario))
    report_root = Path(args.report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    judge = None
    if args.judge:
        from tests.role_sysadmin.judge import build_honesty_judge

        judge = build_honesty_judge(args.judge)

    rows: list[dict[str, Any]] = []
    for arm in arms:
        print(
            f"\n=== arm context_max_tokens={arm} "
            f"(~{arm / _REFERENCE_WINDOW * 100:.0f}% of {_REFERENCE_WINDOW}) "
            f"× {args.repeats} on {scenario.id} / {args.model} ==="
        )
        row = run_arm(
            scenario=scenario,
            model=args.model,
            repeats=args.repeats,
            arm_value=arm,
            arm_dir=report_root / f"cap_{arm}",
            judge=judge,
            dod_gate=not args.no_dod_gate,
        )
        rows.append(row)
        print(
            f"  → pass {row['clean_passes']}/{row['repeats']}, "
            f"tool_pair_400s={row['tool_pair_400s']}, "
            f"compression_passes={row['churn']['compression_passes']}, "
            f"dropped_oldest={row['churn']['dropped_oldest_events']}, "
            f"mean_time={row['mean_elapsed_seconds']}s"
        )

    experiment = {
        "model": args.model,
        "scenario": scenario.id,
        "repeats": args.repeats,
        "reference_window": _REFERENCE_WINDOW,
        "arms": rows,
    }
    (report_root / "experiment_summary.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    _print_table(rows)
    print(f"\nFull results: {report_root / 'experiment_summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
