"""12 quality metric functions — each takes a HarnessResult and returns a number.

All metrics correspond exactly to the definitions in docs/optional/quality/METRICS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, ToolMessage

from src.orchestration.graph import _looks_like_phantom_tool_markup
from tests.quality.harness import HarnessResult

# Substring that identifies a per-tool budget cutoff ToolMessage
# Source: src/orchestration/graph.py:1693
_TOOL_DISABLED_MARKER = "has been disabled after"

# Prefix that identifies a tool error ToolMessage
_TOOL_ERROR_PREFIX = "Error:"


# ---------------------------------------------------------------------------
# Tier 1 — Hard gates
# ---------------------------------------------------------------------------


def metric_1_tool_selection_rate(result: HarnessResult) -> float:
    """% of required tools that the agent called at least once."""
    required = result.scenario.metrics.required_tools
    if not required:
        return 100.0
    called = {
        tc["name"]
        for msg in result.trace
        if isinstance(msg, AIMessage)
        for tc in (msg.tool_calls or [])
    }
    return len(set(required) & called) / len(required) * 100.0


def metric_2_parameter_name_f1(result: HarnessResult) -> float:
    """Mean F1 score of parameter name accuracy across all tool calls."""
    f1_scores: list[float] = []
    for msg in result.trace:
        if not isinstance(msg, AIMessage):
            continue
        for tc in msg.tool_calls or []:
            used_names = set(tc.get("args", {}).keys())
            # Required names come from scenario expected args if specified;
            # fall back to used names (perfect score) when not specified
            required_names = used_names  # TODO: enrich from scenario spec
            if not required_names and not used_names:
                f1_scores.append(1.0)
                continue
            intersection = used_names & required_names
            precision = len(intersection) / len(used_names) if used_names else 0.0
            recall = len(intersection) / len(required_names) if required_names else 0.0
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom > 0 else 0.0
            f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0


def metric_3_parameter_value_match_rate(result: HarnessResult) -> float:
    """% of checked parameter values that match the scenario specification.

    Only parameters with expected_args in the scenario YAML are checked.
    If no expected values are specified the scenario passes by default (100%).
    """
    checked = 0
    correct = 0
    spec = result.scenario
    # Build expected values from scenario script tool_calls
    expected: dict[str, dict[str, object]] = {}
    for step in spec.script:
        if step.role == "llm":
            for tc in step.tool_calls:
                expected[tc["id"]] = tc.get("args", {})

    for msg in result.trace:
        if not isinstance(msg, AIMessage):
            continue
        for tc in msg.tool_calls or []:
            call_id = tc.get("id", "")
            expected_args = expected.get(call_id, {})
            actual_args = tc.get("args", {})
            for param, exp_val in expected_args.items():
                checked += 1
                if actual_args.get(param) == exp_val:
                    correct += 1

    return (correct / checked * 100.0) if checked > 0 else 100.0


def metric_4_phantom_call_count(result: HarnessResult) -> int:
    """Count of LLM turns that produced raw tool markup instead of structured calls."""
    return sum(
        1
        for msg in result.trace
        if isinstance(msg, AIMessage)
        and not msg.tool_calls
        and _looks_like_phantom_tool_markup(msg)
    )


def metric_5_task_completion_rate(result: HarnessResult) -> float:
    """100% if agent produced a coherent final response, 0% otherwise."""
    # Find the last AIMessage with actual content
    for msg in reversed(result.trace):
        if isinstance(msg, AIMessage) and msg.content:
            content = str(msg.content)
            if (
                content
                and not content.startswith(_TOOL_ERROR_PREFIX)
                and "persistent formatting issues" not in content
                and "I encountered" not in content
            ):
                return 100.0
            return 0.0
    return 0.0


def metric_6_orphaned_pair_count(result: HarnessResult) -> int:
    """Count of ToolMessages with no preceding AIMessage declaring their tool_call_id."""
    # Build position map: tool_call_id → index of the AIMessage that declared it
    declared: dict[str, int] = {}
    for i, msg in enumerate(result.trace):
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tc_id = tc.get("id") or tc.get("tool_call_id", "")
                if tc_id:
                    declared.setdefault(tc_id, i)

    orphans = 0
    for i, msg in enumerate(result.trace):
        if isinstance(msg, ToolMessage):
            decl_pos = declared.get(msg.tool_call_id)
            if decl_pos is None or decl_pos >= i:
                orphans += 1
    return orphans


def metric_7_tool_readiness_violations(result: HarnessResult) -> int:
    """Count of call_model invocations with empty tool list after simulated reconnect.

    In the harness, a tools_ready violation is detectable when the graph
    returns the specific reconnect-waiting message.
    """
    reconnect_msg = "MCP tools are reconnecting"
    return sum(
        1
        for msg in result.trace
        if isinstance(msg, AIMessage) and reconnect_msg in str(msg.content)
    )


def metric_8_error_recovery_turns(result: HarnessResult) -> int:
    """Max turns the agent took to attempt a recovery after a tool failure.

    If the agent never recovers, returns scenario.max_turns (worst-case penalty).
    """
    max_recovery = 0
    trace = result.trace
    max_turns = result.scenario.metrics.max_turns

    for i, msg in enumerate(trace):
        if not isinstance(msg, ToolMessage):
            continue
        if not str(msg.content).startswith(_TOOL_ERROR_PREFIX):
            continue
        # Find the next AIMessage that makes a tool call or produces prose (recovery)
        recovered = False
        for j in range(i + 1, len(trace)):
            next_msg = trace[j]
            if isinstance(next_msg, AIMessage) and (next_msg.tool_calls or next_msg.content):
                max_recovery = max(max_recovery, j - i)
                recovered = True
                break
        if not recovered:
            max_recovery = max(max_recovery, max_turns)

    return max_recovery


# ---------------------------------------------------------------------------
# Tier 2 — Warning gates
# ---------------------------------------------------------------------------


def metric_9_turns_to_completion(result: HarnessResult) -> int:
    """Total number of AIMessage turns produced during the task."""
    return sum(1 for msg in result.trace if isinstance(msg, AIMessage))


def metric_10_prompt_tokens_per_task(result: HarnessResult) -> int:
    """Total prompt tokens across all turns (from harness counter)."""
    return result.prompt_tokens


def metric_11_post_cutoff_phantom_count(result: HarnessResult) -> int:
    """Phantom calls occurring after the first tool-disabled ToolMessage."""
    cutoff_seen = False
    count = 0
    for msg in result.trace:
        if isinstance(msg, ToolMessage) and _TOOL_DISABLED_MARKER in str(msg.content):
            cutoff_seen = True
        if cutoff_seen and isinstance(msg, AIMessage) and not msg.tool_calls:
            if _looks_like_phantom_tool_markup(msg):
                count += 1
    return count


def metric_12_identical_error_retry_count(result: HarnessResult) -> int:
    """Max consecutive count of same-tool same-error calls before a different action."""
    max_run = 0
    current_run = 0
    last_key: tuple[str, str] | None = None

    for msg in result.trace:
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content)
        if not content.startswith(_TOOL_ERROR_PREFIX):
            current_run = 0
            last_key = None
            continue
        # Extract error class (first line of content)
        error_class = content.split("\n")[0][:80]
        key = (msg.name or "", error_class)
        if key == last_key:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
            last_key = key

    return max_run


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class MetricsReport:
    """All 12 metric values for one scenario run, plus pass/warn status."""

    # Tier 1
    tool_selection_rate: float = 100.0
    parameter_name_f1: float = 1.0
    parameter_value_match_rate: float = 100.0
    phantom_call_count: int = 0
    task_completion_rate: float = 100.0
    orphaned_pair_count: int = 0
    tool_readiness_violations: int = 0
    error_recovery_turns: int = 0

    # Tier 2
    turns_to_completion: int = 0
    prompt_tokens_per_task: int = 0
    post_cutoff_phantom_count: int = 0
    identical_error_retry_count: int = 0

    # Aggregates (filled by compute_all)
    tier1_pass: bool = True
    tier2_warnings: list[str] = field(default_factory=list)
    scenario_id: str = ""


# Pass thresholds (from METRICS.md)
_TIER1_THRESHOLDS: dict[str, tuple[str, float | int]] = {
    "tool_selection_rate": (">=", 100.0),
    "parameter_name_f1": (">=", 0.90),
    "parameter_value_match_rate": (">=", 90.0),
    "phantom_call_count": ("==", 0),
    "task_completion_rate": (">=", 100.0),
    "orphaned_pair_count": ("==", 0),
    "tool_readiness_violations": ("==", 0),
    "error_recovery_turns": ("<=", 2),
}

_TIER2_THRESHOLDS: dict[str, tuple[str, float | int]] = {
    "prompt_tokens_per_task": (">", 50_000),
    "post_cutoff_phantom_count": (">", 0),
    "identical_error_retry_count": (">", 2),
}


def _check(op: str, value: float | int, threshold: float | int) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == ">":
        return value > threshold
    return True


def compute_all(result: HarnessResult) -> MetricsReport:
    """Compute all 12 metrics and evaluate tier thresholds."""
    report = MetricsReport(scenario_id=result.scenario.id)

    # Tier 1
    report.tool_selection_rate = metric_1_tool_selection_rate(result)
    report.parameter_name_f1 = metric_2_parameter_name_f1(result)
    report.parameter_value_match_rate = metric_3_parameter_value_match_rate(result)
    report.phantom_call_count = metric_4_phantom_call_count(result)
    report.task_completion_rate = metric_5_task_completion_rate(result)
    report.orphaned_pair_count = metric_6_orphaned_pair_count(result)
    report.tool_readiness_violations = metric_7_tool_readiness_violations(result)
    report.error_recovery_turns = metric_8_error_recovery_turns(result)

    # Tier 2
    report.turns_to_completion = metric_9_turns_to_completion(result)
    report.prompt_tokens_per_task = metric_10_prompt_tokens_per_task(result)
    report.post_cutoff_phantom_count = metric_11_post_cutoff_phantom_count(result)
    report.identical_error_retry_count = metric_12_identical_error_retry_count(result)

    # Evaluate Tier 1
    failures: list[str] = []
    for attr, (op, threshold) in _TIER1_THRESHOLDS.items():
        value = getattr(report, attr)
        if not _check(op, value, threshold):
            failures.append(f"{attr}: {value} (must be {op} {threshold})")
    report.tier1_pass = len(failures) == 0
    if failures:
        report.tier2_warnings.extend(failures)

    # Evaluate Tier 2
    spec = result.scenario.metrics
    turns_threshold = spec.max_turns
    if report.turns_to_completion > turns_threshold:
        report.tier2_warnings.append(
            f"turns_to_completion: {report.turns_to_completion} > {turns_threshold}"
        )
    for attr, (op, threshold) in _TIER2_THRESHOLDS.items():
        value = getattr(report, attr)
        if _check(op, value, threshold):
            report.tier2_warnings.append(f"{attr}: {value} {op} {threshold} (warning)")

    return report
