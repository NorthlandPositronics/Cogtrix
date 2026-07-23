# One-Time Backend Tasks

_Previous tasks completed._

---

## BUG-002 — Think mode: full ToT report stored as message content, replaces streamed answer

**Date**: 2026-03-24
**Severity**: P1
**Files**: `src/api/turn_runner.py`, `src/tools/deep_think.py`

### Symptom

In think mode, the user sees the correct LLM answer while streaming, but after the turn completes the message is replaced by the full Tree-of-Thought analysis report:

```
Deep Think — Tree-of-Thought Analysis
Task: ...
Iteration 1
Approaches explored (3):
[0.0/10] ...
...
Final Solution (confidence: 0.0/10)
3 iterations, 5 branches explored, 62.7s elapsed
```

### Root cause

1. `run_agent()` is called with `callbacks=[ws_callback]` — the initial LLM answer is streamed to the client token-by-token.
2. After the agent returns, `_run_think_pipeline()` calls `force_deep_think()` → `deep_think()`.
3. `deep_think()` returns the full `_format_report()` output — the entire iteration scaffold including branch scores, reflection summaries, and the final solution — as a single Markdown string.
4. This full report **replaces** `response_text` and is **persisted to the DB** as the AI message content (`content_json=json.dumps({"text": response_text})`).
5. On `done`, the frontend's `invalidateQueries` refetch returns the stored ToT report, overwriting the initial answer the user already saw streamed.

The `confidence: 0.0/10` case occurs when the model has no access to the required capability (e.g. no internet for an OSINT task) — all branches score 0, and the report contains no useful answer, only scaffolding.

### Fix

In `_run_think_pipeline` (or immediately before persisting the message), extract only the final solution from the report instead of using the full report as the message content.

Minimal approach — add a helper in `turn_runner.py`:

```python
import re

def _extract_final_solution(report: str) -> str:
    match = re.search(
        r"^## Final Solution \(confidence: [\d.]+/10\)\n+(.*?)(?=\n^---|\Z)",
        report,
        re.DOTALL | re.MULTILINE,
    )
    if match:
        sol = match.group(1).strip()
        if sol:
            return sol
    return report  # fallback: never lose content
```

Then in `_run_think_pipeline`, return `_extract_final_solution(result)` instead of `result` directly.

Cleaner long-term: add `return_report: bool = True` to `deep_think()` and pass `False` from the API layer so it returns `final.best_solution` directly.
