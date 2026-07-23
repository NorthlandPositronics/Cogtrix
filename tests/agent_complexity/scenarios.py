"""Default agent complexity test scenarios — 5 tasks spanning the complexity matrix.

Originally documented in personal memory as the agent-test recipe;
codified here under #1930 so the prompts version with the source and
new scenarios can be added without editing a shell heredoc.

Each scenario exercises a different combination of:

  * tool tier — file_ops only, shell + tests, search + fetch, …
  * task length — short (single-file output) vs sustained (multi-phase)
  * recovery surface — does the agent need to handle permission denials,
    tool-name confusion, search dry-holes, etc.

Add a new scenario by appending a :class:`Scenario` to :data:`DEFAULT_SCENARIOS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComplexityTier = Literal["MODERATE", "COMPLEX_ACTION", "COMPLEX_RESEARCH"]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One agent complexity test scenario.

    Attributes:
        slug: Short identifier used for log filenames + container names
            (e.g. ``gas`` → ``test1-gas`` container, ``test1.log`` file).
            ASCII alphanumeric + hyphen, ≤ 32 chars.
        complexity: The classifier tier the agent's complexity framework
            (see ``docs/testing/agent-effectiveness-metrics.md``)
            should assign to the prompt. Used by the runner's summary
            to flag scenarios whose runtime / tool-call count is
            wildly out of line with the tier.
        prompt: The user-facing task description.  Long literals OK —
            triple-quoted strings keep them readable in source.
        expected_tools: Names of tools the agent is likely to use.
            The runner reports which expected tools were NOT invoked
            (a hint that the agent took an unusual recovery path).
            ``None`` = tier-agnostic, don't check.
    """

    slug: str
    complexity: ComplexityTier
    prompt: str
    expected_tools: tuple[str, ...] | None = None


# ── Default 5-task fleet ──────────────────────────────────────────────


_T1_GAS_PROMPT = (
    "Design and build a Google Apps Script (GAS) web app named 'EchoApi' that "
    "exposes: (1) doPost(e) — accepts JSON {message: string, repeat?: "
    "number<=10}, validates inputs, returns {echo: <repeated message joined by "
    "' | '>, timestamp: ISO8601 UTC, length: int} as JSON; (2) doGet() — "
    "returns a small HTML landing page documenting the API with curl examples. "
    "Include proper error handling (return {error: ...} with appropriate "
    "HTTP-like semantics), use HtmlService for the landing page, and provide "
    "deploy-as-web-app instructions in a final notes section. Write the "
    "complete Code.gs and accompanying README.md."
)

_T2_PYTHON_DATA_PROMPT = (
    "Write a Python 3 script analyze_sales.py that reads a CSV file with "
    "columns date,region,product,units,revenue (date as YYYY-MM-DD). Generate "
    "sample data of 30 rows spanning 3 months and 4 regions, then write the "
    "analyzer producing a markdown report with: (a) total revenue and total "
    "units, (b) top-3 products by revenue with their share %, (c) average "
    "units per transaction per region, (d) month-over-month revenue trend "
    "with % change. Use only stdlib (csv, datetime, collections, statistics). "
    "Include 3 unit tests using unittest. Run the tests and the analyzer on "
    "the sample data; include the produced report inline."
)

_T3_SECURITY_PROMPT = (
    "Perform a security audit of this bash script and report findings as a "
    "numbered list (severity: critical/high/medium/low, line ref, issue, "
    "recommended fix). Then write a hardened replacement. Script:\n"
    "#!/bin/bash\n"
    "API_KEY='sk-prod-abc123XYZdef456'\n"
    "USER_INPUT=$1\n"
    "DB_HOST='db.internal'\n"
    "DB_PASS='hunter2'\n"
    'curl -X POST https://api.example.com/v1/lookup?q=$USER_INPUT -H "Authorization: Bearer $API_KEY"\n'
    'RESULT=$(eval "echo $USER_INPUT | mysql -h $DB_HOST -u root -p$DB_PASS")\n'
    "log_path=/tmp/audit_$USER.log\n"
    'echo "$RESULT" >> $log_path\n'
    "chmod 777 $log_path\n"
    "find /home -name '*.bak' -exec rm {} \\;\n"
    "Cover: hardcoded secrets, command injection, eval risk, unsafe "
    "redirection, permission issues, missing input validation, error handling gaps."
)

_T4_WASI_RESEARCH_PROMPT = (
    "Produce a 2000+ word technical report on the current state of "
    "WebAssembly System Interface (WASI) Preview 2 adoption (as of 2026). "
    "Cover: (1) what WASI Preview 2 is and how it differs from Preview 1, "
    "(2) the component model and interface types, (3) which runtimes "
    "implement it (Wasmtime, WasmEdge, Wazero, others) and their conformance "
    "status, (4) major adoption use cases (serverless, plugins, embedded), "
    "(5) known production deployments and their reported wins/pain points, "
    "(6) tooling maturity (cargo-component, jco, wit-bindgen), (7) what's "
    "still missing or blocking wider adoption. Use search and fetch tools to "
    "gather current information. Include source URLs as inline citations. "
    "Format as a structured markdown document with clear section headings."
)

_T5_JQ_PROMPT = (
    "Build a Python 3 CLI tool jq_lite.py that takes a JSON file path and a "
    "dot-path query. Support: (1) nested key access ('.a.b.c'), (2) array "
    "indexing ('.users[0].name'), (3) wildcard array iteration "
    "('.users[].email'), (4) simple equality filtering "
    "('.users[?active==true].name'), (5) length operation ('.users | "
    "length'). On unmatched paths print 'null'; on type mismatch print a "
    "friendly error to stderr with exit 2. Use only stdlib (json, sys, re, "
    "argparse). Include unittest tests (≥12 cases) covering all features, "
    "edge cases (missing keys, out-of-range indices, mixed types), and CLI "
    "integration via subprocess. Run the test suite and report pass count."
)


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        slug="gas",
        complexity="COMPLEX_ACTION",
        prompt=_T1_GAS_PROMPT,
        expected_tools=("write_file",),
    ),
    Scenario(
        slug="pyda",
        complexity="MODERATE",
        prompt=_T2_PYTHON_DATA_PROMPT,
        expected_tools=("write_file", "execute_shell_command"),
    ),
    Scenario(
        slug="sec",
        complexity="MODERATE",
        prompt=_T3_SECURITY_PROMPT,
        expected_tools=("write_file",),
    ),
    Scenario(
        slug="wasi",
        complexity="COMPLEX_RESEARCH",
        prompt=_T4_WASI_RESEARCH_PROMPT,
        expected_tools=("web_search", "http_get"),
    ),
    Scenario(
        slug="jq",
        complexity="MODERATE",
        prompt=_T5_JQ_PROMPT,
        expected_tools=("write_file", "execute_shell_command"),
    ),
)


__all__ = ["Scenario", "ComplexityTier", "DEFAULT_SCENARIOS"]
