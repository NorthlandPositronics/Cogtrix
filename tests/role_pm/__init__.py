"""PM role-test harness (#1948).

A one-shot test harness that puts the Cogtrix agent into a fully
equipped Project Manager role — system prompt + RAG corpus + tool
whitelist — and looks for hallucinations and flawed logic.  Unlike
Gate 2 (objective per-scenario correctness) or the agent-complexity
Docker fleet (short-horizon reasoning), this harness stresses the
whole stack at once.

Entry point: ``python -m tests.role_pm.run``.

See ``tests/role_pm/README.md`` for usage and scorecard semantics.
"""
