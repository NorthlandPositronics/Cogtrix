"""Tests for the ``enforce_tools_available`` scenario flag (#2016).

PM cycle 4 (#2006) catalogued Cluster B (extraneous_tool_calls = 11 across
18 iterations on qwen3-coder).  Cycle 5 (#2006 comment) showed
kimi-k2-6 made zero such calls against the same whitelist, proving the
gap is model-discipline, not orchestration.  This flag, set in the
scenario YAML, asks the harness to append a strict-whitelist clause
to the system prompt so any model — including qwen3-coder — receives
the rule as a quotable directive rather than as inference from the
cycle-2 ``"RAG-only path for corpus facts"`` paragraph.
"""

from __future__ import annotations

from tests.role_pm.run import (
    _TOOL_WHITELIST_PREAMBLE_TEMPLATE,
    _format_whitelist_block,
)

# ── Whitelist formatter ─────────────────────────────────────────────


class TestFormatWhitelistBlock:
    def test_emits_sorted_deduplicated_bullets(self) -> None:
        block = _format_whitelist_block(
            tools_required=["query_knowledge_base", "checkpoint"],
            tools_available=["checkpoint", "query_knowledge_base"],
        )
        # Sorted, deduplicated, prefixed with ``- ``.
        assert block == "- ``checkpoint``\n- ``query_knowledge_base``"

    def test_handles_disjoint_required_and_available(self) -> None:
        block = _format_whitelist_block(
            tools_required=["query_knowledge_base"],
            tools_available=["checkpoint"],
        )
        # Both names appear, sorted.
        assert "``checkpoint``" in block
        assert "``query_knowledge_base``" in block

    def test_empty_inputs_yield_no_tools_placeholder(self) -> None:
        block = _format_whitelist_block(tools_required=[], tools_available=[])
        # Empty fallback must be informative, not an empty string, so
        # the rendered preamble still parses as valid markdown.
        assert "uses no tools" in block


# ── Preamble template ───────────────────────────────────────────────


class TestPreambleTemplate:
    def test_template_contains_explicit_forbidden_examples(self) -> None:
        """The preamble must name specific forbidden tools by example
        so the model has concrete anchors (not just a generic 'whitelist
        only' instruction).  These examples come straight from cycle-4
        evidence — read_file (8 calls in scenario 06 iter 3), get_weather
        (1 call in scenario 03 iter 2), write_file (none yet but the
        natural next mis-fallback).
        """
        rendered = _TOOL_WHITELIST_PREAMBLE_TEMPLATE.format(
            whitelist_block="- ``query_knowledge_base``"
        )
        assert "read_file" in rendered
        assert "write_file" in rendered
        assert "get_weather" in rendered

    def test_template_references_tracking_issues(self) -> None:
        """The preamble's provenance should name the cycle / issue
        numbers so a future operator reading the system prompt can
        trace why this rule exists.  Drops here without warning are
        usually a sign someone removed the cycle-2 / cycle-4 context."""
        rendered = _TOOL_WHITELIST_PREAMBLE_TEMPLATE.format(
            whitelist_block="- ``query_knowledge_base``"
        )
        assert "#1948" in rendered or "#2016" in rendered

    def test_template_directs_to_surface_gap_instead_of_fallback(self) -> None:
        """The Cluster-B failure mode is the model reaching for read_file
        when RAG returns nothing useful.  The preamble must give it the
        explicit fallback verbiage ("I retrieved N chunks but did not
        find X") so it has a script to use instead of a forbidden tool.
        """
        rendered = _TOOL_WHITELIST_PREAMBLE_TEMPLATE.format(
            whitelist_block="- ``query_knowledge_base``"
        )
        assert "surface" in rendered.lower() and "gap" in rendered.lower()


# ── End-to-end via _build_llm_and_graph: preamble appended only when
#    the flag is on; default behaviour is unchanged when the flag is off.
# ────────────────────────────────────────────────────────────────────
#
# These tests stub _build_llm and build_agent_graph so they exercise
# the system_prompt mutation without actually loading langchain /
# starting a real LLM.  The whole point of the flag is that
# system_prompt mutates in exactly one place, and the existing tool
# binding stays untouched.


def _captured_system_prompt(*, enforce: bool) -> str:
    """Run ``_build_llm_and_graph`` with everything else stubbed and
    return the ``system_prompt`` it would pass to ``build_agent_graph``.
    """
    import sys
    import types
    from unittest.mock import patch

    from tests.role_pm import run as run_mod

    captured: dict[str, str] = {}

    # Stub out _build_llm to avoid touching real provider keys.
    # Accepts ``active_key`` as a no-op kwarg — the real signature gained
    # this parameter so the harness can route any model via OpenRouter
    # when its native env_key is absent (see tests/role_pm/run.py:303).
    def _fake_build_llm(_model, *, active_key=None):  # noqa: ANN001, ARG001 — stub
        return object()

    # Stub out build_agent_graph to capture the system_prompt and
    # return a sentinel.  The real build_agent_graph would require
    # langchain + a real LLM.
    def _fake_build_agent_graph(*, system_prompt, **_kwargs):  # noqa: ANN003
        captured["system_prompt"] = system_prompt
        return object()

    # Build a synthetic ModelConfig — just needs to satisfy attribute
    # access in _build_llm (which we've stubbed).
    fake_model = types.SimpleNamespace(
        id="fake-model", provider="openai_compatible", env_key="OPENAI_API_KEY"
    )

    # Patch the imports inside _build_llm_and_graph.  The function
    # imports at call time so the patch path must match the import path.
    with (
        patch("tests.evaluation.runner._build_llm", _fake_build_llm),
        patch("cogtrix_core.orchestration.graph.build_agent_graph", _fake_build_agent_graph),
        # Stub the StructuredTool / query_knowledge_base imports to
        # avoid pulling rag deps.
        patch.dict(
            sys.modules,
            {
                "langchain_core.tools": types.SimpleNamespace(
                    StructuredTool=types.SimpleNamespace(
                        from_function=lambda **_kw: object(),
                    )
                ),
                "cogtrix_core.tools.rag": types.SimpleNamespace(
                    KnowledgeQueryInput=object,
                    query_knowledge_base=lambda **_kw: "",
                ),
            },
        ),
    ):
        run_mod._build_llm_and_graph(
            model=fake_model,
            system_prompt="ORIGINAL SYSTEM PROMPT BODY",
            tools_required=["query_knowledge_base"],
            tools_available=[],
            enforce_tools_available=enforce,
        )

    return captured["system_prompt"]


class TestBuildLLMAndGraphPreambleInjection:
    def test_flag_off_preserves_original_system_prompt(self) -> None:
        sp = _captured_system_prompt(enforce=False)
        # The original body is preserved untouched and no whitelist
        # clause is appended.  This is the default behaviour for any
        # scenario / Gate 2 caller that does NOT opt in.
        assert sp == "ORIGINAL SYSTEM PROMPT BODY"
        assert "Strict tool whitelist" not in sp

    def test_flag_on_appends_strict_whitelist_clause(self) -> None:
        sp = _captured_system_prompt(enforce=True)
        # The original body remains as a prefix (no truncation, no
        # rewrite) — the clause is *appended*, never substituted.
        assert sp.startswith("ORIGINAL SYSTEM PROMPT BODY")
        # And the clause appears verbatim.
        assert "Strict tool whitelist (enforced — #1948 / #2016)" in sp

    def test_flag_on_inlines_actual_whitelist(self) -> None:
        sp = _captured_system_prompt(enforce=True)
        # The model should see the specific tool name from its
        # ``tools_required`` list, not just generic "the whitelist".
        assert "``query_knowledge_base``" in sp


# ── Contract test on the YAMLs themselves ───────────────────────────


class TestAllPMScenariosOptIn:
    """Every PM scenario YAML must carry ``enforce_tools_available: true``
    after #2016 lands.  A future YAML edit that silently drops the flag
    would re-enable Cluster B without warning; this test guards that.
    """

    def test_every_scenario_yaml_has_the_flag(self) -> None:
        from pathlib import Path

        import yaml

        scenario_dir = Path(__file__).parent / "scenarios"
        yaml_paths = sorted(scenario_dir.glob("*.yaml"))
        assert yaml_paths, "no PM scenario YAMLs found"

        for path in yaml_paths:
            data = yaml.safe_load(path.read_text())
            assert data.get("enforce_tools_available") is True, (
                f"{path.name} is missing ``enforce_tools_available: true``; "
                f"see #2016 for why the flag must be on for every PM scenario."
            )
