"""#2213 — per-tool budget classification is content-agnostic, not web-only.

The recursion-aware retrieval ceiling (#2014) originally covered only web/search
tools, so read-only retrieval that iterates the same way — the knowledge base,
file reads, IM-history lookups — fell to the fixed hard cap of 8, too tight for
legitimate progressive lookup (the #2014 runaway was 85 ``query_knowledge_base``
calls, which the web-only list never even covered).

These tests pin the classification now that the policy is hoisted to module level
(importable + centralized).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cogtrix_core.orchestration.graph import (
    _TOOL_BUDGET_ACTION,
    _TOOL_BUDGET_ACTION_CEILING_DIVISOR,
    _TOOL_BUDGET_HARD_EXEMPT,
    _TOOL_BUDGET_RETRIEVAL,
    _TOOL_BUDGET_RETRIEVAL_CEILING_DIVISOR,
    _TOOL_BUDGET_SOFT_EXEMPT,
    _TOOL_CATEGORY_ACTION,
    _TOOL_CATEGORY_CONTROL,
    _TOOL_CATEGORY_RETRIEVAL,
    ToolCategory,
    _resolve_budget_category_sets,
    categorize_tool,
    resolve_tool_category,
)


def _tool(name: str, category: object = "__unset__") -> MagicMock:
    """A tool-like object with .name and optional .metadata['budget_category']."""
    t = MagicMock()
    t.name = name
    t.metadata = {} if category == "__unset__" else {"budget_category": category}
    return t


class TestRetrievalClassification:
    def test_non_web_read_only_retrieval_is_classified(self) -> None:
        for name in (
            "query_knowledge_base",  # RAG / knowledge base
            "read_file",
            "read_pdf",
            "list_directory",
            "file_info",
            "grep",
            "whatsapp_check",  # IM-history lookup
            "telegram_check",
        ):
            assert name in _TOOL_BUDGET_RETRIEVAL, f"{name} must be recursion-ceilinged retrieval"

    def test_web_search_still_classified(self) -> None:
        # Regression: the original web/search coverage must be preserved.
        for name in ("search_web", "web_search", "brave_search", "calendar_search_events"):
            assert name in _TOOL_BUDGET_RETRIEVAL

    def test_retrieval_is_subset_of_hard_exempt(self) -> None:
        # Hard-exempt = soft-exempt ∪ retrieval; the invoker checks retrieval first
        # so these get the recursion-aware ceiling, not full exemption.
        assert _TOOL_BUDGET_RETRIEVAL <= _TOOL_BUDGET_HARD_EXEMPT

    def test_retrieval_tools_still_get_soft_nudge(self) -> None:
        # Retrieval tools must NOT be soft-exempt — the "please synthesize" nudge
        # at _TOOL_BUDGET_SOFT still fires for them; only the hard stop is relaxed
        # to the recursion-aware ceiling.
        assert _TOOL_BUDGET_RETRIEVAL.isdisjoint(_TOOL_BUDGET_SOFT_EXEMPT)

    def test_action_tools_are_not_retrieval(self) -> None:
        # Side-effecting/action tools get their OWN recursion-aware ceiling (#2213
        # Layer 2, looser than retrieval — see TestActionCeilingClassification),
        # not the retrieval ceiling, and stay soft-exempt (no synthesize nudge).
        for name in ("execute_shell_command", "write_file", "patch_file", "append_file"):
            assert name not in _TOOL_BUDGET_RETRIEVAL
            assert name in _TOOL_BUDGET_SOFT_EXEMPT


class TestToolCategoryModel:
    """#2213 Layer 2 (step 1): budget policy keys off a declared category, not a
    flat Internet-centric name allowlist. This slice is behaviour-preserving —
    the derived _TOOL_BUDGET_* sets keep identical values — so these tests pin
    both the classifier and the derivation invariants."""

    def test_categorize_retrieval(self) -> None:
        for name in ("web_search", "query_knowledge_base", "read_file", "grep", "whatsapp_check"):
            assert categorize_tool(name) is ToolCategory.RETRIEVAL

    def test_categorize_control(self) -> None:
        for name in ("request_tools", "report_progress", "checkpoint", "defer_processing"):
            assert categorize_tool(name) is ToolCategory.CONTROL

    def test_categorize_action(self) -> None:
        for name in ("execute_shell_command", "write_file", "append_file", "patch_file"):
            assert categorize_tool(name) is ToolCategory.ACTION

    def test_unknown_and_mcp_tool_defaults_to_standard(self) -> None:
        # Unknown / dynamically-named MCP tools fall to STANDARD (soft nudge +
        # fixed hard cap) — today's default. The issue's "unknown → most
        # restricted" trust default is a deferred semantic change, NOT this slice.
        for name in ("some_unlisted_tool", "mcp__acme__do_thing", ""):
            assert categorize_tool(name) is ToolCategory.STANDARD

    def test_categories_are_disjoint(self) -> None:
        # A tool has exactly one category — the three built-in sets must not overlap.
        assert _TOOL_CATEGORY_RETRIEVAL.isdisjoint(_TOOL_CATEGORY_CONTROL)
        assert _TOOL_CATEGORY_RETRIEVAL.isdisjoint(_TOOL_CATEGORY_ACTION)
        assert _TOOL_CATEGORY_CONTROL.isdisjoint(_TOOL_CATEGORY_ACTION)

    def test_budget_sets_are_derived_from_categories(self) -> None:
        # Behaviour-preservation invariant: the budget sets the invoker consumes
        # are exactly the category unions, so runtime classification is unchanged.
        assert _TOOL_BUDGET_RETRIEVAL == _TOOL_CATEGORY_RETRIEVAL
        assert _TOOL_BUDGET_SOFT_EXEMPT == (_TOOL_CATEGORY_CONTROL | _TOOL_CATEGORY_ACTION)
        assert _TOOL_BUDGET_HARD_EXEMPT == (
            _TOOL_CATEGORY_CONTROL | _TOOL_CATEGORY_ACTION | _TOOL_CATEGORY_RETRIEVAL
        )

    def test_classifier_agrees_with_budget_sets(self) -> None:
        # categorize_tool and the derived sets are two views of one mapping.
        for name in _TOOL_BUDGET_RETRIEVAL:
            assert categorize_tool(name) is ToolCategory.RETRIEVAL
        for name in _TOOL_BUDGET_SOFT_EXEMPT:
            assert categorize_tool(name) in (ToolCategory.CONTROL, ToolCategory.ACTION)


class TestActionCeilingClassification:
    """#2213 Layer 2: action tools get a recursion-aware ceiling (not uncapped),
    looser than retrieval, and stay soft-exempt (no synthesize nudge)."""

    def test_action_set_is_the_action_category(self) -> None:
        assert _TOOL_BUDGET_ACTION == _TOOL_CATEGORY_ACTION
        for name in ("execute_shell_command", "write_file", "append_file", "patch_file"):
            assert name in _TOOL_BUDGET_ACTION

    def test_action_is_not_retrieval(self) -> None:
        assert _TOOL_BUDGET_ACTION.isdisjoint(_TOOL_BUDGET_RETRIEVAL)

    def test_action_stays_soft_exempt(self) -> None:
        # Action tools get a HARD ceiling but no SOFT "please synthesize" nudge —
        # long build/edit sequences shouldn't be nagged toward premature synthesis.
        assert _TOOL_BUDGET_ACTION <= _TOOL_BUDGET_SOFT_EXEMPT

    def test_action_ceiling_is_looser_than_retrieval_but_bounded(self) -> None:
        # Smaller divisor = higher ceiling. Action is looser than retrieval (more
        # headroom for builds) but must stay > 2 so it binds before the
        # ~recursion_limit/2 GraphRecursionError point (~2 super-steps per round).
        assert 2 < _TOOL_BUDGET_ACTION_CEILING_DIVISOR < _TOOL_BUDGET_RETRIEVAL_CEILING_DIVISOR


class TestResolveToolCategoryDeclaration:
    """#2213 Layer 2 — a tool may DECLARE a trusted category (MCP manifest);
    resolve_tool_category honors retrieval/action but not an untrusted escalation
    into an uncapped bucket, and falls back to the name taxonomy otherwise."""

    def test_declared_retrieval_is_honored(self) -> None:
        # A dynamically-named MCP tool no static set could ever enumerate.
        assert resolve_tool_category(_tool("acme_search", "retrieval")) is ToolCategory.RETRIEVAL

    def test_declared_action_is_honored(self) -> None:
        assert resolve_tool_category(_tool("acme_write", "action")) is ToolCategory.ACTION

    def test_declared_control_is_not_trusted(self) -> None:
        # CONTROL is uncapped — a tool must NOT be able to self-exempt into it.
        # Unknown name + untrusted declaration → STANDARD (most-restricted).
        assert resolve_tool_category(_tool("acme_evil", "control")) is ToolCategory.STANDARD

    def test_declared_standard_falls_back(self) -> None:
        assert resolve_tool_category(_tool("acme_thing", "standard")) is ToolCategory.STANDARD

    def test_invalid_declaration_falls_back(self) -> None:
        assert resolve_tool_category(_tool("acme_thing", "banana")) is ToolCategory.STANDARD

    def test_non_string_declaration_falls_back(self) -> None:
        assert resolve_tool_category(_tool("acme_thing", 123)) is ToolCategory.STANDARD

    def test_no_metadata_uses_name_taxonomy(self) -> None:
        # Built-in with no declaration keeps its exact name-based category.
        assert resolve_tool_category(_tool("query_knowledge_base")) is ToolCategory.RETRIEVAL
        assert resolve_tool_category(_tool("execute_shell_command")) is ToolCategory.ACTION
        assert resolve_tool_category(_tool("request_tools")) is ToolCategory.CONTROL
        assert resolve_tool_category(_tool("totally_unknown")) is ToolCategory.STANDARD

    def test_declaration_cannot_downgrade_a_builtin_action_to_retrieval(self) -> None:
        # If a name IS a known action built-in but declares retrieval, we honor the
        # (trusted-declarable) declaration — declarations are opt-in and both are
        # bounded; this documents that intended precedence.
        assert resolve_tool_category(_tool("write_file", "retrieval")) is ToolCategory.RETRIEVAL


class TestResolveBudgetCategorySets:
    """The graph-build helper folds trusted declarations into the budget sets."""

    def test_behaviour_preserving_with_no_declarations(self) -> None:
        # All built-ins (or an empty list) → exactly the static module constants.
        retr, act, soft, hard = _resolve_budget_category_sets(
            [_tool("query_knowledge_base"), _tool("execute_shell_command"), _tool("request_tools")]
        )
        assert retr == _TOOL_BUDGET_RETRIEVAL
        assert act == _TOOL_BUDGET_ACTION
        assert soft == _TOOL_BUDGET_SOFT_EXEMPT
        assert hard == _TOOL_BUDGET_HARD_EXEMPT

    def test_empty_list_equals_static_constants(self) -> None:
        retr, act, soft, hard = _resolve_budget_category_sets([])
        assert (retr, act, soft, hard) == (
            _TOOL_BUDGET_RETRIEVAL,
            _TOOL_BUDGET_ACTION,
            _TOOL_BUDGET_SOFT_EXEMPT,
            _TOOL_BUDGET_HARD_EXEMPT,
        )

    def test_mcp_retrieval_tool_joins_retrieval_and_hard_exempt(self) -> None:
        retr, act, _soft, hard = _resolve_budget_category_sets(
            [_tool("mcp_docs_search", "retrieval")]
        )
        assert "mcp_docs_search" in retr
        assert "mcp_docs_search" in hard  # retrieval ⊆ hard_exempt (gets the ceiling)
        assert "mcp_docs_search" not in act

    def test_mcp_action_tool_joins_action_and_soft_exempt(self) -> None:
        retr, act, soft, _hard = _resolve_budget_category_sets([_tool("mcp_deploy", "action")])
        assert "mcp_deploy" in act
        assert "mcp_deploy" in soft  # action ⊆ soft_exempt (no synthesize nudge)
        assert "mcp_deploy" not in retr

    def test_untrusted_control_declaration_stays_standard(self) -> None:
        # An MCP tool declaring control must NOT land in any exempt set — it stays
        # STANDARD (fixed cap 8), the most-restricted bucket.
        retr, act, soft, hard = _resolve_budget_category_sets([_tool("mcp_sneaky", "control")])
        assert "mcp_sneaky" not in retr
        assert "mcp_sneaky" not in act
        assert "mcp_sneaky" not in soft
        assert "mcp_sneaky" not in hard

    def test_builtins_preserved_alongside_declared_mcp_tool(self) -> None:
        retr, act, _soft, _hard = _resolve_budget_category_sets(
            [_tool("mcp_docs_search", "retrieval"), _tool("mcp_deploy", "action")]
        )
        assert _TOOL_BUDGET_RETRIEVAL <= retr  # static built-ins still present
        assert _TOOL_BUDGET_ACTION <= act
        assert "mcp_docs_search" in retr and "mcp_deploy" in act


class TestResolveToolCategoryHardening:
    """#2437 defect 2 + #2438 — resolve_tool_category must not crash on odd
    metadata and must only trust the namespaced ``budget_category`` key."""

    def test_non_dict_metadata_does_not_crash(self) -> None:
        # A truthy non-dict .metadata (list/str/int/obj) must fall back safely,
        # not AttributeError at graph-build time.
        for bad in (["action"], "action", 123, object()):
            t = MagicMock()
            t.name = "acme_thing"
            t.metadata = bad
            assert resolve_tool_category(t) is ToolCategory.STANDARD

    def test_registry_style_category_key_is_ignored(self) -> None:
        # The registry/delegate vocabulary lives under metadata['category']; budget
        # policy must ONLY trust metadata['budget_category'], so a registry-style
        # 'category' (even one that happens to read 'action') is NOT trusted.
        t = MagicMock()
        t.name = "totally_unknown"
        t.metadata = {"category": "action"}
        assert resolve_tool_category(t) is ToolCategory.STANDARD

    def test_none_metadata_falls_back_to_name(self) -> None:
        t = MagicMock()
        t.name = "query_knowledge_base"
        t.metadata = None
        assert resolve_tool_category(t) is ToolCategory.RETRIEVAL


class TestBudgetCategoryVocabularyIsolation:
    """#2438 — the budget ToolCategory vocabulary must never intersect the
    registry/delegate 'category' vocabulary. Namespacing the key structurally
    prevents leakage; this pins the value-spaces disjoint too, so a future rename
    (e.g. adding 'action' to the delegate taxonomy) fails loudly here instead of
    silently re-budgeting a tool."""

    _KNOWN_DELEGATE_VOCAB = frozenset(
        {"readonly", "mutation", "privacy", "recursive", "messaging", "scheduling", "confirmation"}
    )

    def test_budget_values_disjoint_from_delegate_vocabulary(self) -> None:
        budget_values = {c.value for c in ToolCategory}
        assert budget_values.isdisjoint(self._KNOWN_DELEGATE_VOCAB)

    def test_budget_values_disjoint_from_live_registry(self) -> None:
        # Whatever tools have actually registered (delegate._TOOL_CATEGORIES,
        # populated at import) must not intersect budget policy's vocabulary.
        from cogtrix_core.tools.delegate import _TOOL_CATEGORIES

        budget_values = {c.value for c in ToolCategory}
        assert budget_values.isdisjoint(set(_TOOL_CATEGORIES.values()))
