"""Tests for pure-logic functions in src/agent/core.py.

Tests only deterministic, side-effect-free functions that do not require
a live LLM, FAISS, or any external service.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.core import (
    _estimate_msg_tokens,
    _format_model_detail,
    _format_models_table,
    _trim_to_token_budget,
    _truncate_content,
    build_system_prompt,
    format_milestone_instructions,
    prepare_messages_with_context,
)

# ---------------------------------------------------------------------------
# _format_model_detail
# ---------------------------------------------------------------------------


class TestFormatModelDetail:
    def test_string_passthrough(self):
        assert _format_model_detail("gpt-4") == "gpt-4"

    def test_dict_basic(self):
        result = _format_model_detail({"provider": "openai", "model": "gpt-4"})
        assert result == "openai/gpt-4"

    def test_dict_with_temperature(self):
        result = _format_model_detail({"provider": "openai", "model": "gpt-4", "temperature": 0.7})
        assert "temp=0.7" in result

    def test_dict_with_context_window(self):
        result = _format_model_detail(
            {"provider": "ollama", "model": "qwen3", "context_window": 8192}
        )
        assert "ctx=8192" in result

    def test_dict_with_num_ctx(self):
        result = _format_model_detail({"provider": "ollama", "model": "qwen3", "num_ctx": 4096})
        assert "ctx=4096" in result

    def test_dict_missing_keys_uses_question_mark(self):
        result = _format_model_detail({})
        assert "?" in result

    def test_model_config_object(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")
        result = _format_model_detail(mc)
        assert "anthropic/claude-sonnet-4-5" in result

    def test_model_config_with_temperature(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="openai", model="gpt-4", temperature=0.5)
        result = _format_model_detail(mc)
        assert "temp=0.5" in result

    def test_model_config_with_context_window(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="openai", model="gpt-4", context_window=16384)
        result = _format_model_detail(mc)
        assert "ctx=16384" in result

    def test_unknown_type_returns_str(self):
        result = _format_model_detail(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# _format_models_table
# ---------------------------------------------------------------------------


class TestFormatModelsTable:
    def test_empty_dict_returns_empty_string(self):
        assert _format_models_table({}) == ""

    def test_single_model(self):
        result = _format_models_table({"default": "gpt-4"})
        assert "default" in result
        assert "gpt-4" in result

    def test_delegation_models_section(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast"])
        assert "Delegation targets" in result
        assert "fast" in result

    def test_delegation_models_others_section(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast"])
        # "smart" should be in the "Other models" section
        assert "Other models" in result
        assert "smart" in result

    def test_delegation_models_all_delegated(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast", "smart"])
        # No "Other models" section when all are delegation targets
        assert "Other models" not in result

    def test_delegation_model_not_in_registry_skipped(self):
        models = {"fast": "gpt-3.5"}
        result = _format_models_table(models, delegation_models=["nonexistent"])
        # nonexistent not in models, shouldn't appear in table
        assert "nonexistent" not in result

    def test_returns_string_with_header(self):
        result = _format_models_table({"m": "v"})
        assert "Available Models" in result


# ---------------------------------------------------------------------------
# format_milestone_instructions
# ---------------------------------------------------------------------------


class TestFormatMilestoneInstructions:
    def test_empty_list(self):
        result = format_milestone_instructions([])
        assert "Milestones" in result
        assert "report_progress" in result

    def test_single_milestone(self):
        m = MagicMock()
        m.index = 1
        m.title = "Research phase"
        result = format_milestone_instructions([m])
        assert "1. Research phase" in result

    def test_multiple_milestones(self):
        milestones = []
        for i, title in enumerate(["Plan", "Build", "Test"], start=1):
            m = MagicMock()
            m.index = i
            m.title = title
            milestones.append(m)
        result = format_milestone_instructions(milestones)
        assert "1. Plan" in result
        assert "2. Build" in result
        assert "3. Test" in result

    def test_includes_focus_rule(self):
        result = format_milestone_instructions([])
        assert "Focus rule" in result


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_default_prompt_used_when_base_none(self):
        from src.agent.core import DEFAULT_SYSTEM_PROMPT

        result = build_system_prompt()
        assert DEFAULT_SYSTEM_PROMPT in result

    def test_custom_base_prompt(self):
        result = build_system_prompt(base_prompt="My custom instructions")
        assert "My custom instructions" in result

    def test_mode_additions_appended(self):
        result = build_system_prompt(mode_additions="## Code mode context")
        assert "## Code mode context" in result

    def test_tool_instructions_appended(self):
        result = build_system_prompt(tool_instructions="Use JSON for tool calls.")
        assert "Use JSON for tool calls." in result

    def test_milestone_instructions_appended(self):
        result = build_system_prompt(milestone_instructions="## Milestones\n1. Done")
        assert "## Milestones" in result

    def test_models_table_included_when_no_active_tools(self):
        models = {"fast": "gpt-3.5"}
        result = build_system_prompt(models=models)
        assert "fast" in result

    def test_models_table_excluded_when_no_delegation_tool_active(self):
        models = {"fast": "gpt-3.5"}
        # active_tool_names without delegate_task or delegate_parallel
        result = build_system_prompt(models=models, active_tool_names={"web_search", "shell"})
        assert "fast" not in result

    def test_models_table_included_when_delegate_task_active(self):
        models = {"fast": "gpt-3.5"}
        result = build_system_prompt(
            models=models, active_tool_names={"delegate_task", "web_search"}
        )
        assert "fast" in result

    def test_parts_joined_with_double_newline(self):
        result = build_system_prompt(
            base_prompt="base", mode_additions="mode", tool_instructions="tools"
        )
        assert "\n\n" in result

    def test_empty_models_no_table(self):
        result = build_system_prompt(models={})
        assert "Available Models" not in result

    def test_current_date_injected_into_system_prompt(self):
        # Regression: issue #886 — models with pre-current-year training cutoffs
        # discard valid search results as "fictional" when the system prompt
        # lacks an explicit date anchor. build_system_prompt() must prepend
        # the current UTC date so the model knows what "today" is before
        # reasoning begins.
        result = build_system_prompt()
        assert "Today's date is" in result
        # Date must be in the format "Month DD, YYYY (UTC)"
        import re

        date_pattern = r"Today's date is [A-Z][a-z]+ \d{1,2}, \d{4} \(UTC\)"
        assert re.search(
            date_pattern, result
        ), f"Date prefix not found or malformed. Got: {result[:120]!r}"
        # Date must appear at the very start of the prompt
        assert result.startswith(
            "Today's date is"
        ), f"Date must be the first thing in the prompt. Got start: {result[:80]!r}"

    def test_current_date_injected_with_custom_base_prompt(self):
        # Date injection must work even when a custom base_prompt is provided.
        result = build_system_prompt(base_prompt="Do something useful.")
        assert "Today's date is" in result
        # Custom base must appear after the date prefix
        assert "Do something useful." in result
        assert result.index("Do something useful.") > result.index("Today's date is")

    def test_current_date_injected_with_all_additions(self):
        # Date injection must not be disrupted by mode_additions, tool_instructions,
        # milestone_instructions, decision_accountability_prompt, or
        # pre_action_confirmation_prompt.
        result = build_system_prompt(
            base_prompt="Base.",
            mode_additions="Mode.",
            tool_instructions="Tools.",
            milestone_instructions="Milestones.",
            decision_accountability_prompt="Accountability.",
            pre_action_confirmation_prompt="PAC.",
        )
        assert "Today's date is" in result
        # All additions must still be present
        for addition in [
            "Base.",
            "Mode.",
            "Tools.",
            "Milestones.",
            "Accountability.",
            "PAC.",
        ]:
            assert addition in result

    def test_default_prompt_forbids_speculation_on_empty_prompt(self):
        # Bug L #1735 regression — when the user's message provides no
        # actionable task ("Do it now please.", "Help me", "Proceed"),
        # the agent must ask one clarifying question rather than firing
        # speculative tool calls (request_tools / list_goals /
        # list_tasks / read_agent_inbox). The 2026-05-22 round-5
        # corpus replay F01 reproducer showed the agent fired 9 tool
        # calls before producing a paragraph; the expected_shape was
        # ``clarifying-question``.
        #
        # The existing Clarification Policy block only triggers on
        # "irreversible AND ambiguous" — "Do it now" is neither
        # irreversible nor scope-ambiguous-in-the-named-sense, so
        # the model skipped that rule and explored. The new clause
        # covers the zero-task-stated case explicitly.
        result = build_system_prompt()
        lower = result.lower()

        # The rule must name the "no actionable task" case so the
        # model recognises empty / filler prompts.
        assert "no actionable task" in lower, (
            "Empty-prompt rule missing from default system prompt — "
            "F01-style 'Do it now please.' prompts will trigger "
            "speculative tool calls (Bug L #1735)"
        )

        # The concrete failure phrases the reproducer showed must be
        # named so the model can pattern-match.
        assert '"do it now' in lower or "'do it now" in lower, (
            "Default prompt must include 'Do it now' as a concrete "
            "example so the model recognises filler prompts"
        )

        # The directive must explicitly forbid speculative
        # tool-listing — that's the exact behaviour the F01 trace
        # showed (request_tools + list_goals + list_tasks +
        # read_agent_inbox before any user task is identified).
        assert "do not speculate" in lower or "do not call any tools" in lower, (
            "Default prompt must forbid speculative tool calls on " "empty / filler prompts"
        )

        # And it must explicitly mandate the clarifying-question
        # shape so the expected_shape: clarifying-question check
        # passes downstream.
        assert "clarifying question" in lower

    def test_default_prompt_forbids_sycophantic_prefix_on_unchanged_answer(self):
        # Bug G #1713 regression — when the user contradicts the agent
        # or provides new context, the agent must not preface a repeated
        # answer with "You're absolutely right" / "I apologize" /
        # "You're raising an important point". cogtrix56 turns 3-5
        # showed three consecutive turns starting with such phrases,
        # each followed by byte-identical (or near-identical) content
        # to the prior turn — the apology gave the illusion of update
        # without any actual update, amplifying the user's trust loss
        # when they noticed. The fix is a system-prompt rule that
        # forbids the validation prefix unless the answer was
        # substantively revised AND explicitly demands an
        # "unchanged" disclosure when the conclusion didn't move.
        result = build_system_prompt()
        lower = result.lower()

        # The forbidden phrases must be named so the model treats them
        # as filtered output. Without enumeration the rule is too vague
        # and the RLHF-agreeable bias wins.
        assert "you're absolutely right" in lower, (
            "Default prompt must explicitly name 'You're absolutely right' "
            "as a forbidden prefix — vague 'avoid sycophancy' guidance "
            "fails to counter the RLHF bias"
        )
        assert "i apologize" in lower
        # "You're raising an important point" was the turn-4 cogtrix56
        # opener — must also be banned.
        assert "raising an important point" in lower or "you're right" in lower, (
            "Default prompt must also forbid 'You're raising an important "
            "point' / 'You're right' family phrases"
        )

        # The rule must allow the apology prefix ONLY when the answer
        # is actually being revised — without that exception clause,
        # the agent can't apologise for a real mistake.
        assert "substantively revise" in lower or "substantively revised" in lower, (
            "Rule must explicitly say the prefix is allowed when "
            "substantively revising — otherwise genuine error correction "
            "is suppressed"
        )

        # And when the conclusion is unchanged, the model must say so
        # explicitly with the phrase "conclusion is unchanged". Pinning
        # this exact phrase makes the orchestrator-side stuck-conclusion
        # nudge (call_model.py) line up with the prompt rule.
        assert "conclusion is unchanged" in lower, (
            "Default prompt must require the explicit "
            "'conclusion is unchanged' phrasing when no revision occurs — "
            "this is the load-bearing escape hatch the model uses to "
            "remain honest when it can't change its answer"
        )

    def test_default_prompt_directs_http_get_for_user_provided_urls(self):
        # Bug I #1718 regression — when the user explicitly provides a URL
        # ("check this page: https://...") the agent must prefer http_get
        # over web_search with a `site:` query. The orchestrator can't
        # rewrite tool calls after the fact; the model has to choose right
        # the first time, so this guidance must be in the default system
        # prompt. cogtrix57 turn-5 reproducer: agent emitted 5+ web_search
        # calls with `site:scnsoft.com/management-team ...` instead of one
        # http_get on the exact URL the user supplied, wasting search
        # budget and missing the page content entirely.
        result = build_system_prompt()
        # The guidance must name http_get as the FIRST action for
        # user-supplied URLs and call out the discovery-vs-retrieval split.
        lower = result.lower()
        assert "user explicitly provides a url" in lower or (
            "user provides a url" in lower and "explicitly" in lower
        ), "URL-handling rule missing from default system prompt"
        # Must explicitly tell the agent to use http_get first.
        assert "http_get" in result, "Default prompt must mention http_get for URL retrieval"
        # Must explicitly contrast with web_search to avoid the
        # site:-query bias the cogtrix57 reproducer pinned.
        assert "web_search" in result.lower()
        # Pin the specific guidance phrase so a future copy-edit can't
        # silently weaken the rule.
        assert "discovery" in result.lower() and "retrieval" in result.lower(), (
            "Default prompt must explain the discovery (web_search) vs "
            "retrieval (http_get) distinction"
        )

    def test_default_prompt_directs_http_get_for_snippet_only_results(self):
        # Bug M #1738 regression — when web_search returns sources but
        # the fetcher couldn't extract page content (snippet-only
        # status), the agent must try ``http_get`` on the surfaced
        # URL(s) before refusing. cogtrix63 turn 11081 reproducer:
        # agent searched for AAPL stock price, web_search returned
        # marketwatch.com / nasdaq.com / investing.com URLs all
        # marked ``snippet-only`` (fetcher blocked by 403/timeout),
        # agent flat-refused without trying ``http_get`` directly.
        # ``http_get`` uses a different fetch path (different timeout,
        # different headers) and often succeeds where the search
        # fan-out fetcher did not.
        result = build_system_prompt()
        lower = result.lower()

        # The rule must name the "snippet-only status" case so the
        # model recognises the signal vs treating it as a verdict.
        assert "snippet-only" in lower, (
            "Snippet-only fallback rule missing from default system "
            "prompt — agents will refuse when web_search returns "
            "snippet-only sources instead of trying http_get on the "
            "surfaced URLs (Bug M #1738)"
        )

        # Must explicitly direct http_get as the fallback.
        assert "http_get" in result and (
            "before refusing" in lower or "before you refuse" in lower
        ), (
            "Rule must explicitly say to try http_get BEFORE refusing "
            "when snippet-only results appear"
        )

        # Must frame snippet-only as a SIGNAL not a VERDICT — without
        # that framing the model takes the easy "refuse" path.
        assert (
            "signal, not a verdict" in lower
            or "signal not a verdict" in lower
            or ("signal" in lower and "verdict" in lower)
        ), (
            "Rule must frame snippet-only as a signal (not a final "
            "verdict) so the model attempts the http_get fallback "
            "instead of refusing"
        )

    def test_default_prompt_directs_http_get_for_named_services(self):
        # Bug J #1719 regression — when the user names a specific service
        # ("use the Wayback Machine", "check the GitHub releases"), the
        # agent must reach for ``http_get`` against the canonical URL of
        # that service, not ``web_search`` *about* the service.
        # cogtrix57 turn-8 reproducer: user said "use the wayback machine
        # to see how the website looked in mid-2022"; agent did 4
        # web_searches with queries like "scnsoft.com wayback machine
        # archive 2022", concluded "I could not retrieve archived
        # snapshots" — never actually queried web.archive.org.
        result = build_system_prompt()
        lower = result.lower()

        # The rule must name the "user names a specific service" case
        # distinctly from the URL-handling case (Bug I), so a future
        # refactor can't conflate them and silently drop one.
        assert "user names a specific service" in lower or (
            "names a specific service" in lower and "user" in lower
        ), "Named-service rule missing from default system prompt"

        # The Wayback Machine is the canonical reproducer — must be
        # named so the model has a concrete anchor for the pattern.
        assert "wayback machine" in lower

        # The contrast that anchors the rule:  http_get against the
        # service vs. web_search *about* the service. Both terms must
        # appear so a future copy-edit can't silently weaken the rule.
        assert "http_get" in result
        assert "web_search" in lower

        # GitHub releases is the second concrete example the issue
        # called out — its inclusion proves the pattern generalises
        # beyond Wayback Machine. We only require the named example,
        # not a literal URL template (small models echoed literal
        # template URLs verbatim into refusal responses, which tripped
        # other scenarios' `response_not_contains: github.com/` checks
        # — see #1727 Gate 2 shard D × gpt-oss-20b-fireworks failure).
        assert "github" in lower and "release" in lower, (
            "Default prompt must include the GitHub releases example "
            "so the named-service pattern generalises beyond Wayback "
            "Machine"
        )

        # Must explicitly tell the agent NOT to downgrade to web_search
        # ABOUT the service when the API call fails — that's the
        # exact false-negative the cogtrix57 reproducer surfaced.
        assert (
            "do not downgrade" in lower
            or "do not" in lower
            and ("web_search about" in lower or "web_search *about*" in lower)
        ), (
            "Default prompt must tell the agent not to fall back to "
            "web_search about the service on http_get failure — "
            "that's how the cogtrix57 false-negative happened"
        )

        # Hard guard: literal URL templates must NOT appear in the
        # prompt. gpt-oss-20b on Gate 2 shard D's persist_before_refusing
        # scenario echoed `api.github.com/repos/<owner>/<repo>/releases`
        # verbatim into a "could not find Captain Claw" refusal response,
        # tripping the `response_not_contains: github.com/` check. The
        # rule's value is the http_get-vs-web_search distinction, NOT
        # the URL spelling — the model already knows the URL surfaces.
        assert "api.github.com/repos/" not in result, (
            "Literal api.github.com/repos/ URL template must not appear "
            "in the prompt — small models leak it verbatim into refusal "
            "responses and trip downstream `response_not_contains: "
            "github.com/` checks (#1727 Gate 2 shard D × gpt-oss-20b)"
        )
        # Also forbid literal angle-bracket placeholders that small
        # models can copy literally into tool-call URLs.
        assert "<owner>" not in result and "<repo>" not in result
        assert "<YYYYMMDD>" not in result
        assert "<url>" not in result

    def test_default_prompt_directs_semantic_scholar_for_citation_queries(self):
        # Bug #1889 regression — when the user asks about citation counts /
        # "most-cited" / "top-cited" academic papers, the agent must reach
        # for the Semantic Scholar API (which exposes citationCount per
        # paper), not arXiv.org (which doesn't publish citation counts)
        # and not raw web_search for "most cited X papers" (which returns
        # subjective blog rankings, not citation-sorted results).
        #
        # README-flagship reproducer: "Find the five most-cited deep-
        # learning papers from arXiv in 2025…" — without this rule, the
        # agent has three failure modes: (a) fabricate arXiv IDs from
        # web_search snippets, (b) substitute "recent" for "most-cited"
        # without admitting it, (c) refuse honestly. None of those make
        # the README's flagship promise true.
        result = build_system_prompt()
        lower = result.lower()

        # The rule must name the citation-query trigger phrases so the
        # model recognises the scenario. Both "citation count" and
        # "most-cited" are common surface forms.
        assert "citation count" in lower, (
            "Citation-query rule missing from default system prompt — "
            "README's flagship 'most-cited arXiv papers' prompt cannot be "
            "answered honestly without Semantic Scholar (#1889)"
        )
        assert "most-cited" in lower or "most cited" in lower

        # Semantic Scholar must be named as the canonical service — the
        # whole point of the rule is to give the model a concrete service
        # to reach for, mirroring the Wayback Machine anchor in the named-
        # services rule above.
        assert "semantic scholar" in lower, (
            "Semantic Scholar must be named as the citation-data source — "
            "without a named service the model defaults back to "
            "web_search for 'most cited X' and gets blog rankings"
        )

        # The contrast that anchors the rule: arXiv does NOT publish
        # citation counts. Without this, the README's depiction
        # (http_get against arxiv.org/abs/...) misleads the model into
        # thinking arxiv is the right surface for citation data.
        assert "arxiv" in lower and (
            "does not publish citation" in lower
            or "doesn't publish citation" in lower
            or "not publish citation counts" in lower
        ), (
            "Rule must explicitly state arXiv doesn't publish citation "
            "counts — otherwise the model treats arxiv.org as the "
            "citation-data surface and fabricates rankings"
        )

        # http_get must be named as the access method. web_search must
        # also appear in the same clause so a future copy-edit can't
        # silently weaken the contrast.
        assert "http_get" in result
        assert "web_search" in lower

        # The response field the model needs to sort by must be named —
        # without that the model can't even attempt the ranking step.
        assert "citationcount" in lower or "citation_count" in lower, (
            "Rule must name the citationCount field so the model knows "
            "what to sort the Semantic Scholar response by"
        )

        # Hard guard, same as the named-services rule: no literal API
        # URL template in the prompt. Small models echo URL templates
        # verbatim into refusal responses, tripping downstream
        # response_not_contains checks (#1727 Gate 2 shard D pattern).
        assert "api.semanticscholar.org/graph/v1" not in result, (
            "Literal api.semanticscholar.org/graph/v1 URL template "
            "must not appear in the prompt — small models leak it "
            "verbatim into refusal responses (#1727 pattern)"
        )

        # Forbid the "ask first when ambiguous" escape hatch from
        # silently disappearing — without it, the rule punishes
        # legitimate "top recent papers" queries by routing them to
        # the citation API when "most recent" would have been correct.
        assert "most cited" in lower and "most recent" in lower, (
            "Rule must offer the 'ask once: most cited vs most recent' "
            "branch so non-citation 'top papers' queries route correctly"
        )


# ---------------------------------------------------------------------------
# _estimate_msg_tokens
# ---------------------------------------------------------------------------


class TestEstimateMsgTokens:
    def test_message_with_content_attr(self):
        msg = MagicMock()
        msg.content = "hello world"  # 11 chars → ~2 tokens
        result = _estimate_msg_tokens(msg)
        assert result >= 1

    def test_dict_with_content(self):
        result = _estimate_msg_tokens({"content": "a" * 400})  # 400 chars → 100 tokens
        assert result == 100

    def test_empty_message_returns_overhead(self):
        msg = MagicMock()
        msg.content = ""
        result = _estimate_msg_tokens(msg)
        assert result == 10

    def test_list_content_summed(self):
        msg = MagicMock()
        msg.content = ["hello", "world"]  # 10 chars → 2 tokens
        result = _estimate_msg_tokens(msg)
        assert result >= 1

    def test_minimum_is_one_for_non_empty(self):
        msg = MagicMock()
        msg.content = "a"  # 1 char → max(0, 1) = 1
        result = _estimate_msg_tokens(msg)
        assert result >= 1


# ---------------------------------------------------------------------------
# _truncate_content
# ---------------------------------------------------------------------------


class TestTruncateContent:
    def test_short_content_unchanged(self):
        text = "hello"
        assert _truncate_content(text, max_tokens=100) == text

    def test_long_content_truncated(self):
        text = "a" * 10000
        result = _truncate_content(text, max_tokens=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_truncated_keeps_both_ends(self):
        # Build text with distinguishable start and end
        text = "START" + "x" * 5000 + "END"
        result = _truncate_content(text, max_tokens=50)
        assert "START" in result
        assert "END" in result

    def test_exact_limit_not_truncated(self):
        max_tokens = 100
        text = "a" * (max_tokens * 4)  # exactly at limit
        assert _truncate_content(text, max_tokens) == text

    def test_non_positive_max_tokens_returns_unchanged(self):
        text = "hello world"
        assert _truncate_content(text, max_tokens=0) == text
        assert _truncate_content(text, max_tokens=-1) == text
        assert _truncate_content(text, max_tokens=-100) == text


# ---------------------------------------------------------------------------
# prepare_messages_with_context
# ---------------------------------------------------------------------------


class TestPrepareMessagesWithContext:
    def test_basic_user_input(self):
        result = prepare_messages_with_context([], "hello")
        # Last message is the user input
        last = result[-1]
        content = last.content if hasattr(last, "content") else last.get("content", "")
        assert content == "hello"

    def test_history_included(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        history = [HumanMessage(content="past message")]
        result = prepare_messages_with_context(history, "new input")
        assert len(result) >= 2

    def test_context_prefix_injected_as_message(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        result = prepare_messages_with_context([], "hello", context_prefix="Some context")
        # First message should contain the context prefix (HumanMessage for
        # strict-provider compatibility — Qwen3/vLLM reject SystemMessage
        # outside position 0).
        first = result[0]
        assert isinstance(first, HumanMessage)
        assert "Some context" in first.content

    def test_no_context_prefix_no_system_message(self):
        try:
            from langchain_core.messages import SystemMessage
        except ImportError:
            pytest.skip("langchain not installed")

        result = prepare_messages_with_context([], "hello")
        for msg in result:
            assert not isinstance(msg, SystemMessage)

    def test_token_budget_trims_history(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        # Create a very large history that exceeds a small token budget
        big_text = "x" * 8000
        history = [HumanMessage(content=big_text) for _ in range(5)]
        result = prepare_messages_with_context(history, "new input", max_context_tokens=512)
        # Result should be smaller than original history + input
        assert len(result) <= len(history) + 1

    def test_fallback_without_langchain(self):
        from unittest.mock import patch

        with patch("src.agent.core.HumanMessage", None):
            result = prepare_messages_with_context(
                [{"type": "human", "content": "history"}], "new input"
            )
        # Fallback returns list with history + new input dict
        assert len(result) >= 1
        last = result[-1]
        assert isinstance(last, dict)
        assert last["content"] == "new input"


# ---------------------------------------------------------------------------
# _trim_to_token_budget — role alternation guard
# ---------------------------------------------------------------------------


class TestTrimToTokenBudget:
    """Verify _trim_to_token_budget never produces a leading AIMessage."""

    def _human(self, text: str):
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=text)

    def _ai(self, text: str):
        from langchain_core.messages import AIMessage

        return AIMessage(content=text)

    def _tool(self, text: str, tool_call_id: str = "tc1"):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=text, tool_call_id=tool_call_id)

    def test_leading_ai_message_is_removed(self):
        """If trimming exposes an AIMessage at the head, drop it."""
        msgs = [
            self._human("prefix"),
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        # Budget small enough that prefix + h1 are dropped, exposing a1
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        # First message after any internal dropping must not be AIMessage
        assert not isinstance(result[0], type(self._ai("")))

    def test_leading_tool_message_after_ai_removal_is_also_dropped(self):
        """Dropping an AIMessage may orphan a following ToolMessage — remove both."""
        msgs = [
            self._human("prefix"),
            self._human("h1"),
            self._ai("a1"),
            self._tool("t1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        assert not isinstance(result[0], type(self._ai("")))
        assert not isinstance(result[0], type(self._tool("")))

    def test_valid_history_unchanged_when_under_budget(self):
        """When everything fits, the message order is preserved."""
        msgs = [
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=8192)
        assert len(result) == len(msgs)
        assert result[0].content == "h1"
        assert result[-1].content == "tail"

    def test_system_message_preserved_as_fixed_head(self):
        """SystemMessage stays at position 0 even when history is trimmed."""
        from langchain_core.messages import SystemMessage

        msgs = [
            SystemMessage(content="sys"),
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        assert isinstance(result[0], SystemMessage)
        # Ensure no AIMessage immediately follows SystemMessage if h1 was dropped
        if len(result) > 1:
            assert not isinstance(result[1], type(self._ai("")))
