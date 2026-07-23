"""Regression tests for action-intent / phantom give-up answer synthesis.

When the recovery loops exhaust their retries, the user must NOT see the
model's last stuck-thinking output (which is often meta-analysis with
embedded XML tool-call markup).  Instead, the give-up branches must
synthesize a clean answer from accumulated checkpoints / tool results.

These tests guard against the regression observed in the May 2026 user
run where qwen3-coder produced a meta-analysis as its final response and
the recovery path passed it straight through to the user UI.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.orchestration.nodes.recovery import (
    build_handle_action_intent_node,
    build_handle_phantom_node,
)
from cogtrix_core.orchestration.phases import (
    _extract_checkpoint_findings,
    extract_deepseek_native_tool_calls,
    normalize_native_tool_calls,
    strip_foreign_tool_call_xml,
    synthesize_answer_from_state,
)

# ── strip_foreign_tool_call_xml ─────────────────────────────────────────


class TestStripForeignToolCallXml:
    def test_strips_qwen3_xml_tool_call(self):
        content = (
            "I'll search for that.\n\n"
            "<tool_call>\n<function=search_web>\n</function>\n</tool_call>\n"
        )
        cleaned = strip_foreign_tool_call_xml(content)
        assert "<tool_call>" not in cleaned
        assert "<function=" not in cleaned
        assert cleaned.startswith("I'll search for that.")

    def test_strips_inline_function_tag(self):
        content = "Let me check. <function=get_current_datetime></function> Done."
        cleaned = strip_foreign_tool_call_xml(content)
        assert "<function=" not in cleaned
        assert "Let me check." in cleaned

    def test_preserves_normal_text(self):
        content = "Mattermost supports OAuth 2.0 and bot accounts."
        assert strip_foreign_tool_call_xml(content) == content

    def test_collapses_excessive_blank_lines_after_strip(self):
        content = (
            "First line.\n\n"
            "<tool_call><function=x></function></tool_call>\n\n\n\n"
            "Second line."
        )
        cleaned = strip_foreign_tool_call_xml(content)
        assert "\n\n\n" not in cleaned
        assert "First line." in cleaned
        assert "Second line." in cleaned

    def test_non_string_input_is_returned_unchanged(self):
        assert strip_foreign_tool_call_xml(None) is None
        assert strip_foreign_tool_call_xml([1, 2]) == [1, 2]


# ── _extract_checkpoint_findings ────────────────────────────────────────


class TestExtractCheckpointFindings:
    def test_pulls_finding_text_from_tool_calls(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint",
                    "args": {"finding": "Mattermost has OAuth 2.0 support."},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )
        findings = _extract_checkpoint_findings([ai])
        assert findings == ["Mattermost has OAuth 2.0 support."]

    def test_skips_non_checkpoint_tools(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "x"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )
        assert _extract_checkpoint_findings([ai]) == []

    def test_returns_findings_in_order(self):
        ai1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint",
                    "args": {"finding": "First."},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        )
        ai2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint",
                    "args": {"finding": "Second."},
                    "id": "2",
                    "type": "tool_call",
                }
            ],
        )
        assert _extract_checkpoint_findings([ai1, ai2]) == ["First.", "Second."]

    def test_skips_empty_findings(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "checkpoint", "args": {"finding": ""}, "id": "1", "type": "tool_call"},
                {"name": "checkpoint", "args": {"finding": "   "}, "id": "2", "type": "tool_call"},
            ],
        )
        assert _extract_checkpoint_findings([ai]) == []


# ── synthesize_answer_from_state ────────────────────────────────────────


class TestSynthesizeAnswerFromState:
    def test_prefers_checkpoint_findings_over_meta_analysis(self):
        """Regression: the May 2026 user run produced a meta-analysis as
        the last AIMessage; the actual answer was only in checkpoint
        findings.  Synthesis MUST return the finding, not the meta."""
        msgs = [
            HumanMessage(content="Check Mattermost OAuth support"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "checkpoint",
                        "args": {
                            "finding": (
                                "Mattermost has OAuth 2.0, bot accounts, "
                                "personal access tokens, and webhooks."
                            )
                        },
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Checkpoint #1 recorded.", tool_call_id="1", name="checkpoint"),
            AIMessage(
                content=(
                    "Let me analyze my work systematically:\n"
                    "**What has WORKED so far:** Nothing.\n"
                    "**What has FAILED:** Multiple attempts.\n"
                    "<tool_call><function=get_current_datetime></function></tool_call>"
                ),
            ),
        ]
        result = synthesize_answer_from_state(msgs)
        assert result is not None
        assert "OAuth 2.0" in result
        assert "bot accounts" in result
        # Meta-analysis prose must NOT bleed into the answer
        assert "What has WORKED" not in result
        assert "Let me analyze" not in result
        # Foreign XML must NOT be present
        assert "<tool_call>" not in result
        assert "<function=" not in result

    def test_returns_latest_finding_when_multiple(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "checkpoint",
                        "args": {"finding": "Initial finding."},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "checkpoint",
                        "args": {"finding": "Refined finding with more detail."},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        result = synthesize_answer_from_state(msgs)
        # The latest checkpoint usually subsumes earlier ones
        assert result == "Refined finding with more detail."

    def test_falls_back_to_clean_ai_content_when_no_checkpoints(self):
        msgs = [
            AIMessage(content="Mattermost supports OAuth 2.0 for third-party app integration."),
        ]
        result = synthesize_answer_from_state(msgs)
        assert result is not None
        assert "OAuth" in result

    def test_strips_xml_from_fallback_content(self):
        msgs = [
            AIMessage(
                content=(
                    "Here is the comprehensive answer about Mattermost: "
                    "It supports OAuth 2.0, bot accounts, personal access "
                    "tokens, and webhooks for AI agent communication.\n"
                    "<tool_call><function=x></function></tool_call>"
                )
            )
        ]
        result = synthesize_answer_from_state(msgs)
        assert result is not None
        assert "<tool_call>" not in result
        assert "OAuth" in result

    def test_returns_none_when_no_usable_content(self):
        # Only short content with stripped XML producing tiny remnants
        msgs = [AIMessage(content="<tool_call><function=x></function></tool_call>")]
        # Stripped content is empty → falls through; no other sources
        assert synthesize_answer_from_state(msgs) is None


# ── handle_action_intent give-up path ───────────────────────────────────


class TestActionIntentGiveup:
    def test_giveup_returns_synthesized_answer_not_meta_analysis(self):
        """Regression for the May 2026 user-experience disaster.

        Simulates the exact failure mode: the model recorded checkpoints
        with the answer, then drifted into meta-analysis text containing
        XML tool calls.  The give-up path MUST replace the meta-analysis
        with the synthesized answer.
        """
        node = build_handle_action_intent_node(action_intent_count=[3], max_retries=3)
        meta_msg = AIMessage(
            content=(
                "**What has WORKED so far:** Nothing.\n"
                "**What has FAILED:** All searches.\n"
                "<tool_call><function=get_current_datetime></function></tool_call>"
            )
        )
        meta_msg.id = "meta-msg-1"

        ckpt_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint",
                    "args": {"finding": "Mattermost has OAuth 2.0 and bot accounts."},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )

        state = {"messages": [ckpt_msg, meta_msg]}
        result = node(state)

        # We expect: RemoveMessage(meta_msg.id) + synthesized AIMessage
        assert "messages" in result
        out = result["messages"]
        assert len(out) == 2
        # First entry removes the meta-analysis
        assert getattr(out[0], "id", None) == "meta-msg-1"
        # Second entry is the clean synthesized answer
        synthesized = out[1]
        assert isinstance(synthesized, AIMessage)
        assert "OAuth 2.0" in synthesized.content
        assert "<tool_call>" not in synthesized.content
        assert "What has WORKED" not in synthesized.content

    def test_below_threshold_uses_context_aware_nudge_for_unloaded_tool(self):
        """When the previous turn's failure was 'tool not loaded', the
        nudge must tell the model to call request_tools(add=[...])."""
        node = build_handle_action_intent_node(action_intent_count=[0], max_retries=3)
        state = {
            "messages": [
                AIMessage(content="Let me search."),
                ToolMessage(
                    content=(
                        "Tool 'search_web' is in the catalog but not loaded. "
                        'Call request_tools(add=["search_web"]) ...'
                    ),
                    tool_call_id="tc1",
                    name="search_web",
                ),
                AIMessage(content="I'll try again with the right tool."),
            ]
        }
        result = node(state)
        msgs = result["messages"]
        assert len(msgs) == 1
        nudge = msgs[0]
        assert isinstance(nudge, HumanMessage)
        assert "search_web" in nudge.content
        assert 'request_tools(add=["search_web"])' in nudge.content

    def test_below_threshold_uses_generic_nudge_when_no_unloaded_tool(self):
        node = build_handle_action_intent_node(action_intent_count=[0], max_retries=3)
        state = {
            "messages": [
                AIMessage(content="I will write a file now."),
            ]
        }
        result = node(state)
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert "did not call any tools" in msgs[0].content


# ── handle_phantom give-up path ─────────────────────────────────────────


class TestPhantomGiveup:
    def test_giveup_returns_synthesized_answer(self):
        node = build_handle_phantom_node(phantom_count=[3], max_retries=3)
        ckpt_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint",
                    "args": {"finding": "Found relevant docs."},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )
        last = AIMessage(content="")
        last.id = "phantom-1"
        state = {"messages": [ckpt_msg, last]}
        result = node(state)
        out = result["messages"]
        assert len(out) == 2
        assert getattr(out[0], "id", None) == "phantom-1"


# ── DeepSeek native tool-call parsing ───────────────────────────────────


# DeepSeek's chat-template special tokens.  Defined as constants so the
# tests are readable without hunting for unicode glyphs in regex source.
_DS_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALLS_END = "<｜tool▁calls▁end｜>"
_DS_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DS_CALL_END = "<｜tool▁call▁end｜>"
_DS_SEP = "<｜tool▁sep｜>"


def _ds_tool_call(name: str, args_json: str) -> str:
    return (
        _DS_CALL_BEGIN
        + "function"
        + _DS_SEP
        + name
        + "\n```json\n"
        + args_json
        + "\n```\n"
        + _DS_CALL_END
        + "\n"
    )


def _wrap_calls(*calls: str) -> str:
    return _DS_CALLS_BEGIN + "\n" + "".join(calls) + _DS_CALLS_END


class TestExtractDeepseekNativeToolCalls:
    def test_extracts_single_tool_call(self):
        call = _ds_tool_call("classify_invoice", '{"query": "INV-001"}')
        content = "Let me classify this invoice.\n\n" + _wrap_calls(call)
        calls, cleaned = extract_deepseek_native_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "classify_invoice"
        assert calls[0]["args"] == {"query": "INV-001"}
        assert calls[0]["type"] == "tool_call"
        assert "classify_invoice" not in cleaned
        assert "tool" + chr(0x2581) + "call" not in cleaned
        assert "Let me classify this invoice." in cleaned

    def test_extracts_multiple_tool_calls_in_order(self):
        content = _wrap_calls(
            _ds_tool_call("classify_invoice", '{"query": "X"}'),
            _ds_tool_call("route_for_approval", '{"query": "Y"}'),
            _ds_tool_call("notify_approver", '{"query": "Z"}'),
        )
        calls, cleaned = extract_deepseek_native_tool_calls(content)
        assert [c["name"] for c in calls] == [
            "classify_invoice",
            "route_for_approval",
            "notify_approver",
        ]
        assert calls[0]["args"] == {"query": "X"}
        assert calls[1]["args"] == {"query": "Y"}
        assert calls[2]["args"] == {"query": "Z"}
        assert cleaned == ""

    def test_no_tokens_returns_empty_list_and_unchanged_content(self):
        content = "Hello world, no tool calls here."
        calls, cleaned = extract_deepseek_native_tool_calls(content)
        assert calls == []
        assert cleaned == content

    def test_malformed_json_args_skips_call(self):
        content = _wrap_calls(
            _DS_CALL_BEGIN
            + "function"
            + _DS_SEP
            + "broken_tool\n```json\n{not valid json\n```\n"
            + _DS_CALL_END
            + "\n"
        )
        calls, _cleaned = extract_deepseek_native_tool_calls(content)
        assert calls == []

    def test_non_string_input_returns_empty_unchanged(self):
        calls, cleaned = extract_deepseek_native_tool_calls(None)
        assert calls == []
        assert cleaned is None

    def test_skips_non_function_kind(self):
        content = _wrap_calls(
            _DS_CALL_BEGIN
            + "custom_kind"
            + _DS_SEP
            + 'some_tool\n```json\n{"a": 1}\n```\n'
            + _DS_CALL_END
            + "\n"
        )
        calls, _cleaned = extract_deepseek_native_tool_calls(content)
        # Only "function" kind is supported (matches DeepSeek's chat template).
        assert calls == []

    def test_call_id_is_unique_per_call(self):
        content = _wrap_calls(
            _ds_tool_call("t", '{"a": 1}'),
            _ds_tool_call("t", '{"a": 2}'),
        )
        calls, _ = extract_deepseek_native_tool_calls(content)
        assert calls[0]["id"] != calls[1]["id"]


class TestNormalizeNativeToolCalls:
    def test_pulls_deepseek_calls_into_structured_tool_calls(self):
        content = "I'll handle this.\n\n" + _wrap_calls(
            _ds_tool_call("classify_invoice", '{"query": "INV"}')
        )
        msg = AIMessage(content=content)
        result = normalize_native_tool_calls(msg)
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "classify_invoice"
        assert result.tool_calls[0]["args"] == {"query": "INV"}
        # Tokens stripped from content
        assert "classify_invoice" not in result.content
        assert "I'll handle this." in result.content

    def test_appends_to_existing_tool_calls_does_not_replace(self):
        content = _wrap_calls(_ds_tool_call("ds_tool", "{}"))
        msg = AIMessage(
            content=content,
            tool_calls=[{"name": "existing", "args": {}, "id": "ex1", "type": "tool_call"}],
        )
        result = normalize_native_tool_calls(msg)
        names = [c["name"] for c in result.tool_calls]
        assert "existing" in names
        assert "ds_tool" in names

    def test_strips_qwen3_xml_too(self):
        msg = AIMessage(
            content=("Answer here.\n" "<tool_call><function=get_x></function></tool_call>")
        )
        result = normalize_native_tool_calls(msg)
        assert "<tool_call>" not in result.content
        assert "Answer here." in result.content

    def test_no_op_when_no_native_tokens(self):
        msg = AIMessage(content="Just a regular reply.")
        result = normalize_native_tool_calls(msg)
        assert result.content == "Just a regular reply."

    def test_non_aimessage_returned_unchanged(self):
        msg = HumanMessage(content="hi")
        result = normalize_native_tool_calls(msg)
        assert result is msg
