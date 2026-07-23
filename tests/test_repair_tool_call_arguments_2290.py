"""#2290 — tool_call arguments with trailing data ("Extra data" JSON) → provider 400.

Some open-weight models (seen via OpenRouter → AtlasCloud/AkashML/Parasail) emit a
tool call whose ``arguments`` string is valid JSON followed by extra bytes, e.g.
``'{"add": ["web_search"]} <junk>'``. LangChain's ``parse_tool_call`` does
``json.loads(arguments)`` → raises ``JSONDecodeError: Extra data`` → the call is
demoted to ``invalid_tool_calls`` carrying the **raw string verbatim**. On the next
request ``_lc_invalid_tool_call_to_openai_tool_call`` echoes that raw string, the
strict provider ``json.loads`` it, and rejects with HTTP 400 — killing the turn.

``repair_tool_call_arguments`` re-parses each invalid call's args with
``raw_decode`` (reads the first complete JSON value, ignores the trailing data) and
promotes recoverable ones back to valid ``tool_calls`` so they both serialise as
clean JSON and execute as the model intended.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from cogtrix_core.orchestration.phases import repair_tool_call_arguments


def _invalid(name: str, args: str, id: str = "call_1") -> dict:
    # InvalidToolCall is a TypedDict; a plain dict is the runtime shape LangChain
    # stores (and what _convert_message_to_dict / our repair both read).
    return {
        "name": name,
        "args": args,
        "id": id,
        "error": "Extra data",
        "type": "invalid_tool_call",
    }


def _outbound_arguments(msg: AIMessage) -> list[str]:
    """The literal ``arguments`` strings LangChain would send to the provider."""
    d = _convert_message_to_dict(msg)
    return [tc["function"]["arguments"] for tc in d.get("tool_calls", [])]


class TestRepairToolCallArguments:
    def test_trailing_data_recovered_to_valid_tool_call(self) -> None:
        """The exact prod scenario: args = valid JSON + trailing junk."""
        msg = AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[_invalid("request_tools", '{"add": ["web_search"]} <extra>')],
        )
        fixed = repair_tool_call_arguments(msg)

        assert len(fixed.tool_calls) == 1
        assert len(fixed.invalid_tool_calls) == 0
        tc = fixed.tool_calls[0]
        assert tc["name"] == "request_tools"
        assert tc["args"] == {"add": ["web_search"]}  # parsed, trailing data dropped
        assert tc["id"] == "call_1"

    def test_repaired_message_serialises_to_clean_json(self) -> None:
        """After repair, the outbound arguments parse cleanly (no provider 400)."""
        msg = AIMessage(
            content="",
            invalid_tool_calls=[_invalid("web_search", '{"query": "cogtrix"}trailing')],
        )
        fixed = repair_tool_call_arguments(msg)
        for args in _outbound_arguments(fixed):
            json.loads(args)  # must not raise "Extra data"

    def test_unrepaired_message_would_400(self) -> None:
        """Guard the premise: the *unrepaired* message serialises to invalid JSON."""
        msg = AIMessage(
            content="",
            invalid_tool_calls=[_invalid("web_search", '{"query": "x"} junk')],
        )
        raised = False
        for args in _outbound_arguments(msg):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                raised = True
        assert raised, "expected the unrepaired raw args to fail json.loads"

    def test_preserves_existing_valid_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "a", "args": {"x": 1}, "id": "a1", "type": "tool_call"}],
            invalid_tool_calls=[_invalid("b", '{"y": 2} extra', id="b1")],
        )
        fixed = repair_tool_call_arguments(msg)
        names = {tc["name"] for tc in fixed.tool_calls}
        assert names == {"a", "b"}
        assert len(fixed.invalid_tool_calls) == 0

    def test_genuinely_unparseable_stays_invalid(self) -> None:
        """An invalid call whose args isn't JSON at all is left untouched (no-op)."""
        msg = AIMessage(content="", invalid_tool_calls=[_invalid("y", "not json at all")])
        fixed = repair_tool_call_arguments(msg)
        assert fixed is msg  # nothing recovered → identity preserved
        assert len(fixed.invalid_tool_calls) == 1
        assert len(fixed.tool_calls) == 0

    def test_clean_message_is_noop_identity(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "x", "args": {"a": 1}, "id": "i", "type": "tool_call"}],
        )
        assert repair_tool_call_arguments(msg) is msg

    def test_no_tool_calls_is_noop_identity(self) -> None:
        msg = AIMessage(content="plain answer")
        assert repair_tool_call_arguments(msg) is msg

    def test_non_aimessage_passthrough(self) -> None:
        hm = HumanMessage(content="hello")
        assert repair_tool_call_arguments(hm) is hm

    def test_drops_stale_additional_kwargs_tool_calls(self) -> None:
        """The stale raw copy in additional_kwargs must not survive to override the
        repaired structured tool_calls during serialisation."""
        msg = AIMessage(
            content="",
            invalid_tool_calls=[_invalid("request_tools", '{"add": ["x"]} junk')],
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "request_tools", "arguments": '{"add": ["x"]} junk'},
                    }
                ]
            },
        )
        fixed = repair_tool_call_arguments(msg)
        assert "tool_calls" not in fixed.additional_kwargs
        for args in _outbound_arguments(fixed):
            json.loads(args)  # clean
