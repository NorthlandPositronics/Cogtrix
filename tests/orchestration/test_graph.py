"""Tests for src/orchestration/graph.py helper functions."""

from types import SimpleNamespace

from src.orchestration.graph import _detect_tool_request, _is_action_intent


class TestDetectToolRequest:
    """Unit tests for _detect_tool_request."""

    def _ai_msg(self, tool_calls):
        """Create a fake AIMessage with given tool_calls."""
        return SimpleNamespace(tool_calls=tool_calls)

    def _tool_msg(self, content="ok", name="request_tools", tool_call_id="tc1"):
        """Create a fake ToolMessage (no tool_calls attribute)."""
        return SimpleNamespace(content=content, name=name, tool_call_id=tool_call_id)

    def test_returns_none_for_empty_messages(self):
        assert _detect_tool_request([], start_idx=0) is None

    def test_extracts_add_from_ai_message(self):
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": ["search_web", "http_get"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["search_web", "http_get"]
        assert result.remove == []

    def test_extracts_remove_from_ai_message(self):
        ai = self._ai_msg([{"name": "request_tools", "args": {"remove": ["shell"]}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.remove == ["shell"]

    def test_legacy_names_fallback(self):
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"names": ["calculator"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]

    def test_ignores_tool_messages(self):
        """Regression test for BUG-076: ToolMessages have no tool_calls attribute.

        _detect_tool_request must be called with the AIMessage that contains
        tool_calls, NOT with ToolMessage results.  If only ToolMessages are
        passed, the function must return None.
        """
        tool_msg = self._tool_msg(content="Tools loaded: search_web. They are now active.")
        result = _detect_tool_request([tool_msg], start_idx=0)
        assert result is None

    def test_ai_message_with_non_request_tools_ignored(self):
        ai = self._ai_msg([{"name": "search_web", "args": {"query": "test"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is None

    def test_mixed_add_and_remove(self):
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["http_get"], "remove": ["calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["http_get"]
        assert result.remove == ["calculator"]
        assert result.has_changes is True

    def test_start_idx_skips_earlier_messages(self):
        ai1 = self._ai_msg([{"name": "request_tools", "args": {"add": ["shell"]}, "id": "tc1"}])
        ai2 = self._ai_msg(
            [{"name": "request_tools", "args": {"add": ["calculator"]}, "id": "tc2"}]
        )
        result = _detect_tool_request([ai1, ai2], start_idx=1)
        assert result is not None
        assert result.add == ["calculator"]

    def test_multiple_request_tools_calls_in_single_message(self):
        """GAP-5: Multiple parallel request_tools calls are aggregated."""
        ai = self._ai_msg(
            [
                {"name": "request_tools", "args": {"add": ["tool_a"]}, "id": "tc1"},
                {"name": "request_tools", "args": {"add": ["tool_b"]}, "id": "tc2"},
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["tool_a", "tool_b"]
        assert result.has_changes is True

    def test_mixed_request_tools_and_regular_calls(self):
        """Only request_tools calls are extracted; regular tool calls are ignored."""
        ai = self._ai_msg(
            [
                {"name": "search_web", "args": {"query": "test"}, "id": "tc1"},
                {"name": "request_tools", "args": {"add": ["calculator"]}, "id": "tc2"},
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]

    def test_empty_add_and_remove_returns_none(self):
        """request_tools with empty lists returns None (no changes)."""
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": [], "remove": []}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is None

    # BUG-204 — string arg normalization

    def test_add_as_bare_string(self):
        """BUG-204: LLM sends {"add": "web_search"} — must be normalised to ["web_search"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"add": "web_search"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["web_search"]
        assert result.remove == []

    def test_remove_as_bare_string(self):
        """BUG-204: LLM sends {"remove": "shell"} — must be normalised to ["shell"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"remove": "shell"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.remove == ["shell"]
        assert result.add == []

    def test_legacy_names_as_bare_string(self):
        """BUG-204: LLM sends {"names": "calculator"} — must be normalised to ["calculator"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"names": "calculator"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]


class TestParallelToolTimeout:
    """BUG-202: parallel tool futures must time out instead of hanging indefinitely."""

    def test_future_timeout_produces_error_message(self):
        """Simulate the timeout path: future.result(timeout=0) raises TimeoutError."""
        import concurrent.futures

        from langchain_core.messages import ToolMessage

        call = {"name": "slow_tool", "id": "tc-slow", "args": {}}
        future: concurrent.futures.Future = concurrent.futures.Future()

        try:
            future.result(timeout=0)
            raise AssertionError("Should have raised TimeoutError")
        except (TimeoutError, concurrent.futures.TimeoutError):
            msg = ToolMessage(
                content=f"Error: tool '{call['name']}' timed out after 10 minutes",
                tool_call_id=call["id"],
                name=call["name"],
            )

        assert "timed out" in msg.content
        assert "slow_tool" in msg.content
        assert msg.tool_call_id == "tc-slow"

    def test_source_uses_600s_timeout(self):
        """Verify the timeout constant is 600 seconds (10 minutes) in graph.py."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "future.result(timeout=600)" in source

    def test_source_handles_both_timeout_exception_types(self):
        """Both built-in TimeoutError and concurrent.futures.TimeoutError are caught."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "concurrent.futures.TimeoutError" in source

    def test_timeout_error_message_format(self):
        """The error ToolMessage content matches the expected human-readable format."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "timed out after 10 minutes" in source


class TestDetectToolRequestEdgeCases:
    """Edge-case tests for _detect_tool_request normalization."""

    def _ai_msg(self, tool_calls):
        return SimpleNamespace(tool_calls=tool_calls)

    def test_mixed_string_add_and_list_remove(self):
        """BUG-204 edge: add is a bare string, remove is a list."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": "web_search", "remove": ["shell", "calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["web_search"]
        assert result.remove == ["shell", "calculator"]

    def test_mixed_list_add_and_string_remove(self):
        """BUG-204 edge: add is a list, remove is a bare string."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["http_get", "calculator"], "remove": "shell"},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["http_get", "calculator"]
        assert result.remove == ["shell"]

    def test_empty_string_add_returns_none(self):
        """Edge: empty string add is falsy — triggers names fallback (also empty) → None."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"add": ""}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        # "" is falsy, so the legacy names fallback triggers (also empty) → no changes
        assert result is None

    def test_integer_in_add_list_coerced_to_str(self):
        """Edge: non-string values in list are coerced via str()."""
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": [123, "real_tool"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["123", "real_tool"]

    def test_legacy_names_string_with_add_empty_list(self):
        """Legacy fallback: names is used only when add and remove are both empty."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": [], "remove": [], "names": "calculator"},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        # add=[] and remove=[] are falsy, so names fallback triggers
        assert result is not None
        assert result.add == ["calculator"]

    def test_add_present_suppresses_legacy_names(self):
        """When add is provided, legacy names is ignored even if also present."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["shell"], "names": ["calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["shell"]
        # calculator from names should NOT appear
        assert "calculator" not in result.add


class TestIsActionIntent:
    """Unit tests for _is_action_intent — action-intent-without-tool-call detection."""

    def _ai(self, content: str, tool_calls=None):
        return SimpleNamespace(content=content, tool_calls=tool_calls or [])

    # ── Positive cases (should return True) ──────────────────────────────────

    def test_ill_create(self):
        assert _is_action_intent(self._ai("I'll create an OpenAI-compatible API server."))

    def test_i_will_write(self):
        assert _is_action_intent(self._ai("I will write the configuration file now."))

    def test_let_me_run(self):
        assert _is_action_intent(self._ai("Let me run the tests to verify the changes."))

    def test_lets_build(self):
        assert _is_action_intent(self._ai("Let's build the Docker image first."))

    def test_let_us_execute(self):
        assert _is_action_intent(self._ai("Let us execute the migration script."))

    def test_im_going_to_generate(self):
        assert _is_action_intent(self._ai("I'm going to generate the report."))

    def test_i_am_going_to_fetch(self):
        assert _is_action_intent(self._ai("I am going to fetch the data from the API."))

    def test_im_about_to_deploy(self):
        assert _is_action_intent(self._ai("I'm about to deploy the service."))

    def test_i_need_to_install(self):
        assert _is_action_intent(self._ai("I need to install the dependencies first."))

    def test_i_have_to_update(self):
        assert _is_action_intent(self._ai("I have to update the config file."))

    def test_i_should_search(self):
        assert _is_action_intent(self._ai("I should search for recent documentation."))

    def test_i_must_implement(self):
        assert _is_action_intent(self._ai("I must implement the retry logic."))

    def test_i_will_now_set_up(self):
        assert _is_action_intent(self._ai("I will now set up the server."))

    def test_ill_now_configure(self):
        assert _is_action_intent(self._ai("I'll now configure the environment variables."))

    def test_im_now_loading(self):
        assert _is_action_intent(self._ai("I'm now loading the dataset."))

    def test_ill_proceed_start(self):
        assert _is_action_intent(self._ai("I'll proceed to start the application."))

    def test_ill_go_ahead_save(self):
        assert _is_action_intent(self._ai("I'll go ahead and save the output to a file."))

    def test_now_ill_fetch(self):
        assert _is_action_intent(self._ai("Now I'll fetch the latest changes from GitHub."))

    def test_now_let_me_check(self):
        assert _is_action_intent(self._ai("Now let me check the current directory structure."))

    def test_first_ill_read(self):
        assert _is_action_intent(self._ai("First, I'll read the existing file."))

    def test_next_let_me_build(self):
        assert _is_action_intent(self._ai("Next, let me build the project."))

    def test_then_i_will_commit(self):
        assert _is_action_intent(self._ai("Then I will commit the changes."))

    def test_finally_ill_deploy(self):
        assert _is_action_intent(self._ai("Finally, I'll deploy to production."))

    def test_additionally_ill_export(self):
        assert _is_action_intent(self._ai("Additionally, I'll export the results."))

    def test_going_to_download(self):
        assert _is_action_intent(self._ai("Going to download the model weights."))

    def test_about_to_launch(self):
        assert _is_action_intent(self._ai("About to launch the service container."))

    def test_time_to_refactor(self):
        assert _is_action_intent(self._ai("Time to refactor this module."))

    def test_i_can_now_upload(self):
        assert _is_action_intent(self._ai("I can now upload the package."))

    def test_ill_send_request(self):
        assert _is_action_intent(self._ai("I'll send a request to the endpoint."))

    def test_let_me_parse(self):
        assert _is_action_intent(self._ai("Let me parse the JSON response."))

    def test_ill_scaffold_project(self):
        assert _is_action_intent(self._ai("I'll scaffold the project structure."))

    def test_let_me_clone(self):
        assert _is_action_intent(self._ai("Let me clone the repository."))

    def test_ill_push_changes(self):
        assert _is_action_intent(self._ai("I'll push the changes to the remote branch."))

    def test_let_me_extract(self):
        assert _is_action_intent(self._ai("Let me extract the archive first."))

    def test_ill_spin_up_server(self):
        assert _is_action_intent(self._ai("I'll spin up a local server on port 8080."))

    def test_multiline_with_intent_later(self):
        """Intent phrase near the end of a multi-sentence response."""
        text = (
            "Looking at the requirements, the approach is clear. "
            "Let me implement the solution now."
        )
        assert _is_action_intent(self._ai(text))

    # ── Negative cases (should return False) ─────────────────────────────────

    def test_returns_false_with_tool_calls(self):
        """Tool calls present — not an action-intent-without-action case."""
        msg = SimpleNamespace(
            content="I'll create the file.",
            tool_calls=[{"name": "write_file", "args": {}, "id": "tc1"}],
        )
        assert not _is_action_intent(msg)

    def test_returns_false_empty_content(self):
        assert not _is_action_intent(self._ai(""))

    def test_returns_false_whitespace_only(self):
        assert not _is_action_intent(self._ai("   \n  "))

    def test_returns_false_non_string_content(self):
        msg = SimpleNamespace(content=["list", "content"], tool_calls=[])
        assert not _is_action_intent(msg)

    def test_returns_false_pure_text_explanation(self):
        """'I'll explain' — 'explain' is not a tool-action verb."""
        assert not _is_action_intent(self._ai("I'll explain how this algorithm works."))

    def test_returns_false_summarize(self):
        assert not _is_action_intent(self._ai("I'll summarize what I found above."))

    def test_returns_false_note(self):
        assert not _is_action_intent(self._ai("I should note that this approach has tradeoffs."))

    def test_returns_false_provide_answer(self):
        assert not _is_action_intent(self._ai("I'll provide the answer directly."))

    def test_returns_false_no_intent_phrase(self):
        """Tool verb present but no intent lead phrase."""
        assert not _is_action_intent(
            self._ai("The build process requires installing dependencies.")
        )

    def test_returns_false_past_tense_completed_action(self):
        """Past tense — action already done, not pending."""
        assert not _is_action_intent(self._ai("I created the file and saved it."))

    def test_returns_false_question_about_tool(self):
        """Question about a tool action — no intent to call it."""
        assert not _is_action_intent(self._ai("Should I run the tests?"))

    def test_returns_false_missing_tool_calls_attribute(self):
        """Object with no tool_calls attribute defaults to no tool calls."""
        msg = SimpleNamespace(content="I'll explain the architecture.")
        assert not _is_action_intent(msg)

    # ── Regression: informational query false positives ──────────────────────

    def test_returns_false_weather_response_with_let_me_know(self):
        """Regression: weather table + 'let me know' triggered false positive.

        'Feel free to let me know' matched _INTENT_LEAD_RE ('let me')
        and 'reading' in the response body matched _TOOL_VERB_RE.
        Fix: 'let me know' is excluded as a conversational phrase, and
        intent+verb must appear in the same sentence.
        """
        text = (
            "**Current Weather in Abu Dhabi (as of 10:17 UTC, 2 Apr 2026)**\n\n"
            "| Item | Value | Source |\n"
            "|------|-------|--------|\n"
            "| Temperature | **69 °F (≈ 20 °C)** | AccuWeather |\n"
            "| Condition | **Hazy sunshine** | AccuWeather |\n\n"
            "**Quick Summary**\n"
            "- The air feels slightly cooler than the thermometer reading "
            "due to the RealFeel® adjustment.\n\n"
            "Feel free to let me know if you'd like a forecast!"
        )
        assert not _is_action_intent(self._ai(text))

    def test_returns_false_let_me_know_with_verb_elsewhere(self):
        """'Let me know' in closing + tool verb in body must not trigger."""
        text = "Here are the search results I found.\n" "Let me know if you need more details."
        assert not _is_action_intent(self._ai(text))

    def test_returns_false_informational_response_with_data_table(self):
        """Data table responses should never trigger action-intent."""
        text = (
            "Here's what I found:\n\n"
            "| City | Temp | Humidity |\n"
            "|------|------|----------|\n"
            "| Abu Dhabi | 24°C | 68% |\n\n"
            "The reading is from open-meteo. Let me know if you need more."
        )
        assert not _is_action_intent(self._ai(text))

    def test_still_triggers_when_intent_and_verb_in_same_sentence(self):
        """Genuine action intent in a single sentence still triggers."""
        assert _is_action_intent(self._ai("Let me search for that information now."))
        assert _is_action_intent(self._ai("I'll create the file with the results."))

    def test_returns_false_verb_in_one_sentence_intent_in_another(self):
        """Verb and intent in different sentences must not trigger."""
        text = "The reading shows 24°C.\n" "I'll explain what this means for your trip."
        assert not _is_action_intent(self._ai(text))
