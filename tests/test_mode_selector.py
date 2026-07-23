"""Tests for heuristic memory mode selector.

No LLM calls — pure regex/heuristic logic.
"""

from __future__ import annotations

from cogtrix_core.memory.mode_selector import classify_memory_mode, should_switch_mode

# ---------------------------------------------------------------------------
# classify_memory_mode
# ---------------------------------------------------------------------------


class TestClassifyMemoryMode:
    def test_empty_string_returns_conversation(self) -> None:
        assert classify_memory_mode("") == "conversation"

    def test_whitespace_only_returns_conversation(self) -> None:
        assert classify_memory_mode("   \n\t  ") == "conversation"

    def test_reasoning_keyword_analyze(self) -> None:
        assert classify_memory_mode("Please analyze the performance bottlenecks.") == "reasoning"

    def test_reasoning_keyword_design(self) -> None:
        assert (
            classify_memory_mode("Design a microservice architecture for this system.")
            == "reasoning"
        )

    def test_reasoning_keyword_plan(self) -> None:
        assert classify_memory_mode("Plan the migration strategy for our database.") == "reasoning"

    def test_reasoning_keyword_tradeoffs(self) -> None:
        assert (
            classify_memory_mode("What are the tradeoffs between REST and GraphQL?") == "reasoning"
        )

    def test_reasoning_keyword_evaluate(self) -> None:
        assert classify_memory_mode("Evaluate the options and pick the best one.") == "reasoning"

    def test_reasoning_keyword_explain(self) -> None:
        assert classify_memory_mode("Explain how transformers work in NLP.") == "reasoning"

    def test_long_prompt_no_code_returns_reasoning(self) -> None:
        # >500 chars with no code signals → reasoning
        long_prompt = (
            "I have been thinking about the organizational dynamics at my company lately. "
            "There seems to be a significant amount of friction between the product team "
            "and the engineering team. This manifests in missed deadlines and unclear "
            "requirements, plus frustration on both sides. I want to understand the root "
            "causes and come up with a comprehensive approach to improving collaboration "
            "going forward. Can you share your thoughts on how to best tackle this kind of "
            "cross-functional challenge at a mid-sized software company where processes "
            "are not yet fully established?"
        )
        assert len(long_prompt) > 500
        assert classify_memory_mode(long_prompt) == "reasoning"

    def test_short_prompt_no_signals_returns_conversation(self) -> None:
        assert classify_memory_mode("How are you today?") == "conversation"

    def test_code_fenced_block(self) -> None:
        prompt = "What does this do?\n```python\nprint('hello')\n```"
        assert classify_memory_mode(prompt) == "code"

    def test_code_extension_py(self) -> None:
        assert classify_memory_mode("Can you review my utils.py file?") == "code"

    def test_code_extension_js(self) -> None:
        assert classify_memory_mode("There is a bug in app.js somewhere.") == "code"

    def test_code_keyword_refactor(self) -> None:
        assert classify_memory_mode("I need to refactor this module.") == "code"

    def test_code_keyword_debug(self) -> None:
        assert classify_memory_mode("Help me debug this issue.") == "code"

    def test_code_keyword_pytest(self) -> None:
        assert classify_memory_mode("How do I write a pytest fixture?") == "code"

    def test_code_keyword_function(self) -> None:
        # "function" is a code keyword; use a prompt without reasoning words
        assert classify_memory_mode("What does this function return?") == "code"

    def test_code_keyword_python(self) -> None:
        assert classify_memory_mode("I am learning Python and need help.") == "code"

    def test_code_keyword_exception_name(self) -> None:
        assert classify_memory_mode("I got a TypeError in my script.") == "code"

    def test_reasoning_takes_priority_over_code_for_keywords(self) -> None:
        # If both reasoning and code signals are present, reasoning wins
        # (rule 1a fires before rule 2)
        prompt = "Analyze the architecture of this Python codebase."
        assert classify_memory_mode(prompt) == "reasoning"

    def test_four_space_indent_signals_code(self) -> None:
        prompt = "What does this snippet do?\n    result = x + y\n    return result"
        assert classify_memory_mode(prompt) == "code"

    def test_casual_greeting_returns_conversation(self) -> None:
        assert classify_memory_mode("Hi there! Nice to meet you.") == "conversation"

    def test_long_prompt_with_code_returns_code_not_reasoning(self) -> None:
        # Long prompt (>500 chars) that also has code signals → code wins (not reasoning via length)
        # because rule 1b only fires when there are NO code signals
        base = "I need help with my application. " * 20  # ~660 chars
        prompt = base + " Check the utils.py file for issues."
        assert len(prompt) > 500
        assert classify_memory_mode(prompt) == "code"


# ---------------------------------------------------------------------------
# should_switch_mode
# ---------------------------------------------------------------------------


class TestShouldSwitchMode:
    def test_empty_list_returns_none(self) -> None:
        assert should_switch_mode("conversation", []) is None

    def test_single_prompt_returns_none(self) -> None:
        assert should_switch_mode("conversation", ["How are you?"]) is None

    def test_two_of_three_code_prompts_suggests_code(self) -> None:
        prompts = [
            "How are you?",
            "Can you debug this Python script?",
            "Help me refactor utils.py.",
        ]
        assert should_switch_mode("conversation", prompts) == "code"

    def test_two_of_three_reasoning_prompts_suggests_reasoning(self) -> None:
        prompts = [
            "What is 2+2?",
            "Analyze the tradeoffs of this design.",
            "Please evaluate the options carefully.",
        ]
        assert should_switch_mode("conversation", prompts) == "reasoning"

    def test_no_majority_returns_none(self) -> None:
        # All different modes → no 2-of-3
        prompts = [
            "How are you?",  # conversation
            "Debug my script.py.",  # code
            "Analyze the architecture.",  # reasoning
        ]
        result = should_switch_mode("conversation", prompts)
        assert result is None

    def test_already_in_target_mode_returns_none(self) -> None:
        # 2-of-3 say code, but we're already in code mode
        prompts = [
            "How are you?",
            "Debug this Python code.",
            "Fix the bug in app.js.",
        ]
        assert should_switch_mode("code", prompts) is None

    def test_uses_only_last_three_prompts(self) -> None:
        # Long history: only last 3 matter
        prompts = [
            "Analyze the system design.",  # reasoning (old — ignored)
            "Analyze the architecture.",  # reasoning (old — ignored)
            "How are you?",  # conversation (in window)
            "Debug my script.py.",  # code (in window)
            "Fix the bug in app.js.",  # code (in window)
        ]
        assert should_switch_mode("conversation", prompts) == "code"

    def test_two_prompts_with_matching_mode_suggests_switch(self) -> None:
        prompts = [
            "Debug this Python error.",
            "Refactor the function.",
        ]
        assert should_switch_mode("conversation", prompts) == "code"

    def test_window_of_two_disagreeing_returns_none(self) -> None:
        prompts = [
            "How are you?",  # conversation
            "Analyze the tradeoffs.",  # reasoning
        ]
        result = should_switch_mode("conversation", prompts)
        assert result is None


# ---------------------------------------------------------------------------
# Config.adaptive_memory field
# ---------------------------------------------------------------------------


class TestConfigAdaptiveMemory:
    def test_default_is_true(self) -> None:
        from cogtrix_core.config import Config

        c = Config()
        assert c.adaptive_memory is True

    def test_can_be_set_false(self) -> None:
        from cogtrix_core.config import Config

        c = Config()
        c.adaptive_memory = False
        assert c.adaptive_memory is False

    def test_parsed_from_yaml_false(self) -> None:
        """adaptive_memory: false in YAML is parsed correctly."""
        import os
        import tempfile
        from pathlib import Path

        from cogtrix_core.config import Config, _apply_config_file

        yaml_content = "adaptive_memory: false\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            c = Config()
            _apply_config_file(c, Path(tmp_path))
            assert c.adaptive_memory is False
        finally:
            os.unlink(tmp_path)

    def test_parsed_from_yaml_true(self) -> None:
        """adaptive_memory: true in YAML is parsed correctly."""
        import os
        import tempfile
        from pathlib import Path

        from cogtrix_core.config import Config, _apply_config_file

        yaml_content = "adaptive_memory: true\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            c = Config()
            _apply_config_file(c, Path(tmp_path))
            assert c.adaptive_memory is True
        finally:
            os.unlink(tmp_path)
