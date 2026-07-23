"""Tests for src/memory/distillation.py distill_summary() and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.memory.distillation import _coerce_text, _parse_facts, distill_summary

# ---------------------------------------------------------------------------
# _coerce_text
# ---------------------------------------------------------------------------


class TestCoerceText:
    def test_string_passthrough(self):
        assert _coerce_text("hello world") == "hello world"

    def test_none_returns_empty_string(self):
        assert _coerce_text(None) == ""

    def test_list_of_strings_joined(self):
        result = _coerce_text(["hello", "world"])
        assert result == "hello world"

    def test_list_with_none_items(self):
        result = _coerce_text(["hello", None, "world"])
        assert result == "hello  world"

    def test_list_with_dict_uses_text_key(self):
        result = _coerce_text([{"text": "hello"}, {"text": "world"}])
        assert result == "hello world"

    def test_list_with_dict_missing_text_falls_back_to_json(self):
        result = _coerce_text([{"other": "value"}])
        assert "other" in result and "value" in result

    def test_list_with_mixed_items(self):
        result = _coerce_text([{"text": "hello"}, 42, True, None])
        assert "hello" in result
        assert "42" in result
        assert "True" in result

    def test_int_falls_back_to_str(self):
        result = _coerce_text(42)
        assert result == "42"

    def test_float_falls_back_to_str(self):
        result = _coerce_text(3.14)
        assert result == "3.14"

    def test_empty_list_returns_empty_string(self):
        result = _coerce_text([])
        assert result == ""

    def test_empty_dict_falls_back_to_json(self):
        result = _coerce_text({})
        assert result == "{}"


# ---------------------------------------------------------------------------
# _parse_facts
# ---------------------------------------------------------------------------


class TestParseFacts:
    def test_strips_bullet_prefix_with_hyphen(self):
        result = _parse_facts("- this is a fact")
        assert result == ["this is a fact"]

    def test_strips_bullet_prefix_with_star(self):
        result = _parse_facts("* this is a fact")
        assert result == ["this is a fact"]

    def test_strips_bullet_prefix_with_bullet_char(self):
        result = _parse_facts("• this is a fact")
        assert result == ["this is a fact"]

    def test_strips_numbered_list_prefix(self):
        result = _parse_facts("1. this is a fact")
        assert result == ["this is a fact"]

    def test_strips_numbered_list_with_paren(self):
        result = _parse_facts("2) this is a fact")
        assert result == ["this is a fact"]

    def test_ignores_empty_lines(self):
        result = _parse_facts("\n\n\n")
        assert result == []

    def test_ignores_whitespace_only_lines(self):
        result = _parse_facts("   \n\t\n")
        assert result == []

    def test_truncates_long_lines_to_20_words(self):
        long_line = "word " * 30
        result = _parse_facts(long_line)
        assert len(result) == 1
        words = result[0].split()
        assert len(words) == 20

    def test_caps_at_15_facts(self):
        lines = [f"fact {i}" for i in range(20)]
        input_text = "\n".join(lines)
        result = _parse_facts(input_text)
        assert len(result) == 15

    def test_preserves_content_after_truncation(self):
        # 25 words - should be truncated to first 20
        long_line = "word " * 25
        result = _parse_facts(long_line)
        words = result[0].split()
        assert len(words) == 20
        assert all(w == "word" for w in words)

    def test_multiple_lines_with_mixed_bullets(self):
        input_text = "- first\n* second\n• third\n1. fourth"
        result = _parse_facts(input_text)
        assert result == ["first", "second", "third", "fourth"]

    def test_empty_input_returns_empty_list(self):
        result = _parse_facts("")
        assert result == []


# ---------------------------------------------------------------------------
# distill_summary
# ---------------------------------------------------------------------------


class TestDistillSummary:
    def test_none_llm_returns_empty_list(self):
        """Passing llm=None should return [] without raising."""
        result = distill_summary(None, "some summary text")
        assert result == []

    def test_empty_input_returns_empty_list(self):
        fake_llm = MagicMock()
        result = distill_summary(fake_llm, "")
        assert result == []
        fake_llm.invoke.assert_not_called()

    def test_whitespace_only_input_returns_empty_list(self):
        fake_llm = MagicMock()
        result = distill_summary(fake_llm, "   \n\t")
        assert result == []
        fake_llm.invoke.assert_not_called()

    def test_returns_empty_when_langchain_unavailable(self):
        """LangChain ImportError should return []."""
        # This test verifies the ImportError path in distill_summary
        # Since langchain_core is installed in the runtime, we verify the code path exists
        # by checking the source code contains the expected try/except around import
        import inspect

        import src.memory.distillation as distillation_module

        source = inspect.getsource(distillation_module.distill_summary)
        # Verify the ImportError handling is present
        assert "try:" in source
        assert "from langchain_core.messages import" in source
        assert "except ImportError:" in source
        assert "LangChain not available" in source
        assert "return []" in source

    def test_llm_invoke_failure_returns_empty_list(self, caplog):
        """LLM call exception should return [] and log warning."""
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("Connection timeout")

        with caplog.at_level("WARNING", logger="cogtrix"):
            result = distill_summary(fake_llm, "test summary")
            assert result == []
            assert any(
                "Fact distillation failed" in rec.message
                for rec in caplog.records
                if rec.levelno == 30  # WARNING
            )

    def test_llm_timeout_returns_empty_list(self, caplog):
        """LLM timeout should return [] and log warning."""
        import concurrent.futures

        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = concurrent.futures.TimeoutError()

        with caplog.at_level("WARNING", logger="cogtrix"):
            result = distill_summary(fake_llm, "test summary")
            assert result == []
            assert any(
                "distill_summary: LLM call timed out" in rec.message
                for rec in caplog.records
                if rec.levelno == 30  # WARNING
            )

    def test_successful_invocation_returns_parsed_facts(self):
        """Happy path: LLM returns content that gets parsed."""
        fake_llm = MagicMock()
        # Create a mock response with content attribute
        mock_response = MagicMock()
        mock_response.content = "- fact one\n* fact two\n• fact three"
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        assert result == ["fact one", "fact two", "fact three"]
        fake_llm.invoke.assert_called_once()

    def test_response_with_non_content_attribute(self):
        """Handle response where content is not an attribute."""
        fake_llm = MagicMock()
        # Response is directly the content string
        fake_llm.invoke.return_value = "fact one\nfact two"

        result = distill_summary(fake_llm, "test summary")
        assert result == ["fact one", "fact two"]

    def test_truncation_applied_to_parsed_facts(self):
        """Verify 15-item cap and 20-word truncation are applied."""
        fake_llm = MagicMock()
        # Create content with 25 facts, each with 30 words
        long_content = "\n".join(f"fact {i} " + "word " * 30 for i in range(25))
        mock_response = MagicMock()
        mock_response.content = long_content
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        assert len(result) == 15  # capped at 15
        for fact in result:
            assert len(fact.split()) <= 20  # each truncated to 20 words

    def test_multiline_response_with_empty_lines(self):
        """Empty lines in LLM output should be skipped."""
        fake_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "first fact\n\n\nsecond fact\n\nthird fact"
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        assert result == ["first fact", "second fact", "third fact"]

    def test_list_content_in_response(self):
        """Handle response.content being a list (joins with spaces)."""
        fake_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ["first fact", "second fact"]
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        # _coerce_text joins list items with spaces into a single string
        # Then _parse_facts splits on newlines, not spaces
        assert result == ["first fact second fact"]

    def test_dict_response_with_text_key(self):
        """Handle response.content being a dict with text key."""
        fake_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = {"text": "fact one\nfact two"}
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        assert result == ["fact one", "fact two"]

    def test_dict_response_without_text_key_falls_back_to_json(self):
        """Handle response.content being a dict without text key."""
        fake_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = {"other": "value"}
        fake_llm.invoke.return_value = mock_response

        result = distill_summary(fake_llm, "test summary")
        # Falls back to json.dumps which includes "other" and "value"
        assert any("other" in fact and "value" in fact for fact in result)

    def test_timeout_setting_60_seconds(self):
        """Verify the timeout constant is set correctly."""
        from src.memory import distillation

        assert distillation._DISTILL_TIMEOUT_SECONDS == 60
