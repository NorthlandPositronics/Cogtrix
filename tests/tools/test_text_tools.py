"""Tests for the text tools module."""

from src.tools.text_tools import (
    TOOL_CONFIGS,
    ExtractEmailsInput,
    ExtractUrlsInput,
    FindReplaceInput,
    SplitTextInput,
    TextCompareInput,
    TrimTextInput,
    WordCountInput,
    extract_emails,
    extract_urls,
    find_replace,
    split_text,
    text_compare,
    trim_text,
    word_count,
)


class TestWordCount:
    """Unit tests for word_count()."""

    def test_normal_text_statistics(self):
        result = word_count("Hello world, this is a test.")
        assert "Characters: 28" in result
        assert "Characters: 23" in result  # without spaces
        assert "Words: 6" in result
        assert "Lines: 1" in result

    def test_empty_string_returns_error(self):
        result = word_count("")
        assert result == "Empty text provided"

    def test_whitespace_only_returns_word_count_zero(self):
        result = word_count("   \n\t  ")
        assert "Words: 0" in result

    def test_single_word(self):
        result = word_count("test")
        assert "Words: 1" in result
        assert "Characters: 4" in result

    def test_multiline_text_counts_lines_and_paragraphs(self):
        result = word_count("Line one.\n\nLine two.\nLine three.")
        assert "Lines: 4" in result
        assert "Paragraphs: 2" in result


class TestFindReplace:
    """Unit tests for find_replace()."""

    def test_simple_text_replacement(self):
        result = find_replace("Hello world", "world", "there")
        assert "Made 1 replacement(s)" in result
        assert "Hello there" in result

    def test_multiple_replacements(self):
        result = find_replace("foo foo foo", "foo", "bar")
        assert "Made 3 replacement(s)" in result
        assert "bar bar bar" in result

    def test_regex_replacement(self):
        result = find_replace("abc123def456", r"\d+", "NUM", use_regex=True)
        assert "Made 2 replacement(s)" in result
        assert "abcNUMdefNUM" in result

    def test_case_insensitive_search(self):
        result = find_replace("Hello HELLO hello", "hello", "Hi", case_sensitive=False)
        assert "Made 3 replacement(s)" in result
        assert "Hi Hi Hi" in result

    def test_case_sensitive_search(self):
        result = find_replace("Hello HELLO hello", "hello", "Hi", case_sensitive=True)
        assert "Made 1 replacement(s)" in result

    def test_no_match_returns_original_with_count_zero(self):
        result = find_replace("Hello world", "goodbye", "hi")
        assert "Made 0 replacement(s)" in result
        assert "Hello world" in result

    def test_empty_text_returns_error(self):
        result = find_replace("", "x", "y")
        assert result == "Empty text provided"

    def test_empty_find_returns_error(self):
        result = find_replace("Hello world", "", "y")
        assert result == "Error: Nothing to find"

    def test_regex_error_invalid_pattern(self):
        result = find_replace("some text", "[invalid", "replacement", use_regex=True)
        assert "Regex error" in result

    def test_regex_case_insensitive(self):
        result = find_replace("ABC abc Abc", r"abc", "XYZ", use_regex=True, case_sensitive=False)
        assert "Made 3 replacement(s)" in result
        assert "XYZ XYZ XYZ" in result


class TestExtractUrls:
    """Unit tests for extract_urls()."""

    def test_multiple_urls_extracted(self):
        result = extract_urls("Check https://example.com and http://test.org for more")
        assert "Found 2 URL(s)" in result
        assert "https://example.com" in result
        assert "http://test.org" in result

    def test_no_urls_returns_message(self):
        result = extract_urls("No URLs here, just plain text.")
        assert result == "No URLs found in text"

    def test_url_with_query_string(self):
        result = extract_urls("Visit https://example.com/path?q=hello&page=1 for details")
        assert "https://example.com/path?q=hello&page=1" in result

    def test_url_with_special_chars_in_path(self):
        result = extract_urls("Link: https://example.com/file-name_v2.pdf")
        assert "https://example.com/file-name_v2.pdf" in result

    def test_duplicate_urls_deduplicated(self):
        result = extract_urls("See https://example.com and also https://example.com")
        assert "Found 1 URL(s)" in result

    def test_url_cleaned_of_trailing_punctuation(self):
        result = extract_urls("Visit https://example.com/path! or https://test.com?")
        assert "https://example.com/path" in result
        assert "https://test.com" in result

    def test_empty_text_returns_error(self):
        result = extract_urls("")
        assert result == "Empty text provided"

    def test_url_with_port_number(self):
        result = extract_urls("See http://localhost:8080 or https://example.com:443/path")
        assert "http://localhost:8080" in result
        assert "https://example.com:443/path" in result


class TestExtractEmails:
    """Unit tests for extract_emails()."""

    def test_multiple_emails_extracted(self):
        result = extract_emails("Contact alice@example.com or bob@test.org today")
        assert "Found 2 email address(es)" in result
        assert "alice@example.com" in result
        assert "bob@test.org" in result

    def test_no_emails_returns_message(self):
        result = extract_emails("No emails here, just plain text.")
        assert result == "No email addresses found in text"

    def test_duplicate_emails_deduplicated(self):
        result = extract_emails("Email test@test.com or TEST@TEST.COM or Test@Test.Com")
        assert "Found 1 email address(es)" in result

    def test_email_with_subdomains(self):
        result = extract_emails("Reach out to dev.team@mail.example.co.uk")
        assert "dev.team@mail.example.co.uk" in result

    def test_empty_text_returns_error(self):
        result = extract_emails("")
        assert result == "Empty text provided"

    def test_email_with_plus_sign(self):
        result = extract_emails("Send to user+tag@example.com for filtering")
        assert "user+tag@example.com" in result


class TestTextCompare:
    """Unit tests for text_compare()."""

    def test_identical_texts_returns_identical_message(self):
        result = text_compare("Hello world", "Hello world")
        assert result == "The texts are identical."

    def test_completely_different_texts(self):
        result = text_compare("abc", "xyz")
        assert "Length: 3 vs 3 characters" in result
        assert "Words: 1 vs 1" in result

    def test_different_lengths(self):
        result = text_compare("Short", "Much longer text here")
        assert "Length: 5 vs 21 characters" in result
        assert "16 difference" in result

    def test_shows_first_difference_context(self):
        result = text_compare("Hello world", "Hello planet")
        assert "First difference at position" in result

    def test_empty_texts(self):
        result = text_compare("", "")
        assert result == "The texts are identical."

    def test_one_empty_one_not(self):
        result = text_compare("", "abc")
        assert "Length: 0 vs 3" in result


class TestSplitText:
    """Unit tests for split_text()."""

    def test_default_newline_delimiter(self):
        result = split_text("line one\nline two\nline three")
        assert "Split into 3 part(s)" in result
        assert "1. line one" in result
        assert "2. line two" in result

    def test_custom_delimiter(self):
        result = split_text("a,b,c", ",")
        assert "Split into 3 part(s)" in result
        assert "1. a" in result
        assert "2. b" in result

    def test_max_parts_limits_splits(self):
        result = split_text("a,b,c,d", ",", max_parts=2)
        assert "Split into 2 part(s)" in result
        assert "1. a" in result
        assert "2. b,c,d" in result  # remainder as single part

    def test_empty_text_returns_error(self):
        result = split_text("")
        assert result == "Empty text provided"

    def test_no_delimiter_match(self):
        result = split_text("no commas here", ",")
        # No delimiter found — full text returned as single non-empty part
        assert "Split into 1 part(s)" in result
        assert "no commas here" in result

    def test_escaped_delimiter_n(self):
        result = split_text("line1\nline2", "\\n")
        assert "Split into 2 part(s)" in result

    def test_whitespace_trimmed_from_parts(self):
        result = split_text("  a  ,  b  ,  c  ", ",")
        assert "Split into 3 part(s)" in result
        # Whitespace is preserved in parts (only empty parts filtered)
        assert "1.   a  " in result

    def test_long_part_truncated_in_preview(self):
        long_text = "a" * 150
        result = split_text(f"{long_text},b", ",")
        assert "Split into 2 part(s)" in result
        # Preview truncates at 100 chars
        assert "..." in result


class TestTrimText:
    """Unit tests for trim_text()."""

    def test_text_shorter_than_max_unchanged(self):
        result = trim_text("Hello world", 100)
        assert result == "Hello world"

    def test_text_longer_than_max_truncated_with_ellipsis(self):
        result = trim_text("This is a long piece of text", 10)
        assert len(result) <= 10
        assert result.endswith("...")

    def test_trim_without_ellipsis(self):
        result = trim_text("This is a long piece of text", 10, add_ellipsis=False)
        assert len(result) <= 10
        assert "..." not in result

    def test_max_length_less_than_4_no_ellipsis(self):
        result = trim_text("This is a long text", 3, add_ellipsis=True)
        assert len(result) <= 3
        assert "..." not in result

    def test_max_length_less_than_1_returns_error(self):
        result = trim_text("Some text", 0)
        assert "Error: max_length must be at least 1" in result

    def test_empty_text_returns_error(self):
        result = trim_text("", 10)
        assert result == "Empty text provided"

    def test_exactly_max_length_returns_unchanged(self):
        result = trim_text("Hello", 5)
        assert result == "Hello"

    def test_trim_preserves_start_of_text(self):
        result = trim_text("Hello world and everyone", 12, add_ellipsis=True)
        # Truncated to 9 chars + ellipsis = "Hello wor..."
        assert result.startswith("Hello wor")
        assert result.endswith("...")


class TestInputSchemas:
    """Unit tests for Pydantic input schemas."""

    def test_word_count_input(self):
        schema = WordCountInput(text="Hello world")
        assert schema.text == "Hello world"

    def test_find_replace_input_defaults(self):
        schema = FindReplaceInput(text="hello", find="l", replace="x")
        assert schema.text == "hello"
        assert schema.find == "l"
        assert schema.replace == "x"
        assert schema.use_regex is False
        assert schema.case_sensitive is True

    def test_find_replace_input_custom_flags(self):
        schema = FindReplaceInput(
            text="hello", find="l", replace="x", use_regex=True, case_sensitive=False
        )
        assert schema.use_regex is True
        assert schema.case_sensitive is False

    def test_extract_urls_input(self):
        schema = ExtractUrlsInput(text="https://example.com")
        assert schema.text == "https://example.com"

    def test_extract_emails_input(self):
        schema = ExtractEmailsInput(text="test@example.com")
        assert schema.text == "test@example.com"

    def test_text_compare_input(self):
        schema = TextCompareInput(text1="abc", text2="def")
        assert schema.text1 == "abc"
        assert schema.text2 == "def"

    def test_split_text_input_defaults(self):
        schema = SplitTextInput(text="a,b,c")
        assert schema.text == "a,b,c"
        assert schema.delimiter == "\n"
        assert schema.max_parts is None

    def test_split_text_input_custom(self):
        schema = SplitTextInput(text="a,b,c", delimiter=",", max_parts=2)
        assert schema.delimiter == ","
        assert schema.max_parts == 2

    def test_trim_text_input_defaults(self):
        schema = TrimTextInput(text="Hello world", max_length=10)
        assert schema.text == "Hello world"
        assert schema.max_length == 10
        assert schema.add_ellipsis is True

    def test_trim_text_input_custom(self):
        schema = TrimTextInput(text="Hello world", max_length=10, add_ellipsis=False)
        assert schema.add_ellipsis is False


class TestToolConfigs:
    """Unit tests for TOOL_CONFIGS registry entries."""

    def test_seven_tool_configs(self):
        assert len(TOOL_CONFIGS) == 7

    def test_word_count_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "word_count")
        assert config["function"] == word_count
        assert config["input_schema"] == WordCountInput
        assert config["requires_confirmation"] is False

    def test_find_replace_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "find_replace")
        assert config["function"] == find_replace
        assert config["input_schema"] == FindReplaceInput

    def test_extract_urls_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "extract_urls")
        assert config["function"] == extract_urls
        assert config["input_schema"] == ExtractUrlsInput

    def test_extract_emails_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "extract_emails")
        assert config["function"] == extract_emails
        assert config["input_schema"] == ExtractEmailsInput

    def test_text_compare_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "text_compare")
        assert config["function"] == text_compare
        assert config["input_schema"] == TextCompareInput

    def test_split_text_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "split_text")
        assert config["function"] == split_text
        assert config["input_schema"] == SplitTextInput

    def test_trim_text_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "trim_text")
        assert config["function"] == trim_text
        assert config["input_schema"] == TrimTextInput
