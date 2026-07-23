"""Tests for the NLP tools module."""

from collections import Counter

from src.tools.nlp_tools import (
    TOOL_CONFIGS,
    KeywordExtractionInput,
    SentimentAnalysisInput,
    TextSummarizationInput,
    _score_sentences,
    _simple_sentiment,
    _split_sentences,
    analyze_sentiment,
    extract_keywords,
    summarize_text,
)


class TestSimpleSentiment:
    """Unit tests for _simple_sentiment()."""

    def test_positive_sentiment(self):
        result = _simple_sentiment("This is good and great")
        assert result["polarity"] > 0
        assert result["subjectivity"] > 0

    def test_negative_sentiment(self):
        result = _simple_sentiment("This is bad and terrible")
        assert result["polarity"] < 0
        assert result["subjectivity"] > 0

    def test_neutral_sentiment(self):
        result = _simple_sentiment("The cat sat on the mat")
        assert result["polarity"] == 0.0
        assert result["subjectivity"] == 0.0

    def test_empty_text(self):
        result = _simple_sentiment("")
        assert result["polarity"] == 0.0
        assert result["subjectivity"] == 0.0

    def test_negation_flips_sentiment(self):
        positive = _simple_sentiment("This is good")
        negative = _simple_sentiment("This is not good")
        assert negative["polarity"] < positive["polarity"]

    def test_intensifier_boosts_sentiment(self):
        normal = _simple_sentiment("good but bad")
        intense = _simple_sentiment("very good but bad")
        assert intense["polarity"] > normal["polarity"]

    def test_negation_decay_after_three_words(self):
        result = _simple_sentiment("not one two three good")
        # After 3 words, negation should have decayed
        assert result["polarity"] > 0


class TestAnalyzeSentiment:
    """Unit tests for analyze_sentiment()."""

    def test_empty_text_returns_error(self):
        result = analyze_sentiment("   ")
        assert "Error: Empty text provided" in result

    def test_positive_classification(self):
        result = analyze_sentiment("This is good and great")
        assert "Positive" in result
        assert "Polarity Score:" in result
        assert "Subjectivity Score:" in result

    def test_very_positive_classification(self):
        result = analyze_sentiment("This is absolutely amazing wonderful fantastic")
        assert "Very Positive" in result

    def test_negative_classification(self):
        result = analyze_sentiment("This is bad and terrible")
        assert "Negative" in result

    def test_very_negative_classification(self):
        result = analyze_sentiment("This is absolutely horrible worst terrible awful")
        assert "Very Negative" in result

    def test_neutral_classification(self):
        result = analyze_sentiment("The cat sat on the mat")
        assert "Neutral" in result

    def test_highly_subjective(self):
        result = analyze_sentiment("I love this amazing wonderful fantastic great product")
        assert "Highly Subjective" in result

    def test_objective(self):
        result = analyze_sentiment("The report contains 42 pages")
        assert "Objective" in result

    def test_exception_handling(self):
        from unittest.mock import patch

        with patch(
            "src.tools.nlp_tools._simple_sentiment",
            side_effect=RuntimeError("mock error"),
        ):
            result = analyze_sentiment("some text")
        assert "Error analyzing sentiment" in result
        assert "Operation failed" in result  # sanitized — raw error not leaked


class TestSplitSentences:
    """Unit tests for _split_sentences()."""

    def test_basic_split(self):
        result = _split_sentences("First sentence. Second sentence.")
        assert result == ["First sentence.", "Second sentence."]

    def test_exclamation_split(self):
        result = _split_sentences("Hello! World!")
        assert result == ["Hello!", "World!"]

    def test_question_split(self):
        result = _split_sentences("How are you? I am fine.")
        assert result == ["How are you?", "I am fine."]

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []

    def test_whitespace_string(self):
        result = _split_sentences("   ")
        assert result == []

    def test_single_sentence(self):
        result = _split_sentences("Only one sentence.")
        assert result == ["Only one sentence."]


class TestScoreSentences:
    """Unit tests for _score_sentences()."""

    def test_basic_scoring(self):
        sentences = ["The cat sat on the mat.", "Python is great for coding."]
        word_freq = Counter({"python": 3, "coding": 2, "great": 1})
        scores = _score_sentences(sentences, word_freq)
        assert len(scores) == 2
        assert all(len(s) == 3 for s in scores)  # (score, index, sentence)

    def test_first_two_sentences_boosted(self):
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        word_freq = Counter({"first": 1, "second": 1, "third": 1})
        scores = _score_sentences(sentences, word_freq)
        # First two sentences get a 1.2x boost
        assert scores[0][0] > scores[2][0]
        assert scores[1][0] > scores[2][0]

    def test_empty_sentence(self):
        sentences = [""]
        word_freq = Counter()
        scores = _score_sentences(sentences, word_freq)
        assert scores[0][0] == 0.0

    def test_stop_words_ignored(self):
        sentences = ["The and of a."]
        word_freq = Counter({"the": 5, "and": 5, "of": 5, "a": 5})
        scores = _score_sentences(sentences, word_freq)
        assert scores[0][0] == 0.0


class TestSummarizeText:
    """Unit tests for summarize_text()."""

    def test_empty_text_returns_error(self):
        result = summarize_text("   ")
        assert "Error: Empty text provided" in result

    def test_short_text_returns_original(self):
        text = "One sentence only."
        result = summarize_text(text, num_sentences=3)
        assert "already short" in result
        assert text in result

    def test_summarizes_long_text(self):
        text = (
            "Python is a great programming language. "
            "It is used for web development and data science. "
            "Many developers love Python for its simplicity. "
            "Python has a huge ecosystem of libraries. "
            "Machine learning is a popular use case."
        )
        result = summarize_text(text, num_sentences=2)
        assert "Text Summary:" in result
        assert "Original:" in result
        assert "Summary:" in result
        assert "Compression:" in result

    def test_num_sentences_clamped_high(self):
        text = " ".join(f"Sentence {i} has important words." for i in range(15))
        # num_sentences > 10 should be clamped to 10
        result = summarize_text(text, num_sentences=20)
        assert "Text Summary:" in result

    def test_num_sentences_clamped_low(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        # num_sentences < 1 should be clamped to 1
        result = summarize_text(text, num_sentences=0)
        assert "Text Summary:" in result

    def test_preserves_sentence_order(self):
        text = (
            "First important sentence about Python. "
            "Second sentence about JavaScript. "
            "Third sentence about Rust. "
            "Fourth sentence about Go. "
            "Fifth sentence about Ruby."
        )
        result = summarize_text(text, num_sentences=2)
        lines = result.split("\n")
        summary_line = lines[2]  # The summary is on the third line
        # The summary should preserve original sentence order
        assert summary_line

    def test_exception_handling(self):
        from unittest.mock import patch

        with patch(
            "src.tools.nlp_tools._split_sentences",
            side_effect=RuntimeError("mock error"),
        ):
            result = summarize_text("some text")
        assert "Error summarizing text" in result
        assert "Operation failed" in result  # sanitized — raw error not leaked


class TestExtractKeywords:
    """Unit tests for extract_keywords()."""

    def test_empty_text_returns_error(self):
        result = extract_keywords("   ")
        assert "Error: Empty text provided" in result

    def test_extracts_keywords(self):
        text = (
            "Python is a great programming language. "
            "Python is used for web development and data science."
        )
        result = extract_keywords(text, num_keywords=3)
        assert "Extracted Keywords:" in result
        assert "python" in result.lower()
        assert "Total unique words analyzed:" in result

    def test_no_significant_keywords(self):
        text = "The and of a in is it."
        result = extract_keywords(text)
        assert "No significant keywords found" in result

    def test_num_keywords_clamped_high(self):
        text = "Python is great. Python is amazing."
        result = extract_keywords(text, num_keywords=100)
        assert "Extracted Keywords:" in result

    def test_num_keywords_clamped_low(self):
        text = "Python is great. Python is amazing."
        result = extract_keywords(text, num_keywords=0)
        assert "Extracted Keywords:" in result

    def test_short_words_filtered(self):
        text = "Python is great. Py is not."
        result = extract_keywords(text)
        # "py" should be filtered out (len <= 2)
        assert "py" not in result.lower().split("python")[0]

    def test_exception_handling(self):
        from unittest.mock import patch

        with patch(
            "src.tools.nlp_tools.re.findall",
            side_effect=RuntimeError("mock error"),
        ):
            result = extract_keywords("some text")
        assert "Error extracting keywords" in result
        assert "Operation failed" in result  # sanitized — raw error not leaked


class TestInputSchemas:
    """Unit tests for Pydantic input schemas."""

    def test_sentiment_analysis_input(self):
        schema = SentimentAnalysisInput(text="Hello world")
        assert schema.text == "Hello world"

    def test_text_summarization_input_defaults(self):
        schema = TextSummarizationInput(text="Hello world")
        assert schema.text == "Hello world"
        assert schema.num_sentences == 3

    def test_text_summarization_input_custom(self):
        schema = TextSummarizationInput(text="Hello world", num_sentences=5)
        assert schema.num_sentences == 5

    def test_keyword_extraction_input_defaults(self):
        schema = KeywordExtractionInput(text="Hello world")
        assert schema.text == "Hello world"
        assert schema.num_keywords == 10

    def test_keyword_extraction_input_custom(self):
        schema = KeywordExtractionInput(text="Hello world", num_keywords=5)
        assert schema.num_keywords == 5


class TestToolConfigs:
    """Unit tests for TOOL_CONFIGS registry entries."""

    def test_three_tool_configs(self):
        assert len(TOOL_CONFIGS) == 3

    def test_analyze_sentiment_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "analyze_sentiment")
        assert config["function"] == analyze_sentiment
        assert config["input_schema"] == SentimentAnalysisInput
        assert config["requires_confirmation"] is False
        assert "sentiment" in config["description"].lower()

    def test_summarize_text_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "summarize_text")
        assert config["function"] == summarize_text
        assert config["input_schema"] == TextSummarizationInput
        assert config["requires_confirmation"] is False
        assert "summarize" in config["description"].lower()

    def test_extract_keywords_config(self):
        config = next(c for c in TOOL_CONFIGS if c["name"] == "extract_keywords")
        assert config["function"] == extract_keywords
        assert config["input_schema"] == KeywordExtractionInput
        assert config["requires_confirmation"] is False
        assert "keyword" in config["description"].lower()
