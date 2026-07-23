"""
Natural Language Processing (NLP) tools.
Provides sentiment analysis and text summarization capabilities.
"""

import re
from collections import Counter

from pydantic import BaseModel, Field

# Try to import TextBlob for sentiment analysis
try:
    from textblob import TextBlob  # type: ignore[import-untyped]

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = None  # type: ignore[misc, assignment]
    TEXTBLOB_AVAILABLE = False


class SentimentAnalysisInput(BaseModel):
    """Input schema for sentiment analysis."""

    text: str = Field(description="The text to analyze for sentiment")


class TextSummarizationInput(BaseModel):
    """Input schema for text summarization."""

    text: str = Field(description="The text to summarize")
    num_sentences: int = Field(
        default=3,
        description="Number of sentences in the summary (default: 3)",
    )


class KeywordExtractionInput(BaseModel):
    """Input schema for keyword extraction."""

    text: str = Field(description="The text to extract keywords from")
    num_keywords: int = Field(
        default=10,
        description="Number of keywords to extract (default: 10)",
    )


# Common English stop words
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "just",
    "should",
    "now",
    "i",
    "you",
    "your",
    "we",
    "our",
    "their",
    "them",
    "his",
    "her",
    "she",
    "him",
    "my",
    "me",
    "do",
    "does",
    "did",
    "done",
    "been",
    "being",
    "would",
    "could",
    "might",
    "must",
    "shall",
    "may",
    "also",
    "into",
    "over",
    "after",
    "before",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "about",
    "up",
    "down",
    "out",
    "off",
    "above",
    "below",
    "any",
    "if",
    "or",
}


def _simple_sentiment(text: str) -> dict:
    """
    Simple rule-based sentiment analysis fallback.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    # Positive and negative word lists
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "awesome",
        "love",
        "happy",
        "joy",
        "pleased",
        "delighted",
        "perfect",
        "best",
        "beautiful",
        "nice",
        "positive",
        "brilliant",
        "outstanding",
        "superb",
        "terrific",
        "marvelous",
        "magnificent",
        "splendid",
        "fine",
        "glad",
        "thankful",
        "grateful",
        "satisfied",
        "excited",
        "thrilled",
    }
    negative_words = {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "poor",
        "worst",
        "hate",
        "sad",
        "angry",
        "upset",
        "disappointed",
        "frustrated",
        "annoyed",
        "negative",
        "wrong",
        "fail",
        "failure",
        "problem",
        "issue",
        "error",
        "difficult",
        "hard",
        "unfortunately",
        "sorry",
        "regret",
        "unhappy",
        "miserable",
        "painful",
        "disgust",
        "fear",
        "worry",
        "anxious",
    }
    intensifiers = {
        "very",
        "really",
        "extremely",
        "absolutely",
        "totally",
        "completely",
    }
    negations = {
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
    }

    words = re.findall(r"\b\w+\b", text.lower())

    pos_count: float = 0.0
    neg_count: float = 0.0
    intensity: float = 1.0
    negate = False
    words_since_modifier = 0

    for _i, word in enumerate(words):
        if word in negations:
            negate = True
            words_since_modifier = 0
            continue
        if word in intensifiers:
            intensity = 1.5
            words_since_modifier = 0
            continue

        words_since_modifier += 1
        if words_since_modifier > 3:
            negate = False
            intensity = 1.0

        if word in positive_words:
            if negate:
                neg_count += intensity
            else:
                pos_count += intensity
            negate = False
            intensity = 1.0
            words_since_modifier = 0
        elif word in negative_words:
            if negate:
                pos_count += intensity
            else:
                neg_count += intensity
            negate = False
            intensity = 1.0
            words_since_modifier = 0

    total = pos_count + neg_count
    if total == 0:
        polarity = 0.0
    else:
        polarity = (pos_count - neg_count) / total

    # Estimate subjectivity based on opinion word density
    subjectivity = min(1.0, total / max(len(words), 1) * 5)

    return {"polarity": polarity, "subjectivity": subjectivity}


def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of text to determine if it's positive, negative, or neutral.

    Returns sentiment classification, polarity score, and subjectivity score.
    Polarity ranges from -1 (negative) to 1 (positive).
    Subjectivity ranges from 0 (objective) to 1 (subjective).

    Args:
        text: The text to analyze

    Returns:
        Sentiment analysis results
    """
    if not text.strip():
        return "Error: Empty text provided"

    try:
        if TEXTBLOB_AVAILABLE and TextBlob is not None:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            sentiment_result = _simple_sentiment(text)
            polarity = sentiment_result["polarity"]
            subjectivity = sentiment_result["subjectivity"]

        # Classify sentiment
        if polarity > 0.1:
            if polarity > 0.5:
                sentiment = "Very Positive 😊"
            else:
                sentiment = "Positive 🙂"
        elif polarity < -0.1:
            if polarity < -0.5:
                sentiment = "Very Negative 😞"
            else:
                sentiment = "Negative 😐"
        else:
            sentiment = "Neutral 😶"

        # Classify subjectivity
        if subjectivity > 0.6:
            subj_label = "Highly Subjective (opinion-based)"
        elif subjectivity > 0.3:
            subj_label = "Moderately Subjective"
        else:
            subj_label = "Objective (fact-based)"

        result = [
            "Sentiment Analysis Results:",
            "",
            f"Overall Sentiment: {sentiment}",
            f"Polarity Score: {polarity:.3f} (range: -1 to +1)",
            f"Subjectivity Score: {subjectivity:.3f} (range: 0 to 1)",
            f"Subjectivity: {subj_label}",
        ]

        if not TEXTBLOB_AVAILABLE:
            result.append("")
            result.append("Note: Using basic analysis. Install textblob for better accuracy:")
            result.append("  pip install textblob")

        return "\n".join(result)

    except Exception as e:
        return f"Error analyzing sentiment: {e}"


def _split_sentences(text: str) -> list:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _score_sentences(sentences: list, word_freq: Counter) -> list:
    """Score sentences based on word frequency."""
    scores = []
    for i, sentence in enumerate(sentences):
        words = re.findall(r"\b\w+\b", sentence.lower())
        if not words:
            scores.append((0.0, i, sentence))
            continue

        # Calculate score based on word frequency
        score: float = float(sum(word_freq.get(w, 0) for w in words if w not in STOP_WORDS))
        # Normalize by sentence length to avoid bias toward long sentences
        score = score / (len(words) ** 0.5)
        # Boost first sentences (often contain key info)
        if i < 2:
            score *= 1.2
        scores.append((score, i, sentence))

    return scores


def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text by extracting the most important sentences.

    Uses extractive summarization based on word frequency analysis.
    The summary preserves the original sentence order.

    Args:
        text: The text to summarize
        num_sentences: Number of sentences to include in summary (default: 3)

    Returns:
        Summarized text
    """
    if not text.strip():
        return "Error: Empty text provided"

    num_sentences = max(1, min(num_sentences, 10))

    try:
        sentences = _split_sentences(text)

        if len(sentences) <= num_sentences:
            return f"Text is already short ({len(sentences)} sentences). Original text:\n\n{text}"

        # Calculate word frequencies
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        word_freq = Counter(words)

        # Score sentences
        scored = _score_sentences(sentences, word_freq)

        # Select top sentences
        top_sentences = sorted(scored, key=lambda x: x[0], reverse=True)[:num_sentences]

        # Restore original order
        top_sentences = sorted(top_sentences, key=lambda x: x[1])

        summary = " ".join(s[2] for s in top_sentences)

        result = [
            "Text Summary:",
            "",
            summary,
            "",
            "---",
            f"Original: {len(sentences)} sentences, {len(text)} characters",
            f"Summary: {num_sentences} sentences, {len(summary)} characters",
            f"Compression: {100 - (len(summary) / len(text) * 100):.1f}%",
        ]

        return "\n".join(result)

    except Exception as e:
        return f"Error summarizing text: {e}"


def extract_keywords(text: str, num_keywords: int = 10) -> str:
    """
    Extract the most important keywords from text.

    Uses word frequency analysis with stop word filtering.

    Args:
        text: The text to extract keywords from
        num_keywords: Number of keywords to extract (default: 10)

    Returns:
        List of keywords with their frequencies
    """
    if not text.strip():
        return "Error: Empty text provided"

    num_keywords = max(1, min(num_keywords, 50))

    try:
        # Extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter stop words and short words
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        if not words:
            return "No significant keywords found in text."

        # Count frequencies
        word_freq = Counter(words)
        top_keywords = word_freq.most_common(num_keywords)

        result = ["Extracted Keywords:", ""]
        for i, (word, count) in enumerate(top_keywords, 1):
            result.append(f"{i:2}. {word} ({count} occurrences)")

        result.append("")
        result.append(f"Total unique words analyzed: {len(word_freq)}")

        return "\n".join(result)

    except Exception as e:
        return f"Error extracting keywords: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "analyze_sentiment",
        "description": (
            "Analyze the sentiment of text to determine if it's positive, negative, or neutral. "
            "Returns polarity score (-1 to +1) and subjectivity score (0 to 1). "
            "Useful for understanding user emotions and tailoring responses."
        ),
        "input_schema": SentimentAnalysisInput,
        "requires_confirmation": False,
        "function": analyze_sentiment,
    },
    {
        "name": "summarize_text",
        "description": (
            "Summarize long text by extracting the most important sentences. "
            "Uses extractive summarization to provide concise information quickly. "
            "Specify num_sentences to control summary length."
        ),
        "input_schema": TextSummarizationInput,
        "requires_confirmation": False,
        "function": summarize_text,
    },
    {
        "name": "extract_keywords",
        "description": (
            "Extract the most important keywords from text. "
            "Useful for identifying main topics and themes in documents."
        ),
        "input_schema": KeywordExtractionInput,
        "requires_confirmation": False,
        "function": extract_keywords,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "analyze_sentiment",
    "summarize_text",
    "extract_keywords",
    "SentimentAnalysisInput",
    "TextSummarizationInput",
    "KeywordExtractionInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
