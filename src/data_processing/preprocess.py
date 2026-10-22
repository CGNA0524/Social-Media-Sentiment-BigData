"""
Text Preprocessing and Normalization Pipeline for Social Media Sentiment Analysis.

Handles:
- HTML entity decoding
- URL, user mention, and special token stripping
- Contraction expansion
- Punctuation and numeric character normalization
- Stopword removal (with robust built-in fallback)
- Optional Porter stemming
- Tokenization utilities
"""

import html
import re
from typing import List, Optional, Set
import pandas as pd

# Try importing NLTK resources with fail-safe built-in fallbacks
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    _NLTK_STOPWORDS: Set[str] = set(stopwords.words("english"))
    _STEMMER = PorterStemmer()
except Exception:
    _NLTK_STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"
    }
    _STEMMER = None

# Important sentiment-bearing negation words to retain even if stopwords are removed
NEGATION_WORDS = {"not", "no", "nor", "neither", "never", "cannot", "hardly", "barely"}
CUSTOM_STOPWORDS = _NLTK_STOPWORDS - NEGATION_WORDS

# Common social media and English contractions map
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "shan't": "shall not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "idk": "i do not know",
    "tbh": "to be honest",
    "imho": "in my honest opinion",
}

# Regex patterns compiled for performance
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
ALPHA_ONLY_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")  # e.g., 'sooo' -> 'soo'


def expand_contractions(text: str) -> str:
    """Expand conversational English contractions and social abbreviations."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.IGNORECASE)
    return text


def clean_text(
    text: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    preserve_case: bool = False,
) -> str:
    """
    Standardize raw social media text for feature extraction and sentiment modeling.

    Args:
        text: Raw tweet text string.
        remove_stopwords: Whether to filter common non-informative words.
        use_stemming: Whether to reduce tokens to root stems via PorterStemmer.
        preserve_case: If False, transforms all characters to lower case.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
    cleaned = html.unescape(text)

    # Remove URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # Remove User Mentions (@username)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    # Unpack hashtags: preserve text, drop hash symbol (e.g., #happy -> happy)
    cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)

    # Compress character repetitions (e.g., 'loveeeeee' -> 'lovee')
    cleaned = REPEATED_CHARS_PATTERN.sub(r"\1\1", cleaned)

    # Expand contractions before stripping punctuation
    cleaned = expand_contractions(cleaned)

    # Lowercase
    if not preserve_case:
        cleaned = cleaned.lower()

    # Remove non-alphabetical characters (punctuation, numbers, special symbols)
    cleaned = ALPHA_ONLY_PATTERN.sub(" ", cleaned)

    # Tokenize by whitespace
    tokens = cleaned.split()

    # Filter stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t) > 1]

    # Optional Porter Stemming
    if use_stemming and _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize a cleaned text string into constituent word tokens."""
    return [t for t in text.split() if t]


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    target_column: str = "clean_text",
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> pd.DataFrame:
    """
    Batch process an entire DataFrame, adding a cleaned text column.

    Args:
        df: Input DataFrame containing text.
        text_column: Source column name.
        target_column: Output column name for cleaned text.
        remove_stopwords: Whether to remove stopwords.
        use_stemming: Whether to apply Porter stemmer.

    Returns:
        DataFrame with new cleaned text column, with blank cleaned rows dropped.
    """
    df_out = df.copy()
    df_out[target_column] = df_out[text_column].astype(str).apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords, use_stemming=use_stemming)
    )
    # Drop rows where cleaned text became completely empty
    df_out = df_out[df_out[target_column].str.strip() != ""].reset_index(drop=True)
    return df_out
