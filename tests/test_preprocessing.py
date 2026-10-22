"""
Tests for Text Preprocessing and Tokenization Pipeline.
"""

import pytest
import pandas as pd
from src.data_processing.preprocess import clean_text, expand_contractions, tokenize, preprocess_dataset


def test_url_and_mention_removal():
    """Ensure URLs and @mentions are completely stripped."""
    raw = "Check this out @friend http://example.com/test and https://t.co/xyz #awesome"
    cleaned = clean_text(raw)
    assert "@friend" not in cleaned
    assert "http" not in cleaned
    assert "example.com" not in cleaned
    assert "awesome" in cleaned  # Hashtag text preserved


def test_contraction_expansion():
    """Ensure conversational contractions are normalized."""
    text = "I can't and won't do this, it's not fair."
    expanded = expand_contractions(text)
    assert "cannot" in expanded
    assert "will not" in expanded


def test_negation_preservation():
    """Crucial for sentiment: verify negations are NOT removed as stopwords."""
    raw = "The product is not good and cannot be recommended"
    cleaned = clean_text(raw, remove_stopwords=True)
    assert "not" in cleaned
    assert "cannot" in cleaned


def test_empty_and_invalid_text_handling():
    """Verify preprocessor handles null, empty, or non-string inputs safely."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text("    ") == ""
    assert clean_text("!@#$%^&*()") == ""


def test_batch_dataframe_preprocessing():
    """Verify batch DataFrame processing cleans text and filters empty rows."""
    df = pd.DataFrame({
        "text": ["Great movie!", "   ", "@user http://spam.com", "Loved it!"]
    })
    processed = preprocess_dataset(df, text_column="text", target_column="clean_text")
    assert "clean_text" in processed.columns
    assert len(processed) >= 2
    assert all(processed["clean_text"].str.strip() != "")
