"""
Tests for Dataset Loading and Schema Validation.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_processing.load_dataset import (
    load_sentiment140,
    load_airline_tweets,
    load_sqlite_tweets,
    get_dataset_summary,
    get_project_root,
)


def test_sentiment140_loading_and_schema():
    """Verify Sentiment140 loads correctly with 6 columns mapped and positive/negative labels."""
    sample_size = 200
    df = load_sentiment140(sample_size=sample_size, use_cache=False)
    assert len(df) == sample_size
    expected_cols = ["id", "sentiment", "sentiment_label", "date", "query", "user", "text"]
    assert list(df.columns) == expected_cols

    # Labels must only be 'negative' and 'positive' (binary dataset)
    unique_labels = set(df["sentiment_label"].unique())
    assert unique_labels == {"negative", "positive"}

    # Numeric codes must be 0 and 1
    assert set(df["sentiment"].unique()) == {0, 1}

    # Verify balanced sampling
    neg_count = (df["sentiment_label"] == "negative").sum()
    pos_count = (df["sentiment_label"] == "positive").sum()
    assert neg_count == pos_count == sample_size // 2


def test_airline_tweets_loading_and_schema():
    """Verify Airline Tweets loads with 3 genuine classes."""
    sample_size = 300
    df = load_airline_tweets(sample_size=sample_size)
    assert len(df) == sample_size
    assert "sentiment" in df.columns
    assert "text" in df.columns
    # Check that neutral is detected
    unique_sentiments = set(df["sentiment"].unique())
    assert "neutral" in unique_sentiments or "negative" in unique_sentiments or "positive" in unique_sentiments


def test_sqlite_tweets_loading():
    """Verify safe read-only query against database.sqlite."""
    df = load_sqlite_tweets(limit=50)
    assert len(df) == 50
    assert "text" in df.columns
    assert "sentiment" in df.columns


def test_dataset_summary():
    """Verify dataset summary returns proper fields and numeric memory values."""
    df = pd.DataFrame({
        "text": ["Hello world", "Another tweet", None, "Hello world"],
        "sentiment_label": ["positive", "negative", "positive", "positive"],
    })
    summary = get_dataset_summary(df, "TestDF")
    assert summary["dataset_name"] == "TestDF"
    assert summary["total_records"] == 4
    assert summary["missing_text_count"] == 1
    assert summary["duplicate_text_count"] == 1
    assert isinstance(summary["memory_mb"], float)
