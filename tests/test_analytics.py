"""
Tests for Data Analytics and Metrics Computation.
"""

import pytest
import pandas as pd
from src.analytics.sentiment_analytics import (
    compute_sentiment_distribution,
    get_top_ngrams,
    compute_dataset_health,
    compute_airline_analytics,
)


def test_compute_sentiment_distribution():
    """Verify sentiment distribution calculates counts and percentages correctly."""
    df = pd.DataFrame({"sentiment_label": ["positive", "positive", "negative", "positive"]})
    dist = compute_sentiment_distribution(df, sentiment_column="sentiment_label")
    assert len(dist) == 2
    pos_row = dist[dist["sentiment"] == "positive"].iloc[0]
    neg_row = dist[dist["sentiment"] == "negative"].iloc[0]
    assert pos_row["count"] == 3
    assert pos_row["percentage"] == 75.0
    assert neg_row["count"] == 1
    assert neg_row["percentage"] == 25.0


def test_get_top_ngrams():
    """Verify n-gram extraction for unigrams and bigrams."""
    df = pd.DataFrame({
        "clean_text": ["big data analytics", "big data spark", "big data mapreduce"],
        "sentiment_label": ["positive", "positive", "negative"],
    })
    top_uni = get_top_ngrams(df, ngram=1, top_n=5)
    assert not top_uni.empty
    assert top_uni.iloc[0]["term"] == "big" or top_uni.iloc[0]["term"] == "data"

    top_bi = get_top_ngrams(df, ngram=2, top_n=5)
    assert not top_bi.empty
    assert "big data" in top_bi["term"].values


def test_compute_dataset_health():
    """Verify dataset health metrics assessment."""
    df = pd.DataFrame({
        "text": ["Tweet 1", "Tweet 2", "Tweet 1", None],
        "sentiment": ["positive", "negative", "positive", "neutral"]
    })
    health = compute_dataset_health(df)
    assert health["total_records"] == 4
    assert health["duplicate_records"] == 1
    assert health["missing_values"]["text"] == 1
