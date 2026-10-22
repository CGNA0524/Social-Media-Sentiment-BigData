"""
Data Analytics and Aggregation Module for Social Media Sentiment Analysis.

Computes:
- Sentiment distribution counts and proportions
- Top positive and negative n-grams (unigrams, bigrams)
- Temporal sentiment trends (hourly / daily)
- Dataset quality and cleanliness metrics
- Airline-specific and entity-specific aggregations
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data_processing.preprocess import clean_text


def compute_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_column: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Calculate frequency counts and percentage distributions for sentiment classes.
    """
    if sentiment_column not in df.columns:
        if "sentiment" in df.columns:
            sentiment_column = "sentiment"
        elif "predicted_sentiment" in df.columns:
            sentiment_column = "predicted_sentiment"
        else:
            raise ValueError(f"Sentiment column '{sentiment_column}' not found.")

    counts = df[sentiment_column].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    total = counts["count"].sum()
    counts["percentage"] = (counts["count"] / total * 100).round(2)
    return counts


def get_top_ngrams(
    df: pd.DataFrame,
    sentiment_filter: Optional[str] = None,
    text_column: str = "clean_text",
    sentiment_column: str = "sentiment_label",
    ngram: int = 1,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract the most frequent unigrams or bigrams for a given sentiment class.

    Args:
        df: DataFrame containing preprocessed text.
        sentiment_filter: 'positive', 'negative', 'neutral', or None for all.
        text_column: Preprocessed text column name.
        sentiment_column: Sentiment label column name.
        ngram: 1 for unigrams, 2 for bigrams.
        top_n: Number of top terms to return.

    Returns:
        DataFrame with columns ['term', 'count'].
    """
    target_df = df
    if sentiment_filter is not None and sentiment_column in df.columns:
        target_df = df[df[sentiment_column].astype(str).str.lower() == sentiment_filter.lower()]

    if text_column not in target_df.columns:
        # Fall back to raw text with cleaning
        texts = target_df["text"].dropna().astype(str).apply(lambda x: clean_text(x, remove_stopwords=True))
    else:
        texts = target_df[text_column].dropna().astype(str)

    term_counts: Counter = Counter()

    for text in texts:
        tokens = [t for t in text.split() if len(t) > 2]
        if ngram == 1:
            term_counts.update(tokens)
        elif ngram == 2 and len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            term_counts.update(bigrams)

    most_common = term_counts.most_common(top_n)
    if not most_common:
        return pd.DataFrame(columns=["term", "count"])

    result = pd.DataFrame(most_common, columns=["term", "count"])
    return result


def compute_temporal_trends(
    df: pd.DataFrame,
    date_column: str = "date",
    sentiment_column: str = "sentiment_label",
    freq: str = "D",
) -> pd.DataFrame:
    """
    Aggregate sentiment counts over time (hourly or daily).
    Supports Sentiment140 date formats and ISO-8601 strings.
    """
    if date_column not in df.columns:
        return pd.DataFrame()

    df_time = df.copy()
    try:
        # Parse flexible dates
        df_time["parsed_date"] = pd.to_datetime(df_time[date_column], errors="coerce")
        df_time = df_time.dropna(subset=["parsed_date"])
        if len(df_time) == 0:
            return pd.DataFrame()

        df_time["period"] = df_time["parsed_date"].dt.to_period(freq).astype(str)
        trends = (
            df_time.groupby(["period", sentiment_column])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        return trends
    except Exception:
        return pd.DataFrame()


def compute_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess quality and cleanliness metrics of the dataset.
    """
    total_rows = len(df)
    missing_vals = df.isna().sum().to_dict()
    text_col = "text" if "text" in df.columns else df.columns[0]
    duplicate_count = int(df[text_col].duplicated().sum())
    duplicate_pct = round((duplicate_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    return {
        "total_records": total_rows,
        "missing_values": {k: int(v) for k, v in missing_vals.items()},
        "duplicate_records": duplicate_count,
        "duplicate_percentage": duplicate_pct,
        "memory_mb": float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
    }


def compute_airline_analytics(df_airline: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute domain-specific breakdown for the US Airline dataset.
    """
    if "airline" not in df_airline.columns or "sentiment" not in df_airline.columns:
        return {}

    by_airline = (
        df_airline.groupby(["airline", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    reasons = pd.DataFrame()
    if "negativereason" in df_airline.columns:
        neg = df_airline[df_airline["sentiment"] == "negative"]
        reason_counts = neg["negativereason"].value_counts().reset_index()
        reason_counts.columns = ["reason", "count"]
        reasons = reason_counts.head(10)

    return {"by_airline": by_airline, "negative_reasons": reasons}
