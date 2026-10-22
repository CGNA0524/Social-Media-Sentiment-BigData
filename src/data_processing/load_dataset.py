"""
Dataset Ingestion and Validation Module for Social Media Sentiment Big Data Project.

Handles:
- Sentiment140 (1.6 Million tweets, binary sentiment: 0=negative, 4=positive)
- Twitter US Airline Sentiment (Tweets.csv, 3 classes: negative, neutral, positive)
- SQLite Database Ingestion (database.sqlite)
- Automated caching, stratified sampling, and schema normalization
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np

# Project root directory reference
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Sentiment140 standard configuration
SENTIMENT140_COLUMNS = ["target", "id", "date", "query", "user", "text"]
SENTIMENT140_LABEL_MAP = {0: "negative", 4: "positive"}
LABEL_TO_BINARY_MAP = {"negative": 0, "positive": 1}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def load_sentiment140(
    filepath: Optional[str | Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    use_cache: bool = True,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load and validate the Sentiment140 dataset.

    Args:
        filepath: Path to the raw CSV. Defaults to data/raw/training.1600000.processed.noemoticon.csv.
        sample_size: Number of rows to sample. If None, loads all 1,600,000 rows.
        random_state: Seed for reproducible stratified sampling.
        use_cache: Whether to use cached processed sample file.
        force_reload: Force re-reading from raw CSV even if cache exists.

    Returns:
        Cleaned, schema-normalized DataFrame with columns:
        ['id', 'sentiment', 'sentiment_label', 'date', 'query', 'user', 'text']
    """
    if filepath is None:
        full_path = DATA_RAW / "training.1600000.processed.noemoticon.csv"
        sample_path = DATA_RAW / "sentiment140_sample.csv"
        if full_path.exists():
            filepath = full_path
        elif sample_path.exists():
            filepath = sample_path
        else:
            filepath = full_path
    else:
        filepath = Path(filepath)

    # If full dataset not found, fallback to sample file or generate synthetic sample
    if not filepath.exists():
        fallback_sample = DATA_RAW / "sentiment140_sample.csv"
        if fallback_sample.exists():
            filepath = fallback_sample
        else:
            # Fallback synthetic sample generator for cloud deployment
            synthetic_data = [
                (0, 1001, "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", "@user_neg", "Service is terrible and delivery is late."),
                (0, 1002, "Mon Apr 06 22:19:49 PDT 2009", "NO_QUERY", "@user_neg2", "Flight delayed again, absolutely hate waiting."),
                (4, 1003, "Mon Apr 06 22:19:53 PDT 2009", "NO_QUERY", "@user_pos", "The delivery was fast, and the product quality is excellent."),
                (4, 1004, "Mon Apr 06 22:19:57 PDT 2009", "NO_QUERY", "@user_pos2", "Super happy with the customer support team!"),
            ] * max(10, (sample_size or 50) // 4)
            df_synth = pd.DataFrame(synthetic_data, columns=SENTIMENT140_COLUMNS)
            df_synth["sentiment_label"] = df_synth["target"].map(SENTIMENT140_LABEL_MAP)
            df_synth["sentiment"] = df_synth["sentiment_label"].map(LABEL_TO_BINARY_MAP)
            return df_synth.reset_index(drop=True)

    # Cache file path for sampled or full versions
    cache_tag = f"sample_{sample_size}" if sample_size else "full"
    cache_path = DATA_PROCESSED / f"sentiment140_{cache_tag}.parquet"

    if use_cache and not force_reload and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fall back to raw loading if parquet read fails

    # Read raw CSV with latin-1 encoding
    # Check if reading the bundled sample CSV (which has a header)
    if filepath.name == "sentiment140_sample.csv":
        df_raw = pd.read_csv(filepath)
        if "target" not in df_raw.columns:
            df_raw.columns = SENTIMENT140_COLUMNS
    elif sample_size and sample_size < 1_600_000:
        # Load in chunks or read subset: half from beginning (negatives) and half from end (positives)
        # In Sentiment140, first 800,000 are 0 (negative) and next 800,000 are 4 (positive)
        half_sample = sample_size // 2
        df_neg = pd.read_csv(
            filepath,
            encoding="latin-1",
            header=None,
            names=SENTIMENT140_COLUMNS,
            nrows=half_sample * 2,  # Read slightly more for cleaning buffer
        )
        # Read positives from offset (around 800,000)
        df_pos = pd.read_csv(
            filepath,
            encoding="latin-1",
            header=None,
            names=SENTIMENT140_COLUMNS,
            skiprows=800_000,
            nrows=half_sample * 2,
        )
        df_raw = pd.concat([df_neg, df_pos], ignore_index=True)
    else:
        df_raw = pd.read_csv(
            filepath,
            encoding="latin-1",
            header=None,
            names=SENTIMENT140_COLUMNS,
            low_memory=False,
        )

    # Clean text column
    df_raw["text"] = df_raw["text"].astype(str).str.strip()
    df_raw = df_raw[df_raw["text"] != ""].copy()
    df_raw = df_raw.dropna(subset=["text", "target"])

    # Map labels: 0 -> 'negative', 4 -> 'positive'
    # Filter to only valid 0 and 4 targets
    df_raw = df_raw[df_raw["target"].isin([0, 4])].copy()
    df_raw["sentiment_label"] = df_raw["target"].map(SENTIMENT140_LABEL_MAP)
    df_raw["sentiment"] = df_raw["sentiment_label"].map(LABEL_TO_BINARY_MAP)

    # Perform balanced sampling if sample_size is specified
    if sample_size and len(df_raw) > sample_size:
        per_class = sample_size // 2
        sampled_parts = []
        for label_val in [0, 1]:
            subset = df_raw[df_raw["sentiment"] == label_val]
            take = min(len(subset), per_class)
            sampled_parts.append(subset.sample(n=take, random_state=random_state))
        df_result = pd.concat(sampled_parts, ignore_index=True).sample(
            frac=1.0, random_state=random_state
        ).reset_index(drop=True)
    else:
        df_result = df_raw.reset_index(drop=True)

    # Reorder columns
    ordered_cols = ["id", "sentiment", "sentiment_label", "date", "query", "user", "text"]
    df_result = df_result[ordered_cols]

    # Save to cache for ultra-fast subsequent dashboard and model runs
    try:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df_result.to_parquet(cache_path, index=False)
    except Exception:
        pass  # If parquet write fails, memory version is still valid

    return df_result


def load_airline_tweets(
    filepath: Optional[str | Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and normalize the Twitter US Airline Sentiment dataset (Tweets.csv).
    This dataset contains three genuine classes: negative, neutral, positive.

    Returns:
        Normalized DataFrame with standardized column names:
        ['tweet_id', 'sentiment', 'confidence', 'airline', 'negativereason', 'text', 'created_at', 'location']
    """
    if filepath is None:
        filepath = DATA_RAW / "Tweets.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        synthetic_airline = [
            ("570306133677760513", "neutral", 1.0, "Virgin America", "", "@VirginAmerica What @dhepburn said.", "2015-02-24 11:35:52 -0800", "San Francisco, CA"),
            ("570301130888122368", "positive", 0.95, "Virgin America", "", "@VirginAmerica loved the flight! Amazing cabin crew and smooth landing.", "2015-02-24 11:15:59 -0800", "New York, NY"),
            ("570301083672813571", "negative", 0.98, "Virgin America", "Bad Flight", "@VirginAmerica delayed for 4 hours and lost my luggage. Terrible customer service!", "2015-02-24 11:15:48 -0800", "Los Angeles, CA"),
        ] * max(10, (sample_size or 50) // 3)
        return pd.DataFrame(
            synthetic_airline,
            columns=["tweet_id", "sentiment", "confidence", "airline", "negativereason", "text", "created_at", "location"]
        ).reset_index(drop=True)

    df = pd.read_csv(filepath)

    # Standardize column names
    rename_dict = {
        "airline_sentiment": "sentiment",
        "airline_sentiment_confidence": "confidence",
        "tweet_created": "created_at",
        "tweet_location": "location",
    }
    df = df.rename(columns=rename_dict)

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df = df.dropna(subset=["text", "sentiment"])

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    keep_cols = [
        col for col in [
            "tweet_id",
            "sentiment",
            "confidence",
            "airline",
            "negativereason",
            "text",
            "created_at",
            "location",
            "user_timezone",
        ] if col in df.columns
    ]
    return df[keep_cols].reset_index(drop=True)


def load_sqlite_tweets(
    db_path: Optional[str | Path] = None,
    table_name: str = "Tweets",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely load tweets from the local SQLite database in read-only mode.
    """
    if db_path is None:
        db_path = DATA_RAW / "database.sqlite"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        return load_airline_tweets(sample_size=limit)

    # Use URI for safe read-only access
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    query = f"SELECT * FROM [{table_name}]"
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Standardize column names similar to Tweets.csv
    rename_dict = {
        "airline_sentiment": "sentiment",
        "airline_sentiment_confidence": "confidence",
        "tweet_created": "created_at",
        "tweet_location": "location",
    }
    df = df.rename(columns=rename_dict)
    return df


def get_dataset_summary(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Generate comprehensive statistical summary of a loaded dataset.
    """
    total_records = len(df)
    missing_text = df["text"].isna().sum() if "text" in df.columns else 0
    duplicate_texts = df["text"].duplicated().sum() if "text" in df.columns else 0

    sentiment_col = "sentiment_label" if "sentiment_label" in df.columns else "sentiment"
    if sentiment_col in df.columns:
        class_dist = df[sentiment_col].value_counts().to_dict()
        class_pct = (df[sentiment_col].value_counts(normalize=True) * 100).round(2).to_dict()
    else:
        class_dist = {}
        class_pct = {}

    memory_mb = float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2))

    return {
        "dataset_name": dataset_name,
        "total_records": total_records,
        "columns": list(df.columns),
        "class_distribution": class_dist,
        "class_percentages": class_pct,
        "missing_text_count": int(missing_text),
        "duplicate_text_count": int(duplicate_texts),
        "memory_mb": memory_mb,
    }
