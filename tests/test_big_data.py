"""
Tests for Big Data Components: MapReduce, Spark Engine, and NoSQL Storage.
"""

import pytest
import pandas as pd
from src.big_data.mapreduce_simulator import (
    MapReduceEngine,
    word_count_mapper,
    sentiment_count_mapper,
    sum_reducer,
)
from src.big_data.spark_engine import spark_manager
from src.big_data.nosql_storage import nosql_storage


def test_mapreduce_word_count_execution():
    """Verify MapReduce simulator correctly computes word frequencies."""
    lines = [
        "hadoop mapreduce big data",
        "spark mapreduce big data",
        "big data compute",
    ]
    engine = MapReduceEngine(num_mappers=2, num_reducers=2, use_combiner=True)
    res = engine.run_job(lines, word_count_mapper, sum_reducer)

    assert res["total_input_records"] == 3
    assert res["num_mappers"] == 2
    assert res["num_reducers"] == 2
    assert "data" in res["results"]
    assert res["results"]["data"] == 3
    assert res["results"]["big"] == 3
    assert res["results"]["mapreduce"] == 2


def test_mapreduce_sentiment_count():
    """Verify MapReduce sentiment counting."""
    data = [
        ("tweet text 1", "positive"),
        ("tweet text 2", "negative"),
        ("tweet text 3", "positive"),
    ]
    engine = MapReduceEngine(num_mappers=2, num_reducers=2, use_combiner=False)
    res = engine.run_job(data, sentiment_count_mapper, sum_reducer)
    assert res["results"]["positive"] == 2
    assert res["results"]["negative"] == 1


def test_spark_engine_telemetry():
    """Verify Spark engine returns honest engine status and handles batch aggregation."""
    info = spark_manager.get_engine_info()
    assert "engine_type" in info
    assert "status" in info
    assert "is_distributed" in info

    test_df = pd.DataFrame({
        "text": ["flight good", "flight bad"],
        "sentiment_label": ["positive", "negative"],
    })
    batch_res = spark_manager.aggregate_sentiment_batch(test_df)
    assert "engine_used" in batch_res
    assert batch_res["records_processed"] == 2


def test_nosql_storage_fallback():
    """Verify NoSQL storage gracefully saves and retrieves records."""
    status = nosql_storage.get_status()
    assert "storage_type" in status

    rec = {
        "original_text": "Unit test tweet text",
        "sentiment": "positive",
        "confidence": 0.88,
        "probabilities": {"negative": 0.12, "positive": 0.88},
    }
    saved = nosql_storage.save_prediction(rec)
    assert saved is True

    recent = nosql_storage.get_recent_predictions(limit=5)
    assert len(recent) >= 1
    assert any(r["text"] == "Unit test tweet text" for r in recent)
