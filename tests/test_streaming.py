"""
Unit tests for Apache Flink Real-Time Stream Processing Module.
"""

import pytest
from src.streaming.sample_stream import create_stream_event, get_sample_tweet_records
from src.streaming.flink_pipeline import (
    validate_stream_record,
    normalize_stream_record,
    enrich_sentiment_and_absa,
    FlinkStreamSimulator,
    FileStreamSink,
)


def test_create_stream_event():
    ev = create_stream_event(text="Great product!", user="@tester")
    assert ev["user"] == "@tester"
    assert ev["text"] == "Great product!"
    assert "timestamp" in ev
    assert "event_time_epoch_ms" in ev


def test_stream_pipeline_stages():
    raw_event = {"tweet_id": "test_1", "user": "@user1", "text": "The camera is excellent, but battery is bad."}

    # 1. Validation
    valid = validate_stream_record(raw_event)
    assert valid is not None
    assert valid["tweet_id"] == "test_1"

    # 2. Normalization
    norm = normalize_stream_record(valid)
    assert "clean_text" in norm
    assert len(norm["clean_text"]) > 0

    # 3. Sentiment + ABSA Enrichment
    enriched = enrich_sentiment_and_absa(norm)
    assert "predicted_sentiment" in enriched
    assert "aspects" in enriched
    assert len(enriched["aspects"]) >= 1


def test_flink_stream_simulator(tmp_path):
    output_file = tmp_path / "test_stream_out.jsonl"
    sink = FileStreamSink(output_path=output_file)
    simulator = FlinkStreamSimulator(sinks=[sink], tumbling_window_seconds=2)

    sample_events = get_sample_tweet_records(count=4)
    summary = simulator.run_stream(event_generator=iter(sample_events), max_events=4, delay_seconds=0.0)

    assert summary["status"] == "completed"
    assert summary["events_processed"] == 4
    assert summary["throughput_eps"] > 0
    assert output_file.exists()
