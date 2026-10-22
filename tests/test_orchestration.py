"""
Unit tests for Apache Airflow Orchestration and Pipeline Tasks.
"""

import pytest
from src.orchestration.dags.social_media_sentiment_pipeline import sentiment_dag
from src.orchestration.pipeline_tasks import (
    run_ingest_and_validate,
    run_clean_and_preprocess,
    run_predict_sentiment,
    run_aspect_sentiment_analysis,
    run_elasticsearch_indexing,
    run_generate_analytics_report,
    run_pipeline_completion_log,
)


def test_airflow_dag_structure():
    assert sentiment_dag.dag_id == "social_media_sentiment_pipeline"
    assert sentiment_dag.schedule_interval == "@daily"
    assert sentiment_dag.catchup is False


def test_independent_pipeline_tasks():
    # 1. Ingest
    res1 = run_ingest_and_validate(sample_size=5)
    assert res1["status"] == "success"
    assert res1["record_count"] == 5

    # 2. Clean
    res2 = run_clean_and_preprocess()
    assert res2["status"] == "success"

    # 3. Predict
    res3 = run_predict_sentiment()
    assert res3["status"] == "success"

    # 4. ABSA
    res4 = run_aspect_sentiment_analysis()
    assert res4["status"] == "success"

    # 5. Indexing
    res5 = run_elasticsearch_indexing()
    assert res5["status"] == "success"

    # 6. Analytics
    res6 = run_generate_analytics_report()
    assert res6["status"] == "success"
    assert "sentiment_distribution" in res6["summary"]

    # 7. Completion
    res7 = run_pipeline_completion_log()
    assert res7["status"] == "success"
