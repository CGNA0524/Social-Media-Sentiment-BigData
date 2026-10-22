"""
Modular Airflow Task Callables for Social Media Sentiment Pipeline.

Implements the 7 core stages required by the orchestration pipeline:
1. Dataset ingestion and validation
2. Data cleaning and preprocessing
3. Sentiment prediction
4. Aspect-based sentiment analysis (ABSA)
5. Elasticsearch indexing
6. Analytics and report generation
7. Pipeline completion and logging

Key Features:
- Tasks are purely decoupled and independently testable
- Heavy models are ONLY loaded when a specific task is actively executed (DAG parsing is instantaneous)
- Safe sample-batch processing (configurable via AIRFLOW_BATCH_SIZE or runtime args)
- Idempotent and non-destructive
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data_processing.load_dataset import get_project_root, load_sentiment140, load_airline_tweets
from src.data_processing.preprocess import clean_text
from src.models.predict import predict_text
from src.models.absa_engine import analyze_aspect_sentiment
from src.search.elasticsearch_client import is_elasticsearch_available, get_elasticsearch_client
from src.search.search_service import ElasticsearchSearchService

logger = logging.getLogger("AirflowPipelineTasks")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = get_project_root()
DEFAULT_BATCH_SIZE = int(os.getenv("AIRFLOW_BATCH_SIZE", "50"))
INTERMEDIATE_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


# =====================================================================
# Task 1: Ingestion & Validation
# =====================================================================

def run_ingest_and_validate(sample_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
    """
    Stage 1: Ingests a controlled sample batch and validates schema.
    """
    logger.info(f"[Stage 1] Ingesting dataset sample (size={sample_size})...")
    start_time = time.perf_counter()

    try:
        # Load sample from raw Sentiment140 or Airline tweets
        df = load_airline_tweets(sample_size=sample_size)
        if "tweet_id" in df.columns and "id" not in df.columns:
            df["id"] = df["tweet_id"]
        if "user" not in df.columns:
            df["user"] = df.get("airline", "@airline_user")

        required_cols = {"id", "text", "sentiment"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Schema validation failed: Missing required columns {missing}")

        # Remove nulls or blank text
        valid_df = df.dropna(subset=["text"]).copy()
        valid_df = valid_df[valid_df["text"].str.strip().str.len() > 0]

        records = valid_df[["id", "text", "sentiment", "user"]].to_dict(orient="records")

        # Save intermediate snapshot for stage 2
        INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage1_ingested.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(records, f)

        duration = round(time.perf_counter() - start_time, 3)
        return {
            "status": "success",
            "stage": "ingest_and_validate",
            "record_count": len(records),
            "duration_s": duration,
            "snapshot_path": str(snapshot_file),
            "sample_records": records[:3],
        }
    except Exception as e:
        logger.error(f"[Stage 1] Ingestion failed: {e}")
        return {"status": "failed", "stage": "ingest_and_validate", "error": str(e)}


# =====================================================================
# Task 2: Data Cleaning & Preprocessing
# =====================================================================

def run_clean_and_preprocess(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Stage 2: Cleans raw text while preserving negation words.
    """
    logger.info("[Stage 2] Cleaning and preprocessing records...")
    start_time = time.perf_counter()

    if records is None:
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage1_ingested.json"
        if not snapshot_file.exists():
            return {"status": "failed", "stage": "clean_and_preprocess", "error": "No input records provided."}
        with open(snapshot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    cleaned_records = []
    for rec in records:
        r = dict(rec)
        cleaned = clean_text(r["text"], remove_stopwords=True)
        r["clean_text"] = cleaned if cleaned.strip() else r["text"]
        cleaned_records.append(r)

    snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage2_cleaned.json"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f)

    duration = round(time.perf_counter() - start_time, 3)
    return {
        "status": "success",
        "stage": "clean_and_preprocess",
        "record_count": len(cleaned_records),
        "duration_s": duration,
        "snapshot_path": str(snapshot_file),
        "sample_records": cleaned_records[:3],
    }


# =====================================================================
# Task 3: Sentiment Prediction
# =====================================================================

def run_predict_sentiment(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Stage 3: Applies machine learning classification for sentiment scoring.
    """
    logger.info("[Stage 3] Predicting sentiment for records...")
    start_time = time.perf_counter()

    if records is None:
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage2_cleaned.json"
        if not snapshot_file.exists():
            return {"status": "failed", "stage": "predict_sentiment", "error": "Input records missing."}
        with open(snapshot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    predicted_records = []
    for rec in records:
        r = dict(rec)
        pred = predict_text(r["text"])
        r["predicted_sentiment"] = pred["sentiment"]
        r["confidence"] = pred["confidence"]
        r["model_name"] = "Multinomial Naive Bayes (Baseline)"
        predicted_records.append(r)

    snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage3_predicted.json"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        json.dump(predicted_records, f)

    duration = round(time.perf_counter() - start_time, 3)
    return {
        "status": "success",
        "stage": "predict_sentiment",
        "record_count": len(predicted_records),
        "duration_s": duration,
        "snapshot_path": str(snapshot_file),
        "sample_records": predicted_records[:3],
    }


# =====================================================================
# Task 4: Aspect-Based Sentiment Analysis (ABSA)
# =====================================================================

def run_aspect_sentiment_analysis(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Stage 4: Extracts opinion target aspects and evaluates per-aspect sentiments.
    """
    logger.info("[Stage 4] Performing Aspect-Based Sentiment Analysis (ABSA)...")
    start_time = time.perf_counter()

    if records is None:
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage3_predicted.json"
        if not snapshot_file.exists():
            return {"status": "failed", "stage": "aspect_sentiment_analysis", "error": "Input records missing."}
        with open(snapshot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    absa_records = []
    total_aspects_extracted = 0

    for rec in records:
        r = dict(rec)
        absa_res = analyze_aspect_sentiment(r["text"])
        r["aspects"] = absa_res["aspects"]
        r["aspect_count"] = absa_res["aspect_count"]
        r["overall_aspect_sentiment"] = absa_res["overall_sentiment"]
        total_aspects_extracted += absa_res["aspect_count"]
        absa_records.append(r)

    snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage4_absa.json"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        json.dump(absa_records, f)

    duration = round(time.perf_counter() - start_time, 3)
    return {
        "status": "success",
        "stage": "aspect_sentiment_analysis",
        "record_count": len(absa_records),
        "total_aspects_found": total_aspects_extracted,
        "duration_s": duration,
        "snapshot_path": str(snapshot_file),
        "sample_records": absa_records[:3],
    }


# =====================================================================
# Task 5: Elasticsearch Indexing
# =====================================================================

def run_elasticsearch_indexing(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Stage 5: Indexes enriched records into Elasticsearch, or writes local fallback file.
    """
    logger.info("[Stage 5] Indexing records into Elasticsearch...")
    start_time = time.perf_counter()

    if records is None:
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage4_absa.json"
        if not snapshot_file.exists():
            return {"status": "failed", "stage": "elasticsearch_indexing", "error": "Input records missing."}
        with open(snapshot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    es_online = is_elasticsearch_available()
    indexed_count = 0
    destination = "local_jsonl_fallback"

    if es_online:
        client = get_elasticsearch_client()
        service = ElasticsearchSearchService(client)
        service.ensure_index()
        # Format records for ES index
        es_docs = []
        for r in records:
            es_docs.append({
                "tweet_id": str(r.get("id", "")),
                "user": r.get("user", "@unknown"),
                "original_text": r.get("text", ""),
                "clean_text": r.get("clean_text", ""),
                "predicted_sentiment": r.get("predicted_sentiment", "neutral"),
                "model_name": r.get("model_name", "Baseline"),
                "confidence": r.get("confidence", 0.5),
                "aspects": r.get("aspects", []),
                "source_dataset": "airflow_pipeline",
            })
        res = service.index_batch(es_docs)
        indexed_count = res.get("indexed_count", 0)
        destination = "elasticsearch_cluster"
    else:
        # Fallback to local JSONL sink
        fallback_file = INTERMEDIATE_DATA_DIR / "airflow_elasticsearch_indexed_records.jsonl"
        with open(fallback_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        indexed_count = len(records)
        destination = str(fallback_file)

    duration = round(time.perf_counter() - start_time, 3)
    return {
        "status": "success",
        "stage": "elasticsearch_indexing",
        "destination": destination,
        "elasticsearch_connected": es_online,
        "indexed_count": indexed_count,
        "duration_s": duration,
    }


# =====================================================================
# Task 6: Analytics & Report Generation
# =====================================================================

def run_generate_analytics_report(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Stage 6: Aggregates sentiments, computes summary KPIs, and outputs report.
    """
    logger.info("[Stage 6] Generating analytics report...")
    start_time = time.perf_counter()

    if records is None:
        snapshot_file = INTERMEDIATE_DATA_DIR / "airflow_stage4_absa.json"
        if not snapshot_file.exists():
            return {"status": "failed", "stage": "generate_analytics_report", "error": "Input records missing."}
        with open(snapshot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    sentiment_counts = {}
    aspect_counts = {}
    total = len(records)

    for r in records:
        sent = r.get("predicted_sentiment", "unknown")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

        for asp_item in r.get("aspects", []):
            asp_term = asp_item.get("aspect", "")
            aspect_counts[asp_term] = aspect_counts.get(asp_term, 0) + 1

    top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records_processed": total,
        "sentiment_distribution": sentiment_counts,
        "sentiment_ratios": {
            k: round(v / max(1, total), 4) for k, v in sentiment_counts.items()
        },
        "top_aspects_mentioned": dict(top_aspects),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "airflow_pipeline_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    duration = round(time.perf_counter() - start_time, 3)
    return {
        "status": "success",
        "stage": "generate_analytics_report",
        "report_path": str(report_file),
        "summary": report,
        "duration_s": duration,
    }


# =====================================================================
# Task 7: Completion & Pipeline Logging
# =====================================================================

def run_pipeline_completion_log(summary_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Stage 7: Finalizes pipeline execution, logs audit trail and telemetry.
    """
    logger.info("[Stage 7] Finalizing pipeline run and recording completion audit...")
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": "social_media_sentiment_airflow_dag",
        "status": "SUCCESS",
        "stages_executed": 7,
        "execution_engine": "Apache Airflow Workflow Runner",
        "details": summary_metadata or {},
    }

    log_file = INTERMEDIATE_DATA_DIR / "pipeline_execution_audit.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "status": "success",
        "stage": "pipeline_completion_log",
        "audit_log": str(log_file),
        "completed_at": log_entry["timestamp"],
    }


# =====================================================================
# Standalone Pipeline Runner (For Windows / Test Execution)
# =====================================================================

class AirflowPipelineRunner:
    """
    Executes the 7 DAG tasks sequentially in Python.
    Enables local testing, demonstration, and validation without an active Airflow scheduler daemon.
    """

    def __init__(self, sample_size: int = DEFAULT_BATCH_SIZE):
        self.sample_size = sample_size
        self.stage_results: Dict[str, Any] = {}

    def run_all_stages(self) -> Dict[str, Any]:
        """Execute all 7 stages sequentially and collect telemetry."""
        total_start = time.perf_counter()

        # Stage 1: Ingest & Validate
        res1 = run_ingest_and_validate(sample_size=self.sample_size)
        self.stage_results["stage1_ingest"] = res1
        if res1["status"] != "success":
            return {"status": "failed", "failed_stage": 1, "details": res1}

        # Stage 2: Clean & Preprocess
        res2 = run_clean_and_preprocess()
        self.stage_results["stage2_clean"] = res2
        if res2["status"] != "success":
            return {"status": "failed", "failed_stage": 2, "details": res2}

        # Stage 3: Predict Sentiment
        res3 = run_predict_sentiment()
        self.stage_results["stage3_predict"] = res3
        if res3["status"] != "success":
            return {"status": "failed", "failed_stage": 3, "details": res3}

        # Stage 4: Aspect-Based Sentiment Analysis
        res4 = run_aspect_sentiment_analysis()
        self.stage_results["stage4_absa"] = res4
        if res4["status"] != "success":
            return {"status": "failed", "failed_stage": 4, "details": res4}

        # Stage 5: Elasticsearch Indexing
        res5 = run_elasticsearch_indexing()
        self.stage_results["stage5_elasticsearch"] = res5
        if res5["status"] != "success":
            return {"status": "failed", "failed_stage": 5, "details": res5}

        # Stage 6: Analytics Report
        res6 = run_generate_analytics_report()
        self.stage_results["stage6_analytics"] = res6
        if res6["status"] != "success":
            return {"status": "failed", "failed_stage": 6, "details": res6}

        # Stage 7: Completion Log
        res7 = run_pipeline_completion_log(summary_metadata={
            "records": self.sample_size,
            "report": res6.get("report_path"),
        })
        self.stage_results["stage7_completion"] = res7

        total_duration = round(time.perf_counter() - total_start, 3)
        return {
            "status": "success",
            "pipeline_name": "social_media_sentiment_pipeline",
            "total_duration_s": total_duration,
            "stages_completed": 7,
            "stages": self.stage_results,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 7-Stage Social Media Sentiment Pipeline")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of records to process")
    args = parser.parse_args()

    print(f"\n==========================================================")
    print(f"Executing 7-Stage Big Data Airflow Pipeline (Sample Size: {args.sample_size})")
    print(f"==========================================================\n")

    runner = AirflowPipelineRunner(sample_size=args.sample_size)
    summary = runner.run_all_stages()

    print(f"\nPipeline Status : {summary['status'].upper()}")
    print(f"Total Duration  : {summary.get('total_duration_s')} seconds")
    print(f"Stages Executed : {summary.get('stages_completed')} / 7\n")
    for stage_key, result in summary.get("stages", {}).items():
        print(f"  * {stage_key:<25}: Status={result.get('status')} | Duration={result.get('duration_s', 0)}s")
    print("\nAnalytics Summary:")
    print(json.dumps(summary["stages"]["stage6_analytics"].get("summary", {}), indent=2))
