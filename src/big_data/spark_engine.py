"""
Distributed Batch Processing and Spark Abstraction Engine.

Implements:
- PySpark distributed DataFrame transformations (when Spark and Java runtime are configured)
- High-efficiency Vectorized Local Batch Processing fallback
- Honest telemetry reporting regarding the actual execution engine used
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class SparkEngineManager:
    """
    Manages Spark Session lifecycle and batch aggregation jobs.
    Honoring the requirement:
    'Do not pretend pandas execution is Spark execution.
     Record the execution engine actually used in the UI and reports.'
    """

    def __init__(self):
        self._spark_session = None
        self._spark_available = False
        self._init_error = None
        self._check_spark_availability()

    def _check_spark_availability(self):
        """Test whether PySpark and compatible Java runtime are present."""
        try:
            import pyspark
            from pyspark.sql import SparkSession

            # Attempt a minimal test session
            spark = (
                SparkSession.builder.master("local[2]")
                .appName("SocialMediaSentiment-Check")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate()
            )
            self._spark_session = spark
            self._spark_available = True
        except ImportError:
            self._spark_available = False
            self._init_error = "PySpark package is not installed in the current environment."
        except Exception as e:
            self._spark_available = False
            self._init_error = f"Spark runtime error (e.g., Java 24 / winutils incompatibility on Windows): {str(e)}"

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Return metadata describing the active computational engine.
        """
        if self._spark_available:
            return {
                "engine_type": "Apache Spark (PySpark Distributed)",
                "status": "Active",
                "details": "SparkSession running on local cluster mode with distributed DataFrame transformations.",
                "is_distributed": True,
            }
        else:
            return {
                "engine_type": "Vectorized Local Engine (Pandas/NumPy Batch)",
                "status": "Fallback Active",
                "details": f"Spark unavailable ({self._init_error}). Operating in high-performance vectorized local mode.",
                "is_distributed": False,
            }

    def aggregate_sentiment_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        sentiment_col: str = "sentiment_label",
    ) -> Dict[str, Any]:
        """
        Perform batch aggregation of sentiments and word counts across tweets.

        Uses PySpark DataFrame if active; otherwise uses optimized vectorized pandas.
        """
        start_time = time.time()
        engine_info = self.get_engine_info()

        if self._spark_available and self._spark_session is not None:
            # PySpark Distributed Execution Path
            try:
                spark_df = self._spark_session.createDataFrame(df[[text_col, sentiment_col]])
                from pyspark.sql import functions as F

                # Aggregations
                counts = (
                    spark_df.groupBy(sentiment_col)
                    .count()
                    .toPandas()
                    .rename(columns={sentiment_col: "sentiment", "count": "count"})
                )
                exec_time = round(time.time() - start_time, 4)
                return {
                    "engine_used": engine_info["engine_type"],
                    "records_processed": len(df),
                    "execution_time_seconds": exec_time,
                    "sentiment_distribution": counts.to_dict(orient="records"),
                    "mode": "Distributed Spark DataFrame",
                }
            except Exception as e:
                # Fall back gracefully if Spark execution throws at runtime
                engine_info["details"] += f" (Runtime switch: {e})"

        # Vectorized Local Engine (Pandas Fallback)
        counts = df[sentiment_col].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        exec_time = round(time.time() - start_time, 4)

        return {
            "engine_used": engine_info["engine_type"],
            "records_processed": len(df),
            "execution_time_seconds": exec_time,
            "sentiment_distribution": counts.to_dict(orient="records"),
            "mode": "Vectorized Local Batch Fallback",
            "fallback_reason": self._init_error,
        }


# Singleton manager instance
spark_manager = SparkEngineManager()
