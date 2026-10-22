"""
Apache Airflow DAG Definition for Social Media Sentiment Big Data Pipeline.

Schedule: @daily (or manual trigger)
Orchestrates:
1. task_ingest_and_validate
2. task_clean_and_preprocess
3. task_predict_sentiment
4. task_aspect_sentiment_analysis
5. task_elasticsearch_indexing
6. task_generate_analytics_report
7. task_pipeline_completion_log

Task Dependencies:
task_ingest_and_validate >> task_clean_and_preprocess >> task_predict_sentiment >> task_aspect_sentiment_analysis >> task_elasticsearch_indexing >> task_generate_analytics_report >> task_pipeline_completion_log
"""

from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to sys.path so Airflow worker can resolve src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Airflow import with mock fallback for non-Linux / Windows environments
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False

    # Standalone mock operator for validation when running outside native Airflow
    class PythonOperator:
        def __init__(self, task_id, python_callable, op_kwargs=None, retries=1, retry_delay=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            self.retries = retries
            self.retry_delay = retry_delay
            self.upstream_tasks = []
            self.downstream_tasks = []

        def __rshift__(self, other):
            self.downstream_tasks.append(other)
            other.upstream_tasks.append(self)
            return other

    class DAG:
        def __init__(self, dag_id, default_args=None, schedule_interval=None, catchup=False, description=""):
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.schedule_interval = schedule_interval
            self.catchup = catchup
            self.description = description
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


# Airflow Default Task Arguments
default_args = {
    "owner": "bigdata_admin",
    "depends_on_past": False,
    "email": ["admin@socialsent.local"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
    "start_date": datetime(2026, 1, 1),
}


# =====================================================================
# DAG Definition
# =====================================================================

def create_dag() -> DAG:
    """Construct and configure the Airflow DAG."""
    dag = DAG(
        dag_id="social_media_sentiment_pipeline",
        default_args=default_args,
        description="End-to-end Big Data Sentiment Analysis and ABSA Pipeline",
        schedule_interval="@daily",
        catchup=False,
    )

    # Delay heavy imports inside task wrappers to guarantee sub-second DAG parsing time
    def _task1(**kwargs):
        from src.orchestration.pipeline_tasks import run_ingest_and_validate
        return run_ingest_and_validate()

    def _task2(**kwargs):
        from src.orchestration.pipeline_tasks import run_clean_and_preprocess
        return run_clean_and_preprocess()

    def _task3(**kwargs):
        from src.orchestration.pipeline_tasks import run_predict_sentiment
        return run_predict_sentiment()

    def _task4(**kwargs):
        from src.orchestration.pipeline_tasks import run_aspect_sentiment_analysis
        return run_aspect_sentiment_analysis()

    def _task5(**kwargs):
        from src.orchestration.pipeline_tasks import run_elasticsearch_indexing
        return run_elasticsearch_indexing()

    def _task6(**kwargs):
        from src.orchestration.pipeline_tasks import run_generate_analytics_report
        return run_generate_analytics_report()

    def _task7(**kwargs):
        from src.orchestration.pipeline_tasks import run_pipeline_completion_log
        return run_pipeline_completion_log()

    # Define the 7 Operators
    t1 = PythonOperator(task_id="task_ingest_and_validate", python_callable=_task1, dag=dag)
    t2 = PythonOperator(task_id="task_clean_and_preprocess", python_callable=_task2, dag=dag)
    t3 = PythonOperator(task_id="task_predict_sentiment", python_callable=_task3, dag=dag)
    t4 = PythonOperator(task_id="task_aspect_sentiment_analysis", python_callable=_task4, dag=dag)
    t5 = PythonOperator(task_id="task_elasticsearch_indexing", python_callable=_task5, dag=dag)
    t6 = PythonOperator(task_id="task_generate_analytics_report", python_callable=_task6, dag=dag)
    t7 = PythonOperator(task_id="task_pipeline_completion_log", python_callable=_task7, dag=dag)

    # Set Strict Linear Task Dependencies
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7

    return dag


# Expose DAG object for Airflow discovery
sentiment_dag = create_dag()
