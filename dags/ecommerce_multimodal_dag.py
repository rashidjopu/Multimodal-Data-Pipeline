"""
dags/ecommerce_multimodal_dag.py
====================================

Apache Airflow DAG orchestrating the Multimodal E-Commerce AI Pipeline:

    run_ingestion >> run_ai_enrichment >> trigger_video_generation
                                              >> run_data_quality_checks >> pipeline_success

Error-handling design:
  - Every task's Python callable raises `AirflowException` on a hard failure
    (rather than letting an unrelated exception type bubble up), so Airflow
    marks the task state clearly and logs a readable reason.
  - `run_data_quality_checks` is the final gate: it calls
    `src.quality_checks.run_post_pipeline_checks()` and raises
    `AirflowException` if any CRITICAL check fails. Because `pipeline_success`
    depends on it with the default `all_success` trigger rule, Airflow
    automatically SKIPS `pipeline_success` (and any further downstream tasks
    you add later, e.g. a warehouse-load or notification task) instead of
    running them against bad data — this is the "abort gracefully" behavior.
  - `alert_on_pipeline_failure` uses `trigger_rule=ONE_FAILED` so it fires
    if *any* upstream task fails, giving you a single place to wire real
    alerting (Slack/PagerDuty/email) without duplicating logic in every task.
  - `run_ingestion` also calls `run_pre_flight_checks()` first (data quality
    "before" the pipeline runs), matching the design of
    `src/quality_checks.py`.

Environment variables (see .env.example):
    PIPELINE_DB_PATH             default: <project_root>/data/raw_warehouse.db
    SENTIMENT_VIDEO_THRESHOLD    default: 0.8
    PIPELINE_SCHEDULE_CRON       default: "0 2 * * *"  (daily at 02:00)
    LLM_PROVIDER, ANTHROPIC_API_KEY, OPENAI_API_KEY   -> AI enrichment stage
    HIGGSFIELD_API_KEY, HIGGSFIELD_API_BASE_URL       -> video generation stage
    FORCE_MOCK_LLM / FORCE_MOCK_HIGGSFIELD ("true"/"false") -> force mock
        clients even if API keys ARE set (useful for staging/demo runs).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Make `src/` importable regardless of where AIRFLOW_HOME/dags actually
# lives on disk. Assumes this file sits at <project_root>/dags/<this file>.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "raw_warehouse.db"


def _db_path() -> Path:
    return Path(os.getenv("PIPELINE_DB_PATH", str(DEFAULT_DB_PATH)))


def _sentiment_threshold() -> float:
    return float(os.getenv("SENTIMENT_VIDEO_THRESHOLD", "0.8"))


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# Task callables
# --------------------------------------------------------------------------
def _run_ingestion(**context) -> None:
    """Task 1: run pre-flight data-quality checks, then ingest mock raw data."""
    from src.ingestion import run as run_ingestion_stage
    from src.quality_checks import run_pre_flight_checks

    db_path = _db_path()

    logger.info("Running pre-flight checks before ingestion...")
    pre_report = run_pre_flight_checks(db_path=db_path)
    if not pre_report.passed:
        failed = [r.name for r in pre_report.critical_failures]
        raise AirflowException(f"Pre-flight data quality checks failed: {failed}. Aborting before ingestion.")

    run_ingestion_stage(db_path=db_path, reset=False)
    context["ti"].xcom_push(key="db_path", value=str(db_path))


def _run_ai_enrichment(**context) -> None:
    """Task 2: batch-enrich raw reviews via the LLM and persist to enriched_reviews."""
    from src.ai_pipeline import enrich_product_reviews

    db_path = _db_path()
    force_mock = _env_flag("FORCE_MOCK_LLM", default=False)

    results = enrich_product_reviews(db_path=db_path, force_mock=force_mock)

    if not results:
        raise AirflowException(
            "AI enrichment produced zero validated EnrichedAIOutput records. "
            "Check the dead-letter file (data/processed/enrichment_dead_letter.jsonl) "
            "and the configured LLM provider credentials."
        )

    context["ti"].xcom_push(key="enriched_count", value=len(results))
    logger.info("AI enrichment succeeded for %d product(s).", len(results))


def _trigger_video_generation(**context) -> None:
    """Task 3: trigger + poll Higgsfield promo videos for high-sentiment products."""
    from src.video_generator import generate_promo_videos

    db_path = _db_path()
    threshold = _sentiment_threshold()
    force_mock = _env_flag("FORCE_MOCK_HIGGSFIELD", default=False)

    videos = generate_promo_videos(db_path=db_path, sentiment_threshold=threshold, force_mock=force_mock)

    # NOTE: zero videos can be a legitimate outcome (no product cleared the
    # sentiment threshold this run) so we do NOT fail the task on an empty
    # list — the downstream data-quality task is what decides whether that's
    # actually a problem (e.g. it *would* flag zero videos as CRITICAL if
    # there WERE high-sentiment candidates but none produced a video row).
    context["ti"].xcom_push(key="videos_triggered", value=len(videos))
    logger.info("Video generation stage complete: %d video(s) triggered.", len(videos))


def _run_data_quality_checks(**context) -> None:
    """Task 4: comprehensive post-pipeline data quality gate. Aborts downstream on CRITICAL failure."""
    from src.quality_checks import run_post_pipeline_checks

    db_path = _db_path()
    threshold = _sentiment_threshold()

    report = run_post_pipeline_checks(db_path=db_path, sentiment_threshold=threshold)

    if not report.passed:
        failed_names = [r.name for r in report.critical_failures]
        raise AirflowException(
            f"Data quality gate FAILED with {len(failed_names)} critical check(s): {failed_names}. "
            "Downstream tasks (e.g. warehouse load, notifications) will be skipped."
        )

    context["ti"].xcom_push(key="dq_passed", value=True)
    logger.info("Data quality gate PASSED. Pipeline run is clean.")


def _alert_on_pipeline_failure(**context) -> None:
    """
    Fires whenever ANY upstream task in this DAG fails (trigger_rule=ONE_FAILED).
    Replace the logger.error call with a real Slack/PagerDuty/email integration.
    """
    dag_run = context["dag_run"]
    failed_tasks = [ti.task_id for ti in dag_run.get_task_instances() if ti.state == "failed"]
    logger.error(
        "PIPELINE ALERT: run_id=%s failed_tasks=%s. Investigate before the next scheduled run.",
        dag_run.run_id,
        failed_tasks,
    )
    # e.g. slack_webhook.send(f"Pipeline failed: {failed_tasks}")


def _task_failure_callback(context) -> None:
    """default_args on_failure_callback: lightweight per-task failure log line."""
    task_instance = context["task_instance"]
    logger.error(
        "Task '%s' failed in DAG '%s' (run_id=%s). Exception: %s",
        task_instance.task_id,
        task_instance.dag_id,
        context["dag_run"].run_id,
        context.get("exception"),
    )


# --------------------------------------------------------------------------
# DAG definition
# --------------------------------------------------------------------------
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _task_failure_callback,
}

with DAG(
    dag_id="ecommerce_multimodal_pipeline",
    description="Ingestion -> AI Enrichment -> Video Generation -> Data Quality Gate",
    default_args=default_args,
    schedule=os.getenv("PIPELINE_SCHEDULE_CRON", "0 2 * * *"),
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "ai", "multimodal", "video-gen"],
) as dag:

    run_ingestion = PythonOperator(
        task_id="run_ingestion",
        python_callable=_run_ingestion,
    )

    run_ai_enrichment = PythonOperator(
        task_id="run_ai_enrichment",
        python_callable=_run_ai_enrichment,
    )

    trigger_video_generation = PythonOperator(
        task_id="trigger_video_generation",
        python_callable=_trigger_video_generation,
    )

    run_data_quality_checks = PythonOperator(
        task_id="run_data_quality_checks",
        python_callable=_run_data_quality_checks,
    )

    # Only reached if run_data_quality_checks passed (default trigger_rule=ALL_SUCCESS).
    # Add a real warehouse-load / notification task downstream of this one later —
    # it will inherit the same "skip on upstream failure" protection for free.
    pipeline_success = EmptyOperator(
        task_id="pipeline_success",
    )

    # Fires if ANY of the four main tasks fail, regardless of where in the chain.
    alert_on_pipeline_failure = PythonOperator(
        task_id="alert_on_pipeline_failure",
        python_callable=_alert_on_pipeline_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ---- main linear chain -----------------------------------------------
    run_ingestion >> run_ai_enrichment >> trigger_video_generation >> run_data_quality_checks >> pipeline_success

    # ---- failure-alert fan-in: watches every main task -------------------
    [run_ingestion, run_ai_enrichment, trigger_video_generation, run_data_quality_checks] >> alert_on_pipeline_failure
