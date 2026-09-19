#!/usr/bin/env bash
# One-time local Airflow setup: init DB, create admin user, symlink DAG.
# Usage: AIRFLOW_HOME=$(pwd)/.airflow bash scripts/setup_airflow.sh
set -euo pipefail

: "${AIRFLOW_HOME:?Set AIRFLOW_HOME before running this script}"

airflow db init

airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

mkdir -p "${AIRFLOW_HOME}/dags"
ln -sf "$(pwd)/orchestration/airflow_dags/ecommerce_pipeline_dag.py" \
  "${AIRFLOW_HOME}/dags/ecommerce_pipeline_dag.py"

echo "Airflow initialized. Start with:"
echo "  airflow webserver --port 8080 &"
echo "  airflow scheduler &"
