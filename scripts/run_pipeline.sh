#!/usr/bin/env bash
# Local end-to-end pipeline run (no Airflow required).
# Usage: bash scripts/run_pipeline.sh
set -euo pipefail

echo "== [1/4] Ingestion: generating mock products & reviews =="
python -m ingestion.mock_data_generator

echo "== [2/4] AI Enrichment: sentiment, key features, promo concepts =="
python -m processing.llm_enrichment

echo "== [3/4] Video Generation: triggering promo videos (sentiment > threshold) =="
python -m video_generation.higgsfield_client

echo "== [4/4] Warehouse Load: writing to SQLite/Postgres =="
python -m warehouse.loader

echo "Pipeline run complete."
