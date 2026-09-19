#!/usr/bin/env bash
###############################################################################
# run.sh — one-command launch for the Multimodal E-Commerce AI Pipeline.
#
# Usage:
#   bash run.sh                # full pipeline run + dashboard
#   bash run.sh --no-dashboard # run the pipeline only, skip Streamlit
#   bash run.sh --reset        # wipe the SQLite DB and re-ingest from scratch
#
# What it does:
#   1. Creates/activates a local .venv (idempotent — safe to re-run).
#   2. Installs requirements.txt (skipped if already satisfied).
#   3. Copies .env.example -> .env on first run (edit it to add real API
#      keys; without them the pipeline automatically uses mock clients).
#   4. Runs ingestion -> AI enrichment -> video generation -> data quality.
#   5. Launches the Streamlit dashboard at http://localhost:8501.
###############################################################################
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

RESET_FLAG=""
LAUNCH_DASHBOARD=1
for arg in "$@"; do
  case "$arg" in
    --reset) RESET_FLAG="--reset" ;;
    --no-dashboard) LAUNCH_DASHBOARD=0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

# ---- 1. Virtual environment -------------------------------------------------
if [ ! -d ".venv" ]; then
  log "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 2. Dependencies --------------------------------------------------------
log "Installing dependencies (this is skipped instantly if already satisfied)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ---- 3. Environment file ----------------------------------------------------
if [ ! -f ".env" ]; then
  log "No .env found — copying .env.example -> .env (edit it to add real API keys)."
  cp .env.example .env
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

# ---- 4. Pipeline stages ------------------------------------------------------
log "Stage 1/4: Ingestion"
python -m src.ingestion ${RESET_FLAG}

log "Stage 2/4: AI Enrichment"
python -m src.ai_pipeline

log "Stage 3/4: Video Generation"
python -m src.video_generator

log "Stage 4/4: Data Quality Checks"
python -m src.quality_checks --stage all

log "Pipeline run complete. Data written to data/raw_warehouse.db"

# ---- 5. Dashboard -------------------------------------------------------------
if [ "$LAUNCH_DASHBOARD" -eq 1 ]; then
  log "Launching Streamlit dashboard -> http://localhost:8501"
  streamlit run app.py
else
  log "Skipping dashboard (--no-dashboard passed). Run 'streamlit run app.py' to view it."
fi
