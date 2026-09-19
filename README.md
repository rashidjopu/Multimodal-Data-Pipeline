# Multimodal E-Commerce Data & AI Pipeline

A production-grade, end-to-end pipeline that ingests e-commerce product/review
data, enriches it with LLM-derived structured insights (sentiment, key
features, promo concepts), auto-triggers 5-second AI promo videos for
high-sentiment products via the Higgsfield AI API, validates data quality at
every boundary, and lands everything in a queryable data warehouse — all
orchestrated by Apache Airflow.

---

## 1. Architecture Overview

The pipeline is a five-stage DAG. Each stage reads a validated contract from
the stage before it and writes a validated contract for the stage after it,
so failures are caught at the boundary rather than propagating downstream.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         MULTIMODAL E-COMMERCE PIPELINE                     │
└────────────────────────────────────────────────────────────────────────────┘

  [1] INGESTION                [2] AI ENRICHMENT             [3] VIDEO GEN
 ┌──────────────────┐        ┌───────────────────────┐    ┌────────────────────┐
 │ Mock Data Gen     │        │ Batch LLM Calls        │    │ Higgsfield Client   │
 │ (Faker)           │  raw   │ (OpenAI / Claude API)  │ AI │ (real or mock)      │
 │ • products.csv    │──────▶ │ • sentiment_score       │──▶│ • poll job status   │
 │ • reviews.json    │  JSON  │ • top_key_features      │  │ • download video    │
 └──────────────────┘        │ • visual_promo_concept │gate│   asset if score    │
         │                    │ Pydantic-validated I/O │>0.8│   > 0.8             │
         │ GE checkpoint       └───────────────────────┘    └────────────────────┘
         ▼                              │                             │
 ┌──────────────────┐                   ▼                             ▼
 │ Great Expectations│        ┌───────────────────────┐    ┌────────────────────┐
 │ raw schema suite   │        │ GE enrichment suite    │    │ GE video-metadata   │
 └──────────────────┘        └───────────────────────┘    │ suite               │
                                                              └────────────────────┘
                                          │                             │
                                          └─────────────┬───────────────┘
                                                         ▼
                                          [4] DATA WAREHOUSE LOADING
                                    ┌───────────────────────────────────┐
                                    │ SQLAlchemy ORM → SQLite/Postgres    │
                                    │  • dim_products                    │
                                    │  • fact_reviews                    │
                                    │  • fact_ai_enrichment               │
                                    │  • fact_promo_videos                │
                                    │ GE checkpoint on load               │
                                    └───────────────────────────────────┘
                                                         │
                                                         ▼
                                          [5] ORCHESTRATION LAYER
                                    ┌───────────────────────────────────┐
                                    │ Apache Airflow DAG                  │
                                    │ ingest ▸ enrich ▸ gate ▸ video ▸    │
                                    │ dq_check ▸ load ▸ notify           │
                                    │ (Mage.ai pipeline is a drop-in     │
                                    │  alternative — see /orchestration) │
                                    └───────────────────────────────────┘
                                                         │
                                                         ▼
                                    ┌───────────────────────────────────┐
                                    │ Streamlit Dashboard (optional)      │
                                    │ browse products, sentiment,        │
                                    │ generated promo video links         │
                                    └───────────────────────────────────┘
```

### Stage-by-stage responsibilities

| # | Stage | Module | Input contract | Output contract |
|---|-------|--------|-----------------|------------------|
| 1 | **Ingestion** | `src/ingestion.py` | — (hand-authored mock reviews) | `RawReview`, `ProductRecord` (Pydantic) → `data/raw_warehouse.db` |
| 2 | **AI Enrichment** | `src/ai_pipeline.py` | `RawReview` batches, per product | `EnrichedAIOutput` (Pydantic: `sentiment_score`, `key_themes`, `promo_video_prompt`) → `enriched_reviews` table |
| 3 | **Video Generation** | `src/video_generator.py` | `EnrichedAIOutput` where `sentiment_score > 0.8` | `VideoMetadata` (job id, status, final asset URL) → `generated_videos` table |
| 4 | **Orchestration** | `dags/ecommerce_multimodal_dag.py` | — | Airflow DAG sequencing stages 1–3 + the quality gate, with retries and failure alerting |
| 5 | **Data Quality** | `src/quality_checks.py` | rows from every table above | pre-flight + post-pipeline `QualityReport` (pass/fail gate) |

Data quality runs both BEFORE the pipeline starts (`run_pre_flight_checks`)
and AFTER every stage completes (`run_post_pipeline_checks`), using
dependency-free custom Python assertions (no Great Expectations required).
A CRITICAL failure raises `AirflowException` in the DAG, so any task wired
downstream of the quality gate is automatically skipped rather than run
against bad data.

---

## 2. Repository Structure

```
multimodal-ecommerce-pipeline/
├── README.md                          # you are here
├── requirements.txt                   # pinned Python dependencies
├── .env.example                       # required environment variables (copy to .env)
├── Dockerfile                         # image used by docker-compose for pipeline + dashboard
├── docker-compose.yml                 # one-command launch: pipeline run -> dashboard
├── run.sh                             # one-command launch without Docker (venv + pipeline + dashboard)
├── app.py                             # Streamlit dashboard (metrics, analytics table, video gallery)
│
├── src/
│   ├── __init__.py
│   ├── schemas.py                     # Pydantic contracts: ProductRecord, RawReview,
│   │                                   #   EnrichedAIOutput, VideoMetadata (+ VideoStatus enum)
│   ├── ingestion.py                   # Stage 1: mock product/review data -> data/raw_warehouse.db
│   ├── ai_pipeline.py                 # Stage 2: AIPipeline + MockLLMClient -> enriched_reviews table
│   ├── video_generator.py             # Stage 3: HiggsfieldVideoClient -> generated_videos table
│   └── quality_checks.py              # Stage 5: pre-flight + post-pipeline data quality gate
│
├── dags/
│   └── ecommerce_multimodal_dag.py    # Stage 4: Airflow DAG orchestrating stages 1-3 + the DQ gate
│
├── tests/
│   ├── conftest.py                    # sys.path setup so `src` imports resolve under pytest
│   └── test_pipeline.py               # Pydantic schema tests + mocked LLM/Higgsfield client tests
│
├── data/
│   ├── raw_warehouse.db               # SQLite warehouse (created by running the pipeline)
│   ├── raw/ processed/ warehouse/     # reserved for file-based intermediates / alternate DB location
│
├── notebooks/
│   └── exploration.ipynb              # ad-hoc EDA on generated/enriched data
│
├── docs/
│   └── architecture.png               # exported diagram (source: this README's ASCII art)
│
└── logs/                              # runtime logs (gitignored)
```

---

## 3. Prerequisites

- **Python** 3.10 or 3.11 (Airflow 2.9.x does not yet support 3.12)
- **pip** ≥ 23 and, ideally, a virtual environment tool (`venv`, `pyenv`, or `conda`)
- **Docker** (optional but recommended) if you want to run Postgres and/or
  Airflow's full webserver+scheduler stack in containers rather than local mode
- API credentials for whichever providers you enable:
  - `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` for the AI enrichment stage
  - `HIGGSFIELD_API_KEY` + `HIGGSFIELD_API_BASE_URL` for real video generation
    (omit both to fall back to `MockHiggsfieldClient`, which simulates job
    submission, polling, and completion latency without any network calls)
- **SQLite** (bundled with Python — zero setup) for local/dev warehouse mode,
  or a running **PostgreSQL** instance for production mode
- (Optional) **Great Expectations** compatible environment — see the note in
  `requirements.txt` about isolating it from Airflow's dependency set if you
  hit resolver conflicts

---

## 4. Quickstart

### 4.1 Clone & set up the environment

```bash
git clone <your-repo-url> multimodal-ecommerce-pipeline
cd multimodal-ecommerce-pipeline

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Configure environment variables

```bash
cp .env.example .env
# then edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   HIGGSFIELD_API_KEY=...            # leave blank to use the mock client
#   HIGGSFIELD_API_BASE_URL=...
#   DATABASE_URL=sqlite:///data/warehouse/ecommerce.db
#   SENTIMENT_VIDEO_THRESHOLD=0.8
```

### 4.3 Fastest path: one-command launch

```bash
bash run.sh
# Runs ingestion -> AI enrichment -> video generation -> data quality checks,
# then launches the Streamlit dashboard at http://localhost:8501.
# Add --reset to wipe and re-ingest, or --no-dashboard to skip the UI.
```

Or with Docker:

```bash
docker compose up --build
# Runs the pipeline once, then serves the dashboard at http://localhost:8501.
```

### 4.4 Run pipeline stages individually

```bash
python -m src.ingestion --reset       # Stage 1: mock data -> data/raw_warehouse.db
python -m src.ai_pipeline             # Stage 2: LLM enrichment -> enriched_reviews table
python -m src.video_generator         # Stage 3: Higgsfield videos -> generated_videos table
python -m src.quality_checks --stage all   # Stage 5: pre-flight + post-pipeline DQ gate
```

Without `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` or `HIGGSFIELD_API_KEY` set,
stages 2 and 3 automatically fall back to deterministic mock clients — the
whole pipeline runs end-to-end for free. Pass `--force-mock` to either
command to use the mock client even if real keys ARE configured.

### 4.5 Run under Apache Airflow

```bash
export AIRFLOW_HOME=$(pwd)/.airflow
airflow db init
airflow users create --username admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com --password admin

mkdir -p "$AIRFLOW_HOME/dags"
ln -sf "$(pwd)/dags/ecommerce_multimodal_dag.py" "$AIRFLOW_HOME/dags/"

airflow webserver --port 8080 &
airflow scheduler &
# open http://localhost:8080, enable and trigger "ecommerce_multimodal_pipeline"
```

### 4.6 Dashboard only

```bash
streamlit run app.py
```

### 4.7 Run tests

```bash
pytest tests/test_pipeline.py -v
```

---

## 5. Design Notes

- **Strict LLM output contracts.** Every LLM call requests JSON-only output
  and immediately validates it against `EnrichedAIOutput` (`src/schemas.py`).
  A malformed response gets one corrective re-prompt, then — if still
  invalid — is routed to `data/processed/enrichment_dead_letter.jsonl`
  instead of corrupting the warehouse.
- **Mock-first design.** `MockLLMClient` (`src/ai_pipeline.py`) and the
  built-in mock mode of `HiggsfieldVideoClient` (`src/video_generator.py`)
  simulate realistic output and job-polling latency, so the entire pipeline
  runs end-to-end with zero API keys and zero cost.
- **Idempotent SQLite upserts.** Every stage upserts on its natural key
  (`product_id`, `review_id`, `video_id`), so re-running the pipeline never
  creates duplicate rows.
- **Data quality as a first-class citizen.** `src/quality_checks.py` runs
  dependency-free custom Python assertions before AND after the pipeline —
  null checks, range checks, URL format validation, referential integrity,
  and full Pydantic schema re-validation of every persisted row. A CRITICAL
  failure raises `AirflowException` in the DAG, so downstream tasks abort
  gracefully instead of running against bad data.
- **Provider-agnostic AI enrichment.** `AIPipeline` drives either Anthropic
  or OpenAI via a `--provider` flag / `LLM_PROVIDER` env var, with the same
  downstream code path either way.

---

## 6. Current Status

All five stages are implemented and tested end-to-end (offline, via mocks):

- [x] `src/schemas.py` — Pydantic contracts for every stage boundary
- [x] `src/ingestion.py` — mock data generation + SQLite load
- [x] `src/ai_pipeline.py` — LLM enrichment with dead-letter handling
- [x] `src/video_generator.py` — Higgsfield client with mock fallback
- [x] `src/quality_checks.py` — pre-flight + post-pipeline DQ gate
- [x] `dags/ecommerce_multimodal_dag.py` — Airflow orchestration
- [x] `app.py` — Streamlit dashboard
- [x] `tests/test_pipeline.py` — pytest suite (schemas + mocked clients)
- [x] `run.sh` / `docker-compose.yml` — one-command launch

**Possible next steps:** a real Postgres-backed warehouse loader (star
schema) as a 5th DAG task, a Mage.ai alternative pipeline, and CI (GitHub
Actions) running `pytest` on every push.
