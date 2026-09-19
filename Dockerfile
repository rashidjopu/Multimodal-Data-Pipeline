# Multimodal E-Commerce AI Pipeline — container image
# Runs either a pipeline stage (via CMD override) or the Streamlit dashboard.
FROM python:3.11-slim

WORKDIR /app

# System deps: psycopg2 needs libpq at build time if compiling from source;
# slim images are fine with the binary wheel but this keeps builds robust.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Airflow is intentionally NOT installed in this image — it has its own
# strict dependency constraints (see requirements.txt notes) and isn't
# needed to run the pipeline stages or the dashboard via docker-compose.
RUN grep -v -i '^apache-airflow' requirements.txt > requirements.no-airflow.txt \
    && pip install --no-cache-dir -r requirements.no-airflow.txt

COPY . .

EXPOSE 8501

# Default command launches the dashboard; docker-compose overrides this
# for the one-shot `pipeline` service.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
