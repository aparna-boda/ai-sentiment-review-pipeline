# ai-sentiment-review-pipeline

A generic, config-driven batch pipeline that ingests customer reviews, scores sentiment with DistilBERT, validates data quality, loads results into Supabase PostgreSQL, and surfaces trends on a Streamlit dashboard.

Demo dataset: Amazon Books reviews (HuggingFace `McAuley-Lab/Amazon-Reviews-2023`, `raw_review_Books` subset). The pipeline is source-agnostic — swap the dataset via `config/config.yaml`, no code changes required.

Full requirements and design are in [PRD.md](PRD.md).

## Architecture

```
HuggingFace dataset (streamed)
        │
        ▼
src/ingestion/loader.py     load + normalise schema
        │
        ▼
src/scoring/sentiment.py    DistilBERT sentiment scoring
        │
        ▼
src/quality/validator.py    schema / null / range checks
        │
        ▼
src/loader/db.py            batch upsert → Supabase PostgreSQL
        │
        ▼
dashboard/app.py            Streamlit dashboard (reads DB directly)
```

Orchestrated end-to-end by `run_pipeline.py`.

## Repo structure

```
ai-sentiment-review-pipeline/
├── src/
│   ├── ingestion/loader.py     # load + normalise raw review data
│   ├── scoring/sentiment.py    # DistilBERT sentiment scoring
│   ├── quality/validator.py    # data quality checks
│   └── loader/db.py            # batch upsert to Supabase
├── dashboard/app.py            # Streamlit dashboard
├── config/config.yaml          # pipeline configuration
├── logs/failed_records.log     # records that failed quality checks (generated at runtime)
├── tests/                      # unit tests
├── run_pipeline.py             # entrypoint — runs all stages in order
├── requirements.txt
├── .env.example
└── PRD.md
```

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in real values:

   ```bash
   cp .env.example .env
   ```

   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_DB_PASSWORD=your_db_password
   DATABASE_URL=your_postgres_connection_string
   ```

   Notes on `DATABASE_URL`:
   - Use the Supabase **Connection Pooling (Session mode)** connection string, not the direct `db.<ref>.supabase.co` host — the direct host resolves IPv6-only on newer Supabase projects and many local/CI networks have no IPv6 route.
     Format: `postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`
   - If your database password contains special characters (`@`, `/`, `:`, `#`, etc.), percent-encode them in the URL — e.g. a literal `@` must be written as `%40`. SQLAlchemy's URL parser splits on the *first* `@`, so an unescaped `@` in the password will silently produce the wrong host.

3. Adjust `config/config.yaml` if you want a different dataset, batch size, or thresholds.

## Running the pipeline

```bash
python run_pipeline.py
```

This ingests, scores, validates, and upserts a batch of reviews, then prints a summary:

```
==================================================
PIPELINE SUMMARY
==================================================
Records ingested:   ...
Records scored:     ...
Records passed QC:  ...
Records failed QC:  ...
Records loaded:     ...
Duration (seconds): ...
Status:             SUCCESS
==================================================
```

Every run also inserts one row into the `pipeline_runs` table with the same counts, so history is queryable. Loads are idempotent — re-running against the same data upserts on `review_id` (`ON CONFLICT DO NOTHING`), so `records_loaded` drops to `0` on a repeat run.

Failed-quality-check records are appended to `logs/failed_records.log` as JSON Lines.

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Reads `DATABASE_URL` from `.env` locally, or from Streamlit Cloud Secrets when deployed — no code changes needed for deployment. Shows a friendly empty-state message if the pipeline hasn't been run yet.

## Testing

```bash
pytest
```

## Tech stack

| Layer | Tool |
|---|---|
| Data source | HuggingFace `datasets` (streaming) |
| Sentiment model | `distilbert-base-uncased-finetuned-sst-2-english` via `transformers` |
| Data processing | Pandas |
| Database | Supabase (PostgreSQL) via SQLAlchemy / psycopg2 |
| Dashboard | Streamlit + Plotly |
| Config | `python-dotenv` + `config.yaml` |
