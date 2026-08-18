# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real Supabase credentials

# run the full pipeline (ingest → score → validate → load)
python run_pipeline.py

# run the dashboard
streamlit run dashboard/app.py

# tests
pytest
pytest tests/test_validator.py::test_name   # single test
```

There is no separate lint/format command configured in this repo.

## Architecture

This is a linear, five-stage batch pipeline. Each stage is a pure function that takes a
DataFrame in and returns a DataFrame out — there is no shared mutable state between stages,
and `run_pipeline.py` contains no business logic, only orchestration and count-tracking.

```
src/ingestion/loader.py   load_reviews(config) -> DataFrame
src/scoring/sentiment.py  score_reviews(df, config) -> DataFrame
src/quality/validator.py  validate_reviews(df, config) -> DataFrame
src/loader/db.py          load_reviews(df, config) -> int   (+ record_pipeline_run(dict))
dashboard/app.py          standalone Streamlit app, reads DB directly (no import from src/)
```

- **Ingestion** streams from a HuggingFace dataset (`datasets.load_dataset(..., streaming=True)`
  + `itertools.islice`) rather than downloading it in full — the source dataset is multi-GB.
  `review_id` is derived as `sha256(user_id|asin|timestamp)`, not taken from the source data.
- **Scoring** runs the HF `sentiment-analysis` pipeline CPU-only (`device=-1`), batched by
  `scoring.model_batch_size`. Reviews under `quality.min_review_length` words are dropped before
  scoring. Any prediction with `confidence_score < scoring.neutral_confidence_threshold` is
  relabeled `Neutral` regardless of the model's raw Positive/Negative output — this is the only
  place `Neutral` labels come from (SST-2 is a binary model).
- **Validation** is fully vectorized (no row-by-row loops). Rows that fail any check get a
  `failure_reason` column (all failing checks joined) and are appended as JSON Lines to
  `logs/failed_records.log`; only passing rows continue downstream.
- **Load** uses SQLAlchemy Core (not the ORM) with
  `postgresql.insert(...).on_conflict_do_nothing(index_elements=["review_id"])` for idempotency —
  re-running the pipeline on the same data must yield `records_loaded == 0` on the second run.
  `_ensure_schema()` creates the `reviews` / `pipeline_runs` tables on first run
  (`metadata.create_all(engine, checkfirst=True)`). `record_pipeline_run()` is called from
  `run_pipeline.py`'s `finally` block wrapped in its own try/except, so a DB failure during the
  main load still gets a `FAILED` row recorded if the DB is reachable at all.
- **Config** (`config/config.yaml`) drives dataset source/subset/split, batch sizes, the model
  name, and all quality thresholds — nothing pipeline-specific is hardcoded in the stage modules.
  Secrets (`SUPABASE_URL`, `SUPABASE_DB_PASSWORD`, `DATABASE_URL`) live only in `.env`.
- **Dashboard** is intentionally standalone (does not import `src/`) so it can be deployed to
  Streamlit Cloud as-is. Engine is `st.cache_resource`; every query function is
  `st.cache_data(ttl=300)`. `DATABASE_URL` is read via `os.getenv`, populated from `.env` locally
  or Streamlit Cloud Secrets in deployment — same code path either way.

## Known gotchas

- **Supabase connection string**: use the Connection Pooling (Session mode) host
  (`aws-0-<region>.pooler.supabase.com`, user `postgres.<project-ref>`), not the direct
  `db.<ref>.supabase.co` host — the direct host resolves IPv6-only on newer projects and many
  sandboxes/CI runners have no IPv6 route.
- **Password with `@`**: SQLAlchemy's URL parser splits userinfo/host on the *first* `@` in the
  URL (Python's own `urllib.parse.urlsplit` splits on the last one) — any literal `@` in
  `SUPABASE_DB_PASSWORD` must be percent-encoded (`%40`) when composing `DATABASE_URL`.
- **`python-dotenv` `load_dotenv()`** does not override keys already present in `os.environ`. When
  testing code paths that read `os.getenv("DATABASE_URL")` with the var unset, use
  `os.environ["DATABASE_URL"] = ""` rather than `os.environ.pop(...)`, or a stale real value can
  leak back in via `load_dotenv()`.
- Pandas nullable dtypes (`Int64`) are used for `star_rating`; convert `pd.NA`/`NaT` to `None`
  before binding parameters for psycopg2 — it doesn't accept pandas' null sentinels directly.
