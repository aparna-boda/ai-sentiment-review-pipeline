# Product Requirements Document — AI Sentiment Review Pipeline

**Repository:** `ai-sentiment-review-pipeline`
**Author:** RK
**Status:** In Progress
**Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Goals](#goals)
4. [Non-Goals](#non-goals)
5. [Users](#users)
6. [Functional Requirements](#functional-requirements)
7. [Non-Functional Requirements](#non-functional-requirements)
8. [System Architecture](#system-architecture)
9. [Data Flow](#data-flow)
10. [Database Schema](#database-schema)
11. [Dashboard Requirements](#dashboard-requirements)
12. [Tech Stack](#tech-stack)
13. [Repo Structure](#repo-structure)
14. [Configuration](#configuration)
15. [Success Metrics](#success-metrics)

---

## Overview

The AI Sentiment Review Pipeline is a generic, reusable, end-to-end data engineering pipeline that ingests customer reviews from any source, scores each review for sentiment using a pretrained NLP model, validates data quality, loads results into a cloud database, and surfaces trends on an interactive dashboard.

The pipeline is source-agnostic — it is designed to work with any review dataset (product reviews, book reviews, restaurant reviews, movie reviews) with minimal configuration changes.

**Demo dataset:** Amazon Books reviews (HuggingFace `McAuley-Lab/Amazon-Reviews-2023`, `raw_review_Books` subset — the original `amazon_us_reviews` dataset was discontinued by its data provider and is no longer accessible)

---

## Problem Statement

Businesses receive thousands of customer reviews every day across their product catalogues. Manually reading and categorising reviews to understand customer sentiment is not scalable. Without automated tooling, teams face:

- No visibility into whether sentiment is improving or declining over time
- No way to identify which products or categories are generating the most negative feedback
- Reactive decision-making — problems are discovered late, after they have already impacted revenue or reputation
- One-off, manual analysis that cannot be repeated consistently

**The gap:** There is no automated, repeatable pipeline that takes raw review text, scores sentiment at scale, and makes the results queryable and visual for business teams.

---

## Goals

- Build a generic, config-driven pipeline that can process reviews from any source
- Score sentiment for each review (Positive / Negative / Neutral) using a pretrained model
- Validate data quality at every stage before loading to the database
- Store enriched review data in a cloud PostgreSQL database (Supabase)
- Surface sentiment trends, category breakdowns, and negative review flags on a Streamlit dashboard
- Deploy the dashboard publicly so it is accessible via a live URL
- Write clean, modular, well-documented code that demonstrates production engineering standards

---

## Non-Goals

- Real-time or streaming sentiment scoring — this is a batch pipeline
- Training or fine-tuning a custom sentiment model
- Supporting languages other than English
- User authentication on the dashboard
- Expanding beyond the Books category for the demo dataset (other datasets can be plugged in via config)
- Automated scheduling or orchestration (Airflow, Prefect) — pipeline is triggered manually via a single script

---

## Users

### Data Engineer
Runs and maintains the pipeline. Needs:
- Clear setup instructions
- Configurable pipeline settings (source, batch size, DB connection)
- Informative logs at each pipeline stage
- Easy re-runnability without data duplication

### Business Analyst / Stakeholder
Reads the dashboard. Needs:
- Overall sentiment breakdown (Positive / Negative / Neutral counts)
- Sentiment trend over time
- Category or source-level breakdown
- List of most negative reviews for action

---

## Functional Requirements

### FR-1 Data Ingestion
- The pipeline must load review data from a configurable source
- For the demo, source is HuggingFace `McAuley-Lab/Amazon-Reviews-2023`, `raw_review_Books` subset
- Ingestion must support loading a configurable number of records (batch size)
- Raw records must include: review text, review title, star rating, product category, review date
- Missing or null review text must be flagged and excluded before scoring

### FR-2 Sentiment Scoring
- Each review must be scored using a pretrained DistilBERT model via HuggingFace `transformers` pipeline
- Output classes: Positive, Negative, Neutral
- Each scored record must include: sentiment label, confidence score
- Reviews with review text shorter than 3 words must be skipped and logged

### FR-3 Data Quality Validation
- Before loading to the database, the pipeline must run the following checks:
  - No null values in required fields (review text, sentiment label, confidence score)
  - Sentiment label is one of: Positive, Negative, Neutral
  - Confidence score is between 0.0 and 1.0
  - Review date is a valid date format
- Records that fail validation must be written to a separate `failed_records` log file
- Pipeline must log how many records passed and failed validation

### FR-4 Database Load
- Validated records must be loaded into Supabase PostgreSQL
- Pipeline must be idempotent — re-running on the same data must not create duplicate records
- Deduplication must be handled via a unique constraint on review ID
- Load must be done in batches, not row by row

### FR-5 Dashboard
- Dashboard must display:
  - Total reviews processed
  - Sentiment breakdown (Positive / Negative / Neutral) as a donut chart
  - Sentiment trend over time as a line chart
  - Sentiment breakdown by product category as a bar chart
  - Top 10 most negative reviews table (review text, score, date)
  - Pipeline last run timestamp and record count
- Dashboard must be deployed to Streamlit Cloud with a public URL

### FR-6 Logging
- Every pipeline stage must log: stage name, start time, record count in, record count out, duration
- Errors must be caught and logged with full context before the pipeline exits gracefully

---

## Non-Functional Requirements

### NFR-1 Reusability
- Dataset source, batch size, model name, and database connection must all be controlled via a config file
- Swapping the review source must require only a config change, not a code change

### NFR-2 Idempotency
- Re-running the pipeline on the same dataset must produce the same result in the database
- No duplicate records must be created on re-run

### NFR-3 Modularity
- Code must be structured as a `src/` package with separate modules per pipeline stage
- No business logic in the entrypoint script (`run_pipeline.py`)

### NFR-4 Observability
- Every stage must produce structured logs
- Failed records must be written to a separate file for inspection
- Pipeline summary (records in, records out, failures, duration) must be printed at the end of every run

### NFR-5 Portability
- The project must run locally with a single command after `.env` setup
- Dependencies must be pinned in `requirements.txt`

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Source                          │
│      HuggingFace McAuley-Lab/Amazon-Reviews-2023 (Books)    │
│              (swappable via config)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Ingestion Layer                         │
│              src/ingestion/loader.py                        │
│     Load N records → normalise schema → raw DataFrame       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Scoring Layer                           │
│              src/scoring/sentiment.py                       │
│   DistilBERT → sentiment label + confidence score           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Quality Layer                         │
│              src/quality/validator.py                       │
│     Schema checks → null checks → range checks             │
│     Pass → continue │ Fail → failed_records.log            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Load Layer                             │
│               src/loader/db.py                              │
│         Batch upsert → Supabase PostgreSQL                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard Layer                           │
│                  dashboard/app.py                           │
│         Streamlit → queries Supabase → live charts          │
│         Deployed on Streamlit Cloud (public URL)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Raw Review Record
        │
        ▼
┌───────────────────┐
│  review_id        │  unique identifier
│  review_text      │  raw text input to model
│  review_title     │  supporting context
│  star_rating      │  1–5
│  product_category │  e.g. Books
│  review_date      │  date of review
└───────────────────┘
        │
        ▼  [DistilBERT scoring]
┌───────────────────┐
│  sentiment_label  │  Positive / Negative / Neutral
│  confidence_score │  0.0 – 1.0
│  scored_at        │  pipeline run timestamp
└───────────────────┘
        │
        ▼  [quality validation]
        │
        ▼  [upsert to Supabase]
```

---

## Database Schema

### Table: `reviews`

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Primary key, auto-generated |
| `review_id` | `TEXT` | Unique ID from source dataset (unique constraint) |
| `review_text` | `TEXT` | Raw review text |
| `review_title` | `TEXT` | Review title |
| `star_rating` | `INTEGER` | 1–5 star rating |
| `product_category` | `TEXT` | Category label (e.g. Books) |
| `review_date` | `DATE` | Date review was written |
| `sentiment_label` | `TEXT` | Positive / Negative / Neutral |
| `confidence_score` | `FLOAT` | Model confidence 0.0–1.0 |
| `source` | `TEXT` | Dataset source identifier (e.g. McAuley-Lab/Amazon-Reviews-2023) |
| `scored_at` | `TIMESTAMP` | When the pipeline processed this record |
| `created_at` | `TIMESTAMP` | Row insertion time, auto-generated |

### Table: `pipeline_runs`

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Primary key |
| `run_at` | `TIMESTAMP` | When the pipeline was triggered |
| `source` | `TEXT` | Dataset source used |
| `records_ingested` | `INTEGER` | Total records loaded from source |
| `records_scored` | `INTEGER` | Records successfully scored |
| `records_passed_qc` | `INTEGER` | Records that passed quality checks |
| `records_failed_qc` | `INTEGER` | Records that failed quality checks |
| `records_loaded` | `INTEGER` | Records successfully upserted to DB |
| `duration_seconds` | `FLOAT` | Total pipeline run time |
| `status` | `TEXT` | SUCCESS / FAILED |

---

## Dashboard Requirements

| View | Chart type | Data source |
|---|---|---|
| Total reviews processed | Metric card | `COUNT(*)` from `reviews` |
| Sentiment breakdown | Donut chart | `COUNT` grouped by `sentiment_label` |
| Sentiment trend over time | Line chart | Daily `COUNT` grouped by `sentiment_label` and `review_date` |
| Sentiment by category | Bar chart | `COUNT` grouped by `product_category` and `sentiment_label` |
| Top 10 most negative reviews | Table | `sentiment_label = Negative` ordered by `confidence_score DESC` |
| Last pipeline run | Metric card | Latest row from `pipeline_runs` |

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Language | Python 3.11 | Standard for data and AI engineering |
| Data source | HuggingFace `datasets` | Free, large, no API key required |
| Sentiment model | `distilbert-base-uncased-finetuned-sst-2-english` | Pretrained, fast, no GPU required |
| NLP library | HuggingFace `transformers` | Industry standard |
| Data processing | Pandas | Pipeline transformations |
| Database | Supabase (PostgreSQL) | Free tier, cloud-hosted, production-grade |
| DB client | `psycopg2` / `sqlalchemy` | Standard PostgreSQL connectors |
| Dashboard | Streamlit | Fast to build, easy to deploy |
| Deployment | Streamlit Cloud | Free, public URL, connects to GitHub |
| Config | `python-dotenv` + `config.yaml` | Keeps secrets out of code |
| Logging | Python `logging` module | Structured pipeline logs |
| Dependency management | `requirements.txt` | Simple, portable |

---

## Repo Structure

```
ai-sentiment-review-pipeline/
│
├── src/
│   ├── ingestion/
│   │   └── loader.py          # loads and normalises raw review data
│   ├── scoring/
│   │   └── sentiment.py       # runs DistilBERT sentiment scoring
│   ├── quality/
│   │   └── validator.py       # data quality checks
│   ├── loader/
│   │   └── db.py              # batch upsert to Supabase
│   └── utils/
│       └── logger.py          # shared logging setup
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── config/
│   └── config.yaml            # pipeline configuration
│
├── logs/
│   └── failed_records.log     # records that failed quality checks
│
├── tests/
│   └── test_validator.py      # unit tests for quality checks
│
├── run_pipeline.py            # entrypoint — runs all stages in order
├── requirements.txt           # pinned dependencies
├── .env.example               # template for environment variables
├── .gitignore                 # excludes .env, logs, __pycache__
├── PRD.md                     # this document
└── README.md                  # setup guide and project overview
```

---

## Configuration

All pipeline settings are controlled via `config/config.yaml` and `.env`:

**config.yaml**
```yaml
pipeline:
  source: "McAuley-Lab/Amazon-Reviews-2023"
  subset: "raw_review_Books"
  split: "full"
  category: "Books"
  batch_size: 5000
  model_name: "distilbert-base-uncased-finetuned-sst-2-english"

quality:
  min_review_length: 3
  valid_sentiment_labels: ["Positive", "Negative", "Neutral"]
  confidence_range: [0.0, 1.0]
```

**.env**
```
SUPABASE_URL=your_supabase_url
SUPABASE_DB_PASSWORD=your_db_password
DATABASE_URL=your_postgres_connection_string
```

---

## Success Metrics

| Metric | Target |
|---|---|
| Pipeline runs end-to-end without errors | 100% of runs |
| Data quality pass rate | > 95% of ingested records |
| No duplicate records on re-run | 0 duplicates |
| Dashboard loads successfully | Every time |
| Dashboard is publicly accessible via live URL | Yes |
| README enables a new developer to run the pipeline from scratch | Yes |

---

> "A pipeline that runs once is a script. A pipeline that runs correctly every time, handles failures gracefully, and makes results visible to stakeholders — that is engineering."
