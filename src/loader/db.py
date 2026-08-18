import logging
import os
import time
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Column, Date, DateTime, Float, Integer, MetaData, Table, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

load_dotenv()

metadata = MetaData()

reviews_table = Table(
    "reviews",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("review_id", Text, nullable=False, unique=True),
    Column("review_text", Text, nullable=False),
    Column("review_title", Text),
    Column("star_rating", Integer),
    Column("product_category", Text),
    Column("review_date", Date),
    Column("sentiment_label", Text, nullable=False),
    Column("confidence_score", Float, nullable=False),
    Column("source", Text),
    Column("scored_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

pipeline_runs_table = Table(
    "pipeline_runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("run_at", DateTime(timezone=True), nullable=False),
    Column("source", Text),
    Column("records_ingested", Integer),
    Column("records_scored", Integer),
    Column("records_passed_qc", Integer),
    Column("records_failed_qc", Integer),
    Column("records_loaded", Integer),
    Column("duration_seconds", Float),
    Column("status", Text),
)

REVIEW_COLUMNS = [
    "review_id", "review_text", "review_title", "star_rating",
    "product_category", "review_date", "sentiment_label",
    "confidence_score", "source", "scored_at",
]


def _normalise_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


@lru_cache(maxsize=1)
def _get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
    return create_engine(_normalise_database_url(database_url))


def _ensure_schema(engine):
    metadata.create_all(engine, checkfirst=True)


def _records_for_insert(df: pd.DataFrame) -> list:
    frame = df[REVIEW_COLUMNS].copy()
    frame["review_date"] = pd.to_datetime(frame["review_date"]).dt.date
    frame = frame.astype(object).where(pd.notnull(frame), None)
    return frame.to_dict(orient="records")


def load_reviews(df: pd.DataFrame, config: dict) -> int:
    """Batch-upsert validated, scored reviews into Supabase Postgres.

    Uses INSERT ... ON CONFLICT (review_id) DO NOTHING so re-running the
    pipeline on the same source data never creates duplicate rows. Returns
    the number of rows actually inserted — rows skipped due to a conflict
    are not counted.
    """
    stage = "db_load"
    start = time.perf_counter()
    batch_size = config["pipeline"]["batch_size"]

    records_received = len(df)
    logger.info("[%s] start | records_received=%d batch_size=%d", stage, records_received, batch_size)

    try:
        engine = _get_engine()
        _ensure_schema(engine)
    except SQLAlchemyError:
        logger.error("[%s] failed to connect to the database", stage, exc_info=True)
        raise

    records = _records_for_insert(df)
    total_loaded = 0

    try:
        with engine.begin() as conn:
            for batch_number, start_idx in enumerate(range(0, len(records), batch_size), start=1):
                batch = records[start_idx:start_idx + batch_size]
                stmt = pg_insert(reviews_table).values(batch).on_conflict_do_nothing(
                    index_elements=["review_id"]
                )
                result = conn.execute(stmt)
                total_loaded += result.rowcount
                logger.info(
                    "[%s] batch %d | records_in_batch=%d records_inserted=%d",
                    stage, batch_number, len(batch), result.rowcount,
                )
    except SQLAlchemyError:
        logger.error("[%s] failed while loading reviews", stage, exc_info=True)
        raise

    duration = time.perf_counter() - start
    logger.info(
        "[%s] done | records_received=%d records_loaded=%d duration=%.2fs",
        stage, records_received, total_loaded, duration,
    )

    return total_loaded


def record_pipeline_run(run_summary: dict) -> None:
    """Insert one summary row into pipeline_runs.

    run_summary must contain: run_at, source, records_ingested,
    records_scored, records_passed_qc, records_failed_qc, records_loaded,
    duration_seconds, status.
    """
    stage = "db_load"
    try:
        engine = _get_engine()
        _ensure_schema(engine)
        with engine.begin() as conn:
            conn.execute(pipeline_runs_table.insert().values(**run_summary))
    except SQLAlchemyError:
        logger.error("[%s] failed to record pipeline_runs row", stage, exc_info=True)
        raise

    logger.info("[%s] recorded pipeline_runs row | status=%s", stage, run_summary.get("status"))
