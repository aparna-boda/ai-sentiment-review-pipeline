import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FAILED_RECORDS_LOG_PATH = Path("logs/failed_records.log")


def validate_reviews(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run data quality checks on scored reviews.

    Records failing any check are appended to `logs/failed_records.log`
    (one JSON object per line, including a `failure_reason` column listing
    every check that failed). Only records that pass every check are
    returned.
    """
    stage = "quality"
    start = time.perf_counter()

    quality_cfg = config["quality"]
    min_review_length = quality_cfg["min_review_length"]
    valid_labels = set(quality_cfg["valid_sentiment_labels"])
    confidence_min, confidence_max = quality_cfg["confidence_range"]

    records_received = len(df)
    logger.info("[%s] start | records_received=%d", stage, records_received)

    word_counts = df["review_text"].str.split().str.len()
    parsed_dates = pd.to_datetime(df["review_date"], errors="coerce")

    checks = {
        "null_review_text": df["review_text"].isna(),
        "null_sentiment_label": df["sentiment_label"].isna(),
        "invalid_sentiment_label": df["sentiment_label"].notna() & ~df["sentiment_label"].isin(valid_labels),
        "confidence_score_out_of_range": (
            df["confidence_score"].isna()
            | (df["confidence_score"] < confidence_min)
            | (df["confidence_score"] > confidence_max)
        ),
        "invalid_review_date": parsed_dates.isna(),
        "review_text_too_short": word_counts.isna() | (word_counts < min_review_length),
    }
    checks_df = pd.DataFrame(checks, index=df.index)

    any_failed = checks_df.any(axis=1)
    failure_reason = checks_df.apply(
        lambda row: "; ".join(name for name, failed in row.items() if failed), axis=1
    )

    passed_df = df.loc[~any_failed].reset_index(drop=True)

    records_failed = int(any_failed.sum())
    if records_failed:
        failed_df = df.loc[any_failed].copy()
        failed_df["failure_reason"] = failure_reason.loc[any_failed]
        failed_df = failed_df.reset_index(drop=True)

        FAILED_RECORDS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        failed_df.to_json(
            FAILED_RECORDS_LOG_PATH, orient="records", lines=True, date_format="iso", mode="a"
        )
        logger.info(
            "[%s] wrote %d failed record(s) to %s", stage, records_failed, FAILED_RECORDS_LOG_PATH
        )

    records_passed = len(passed_df)
    pass_rate = (records_passed / records_received * 100) if records_received else 0.0
    duration = time.perf_counter() - start

    logger.info(
        "[%s] done | records_received=%d records_passed=%d records_failed=%d pass_rate=%.2f%% duration=%.2fs",
        stage, records_received, records_passed, records_failed, pass_rate, duration,
    )

    return passed_df
