import hashlib
import itertools
import logging
import time

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

NORMALISED_COLUMNS = [
    "review_id",
    "review_text",
    "review_title",
    "star_rating",
    "product_category",
    "review_date",
    "source",
]


def _build_review_id(user_id, asin, timestamp) -> str:
    raw = f"{user_id}|{asin}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_reviews(config: dict) -> pd.DataFrame:
    """Load a batch of reviews from the configured HuggingFace dataset and
    normalise them to the pipeline's review schema.
    """
    stage = "ingestion"
    start = time.perf_counter()

    pipeline_cfg = config["pipeline"]
    source = pipeline_cfg["source"]
    subset = pipeline_cfg["subset"]
    split = pipeline_cfg["split"]
    category = pipeline_cfg["category"]
    batch_size = pipeline_cfg["batch_size"]

    logger.info(
        "[%s] start | source=%s subset=%s split=%s batch_size=%d",
        stage, source, subset, split, batch_size,
    )

    dataset = load_dataset(
        source, subset, split=split, streaming=True, trust_remote_code=True
    )
    raw_records = list(itertools.islice(dataset, batch_size))
    records_loaded = len(raw_records)

    normalised = [
        {
            "review_id": _build_review_id(
                rec.get("user_id"), rec.get("asin"), rec.get("timestamp")
            ),
            "review_text": rec.get("text"),
            "review_title": rec.get("title"),
            "star_rating": rec.get("rating"),
            "product_category": category,
            "review_date": rec.get("timestamp"),
            "source": source,
        }
        for rec in raw_records
    ]

    df = pd.DataFrame(normalised, columns=NORMALISED_COLUMNS)
    df["review_date"] = pd.to_datetime(df["review_date"], unit="ms", errors="coerce")
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").astype("Int64")

    records_before_drop = len(df)
    df = df[df["review_text"].notna()]
    records_dropped = records_before_drop - len(df)
    df = df.reset_index(drop=True)

    duration = time.perf_counter() - start
    logger.info(
        "[%s] done | records_in=%d records_out=%d records_dropped_null_text=%d duration=%.2fs",
        stage, records_loaded, len(df), records_dropped, duration,
    )

    return df
