import logging
import time
from functools import lru_cache

import pandas as pd
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

MODEL_LABEL_MAP = {
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
}


@lru_cache(maxsize=None)
def _get_sentiment_pipeline(model_name: str):
    return hf_pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1)


def score_reviews(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Score review sentiment using a pretrained DistilBERT model.

    Reviews with fewer than `quality.min_review_length` words are skipped
    and excluded before scoring. Reviews whose model confidence is below
    `scoring.neutral_confidence_threshold` are labelled Neutral regardless
    of the model's predicted label.
    """
    stage = "scoring"
    start = time.perf_counter()

    model_name = config["pipeline"]["model_name"]
    model_batch_size = config["scoring"]["model_batch_size"]
    neutral_threshold = config["scoring"]["neutral_confidence_threshold"]
    min_review_length = config["quality"]["min_review_length"]

    records_received = len(df)
    logger.info(
        "[%s] start | records_received=%d model=%s model_batch_size=%d",
        stage, records_received, model_name, model_batch_size,
    )

    word_counts = df["review_text"].str.split().str.len()
    keep_mask = word_counts >= min_review_length
    records_skipped = int((~keep_mask).sum())

    if records_skipped:
        skipped_ids = df.loc[~keep_mask, "review_id"].tolist()
        logger.info(
            "[%s] skipped %d review(s) with fewer than %d words | review_ids=%s",
            stage, records_skipped, min_review_length, skipped_ids,
        )

    scored_df = df.loc[keep_mask].reset_index(drop=True)

    if len(scored_df) > 0:
        sentiment_pipeline = _get_sentiment_pipeline(model_name)
        predictions = sentiment_pipeline(
            scored_df["review_text"].tolist(), batch_size=model_batch_size, truncation=True
        )

        labels, scores = [], []
        for pred in predictions:
            confidence = float(pred["score"])
            if confidence < neutral_threshold:
                label = "Neutral"
            else:
                label = MODEL_LABEL_MAP.get(pred["label"], pred["label"])
            labels.append(label)
            scores.append(confidence)

        scored_df["sentiment_label"] = labels
        scored_df["confidence_score"] = scores
        scored_df["scored_at"] = pd.Timestamp.now(tz="UTC")
    else:
        scored_df["sentiment_label"] = pd.Series(dtype="object")
        scored_df["confidence_score"] = pd.Series(dtype="float64")
        scored_df["scored_at"] = pd.Series(dtype="datetime64[ns, UTC]")

    duration = time.perf_counter() - start
    logger.info(
        "[%s] done | records_received=%d records_scored=%d records_skipped=%d duration=%.2fs",
        stage, records_received, len(scored_df), records_skipped, duration,
    )

    return scored_df
