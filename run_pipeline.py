import logging
import sys
import time
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

from src.ingestion.loader import load_reviews as ingest_reviews
from src.loader.db import load_reviews as load_reviews_to_db
from src.loader.db import record_pipeline_run
from src.quality.validator import validate_reviews
from src.scoring.sentiment import score_reviews

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_summary(counts: dict, duration_seconds: float, status: str) -> None:
    print()
    print("=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Records ingested:   {counts['records_ingested']}")
    print(f"Records scored:     {counts['records_scored']}")
    print(f"Records passed QC:  {counts['records_passed_qc']}")
    print(f"Records failed QC:  {counts['records_failed_qc']}")
    print(f"Records loaded:     {counts['records_loaded']}")
    print(f"Duration (seconds): {duration_seconds:.2f}")
    print(f"Status:             {status}")
    print("=" * 50)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    config = load_config()

    run_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    counts = {
        "records_ingested": 0,
        "records_scored": 0,
        "records_passed_qc": 0,
        "records_failed_qc": 0,
        "records_loaded": 0,
    }
    status = "FAILED"

    try:
        raw_df = ingest_reviews(config)
        counts["records_ingested"] = len(raw_df)

        scored_df = score_reviews(raw_df, config)
        counts["records_scored"] = len(scored_df)

        validated_df = validate_reviews(scored_df, config)
        counts["records_passed_qc"] = len(validated_df)
        counts["records_failed_qc"] = counts["records_scored"] - counts["records_passed_qc"]

        counts["records_loaded"] = load_reviews_to_db(validated_df, config)

        status = "SUCCESS"
    except Exception:
        logger.error("Pipeline run failed", exc_info=True)
        status = "FAILED"
    finally:
        duration_seconds = time.perf_counter() - start

        try:
            record_pipeline_run(
                {
                    "run_at": run_at,
                    "source": config["pipeline"]["source"],
                    "records_ingested": counts["records_ingested"],
                    "records_scored": counts["records_scored"],
                    "records_passed_qc": counts["records_passed_qc"],
                    "records_failed_qc": counts["records_failed_qc"],
                    "records_loaded": counts["records_loaded"],
                    "duration_seconds": duration_seconds,
                    "status": status,
                }
            )
        except Exception:
            logger.error("Failed to record pipeline_runs summary", exc_info=True)

        print_summary(counts, duration_seconds, status)

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
