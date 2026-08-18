import json

import pandas as pd
import pytest

from src.quality import validator


@pytest.fixture
def config():
    return {
        "quality": {
            "min_review_length": 3,
            "valid_sentiment_labels": ["Positive", "Negative", "Neutral"],
            "confidence_range": [0.0, 1.0],
        }
    }


@pytest.fixture(autouse=True)
def redirect_failed_records_log(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "FAILED_RECORDS_LOG_PATH", tmp_path / "failed_records.log")


def _base_row(**overrides):
    row = {
        "review_id": "r1",
        "review_text": "this book was genuinely great",
        "sentiment_label": "Positive",
        "confidence_score": 0.95,
        "review_date": "2024-01-15",
    }
    row.update(overrides)
    return row


def test_all_valid_records_pass(config):
    df = pd.DataFrame([_base_row(), _base_row(review_id="r2")])

    result = validator.validate_reviews(df, config)

    assert len(result) == 2
    assert list(result["review_id"]) == ["r1", "r2"]


def test_null_review_text_fails(config):
    df = pd.DataFrame([_base_row(review_text=None)])

    result = validator.validate_reviews(df, config)

    assert result.empty


def test_null_sentiment_label_fails(config):
    df = pd.DataFrame([_base_row(sentiment_label=None)])

    result = validator.validate_reviews(df, config)

    assert result.empty


def test_invalid_sentiment_label_fails(config):
    df = pd.DataFrame([_base_row(sentiment_label="Mixed")])

    result = validator.validate_reviews(df, config)

    assert result.empty


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_confidence_score_out_of_range_fails(config, confidence):
    df = pd.DataFrame([_base_row(confidence_score=confidence)])

    result = validator.validate_reviews(df, config)

    assert result.empty


def test_confidence_score_boundaries_pass(config):
    df = pd.DataFrame([_base_row(confidence_score=0.0), _base_row(review_id="r2", confidence_score=1.0)])

    result = validator.validate_reviews(df, config)

    assert len(result) == 2


def test_invalid_review_date_fails(config):
    df = pd.DataFrame([_base_row(review_date="not-a-date")])

    result = validator.validate_reviews(df, config)

    assert result.empty


def test_review_text_too_short_fails(config):
    df = pd.DataFrame([_base_row(review_text="bad")])

    result = validator.validate_reviews(df, config)

    assert result.empty


def test_review_text_meets_minimum_length_passes(config):
    df = pd.DataFrame([_base_row(review_text="not bad here")])

    result = validator.validate_reviews(df, config)

    assert len(result) == 1


def test_mixed_pass_and_fail_returns_only_passing_rows(config):
    df = pd.DataFrame(
        [
            _base_row(review_id="pass-1"),
            _base_row(review_id="fail-1", review_text=None),
            _base_row(review_id="pass-2"),
        ]
    )

    result = validator.validate_reviews(df, config)

    assert sorted(result["review_id"]) == ["pass-1", "pass-2"]


def test_failed_records_are_written_with_failure_reason(config, tmp_path):
    df = pd.DataFrame([_base_row(review_text=None, confidence_score=5.0)])

    validator.validate_reviews(df, config)

    log_path = validator.FAILED_RECORDS_LOG_PATH
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "null_review_text" in record["failure_reason"]
    assert "confidence_score_out_of_range" in record["failure_reason"]


def test_no_log_written_when_nothing_fails(config):
    df = pd.DataFrame([_base_row()])

    validator.validate_reviews(df, config)

    assert not validator.FAILED_RECORDS_LOG_PATH.exists()


def test_empty_input_returns_empty_output_without_error(config):
    df = pd.DataFrame(
        columns=["review_id", "review_text", "sentiment_label", "confidence_score", "review_date"]
    )

    result = validator.validate_reviews(df, config)

    assert result.empty
