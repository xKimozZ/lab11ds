"""Unit tests for the pipeline modules."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import clean_data, DataConfig as PreprocessDataConfig
from src.data.validate import build_validation_report, DataConfig as ValidateDataConfig
from src.features.engineer import engineer_features, DataConfig as EngineerDataConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level",
]
CATEGORICAL_COLS = ["gender", "platform_usage", "social_interaction_level"]
TARGET_COL = "depression_label"


def _make_sample_df(n: int = 20) -> pd.DataFrame:
    """Return a minimal valid DataFrame mimicking the raw dataset."""
    import numpy as np

    rng = pd.np if hasattr(pd, "np") else __import__("numpy").random.default_rng(42)
    data = {
        "age": [15, 16, 17, 18, 15, 16, 17, 18, 15, 16,
                15, 16, 17, 18, 15, 16, 17, 18, 15, 16],
        "daily_social_media_hours": [2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 1.5, 5.0, 2.0, 3.0,
                                     2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 1.5, 5.0, 2.0, 3.0],
        "sleep_hours": [7.0, 6.5, 8.0, 5.0, 7.5, 6.0, 8.5, 4.5, 7.0, 6.5,
                        7.0, 6.5, 8.0, 5.0, 7.5, 6.0, 8.5, 4.5, 7.0, 6.5],
        "screen_time_before_sleep": [1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.0, 3.5, 1.0, 2.0,
                                     1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.0, 3.5, 1.0, 2.0],
        "academic_performance": [3.0, 2.5, 3.5, 2.0, 3.0, 2.5, 3.5, 1.5, 3.0, 2.5,
                                 3.0, 2.5, 3.5, 2.0, 3.0, 2.5, 3.5, 1.5, 3.0, 2.5],
        "physical_activity": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5, 0.0, 1.0, 2.0,
                               1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5, 0.0, 1.0, 2.0],
        "stress_level": [3, 4, 2, 5, 3, 4, 2, 5, 3, 4,
                         3, 4, 2, 5, 3, 4, 2, 5, 3, 4],
        "anxiety_level": [2, 3, 1, 4, 2, 3, 1, 4, 2, 3,
                          2, 3, 1, 4, 2, 3, 1, 4, 2, 3],
        "addiction_level": [3, 4, 2, 5, 3, 4, 2, 5, 3, 4,
                            3, 4, 2, 5, 3, 4, 2, 5, 3, 4],
        "gender": ["Male", "Female"] * 10,
        "platform_usage": ["Instagram", "TikTok", "YouTube", "Twitter", "Instagram"] * 4,
        "social_interaction_level": ["Low", "Medium", "High"] * 6 + ["Low", "Medium"],
        "depression_label": [0, 1] * 10,
    }
    return pd.DataFrame(data)[:n]


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------


def test_clean_data_removes_duplicates():
    df = _make_sample_df()
    df_with_dups = pd.concat([df, df.iloc[:3]], ignore_index=True)
    config = PreprocessDataConfig(
        raw_data_path="",
        cleaned_data_path="",
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    cleaned, log = clean_data(df_with_dups, config)
    assert log["duplicates_removed"] == 3
    assert len(cleaned) == len(df)


def test_clean_data_imputes_numeric_missing():
    df = _make_sample_df()
    df.loc[0, "sleep_hours"] = None
    config = PreprocessDataConfig(
        raw_data_path="",
        cleaned_data_path="",
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    cleaned, log = clean_data(df, config)
    assert cleaned["sleep_hours"].isna().sum() == 0
    assert log["sleep_hours_missing_imputed"] == 1


def test_clean_data_final_row_count():
    df = _make_sample_df(n=10)
    config = PreprocessDataConfig(
        raw_data_path="",
        cleaned_data_path="",
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    cleaned, log = clean_data(df, config)
    assert log["final_row_count"] == len(cleaned)


# ---------------------------------------------------------------------------
# validate tests
# ---------------------------------------------------------------------------


def test_build_validation_report_structure():
    df = _make_sample_df()
    config = ValidateDataConfig(
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    report = build_validation_report(df, config)
    assert "row_count" in report
    assert "duplicate_rows" in report
    assert "missing_by_column" in report
    assert "numeric_summary" in report
    assert "categorical_summary" in report
    assert "target_distribution" in report


def test_build_validation_report_row_count():
    df = _make_sample_df(n=15)
    config = ValidateDataConfig(
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    report = build_validation_report(df, config)
    assert report["row_count"] == 15


# ---------------------------------------------------------------------------
# engineer tests
# ---------------------------------------------------------------------------


def test_engineer_features_output_shape():
    df = _make_sample_df()
    config = EngineerDataConfig(
        cleaned_data_path="",
        featured_data_path="",
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    featured_df, log = engineer_features(df, config)
    # target column preserved + scaled numeric + encoded categoricals + interactions
    assert TARGET_COL in featured_df.columns
    assert log["output_rows"] == len(df)
    assert log["output_feature_count"] == featured_df.shape[1] - 1


def test_engineer_features_log_keys():
    df = _make_sample_df()
    config = EngineerDataConfig(
        cleaned_data_path="",
        featured_data_path="",
        target_column=TARGET_COL,
        numeric_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS,
    )
    _, log = engineer_features(df, config)
    for key in ("input_rows", "output_rows", "numeric_features_scaled",
                "categorical_features_encoded", "output_feature_count"):
        assert key in log
