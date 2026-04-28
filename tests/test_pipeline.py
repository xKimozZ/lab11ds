"""Unit tests for the pipeline modules."""

import pandas as pd

from src.data.preprocess import DataConfig as PreprocessDataConfig
from src.data.preprocess import clean_data
from src.data.validate import DataConfig as ValidateDataConfig
from src.data.validate import build_validation_report
from src.features.engineer import DataConfig as EngineerDataConfig
from src.features.engineer import engineer_features


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
    ages = ([15, 16, 17, 18] * 5)[:n]
    hours = ([2.0, 3.5, 1.0, 4.0, 2.5] * 4)[:n]
    sleep = ([7.0, 6.5, 8.0, 5.0, 7.5] * 4)[:n]
    screen = ([1.0, 2.0, 0.5, 3.0, 1.5] * 4)[:n]
    academic = ([3.0, 2.5, 3.5, 2.0, 3.0] * 4)[:n]
    activity = ([1.0, 2.0, 3.0, 0.5, 1.5] * 4)[:n]
    stress = ([3, 4, 2, 5, 3] * 4)[:n]
    anxiety = ([2, 3, 1, 4, 2] * 4)[:n]
    addiction = ([3, 4, 2, 5, 3] * 4)[:n]
    gender = (["Male", "Female"] * 10)[:n]
    platform = (["Instagram", "TikTok", "YouTube", "Twitter", "Instagram"] * 4)[:n]
    interaction = (["Low", "Medium", "High"] * 7)[:n]
    label = ([0, 1] * 10)[:n]
    return pd.DataFrame(
        {
            "age": ages,
            "daily_social_media_hours": hours,
            "sleep_hours": sleep,
            "screen_time_before_sleep": screen,
            "academic_performance": academic,
            "physical_activity": activity,
            "stress_level": stress,
            "anxiety_level": anxiety,
            "addiction_level": addiction,
            "gender": gender,
            "platform_usage": platform,
            "social_interaction_level": interaction,
            "depression_label": label,
        }
    )


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
    for key in [
        "input_rows",
        "output_rows",
        "numeric_features_scaled",
        "categorical_features_encoded",
        "output_feature_count",
    ]:
        assert key in log
