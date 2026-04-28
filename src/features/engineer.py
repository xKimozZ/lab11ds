"""Generate model features from cleaned data."""

import argparse
import json
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler


class DataConfig(BaseModel):
    cleaned_data_path: str
    featured_data_path: str
    target_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]


class ReportsConfig(BaseModel):
    feature_log_path: str


class AppConfig(BaseModel):
    data: DataConfig
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    """Load and validate config."""
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


def engineer_features(df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Encode categorical columns, scale numerical columns, and add interactions."""
    work_df = df.copy()
    target = work_df[config.target_column]

    numeric_cols = [c for c in config.numeric_columns if c in work_df.columns]
    cat_cols = [c for c in config.categorical_columns if c in work_df.columns]

    scaler = StandardScaler()
    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(work_df[numeric_cols]),
        columns=[f"{c}_scaled" for c in numeric_cols],
        index=work_df.index,
    )

    encoded_categorical = pd.get_dummies(work_df[cat_cols], drop_first=True)
    feature_df = pd.concat([scaled_numeric, encoded_categorical], axis=1)

    if {"stress_level", "anxiety_level"}.issubset(work_df.columns):
        feature_df["stress_anxiety_interaction"] = (
            work_df["stress_level"] * work_df["anxiety_level"]
        )
    if {"screen_time_before_sleep", "sleep_hours"}.issubset(work_df.columns):
        feature_df["screen_sleep_ratio"] = work_df["screen_time_before_sleep"] / (
            work_df["sleep_hours"] + 1e-6
        )
    if {"daily_social_media_hours", "physical_activity"}.issubset(work_df.columns):
        feature_df["social_activity_gap"] = (
            work_df["daily_social_media_hours"] - work_df["physical_activity"]
        )

    feature_df[config.target_column] = target.values
    log = {
        "input_rows": int(len(df)),
        "output_rows": int(len(feature_df)),
        "numeric_features_scaled": len(numeric_cols),
        "categorical_features_encoded": len(cat_cols),
        "output_feature_count": int(feature_df.shape[1] - 1),
    }
    return feature_df, log


def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save feature table."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Feature dataset saved -> {filepath}")


def save_log(payload: dict, filepath: str) -> None:
    """Save feature engineering log."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Feature log saved -> {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    app_config = load_config(args.config)
    source_df = pd.read_csv(app_config.data.cleaned_data_path)
    featured_df, feature_log = engineer_features(source_df, app_config.data)
    save_csv(featured_df, app_config.data.featured_data_path)
    save_log(feature_log, app_config.reports.feature_log_path)
