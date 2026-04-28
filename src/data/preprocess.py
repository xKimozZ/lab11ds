"""Clean raw teen mental-health data."""

import argparse
import json
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_data_path: str
    cleaned_data_path: str
    target_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]


class ReportsConfig(BaseModel):
    cleaning_log_path: str


class AppConfig(BaseModel):
    data: DataConfig
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    """Load and validate the TOML config with pydantic."""
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)  # pydantic v2
    except AttributeError:
        return AppConfig.parse_obj(raw_config)  # pydantic v1


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Impute, cap outliers, and deduplicate while preserving target."""
    cleaned = df.copy()
    log: dict[str, int] = {}

    before_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    log["duplicates_removed"] = before_rows - len(cleaned)

    numeric_cols = [c for c in config.numeric_columns if c in cleaned.columns]
    cat_cols = [c for c in config.categorical_columns if c in cleaned.columns]

    for col in numeric_cols:
        missing_before = int(cleaned[col].isna().sum())
        if missing_before > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((cleaned[col] < lower) | (cleaned[col] > upper)).sum())
        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
        log[f"{col}_missing_imputed"] = missing_before
        log[f"{col}_outliers_capped"] = outliers

    for col in cat_cols:
        missing_before = int(cleaned[col].isna().sum())
        if missing_before > 0:
            mode_value = cleaned[col].mode(dropna=True)
            replacement = mode_value.iloc[0] if not mode_value.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(replacement)
        log[f"{col}_missing_imputed"] = missing_before

    required_cols = numeric_cols + cat_cols + [config.target_column]
    required_cols = [c for c in required_cols if c in cleaned.columns]
    cleaned = cleaned[required_cols].dropna()
    log["final_row_count"] = len(cleaned)
    return cleaned, log


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data -> {filepath}  ({len(df)} rows)")


def save_cleaning_log(payload: dict, filepath: str) -> None:
    """Write cleaning summary to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Cleaning log saved -> {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    app_config = load_config(args.config)

    df = load_raw_data(app_config.data.raw_data_path)
    cleaned_df, cleaning_log = clean_data(df, app_config.data)
    save_data(cleaned_df, app_config.data.cleaned_data_path)
    save_cleaning_log(cleaning_log, app_config.reports.cleaning_log_path)
