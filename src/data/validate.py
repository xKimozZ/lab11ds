"""Run data validation checks and save a JSON report."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel


class DataConfig(BaseModel):
    target_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]


class AppConfig(BaseModel):
    data: DataConfig


def load_config(filepath: str) -> AppConfig:
    """Load project config."""
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


def build_validation_report(df: pd.DataFrame, config: DataConfig) -> dict:
    """Build validation metrics."""
    missing_raw = df.isna().sum().to_dict()
    missing_by_column = {k: int(v) for k, v in missing_raw.items()}
    dtypes_raw = df.dtypes.to_dict()
    report: dict = {
        "generated_at": datetime.utcnow().isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": missing_by_column,
        "dtypes": {k: str(v) for k, v in dtypes_raw.items()},
    }

    numeric_summary: dict[str, dict] = {}
    for col in config.numeric_columns:
        if col in df.columns:
            numeric_summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    report["numeric_summary"] = numeric_summary

    categorical_summary: dict[str, dict] = {}
    for col in config.categorical_columns:
        if col in df.columns:
            top_raw = df[col].value_counts().head(5).to_dict()
            categorical_summary[col] = {
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_values": {k: int(v) for k, v in top_raw.items()},
            }
    report["categorical_summary"] = categorical_summary

    target = config.target_column
    if target in df.columns:
        dist_raw = df[target].value_counts().to_dict()
        report["target_distribution"] = {k: int(v) for k, v in dist_raw.items()}
    return report


def save_report(report: dict, filepath: str) -> None:
    """Persist validation report as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved -> {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--output", required=True, help="Path to JSON report")
    args = parser.parse_args()

    config = load_config(args.config)
    dataframe = pd.read_csv(args.input)
    report_payload = build_validation_report(dataframe, config.data)
    save_report(report_payload, args.output)
