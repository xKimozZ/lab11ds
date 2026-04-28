"""Generate a markdown report for the full ML pipeline."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import toml
from pydantic import BaseModel


class ReportsConfig(BaseModel):
    validation_raw_path: str
    validation_cleaned_path: str
    cleaning_log_path: str
    feature_log_path: str
    metrics_path: str
    classification_metrics_path: str
    pipeline_report_path: str


class AppConfig(BaseModel):
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    """Load project config."""
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


def load_json(filepath: str) -> dict:
    """Load JSON helper."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_report(
    raw_validation: dict,
    cleaned_validation: dict,
    cleaning_log: dict,
    feature_log: dict,
    baseline_metrics: dict,
    classification_metrics: dict,
) -> str:
    """Create markdown report text."""
    return f"""# End-to-End ML Pipeline Report

Generated at: {datetime.utcnow().isoformat()} UTC

## 1) Validation Before Cleaning
- Rows: {raw_validation.get("row_count")}
- Duplicate rows: {raw_validation.get("duplicate_rows")}
- Missing values total: {sum(raw_validation.get("missing_by_column", {}).values())}

## 2) Cleaning Summary
- Duplicates removed: {cleaning_log.get("duplicates_removed")}
- Final cleaned rows: {cleaning_log.get("final_row_count")}

## 3) Validation After Cleaning
- Rows: {cleaned_validation.get("row_count")}
- Duplicate rows: {cleaned_validation.get("duplicate_rows")}
- Missing values total: {sum(cleaned_validation.get("missing_by_column", {}).values())}

## 4) Feature Engineering
- Input rows: {feature_log.get("input_rows")}
- Output rows: {feature_log.get("output_rows")}
- Output feature count: {feature_log.get("output_feature_count")}

## 5) Baseline Training
- Baseline model accuracy: {baseline_metrics.get("accuracy")}
- Baseline model F1: {baseline_metrics.get("f1_score")}

## 6) Classifier Benchmark
- Best model: {classification_metrics.get("best_model")}
- Best F1 score: {classification_metrics.get("best_f1_score")}

## 7) Notes
- This report is generated automatically from pipeline artifacts in `reports/`.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    app_config = load_config(args.config)
    rpts = app_config.reports

    raw_validation = load_json(rpts.validation_raw_path)
    cleaned_validation = load_json(rpts.validation_cleaned_path)
    cleaning_log = load_json(rpts.cleaning_log_path)
    feature_log = load_json(rpts.feature_log_path)
    baseline_metrics = load_json(rpts.metrics_path)
    classification_metrics = load_json(rpts.classification_metrics_path)

    report_text = render_report(
        raw_validation,
        cleaned_validation,
        cleaning_log,
        feature_log,
        baseline_metrics,
        classification_metrics,
    )
    Path(rpts.pipeline_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rpts.pipeline_report_path).write_text(report_text, encoding="utf-8")
    print(f"Pipeline report saved -> {rpts.pipeline_report_path}")
