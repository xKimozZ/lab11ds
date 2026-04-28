"""Train a baseline model and benchmark multiple classifiers.

This module covers two pipeline steps:
  1. Baseline training  – train_baseline()   (Random Forest, raw feature columns)
  2. Classifier comparison – evaluate_models() (multiple algorithms on engineered features)

Run directly to execute **both** steps in sequence.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import toml
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    cleaned_data_path: str
    featured_data_path: str
    train_feature_columns: list[str]
    target_column: str
    test_size: float
    random_state: int


class ModelConfig(BaseModel):
    n_estimators: int
    max_depth: int
    model_output_path: str


class ReportsConfig(BaseModel):
    metrics_path: str
    classification_metrics_path: str


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    """Load and validate project config."""
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_model(model, filepath: str) -> None:
    """Pickle a model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {filepath}")


def _save_json(payload: dict, filepath: str) -> None:
    """Write a dict as indented JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Report saved -> {filepath}")


# ---------------------------------------------------------------------------
# Step 1 – Baseline training (Random Forest on raw feature columns)
# ---------------------------------------------------------------------------


def train_baseline(config: AppConfig, model_version: str = "v1") -> dict:
    """Train a Random Forest baseline on cleaned data and save it.

    Returns the metrics dict.
    """
    df = pd.read_csv(config.data.cleaned_data_path)
    X = df[config.data.train_feature_columns]
    y = df[config.data.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    rf = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        random_state=config.data.random_state,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    print(f"  baseline accuracy : {metrics['accuracy']}")
    print(f"  baseline f1_score : {metrics['f1_score']}")

    model_dir = Path(config.model.model_output_path).parent
    _save_model(rf, str(model_dir / f"model_{model_version}.pkl"))
    _save_json(metrics, config.reports.metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# Step 2 – Multi-classifier comparison (on engineered features)
# ---------------------------------------------------------------------------


def evaluate_models(config: AppConfig) -> tuple[dict, object]:
    """Benchmark several classifiers on the feature-engineered dataset.

    Returns (metrics_payload, best_model).
    """
    df = pd.read_csv(config.data.featured_data_path)
    X = df.drop(columns=[config.data.target_column])
    y = df[config.data.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=300),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "svm_rbf": SVC(kernel="rbf", probability=False),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=config.data.random_state
        ),
    }

    scores: dict = {}
    best_name = ""
    best_score = -1.0
    best_model = None

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        m = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        scores[name] = m
        print(f"  {name:25s}  acc={m['accuracy']}  f1={m['f1_score']}")
        if m["f1_score"] > best_score:
            best_score = m["f1_score"]
            best_name = name
            best_model = clf

    payload = {
        "models": scores,
        "best_model": best_name,
        "best_f1_score": best_score,
        "test_size": config.data.test_size,
    }
    return payload, best_model


# ---------------------------------------------------------------------------
# Entry point – runs both steps
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    model_version = os.getenv("MODEL_VERSION", "v1")

    parser = argparse.ArgumentParser(
        description="Train baseline and benchmark classifiers."
    )
    parser.add_argument("--config", default="configs/config.toml")
    parser.add_argument(
        "--step",
        choices=["baseline", "compare", "both"],
        default="both",
        help="Which step to run (default: both)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ("baseline", "both"):
        print("\n=== Step 1: Baseline training ===")
        train_baseline(cfg, model_version=model_version)

    if args.step in ("compare", "both"):
        print("\n=== Step 2: Classifier comparison ===")
        metrics_payload, winner = evaluate_models(cfg)
        _save_json(metrics_payload, cfg.reports.classification_metrics_path)
        _save_model(winner, cfg.model.model_output_path)
