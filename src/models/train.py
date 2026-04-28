"""Baseline model training – thin wrapper around classify.train_baseline().

Kept as a standalone entry point so `make train` has its own target,
but all logic lives in src/models/classify.py.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classify import load_config, train_baseline  # noqa: E402


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    model_version = os.getenv("MODEL_VERSION", "v1")

    parser = argparse.ArgumentParser(description="Train baseline Random Forest model.")
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=== Baseline training ===")
    train_baseline(cfg, model_version=model_version)
