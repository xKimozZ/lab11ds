# Lab Requirements: Convert Your Student Lab to the Original Project

## Goal

You are given:

- one notebook containing all needed functions
- one dataset

Your task is to convert that notebook work into a clean project structure similar to the original pipeline project.

## Learning Objectives

By the end of this lab, you should be able to:

1. Organize ML code into modules (`src/data`, `src/features`, `src/models`, `src/reports`)
2. Move logic from notebook cells into reusable Python scripts
3. Build a reproducible end-to-end pipeline using `Makefile`
4. Validate, clean, transform, and model data in a consistent workflow
5. Generate JSON and Markdown reports for traceability
6. Use basic CI (GitHub Actions) for quality checks

## Required Project Structure

Your final project should include at least:

- `data/`
  - `raw/`
  - `processed/` (generated)
- `src/`
  - `data/`
    - `preprocess.py`
    - `validate.py`
  - `features/`
    - `engineer.py`
  - `models/`
    - `train.py`
    - `classify.py`
  - `reports/`
    - `generate_report.py`
- `tests/`
- `.github/workflows/`
  - CI workflow file (`.yml`)
- `Makefile`
- `pyproject.toml`
- `.gitignore`
- `__init__.py` (where needed)

## Required Pipeline Steps

Implement and run these steps in order:

1. Validate raw data
2. Clean/preprocess data
3. Validate cleaned data
4. Engineer features
5. Train baseline model
6. Compare classifiers and save best model
7. Generate final pipeline report

## Minimum Configuration Expectations

Your config should define paths and settings for:

- Raw dataset path
- Cleaned dataset path
- Feature dataset path
- Target column
- Numeric and categorical columns
- Train/test split settings
- Model hyperparameters
- Output report paths

## Reports You Must Generate

Your pipeline must produce the following artifacts:

1. `reports/validation_raw.json`
  - validation summary of raw dataset
2. `reports/cleaning_log.json`
  - cleaning actions (duplicates removed, imputation, outlier handling, final rows)
3. `reports/validation_cleaned.json`
  - validation summary after cleaning
4. `reports/feature_log.json`
  - feature engineering summary (input rows, output rows, feature count)
5. `reports/metrics.json`
  - baseline model metrics (at least accuracy and F1)
6. `reports/classification_metrics.json`
  - multi-model benchmark results and selected best model
7. `reports/pipeline_report.md`
  - final human-readable summary of the full pipeline

## Model Artifacts to Save

At minimum, save:

- baseline model file (for example: `models/model_v1.pkl`)
- best classifier model file (for example: `models/baseline_model.pkl`)

## Makefile Requirements

Your `Makefile` should include targets equivalent to:

- `setup`
- `validate`
- `clean_data`
- `features`
- `train`
- `classify`
- `report`
- `pipeline`
- `test`
- code quality targets (`format`, `check`, `lint`, optionally `isort`)

## GitHub Actions (CI) Requirements

You need to create your own gitHub repo with this workflow

Your CI workflow should:

1. Run on `push` and `pull_request`
2. Set up Python
3. Install dependencies
4. Run make pipeline

