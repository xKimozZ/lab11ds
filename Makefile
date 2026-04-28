PYTHON   := py -3.14
CONFIG   := configs/config.toml

.PHONY: help setup validate clean_data features train classify report pipeline \
        test format check lint isort

help:
	@echo "Available targets:"
	@echo "  setup        Install all dependencies"
	@echo "  validate     Validate raw and cleaned data"
	@echo "  clean_data   Preprocess raw data"
	@echo "  features     Engineer features"
	@echo "  train        Train baseline Random Forest"
	@echo "  classify     Benchmark classifiers and save best model"
	@echo "  report       Generate pipeline markdown report"
	@echo "  pipeline     Run full end-to-end pipeline"
	@echo "  test         Run unit tests"
	@echo "  format       Auto-format code with black"
	@echo "  isort        Sort imports with isort"
	@echo "  check        Check formatting without modifying files"
	@echo "  lint         Run flake8 linter"

# ── dependency installation ───────────────────────────────────────────────

setup:
	pip install -e ".[dev]"

# ── pipeline steps ─────────────────────────────────────────────────────────

validate:
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/raw/Teen_Mental_Health_Dataset.csv \
	    --output reports/validation_raw.json
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/processed/cleaned.csv \
	    --output reports/validation_cleaned.json

clean_data:
	$(PYTHON) src/data/preprocess.py --config $(CONFIG)

features:
	$(PYTHON) src/features/engineer.py --config $(CONFIG)

train:
	$(PYTHON) -m src.models.train --config $(CONFIG)

classify:
	$(PYTHON) -m src.models.classify --config $(CONFIG) --step both

report:
	$(PYTHON) src/reports/generate_report.py --config $(CONFIG)

pipeline:
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/raw/Teen_Mental_Health_Dataset.csv \
	    --output reports/validation_raw.json
	$(PYTHON) src/data/preprocess.py --config $(CONFIG)
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/processed/cleaned.csv \
	    --output reports/validation_cleaned.json
	$(PYTHON) src/features/engineer.py --config $(CONFIG)
	$(PYTHON) -m src.models.classify --config $(CONFIG) --step both
	$(PYTHON) src/reports/generate_report.py --config $(CONFIG)

# ── quality ────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

format:
	black src/ tests/

isort:
	isort src/ tests/

check:
	black --check src/ tests/
	isort --check-only src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

# ── housekeeping ───────────────────────────────────────────────────────────

clean:
	rm -rf data/processed reports models __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
