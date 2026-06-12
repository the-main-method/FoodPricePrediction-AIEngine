# Data Pipeline Guide

This document explains how and when to run the scripts in the `scripts/` directory to manage the FoodPricePrediction engine.

---

## Stage 01: Collection
**Goal**: Assemble the historical master dataset for training.

### `01_collection/bootstrap_diesel.py`
- **When to run**: Once, or whenever you have a new batch of historical diesel data in CSV format that isn't available via real-time APIs.
- **How**: `uv run scripts/01_collection/bootstrap_diesel.py --input-file data/raw/diesel_history.csv`

### `01_collection/build_dataset.py`
- **When to run**: Whenever you want to re-train the model on fresh data. It triggers all APIs (World Bank, ACLED, etc.) to fetch history and merges them with local Excel files (`data/raw/`).
- **How**: `uv run scripts/01_collection/build_dataset.py`
- **Output**: Generates `data/ml_ready_global_data.csv`.

---

## Stage 02: Database
**Goal**: Initialize the Feature Store (the "State of the World").

### `02_database/setup_db.py`
- **When to run**: During first-time setup or after a major schema change. It populates the SQLite Feature Store from the master CSV generated in Stage 01.
- **How**: `uv run scripts/02_database/setup_db.py`
- **Output**: Populates `data/feature_store.db`.

---

## Stage 03: Maintenance
**Goal**: Keep real-time data fresh for live predictions.

### `03_maintenance/update_live_features.py`
- **When to run**: Every 24 hours (e.g., via a Cron job). It fetches the latest weather, exchange rates, and inflation from APIs to ensure predictions are context-aware.
- **How**: `uv run scripts/03_maintenance/update_live_features.py --state Lagos`

---

## Stage 04: Training
**Goal**: Persist the brain of the engine.

### `04_training/train_model.py`
- **When to run**: After generating a new master CSV in Stage 01.
- **How**: `uv run scripts/04_training/train_model.py`
- **Output**: Saves a new `.cbm` file in the `models/` directory.

---

## Stage 05: Validation
**Goal**: Ensure everything is working.

### `05_validation/test_api.py`
- **When to run**: After launching the API service to confirm it's communicating correctly with the Feature Store and Model.
- **How**: `uv run scripts/05_validation/test_api.py`
