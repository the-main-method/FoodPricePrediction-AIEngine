# FoodPricePrediction-AIEngine

An AI-powered predictive engine designed to forecast national agricultural commodity prices. This project utilizes CatBoost for high-accuracy regression and SHAP (SHapley Additive exPlanations) to provide transparency into model decisions.

It integrates granular macroeconomic data from the **World Bank Data360 API**, real-time weather from Open Meteo, and conflict data from ACLED to provide a comprehensive view of market drivers.

## Project Structure
- `agri_price/`: Modular library for core logic, data handling, and API fetchers.
- `api/`: FastAPI backend for model serving.
- `dashboard/`: Streamlit dashboard for interactive simulations.
- `data/`: Datasets, raw files, and the SQLite feature store.
- `models/`: Trained model binaries (`.cbm`).
- `scripts/`: Numbered pipeline stages for the full data lifecycle.

## Data Pipeline Workflow

The engine operates in distinct stages. See [PIPELINE.md](PIPELINE.md) for detailed instructions.

1. **Collection**: `scripts/01_collection/build_dataset.py` (API + Local Excel merge)
2. **Database**: `scripts/02_database/setup_db.py` (Initialize Feature Store)
3. **Maintenance**: `scripts/03_maintenance/update_live_features.py` (Nightly API updates)
4. **Training**: `scripts/04_training/train_model.py` (Model persistence)
5. **Validation**: `scripts/05_validation/test_api.py` (API check)

## Quick Start

### 1. Environment Setup
```bash
uv sync
```

### 2. Initialize and Run
```bash
# Set up the database
uv run scripts/02_database/setup_db.py

# Launch services (in separate terminals)
uvicorn api.app:app --reload
streamlit run dashboard/app.py
```

## Documentation
- [API Documentation](api/README.md)
- [Dashboard Documentation](dashboard/README.md)
