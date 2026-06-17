from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime

from agri_price.data.db_manager import load_data
from agri_price.core.utils import get_versioned_path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    db_path = repo_root / 'data' / 'feature_store.db'
    X, y, cat_features = load_data(str(db_path), table_name="historical_data")
    
    cat_features = [col for col in cat_features if col in X.columns]
    
    model = CatBoostRegressor(iterations=138, depth=5, learning_rate=0.1, verbose=0)
    model.fit(X, y, cat_features=cat_features)
    
    model_path = get_versioned_path("agri_price_model", "cbm", repo_root / "models")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()