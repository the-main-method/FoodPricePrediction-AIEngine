from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime

from agri_price.data import load_data


def main():
    repo_root = Path(__file__).resolve().parents[1]
    
    data_path = repo_root / 'data' / 'ml_ready_global_data.csv'
    X, y, cat_features = load_data(str(data_path))
    
    cat_features = [col for col in cat_features if col in X.columns]
    
    model = CatBoostRegressor(iterations=138, depth=5, learning_rate=0.1, verbose=0)
    model.fit(X, y, cat_features=cat_features)
    
    now = datetime.now().strftime("%y%m%d")
    
    model_path = repo_root / 'models' / f'agri_price_model_{now}.cbm'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))


if __name__ == "__main__":
    main()