from catboost import CatBoostRegressor
import shap
import pandas as pd
from typing import Any


def load_model(path: str) -> tuple[CatBoostRegressor, shap.TreeExplainer]:
    print("Loading CatBoost Model...")
    model = CatBoostRegressor().load_model(path)
    explainer = shap.TreeExplainer(model)
    print("Model loaded and ready!")
    
    return model, explainer

def build_input_df(raw_inputs: dict[str, Any], model: CatBoostRegressor) -> pd.DataFrame:
    expected_cols = list(getattr(model, "feature_names_", []) or [])
    cat_indices = list(getattr(model, "get_cat_feature_indices", lambda: [])())
    cat_feature_names = [expected_cols[i] for i in cat_indices if i < len(expected_cols)]

    final_input_dict: dict[str, Any] = {}
    for col in expected_cols:
        if col in raw_inputs:
            val = raw_inputs[col]
            if col in cat_feature_names:
                final_input_dict[col] = str(val).lower().strip()
            else:
                try:
                    final_input_dict[col] = float(val)
                except (TypeError, ValueError):
                    final_input_dict[col] = 0.0
        else:
            final_input_dict[col] = "unknown" if col in cat_feature_names else 0.0

    if not final_input_dict:
        return pd.DataFrame([raw_inputs])

    return pd.DataFrame([final_input_dict])

def run_shap(explainer: shap.TreeExplainer, input_df: pd.DataFrame):
    shap_values = explainer(input_df)
    base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, pd.Series)) else explainer.expected_value
    
    feature_impacts = []
    for feature_name, shap_val, actual_val in zip(input_df.columns, shap_values.values[0], input_df.values[0]):
        if round(shap_val, 3) != 0:
            feature_impacts.append({
                "feature": feature_name,
                "current_value": float(actual_val) if pd.api.types.is_numeric_dtype(type(actual_val)) else str(actual_val),
                "impact_percentage": float(round(shap_val, 2)),
                "direction": "increase" if shap_val > 0 else "decrease"
            })
            
    feature_impacts.sort(key=lambda x: abs(x['impact_percentage']), reverse=True)
    
    return base_value, feature_impacts