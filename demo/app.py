from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
from catboost import CatBoostRegressor
import shap
import numpy as np

# ==========================================
# 1. INITIALIZE SERVER & LOAD AI INTO RAM
# ==========================================
app = FastAPI(title="AgriPrice Prediction API", version="1.0")

print("Loading CatBoost Model...")
model = CatBoostRegressor().load_model("agri_price_model.cbm")
explainer = shap.TreeExplainer(model)
print("Model loaded and ready!")

# ==========================================
# 2. DEFINE THE FRONTEND PAYLOAD
# ==========================================
# The frontend only sends what the user clicked on the dashboard.
class PredictionRequest(BaseModel):
    food_item: str
    item_type: str
    category: str
    vendor_type: str

# ==========================================
# 3. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
async def predict_price(req: PredictionRequest):
    try:
        # Step A: Pull the "State of the World" from the local Feature Store
        try:
            conn = sqlite3.connect('feature_store.db')
            df_context = pd.read_sql_query("SELECT * FROM current_market_state WHERE id=1", conn)
            conn.close()
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"Database Error: Ensure setup_local_db.py was run. Details: {str(db_err)}")
        
        if df_context.empty:
            raise HTTPException(status_code=500, detail="Feature store is empty. Please run setup_local_db.py.")

        # Step B: Gather the inputs we DO have
        raw_inputs = {
            "food_item": req.food_item,
            "item_type": req.item_type,
            "category": req.category,
            "vendor_type": req.vendor_type,
            "Price_Change_1M": df_context['Price_Change_1M'].iloc[0],
            "Price_Change_3M": df_context['Price_Change_3M'].iloc[0],
            "Price_Change_6M": df_context['Price_Change_6M'].iloc[0],
            "Price_Change_1Y": df_context['Price_Change_1Y'].iloc[0],
            "Avg_Temperature_C": df_context['Avg_Temperature_C'].iloc[0],
            "Precipitation_mm": df_context['Precipitation_mm'].iloc[0],
            "General_Inflation_Rate": df_context['General_Inflation_Rate'].iloc[0],
            "Month_Num": df_context['Month_Num'].iloc[0]
        }

        # Step C: Dynamically rebuild the exact DataFrame the model expects
        expected_cols = model.feature_names_
        cat_indices = model.get_cat_feature_indices()
        cat_feature_names = [expected_cols[i] for i in cat_indices]

        final_input_dict = {}
        for col in expected_cols:
            if col in raw_inputs:
                # If we have the data, clean it and use it
                val = raw_inputs[col]
                if col in cat_feature_names:
                    final_input_dict[col] = str(val).lower().strip()
                else:
                    final_input_dict[col] = float(val)
            else:
                # If the model expects a column we didn't explicitly provide, pad it safely
                if col in cat_feature_names:
                    final_input_dict[col] = "unknown"
                else:
                    final_input_dict[col] = 0.0

        # Create the perfect single-row DataFrame
        input_data = pd.DataFrame([final_input_dict])

        # Step D: Run the Prediction
        prediction = model.predict(input_data)[0]

        # Step E: Ask SHAP why it made that prediction
        shap_values = explainer(input_data)
        
        # Handle SHAP base value typing safely
        base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
        feature_impacts = []
        for feature_name, shap_val, actual_val in zip(input_data.columns, shap_values.values[0], input_data.values[0]):
            if round(shap_val, 3) != 0:
                feature_impacts.append({
                    "feature": feature_name,
                    "current_value": float(actual_val) if isinstance(actual_val, (int, float, np.number)) else str(actual_val),
                    "impact_percentage": float(round(shap_val, 2)),
                    "direction": "increase" if shap_val > 0 else "decrease"
                })
                
        # Sort impacts from biggest driver to smallest driver
        feature_impacts.sort(key=lambda x: abs(x['impact_percentage']), reverse=True)

        # Step F: Return the finalized JSON Payload to the Frontend
        return {
            "metadata": {
                "food_item": req.food_item,
                "vendor_type": req.vendor_type
            },
            "forecast_horizon": "1_Month",
            "predicted_price_change_percent": float(round(prediction, 2)),
            "xai_explanation": {
                "base_market_trend": float(round(base_value, 2)),
                "top_driving_features": feature_impacts[:6] # Send top 6 drivers for UI chart
            }
        }

    except Exception as e:
        # If anything breaks, tell the frontend exactly why
        raise HTTPException(status_code=500, detail=str(e))