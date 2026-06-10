from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
import numpy as np

import agri_price.predictor

# ==========================================
# 1. INITIALIZE SERVER & LOAD AI INTO RAM
# ==========================================
app = FastAPI(title="AgriPrice Prediction API", version="1.0")

print("Loading CatBoost Model...")
model, explainer = agri_price.predictor.load_model("models/agri_price_model.cbm")
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
        
        input_data = agri_price.predictor.build_input_df(raw_inputs, model)

        # Step D: Run the Prediction
        prediction = model.predict(input_data)[0]
        base_value, feature_impacts = agri_price.predictor.run_shap(explainer, input_data)

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