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
model, explainer = agri_price.predictor.load_model("models/agri_price_model.cbm")

# ==========================================
# 2. DEFINE THE FRONTEND PAYLOAD
# ==========================================
# The frontend only sends what the user clicked on the dashboard.
class PredictionRequest(BaseModel):
    Food_Item: str
    Item_Type: str
    Category: str
    Vendor_Type: str

# ==========================================
# 3. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
async def predict_price(req: PredictionRequest):
    try:
        # Step A: Pull the "State of the World" from the local Feature Store
        try:
            conn = sqlite3.connect('data/feature_store.db')
            df_context = pd.read_sql_query("SELECT * FROM current_market_state WHERE id=1", conn)
            conn.close()
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"Database Error: Ensure setup_db.py was run. Details: {str(db_err)}")
        
        if df_context.empty:
            raise HTTPException(status_code=500, detail="Feature store is empty. Please run setup_db.py.")

        # Step B: Gather the inputs we DO have
        raw_inputs = {
            "Food_Item": req.Food_Item,
            "Item_Type": req.Item_Type,
            "Category": req.Category,
            "Vendor_Type": req.Vendor_Type,
            "Price_Change_1M_Percent": df_context['Price_Change_1M_Percent'].iloc[0],
            "Price_Change_3M_Percent": df_context['Price_Change_3M_Percent'].iloc[0],
            "Price_Change_6M_Percent": df_context['Price_Change_6M_Percent'].iloc[0],
            "Price_Change_1Y_Percent": df_context['Price_Change_1Y_Percent'].iloc[0],
            "Avg_Temperature_C": df_context['Avg_Temperature_C'].iloc[0],
            "Precipitation_mm": df_context['Precipitation_mm'].iloc[0],
            "General_Inflation_Rate_Percent": df_context['General_Inflation_Rate_Percent'].iloc[0],
            "Month_Num": df_context['Month_Num'].iloc[0]
        }
        
        input_data = agri_price.predictor.build_input_df(raw_inputs, model)

        # Step D: Run the Prediction
        prediction = model.predict(input_data)[0]
        base_value, feature_impacts = agri_price.predictor.run_shap(explainer, input_data)

        # Step F: Return the finalized JSON Payload to the Frontend
        return {
            "metadata": {
                "Food_Item": req.Food_Item,
                "Vendor_Type": req.Vendor_Type
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