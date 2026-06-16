from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3

import agri_price.core.predictor

# ==========================================
# 1. INITIALIZE SERVER & LOAD AI INTO RAM
# ==========================================
app = FastAPI(title="AgriPrice Prediction API", version="1.0")
model, explainer = agri_price.core.predictor.load_model("models/agri_price_model*.cbm")

# ==========================================
# 2. DEFINE THE FRONTEND PAYLOAD & PIPELINE
# ==========================================
# The frontend only sends what the user clicked on the dashboard.
class PredictionRequest(BaseModel):
    Food_Item: str
    Item_Type: str
    Category: str
    Vendor_Type: str

def run_prediction_pipeline(req: PredictionRequest, df_context: pd.DataFrame):
    """
    Executes the shared feature gathering, data building, model prediction, 
    and SHAP calculation pipeline.
    """
    def get_df_value(df, col_names: list[str], default=0.0):
        for col in col_names:
            if col in df.columns:
                val = df[col].iloc[0]
                return float(val) if pd.notna(val) else default
        return default

    # Gather all inputs from request and context
    raw_inputs = {
        "Food_Item": req.Food_Item,
        "Item_Type": req.Item_Type,
        "Category": req.Category,
        "Vendor_Type": req.Vendor_Type,
        "Price_Change_1M_Percent": get_df_value(df_context, ['Price_Change_1M_Percent', 'Price_Change_1M']),
        "Price_Change_3M_Percent": get_df_value(df_context, ['Price_Change_3M_Percent', 'Price_Change_3M']),
        "Price_Change_6M_Percent": get_df_value(df_context, ['Price_Change_6M_Percent', 'Price_Change_6M']),
        "Price_Change_1Y_Percent": get_df_value(df_context, ['Price_Change_1Y_Percent', 'Price_Change_1Y']),
        "Avg_Temperature_C": get_df_value(df_context, ['Avg_Temperature_C']),
        "Precipitation_mm": get_df_value(df_context, ['Precipitation_mm']),
        "Solar_Radiation_MJ": get_df_value(df_context, ['Solar_Radiation_MJ']),
        "General_Inflation_Rate_Percent": get_df_value(df_context, ['General_Inflation_Rate_Percent', 'General_Inflation_Rate']),
        "Weekly_Econ_Sentiment_Score": get_df_value(df_context, ['Weekly_Econ_Sentiment_Score']),
        "Exchange_Rate_NGN_USD": get_df_value(df_context, ['Exchange_Rate_NGN_USD']),
        "Diesel_Price_NGN": get_df_value(df_context, ['Diesel_Price_NGN']),
        "Month_Num": get_df_value(df_context, ['Month_Num'])
    }
    
    input_data = agri_price.core.predictor.build_input_df(raw_inputs, model)
    prediction = model.predict(input_data)[0]
    base_value, feature_impacts = agri_price.core.predictor.run_shap(explainer, input_data)
    
    return prediction, base_value, feature_impacts


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

        prediction, base_value, feature_impacts = run_prediction_pipeline(req, df_context)

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


# ==========================================
# 4. THE SIMULATION ENDPOINT
# ==========================================
class SimulationRequest(PredictionRequest):
    Year: int
    Month: int

@app.post("/simulate")
async def simulate_price(req: SimulationRequest):
    try:
        # Step A: Pull matching historical context from the feature store
        try:
            conn = sqlite3.connect('data/feature_store.db')
            
            # Try to find the exact match for item + date
            query_exact = """
                SELECT * FROM historical_data 
                WHERE LOWER(food_item) = LOWER(?) 
                  AND LOWER(item_type) = LOWER(?) 
                  AND LOWER(category) = LOWER(?) 
                  AND LOWER(vendor_type) = LOWER(?) 
                  AND Year = ? 
                  AND Month = ?
                LIMIT 1
            """
            df_context = pd.read_sql_query(
                query_exact, 
                conn, 
                params=(req.Food_Item, req.Item_Type, req.Category, req.Vendor_Type, req.Year, req.Month)
            )
            
            exact_match_found = not df_context.empty
            
            # Fallback: find any row for that year/month to get general macro & weather context
            if not exact_match_found:
                query_fallback = "SELECT * FROM historical_data WHERE Year = ? AND Month = ? LIMIT 1"
                df_context = pd.read_sql_query(query_fallback, conn, params=(req.Year, req.Month))
                
            conn.close()
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"Database Error: Ensure setup_db.py was run. Details: {str(db_err)}")
        
        if df_context.empty:
            raise HTTPException(status_code=404, detail=f"No historical records found for Year {req.Year}, Month {req.Month}.")

        prediction, base_value, feature_impacts = run_prediction_pipeline(req, df_context)

        # Step E: Determine Actual Target
        actual_price_change = None
        error_delta = None
        if exact_match_found:
            def get_df_value(df, col_names: list[str], default=0.0):
                for col in col_names:
                    if col in df.columns:
                        val = df[col].iloc[0]
                        return float(val) if pd.notna(val) else default
                return default
            actual_val = get_df_value(df_context, ['TARGET_Price_Change_1M'])
            actual_price_change = float(round(actual_val, 2))
            error_delta = float(round(prediction - actual_val, 2))

        # Step F: Return the finalized JSON Payload to the Frontend
        return {
            "metadata": {
                "Food_Item": req.Food_Item,
                "Vendor_Type": req.Vendor_Type,
                "Year": req.Year,
                "Month": req.Month,
                "exact_match_found": exact_match_found
            },
            "forecast_horizon": "1_Month",
            "predicted_price_change_percent": float(round(prediction, 2)),
            "actual_price_change_percent": actual_price_change,
            "error_delta_percent": error_delta,
            "xai_explanation": {
                "base_market_trend": float(round(base_value, 2)),
                "top_driving_features": feature_impacts[:6] # Send top 6 drivers for UI chart
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))