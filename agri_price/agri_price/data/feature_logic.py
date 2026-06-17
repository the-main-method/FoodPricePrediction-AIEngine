import pandas as pd
import sqlite3
import logging

def fetch_market_price_lags(db_path: str):
    """
    Calculates the 1M, 3M, 6M, and 1Y lags from the historical database.
    """
    try:
        conn = sqlite3.connect(db_path)
        # Pull the latest record from historical_data
        df = pd.read_sql_query("SELECT * FROM historical_data ORDER BY Year DESC, Week DESC LIMIT 1", conn)
        conn.close()
        
        if df.empty:
            return None
            
        latest = df.iloc[0]
        # Map CSV/DB columns to the expected feature names
        return {
            "Price_Change_1M_Percent": float(latest.get('Price_Change_1M', latest.get('Price_Change_1M_Percent', 0.0))),
            "Price_Change_3M_Percent": float(latest.get('Price_Change_3M', latest.get('Price_Change_3M_Percent', 0.0))),
            "Price_Change_6M_Percent": float(latest.get('Price_Change_6M', latest.get('Price_Change_6M_Percent', 0.0))),
            "Price_Change_1Y_Percent": float(latest.get('Price_Change_1Y', latest.get('Price_Change_1Y_Percent', 0.0)))
        }
    except Exception as e:
        logging.error(f"Market DB query failed: {e}")
        return None
