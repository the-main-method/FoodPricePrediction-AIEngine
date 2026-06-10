import logging
import sqlite3
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import pandas as pd

from agri_price.ingestion import fetch_live_weather, fetch_live_macro_economics, get_season

# 1. Setup Logging
logging.basicConfig(
    filename='feature_store_updates.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DEFAULT_STATE = 'Lagos'

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

def main(state: str = DEFAULT_STATE):
    logging.info("Starting nightly feature store update...")
    
    repo_root = Path(__file__).resolve().parents[1]
    db_path = str(repo_root / "data" / "feature_store.db")
    
    # 1. Gather all live data
    now = datetime.now()
    weather = fetch_live_weather(state)
    macro = fetch_live_macro_economics()
    market = fetch_market_price_lags(db_path)
    current_month = now.month
    current_season = get_season(current_month)

    # 2. Connect to the local SQLite Feature Store
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 3. Graceful Degradation Logic
    cursor.execute("SELECT * FROM current_market_state WHERE id = 1")
    yesterday_data = cursor.fetchone()
    
    if yesterday_data is None:
        logging.warning("Feature store is empty. Forcing update with available data.")
        yesterday_data = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_month, "Unknown")

    # 4. Construct the Final Update Payload
    updated_values = (
        1,
        macro['General_Inflation_Rate_Percent'] if macro else yesterday_data[1],
        market['Price_Change_1M_Percent'] if market else yesterday_data[2],
        market['Price_Change_3M_Percent'] if market else yesterday_data[3],
        market['Price_Change_6M_Percent'] if market else yesterday_data[4],
        market['Price_Change_1Y_Percent'] if market else yesterday_data[5],
        weather['Avg_Temperature_C'] if weather else yesterday_data[6],
        weather['Precipitation_mm'] if weather else yesterday_data[7],
        weather['Solar_Radiation_MJ'] if weather else yesterday_data[8],
        current_month,
        current_season
    )

    # 5. Push to Database
    cursor.execute('''
        INSERT OR REPLACE INTO current_market_state 
        (id, General_Inflation_Rate_Percent, Price_Change_1M_Percent, Price_Change_3M_Percent, Price_Change_6M_Percent, Price_Change_1Y_Percent, Avg_Temperature_C, Precipitation_mm, Solar_Radiation_MJ, Month_Num, Season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', updated_values)

    conn.commit()
    conn.close()
    
    logging.info("Feature store update completed successfully.")
    print("Feature store updated successfully! Check feature_store_updates.log for details.")

if __name__ == "__main__":
    parser = ArgumentParser(description='Update the feature store for a specific state.')
    parser.add_argument('--state', default=DEFAULT_STATE, help='State to fetch weather for')
    args = parser.parse_args()
    main(args.state)
