import logging
import sqlite3
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import pandas as pd

from agri_price.ingestion.fetchers import weather, inflation
from agri_price.core.utils import get_season
from agri_price.data.feature_logic import fetch_market_price_lags
import agri_price.data.db as db

# 1. Setup Logging
logging.basicConfig(
    filename='feature_store_updates.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DEFAULT_STATE = 'Lagos'

def main(state: str = DEFAULT_STATE):
    logging.info("Starting nightly feature store update...")
    
    repo_root = Path(__file__).resolve().parents[2]
    db_path = str(repo_root / "data" / "feature_store.db")
    
    # 1. Gather all live data
    now = datetime.now()
    weather_data = weather.fetch_live_weather(state)
    macro = inflation.fetch_live_macro_economics()
    market_lags = fetch_market_price_lags(db_path)
    current_month = now.month
    current_season = get_season(current_month)

    # 2. Connect to the local SQLite or Remote Postgres Feature Store
    conn, is_pg = db.get_connection(db_path)

    # 3. Graceful Degradation Logic
    cursor = db.execute_query(conn, is_pg, "SELECT * FROM current_market_state WHERE id = 1")
    yesterday_data = cursor.fetchone()
    cursor.close()
    
    if yesterday_data is None:
        logging.warning("Feature store is empty. Forcing update with available data.")
        # Adjusted for new schema: id, Gen_Inf, Food_Inf, Price_1M, Price_3M, Price_6M, Price_1Y, Temp, Precip, Solar, Month, Season
        yesterday_data = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_month, "Unknown")

    # 4. Construct the Final Update Payload
    updated_values = (
        1,
        macro['General_Inflation_Rate_Percent'] if macro else yesterday_data[1],
        macro['Food_Inflation_Rate_Percent'] if macro else yesterday_data[2],
        market_lags['Price_Change_1M_Percent'] if market_lags else yesterday_data[3],
        market_lags['Price_Change_3M_Percent'] if market_lags else yesterday_data[4],
        market_lags['Price_Change_6M_Percent'] if market_lags else yesterday_data[5],
        market_lags['Price_Change_1Y_Percent'] if market_lags else yesterday_data[6],
        weather_data['Avg_Temperature_C'] if weather_data else yesterday_data[7],
        weather_data['Precipitation_mm'] if weather_data else yesterday_data[8],
        weather_data['Solar_Radiation_MJ'] if weather_data else yesterday_data[9],
        current_month,
        current_season
    )

    # 5. Push to Database
    sql = '''
        INSERT OR REPLACE INTO current_market_state 
        (id, General_Inflation_Rate_Percent, Food_Inflation_Rate_Percent, Price_Change_1M_Percent, Price_Change_3M_Percent, Price_Change_6M_Percent, Price_Change_1Y_Percent, Avg_Temperature_C, Precipitation_mm, Solar_Radiation_MJ, Month_Num, Season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    cursor = db.execute_query(conn, is_pg, sql, updated_values)
    cursor.close()

    conn.commit()
    conn.close()
    
    logging.info("Feature store update completed successfully.")
    print("Feature store updated successfully! Check feature_store_updates.log for details.")

if __name__ == "__main__":
    parser = ArgumentParser(description='Update the feature store for a specific state.')
    parser.add_argument('--state', default=DEFAULT_STATE, help='State to fetch weather for')
    args = parser.parse_args()
    main(args.state)
