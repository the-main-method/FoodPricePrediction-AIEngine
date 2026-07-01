import sqlite3
import pandas as pd
import os
from pathlib import Path
from agri_price.core.utils import get_latest_file
import agri_price.data.db as db

def setup_db():
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / 'data' / 'feature_store.db'
    
    try:
        csv_path = get_latest_file("ml_ready_global_data*.csv", repo_root / 'data')
        print(f"Found latest dataset: {csv_path}")
    except FileNotFoundError:
        print("Error: No ml_ready_global_data CSV found. Cannot initialize Feature Store.")
        return
    
    conn, is_pg = db.get_connection(str(db_path))
    
    # 1. Create/Update the current_market_state table
    cursor = db.execute_query(conn, is_pg, 'DROP TABLE IF EXISTS current_market_state')
    cursor.close()
    
    sql_create = '''
        CREATE TABLE IF NOT EXISTS current_market_state (
            id INTEGER PRIMARY KEY,
            General_Inflation_Rate_Percent REAL,
            Food_Inflation_Rate_Percent REAL,
            Price_Change_1M_Percent REAL,
            Price_Change_3M_Percent REAL,
            Price_Change_6M_Percent REAL,
            Price_Change_1Y_Percent REAL,
            Avg_Temperature_C REAL,
            Precipitation_mm REAL,
            Solar_Radiation_MJ REAL,
            Month_Num REAL,
            Season TEXT
        )
    '''
    cursor = db.execute_query(conn, is_pg, sql_create)
    cursor.close()
    
    # 2. Port CSV to historical_data table
    if os.path.exists(csv_path):
        print(f"Porting {csv_path} to historical_data table...")
        df_historical = pd.read_csv(csv_path)
        db.to_sql(df_historical, 'historical_data', str(db_path), if_exists='replace')
        print("Successfully ported CSV to database.")
        
        # 3. Pull the most recent record to update current_market_state
        print("Updating current_market_state with the latest historical data...")
        # Get the latest row (assuming last row is most recent)
        latest_row = df_historical.iloc[-1]
        
        month_num = float(latest_row.get('Month_Num', 1.0))
        # Derive Season (Nigeria context: Wet season is roughly April-October)
        season = 'Wet' if 4 <= month_num <= 10 else 'Dry'
        
        # Mapping CSV columns to current_market_state schema
        sql_insert = '''
            INSERT OR REPLACE INTO current_market_state 
            (id, General_Inflation_Rate_Percent, Food_Inflation_Rate_Percent, Price_Change_1M_Percent, Price_Change_3M_Percent, Price_Change_6M_Percent, Price_Change_1Y_Percent, Avg_Temperature_C, Precipitation_mm, Solar_Radiation_MJ, Month_Num, Season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor = db.execute_query(conn, is_pg, sql_insert, (
            1, 
            float(latest_row.get('General_Inflation_Rate_Percent', latest_row.get('General_Inflation_Rate', 32.7))),
            float(latest_row.get('Food_Inflation_Rate_Percent', 40.0)),
            float(latest_row.get('Price_Change_1M_Percent', latest_row.get('Price_Change_1M', 0.0))),
            float(latest_row.get('Price_Change_3M_Percent', latest_row.get('Price_Change_3M', 0.0))),
            float(latest_row.get('Price_Change_6M_Percent', latest_row.get('Price_Change_6M', 0.0))),
            float(latest_row.get('Price_Change_1Y_Percent', latest_row.get('Price_Change_1Y', 0.0))),
            float(latest_row.get('Avg_Temperature_C', 25.0)),
            float(latest_row.get('Precipitation_mm', 0.0)),
            float(latest_row.get('Solar_Radiation_MJ', 15.0)),
            month_num,
            season
        ))
        cursor.close()
    else:
        print(f"Error: {csv_path} not found. Cannot initialize Feature Store without data.")
    
    conn.commit()
    conn.close()
    print("Local SQLite Feature Store updated successfully!")

if __name__ == "__main__":
    setup_db()