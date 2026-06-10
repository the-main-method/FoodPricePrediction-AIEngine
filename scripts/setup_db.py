import sqlite3
import pandas as pd
import os

def setup_db():
    db_path = 'data/feature_store.db'
    csv_path = 'data/ml_ready_global_data.csv'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create/Update the current_market_state table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS current_market_state (
            id INTEGER PRIMARY KEY,
            General_Inflation_Rate_Percent REAL,
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
    ''')
    
    # Inserting MOCK data (You'll need to get real data)
    cursor.execute('''
        INSERT OR REPLACE INTO current_market_state 
        (id, General_Inflation_Rate_Percent, Price_Change_1M_Percent, Price_Change_3M_Percent, Price_Change_6M_Percent, Price_Change_1Y_Percent, Avg_Temperature_C, Precipitation_mm, Solar_Radiation_MJ, Month_Num, Season)
        VALUES (1, 32.7, 5.2, -16.19, 12.4, 135.17, 25.93, 29.17, 15.2, 9.0, 'Dry')
    ''')
    
    # 2. Port CSV to historical_data table
    if os.path.exists(csv_path):
        print(f"Porting {csv_path} to historical_data table...")
        df_historical = pd.read_csv(csv_path)
        df_historical.to_sql('historical_data', conn, if_exists='replace', index=False)
        print("Successfully ported CSV to SQLite.")
    else:
        print(f"Warning: {csv_path} not found. Skipping historical data port.")
    
    conn.commit()
    conn.close()
    print("Local SQLite Feature Store updated successfully!")

if __name__ == "__main__":
    setup_db()