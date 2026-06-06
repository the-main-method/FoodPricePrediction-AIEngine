import sqlite3

def setup_db():
    conn = sqlite3.connect('feature_store.db')
    cursor = conn.cursor()
    
    # Create the table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS current_market_state (
            id INTEGER PRIMARY KEY,
            General_Inflation_Rate REAL,
            Price_Change_1M REAL,
            Price_Change_3M REAL,
            Price_Change_6M REAL,
            Price_Change_1Y REAL,
            Avg_Temperature_C REAL,
            Precipitation_mm REAL,
            Month_Num REAL
        )
    ''')
    
    # Inserting MOCK data (You'll need to get real data)
    cursor.execute('''
        INSERT OR REPLACE INTO current_market_state 
        (id, General_Inflation_Rate, Price_Change_1M, Price_Change_3M, Price_Change_6M, Price_Change_1Y, Avg_Temperature_C, Precipitation_mm, Month_Num)
        VALUES (1, 32.7, 5.2, -16.19, 12.4, 135.17, 25.93, 29.17, 9.0)
    ''')
    
    conn.commit()
    conn.close()
    print("Local SQLite Feature Store created successfully!")

if __name__ == "__main__":
    setup_db()