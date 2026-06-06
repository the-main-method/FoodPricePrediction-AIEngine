import sqlite3
import requests
from datetime import datetime
import logging

# 1. Setup Logging (Crucial for Cron Jobs so we know if it failedearly)
logging.basicConfig(
    filename='feature_store_updates.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_live_weather():
    """Fetches real, live weather data for Kano using the free Open-Meteo API."""
    try:
        # Coordinates for Kano, Nigeria
        url = "https://api.open-meteo.com/v1/forecast?latitude=12.0022&longitude=8.5920&current=temperature_2m,precipitation&timezone=Africa%2FLagos"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "Avg_Temperature_C": data['current']['temperature_2m'],
            "Precipitation_mm": data['current']['precipitation']
        }
    except Exception as e:
        logging.error(f"Weather API failed: {e}")
        return None  # Return None so we know to use graceful degradation

def fetch_macro_economics():
    """
    Fetches the latest inflation rate. 
    (TODO: Replace with actual CBN API call, TradingEconomics, or web scraper)
    """
    try:
        # Mocking the live API call
        return {"General_Inflation_Rate": 33.20}
    except Exception as e:
        logging.error(f"Macro API failed: {e}")
        return None

def fetch_market_price_lags():
    """
    Calculates the 1M, 3M, 6M, and 1Y lags.
    (TODO: Connect this to your live production SQL database where daily vendor prices drop)
    """
    try:
        # Mocking the calculation from your raw vendor database
        return {
            "Price_Change_1M": 5.2,
            "Price_Change_3M": -16.19,
            "Price_Change_6M": 12.4,
            "Price_Change_1Y": 135.17
        }
    except Exception as e:
        logging.error(f"Market DB query failed: {e}")
        return None

def main():
    logging.info("Starting nightly feature store update...")
    
    # 1. Gather all live data
    weather = fetch_live_weather()
    macro = fetch_macro_economics()
    market = fetch_market_price_lags()
    current_month = datetime.now().month

    # 2. Connect to the local SQLite Feature Store
    conn = sqlite3.connect('feature_store.db')
    cursor = conn.cursor()

    # 3. Graceful Degradation Logic
    # We first pull yesterday's data. If any API failed today, we fallback to yesterday's number.
    cursor.execute("SELECT * FROM current_market_state WHERE id = 1")
    yesterday_data = cursor.fetchone()
    
    if yesterday_data is None:
        logging.warning("Feature store is empty. Forcing update with available data.")
        # Create a blank slate if the DB is completely empty (matches the 9 columns we created)
        yesterday_data = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_month)

    # 4. Construct the Final Update Payload
    # Format: id, Inflation, 1M, 3M, 6M, 1Y, Temp, Precip, Month
    updated_values = (
        1,
        macro['General_Inflation_Rate'] if macro else yesterday_data[1],
        market['Price_Change_1M'] if market else yesterday_data[2],
        market['Price_Change_3M'] if market else yesterday_data[3],
        market['Price_Change_6M'] if market else yesterday_data[4],
        market['Price_Change_1Y'] if market else yesterday_data[5],
        weather['Avg_Temperature_C'] if weather else yesterday_data[6],
        weather['Precipitation_mm'] if weather else yesterday_data[7],
        current_month
    )

    # 5. Push to Database
    cursor.execute('''
        INSERT OR REPLACE INTO current_market_state 
        (id, General_Inflation_Rate, Price_Change_1M, Price_Change_3M, Price_Change_6M, Price_Change_1Y, Avg_Temperature_C, Precipitation_mm, Month_Num)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', updated_values)

    conn.commit()
    conn.close()
    
    logging.info("Feature store update completed successfully.")
    print("Feature store updated successfully! Check feature_store_updates.log for details.")

if __name__ == "__main__":
    main()