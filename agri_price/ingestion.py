import os
import pandas as pd
import requests
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from agri_price import insecurity
from agri_price import utils
from agri_price import depotdata

from agri_price.state_coords import state_coords

def fetch_latest_diesel_prices() -> pd.DataFrame:
    """
    Fetches the latest diesel prices from DepotData.ng.
    """
    print("Fetching real-time diesel prices from DepotData.ng...")
    df_depot = depotdata.fetch_depot_prices()
    if not df_depot.empty:
        # Standardize for the feature store
        df_depot = df_depot[df_depot['State'] != 'Unknown']
        if not df_depot.empty:
            # Average by State if multiple depots exist
            df_state = df_depot.groupby('State')['AGO'].mean().reset_index()
            df_state.rename(columns={'AGO': 'Diesel_Price_NGN'}, inplace=True)
            return df_state
            
    return pd.DataFrame()

def get_season(month: int) -> str:
    """Determines the Nigerian season from the month."""
    if 4 <= month <= 10:
        return "Wet"
    return "Dry"

def fetch_live_weather(state: str):
    """Fetches real, live weather data for a Nigerian state using the free Open-Meteo API."""
    if state not in state_coords:
        state = 'Lagos'
    
    lat, lon = state_coords[state]
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,shortwave_radiation&timezone=Africa%2FLagos"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Convert W/m^2 to MJ for the last hour
        watts_per_sq_meter = data['current']['shortwave_radiation']
        joules_per_sq_meter = watts_per_sq_meter * 3600
        megajoules_per_sq_meter = joules_per_sq_meter / 1_000_000
        
        return {
            "Avg_Temperature_C": data['current']['temperature_2m'],
            "Precipitation_mm": data['current']['precipitation'],
            "Solar_Radiation_MJ": megajoules_per_sq_meter
        }
    except Exception as e:
        print(f"Weather API failed: {e}")
        return None

def fetch_live_macro_economics():
    """Fetches the latest inflation rates from World Bank Data360."""
    print("Fetching live inflation data from Data360...")
    
    # 1. Food Price Inflation (Monthly)
    df_food = fetch_data360_indicator("FAO_CP_23014", "FAO_CP")
    # 2. General Consumer Price Inflation
    df_gen = fetch_data360_indicator("WB_WDI_FP_CPI_TOTL_ZG", "WB_WDI")
    
    # Get latest available value for each
    food_rate = df_food['Value'].iloc[-1] if not df_food.empty else 0.0
    gen_rate = df_gen['Value'].iloc[-1] if not df_gen.empty else 0.0
    
    # If latest is 0.0 (and we have other data), maybe try the one before it?
    # Usually the API returns 0.0 for missing, but let's assume Value is what we want.
    
    return {
        "General_Inflation_Rate_Percent": gen_rate,
        "Food_Inflation_Rate_Percent": food_rate
    }

def fetch_historical_crude_oil(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical Brent Crude Oil prices using yfinance."""
    print(f"Fetching Crude Oil prices from {start_date} to {end_date}...")
    data = yf.download("BZ=F", start=start_date, end=end_date)
    if data.empty:
        return pd.DataFrame()
    
    df = data[['Close']].reset_index()
    df.columns = ['Date', 'Crude_Oil_Price_USD']
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.strftime('%W').astype(int)
    
    # Weekly average
    weekly = df.groupby(['Year', 'Week'])['Crude_Oil_Price_USD'].mean().reset_index()
    return weekly

def fetch_historical_exchange_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical USD/NGN exchange rates using yfinance."""
    print(f"Fetching USD/NGN exchange rates from {start_date} to {end_date}...")
    data = yf.download("NGN=X", start=start_date, end=end_date)
    if data.empty:
        # Fallback to NGNUSD=X if NGN=X is unavailable
        data = yf.download("NGNUSD=X", start=start_date, end=end_date)
        if not data.empty:
             data['Close'] = 1 / data['Close']

    if data.empty:
        return pd.DataFrame()
    
    df = data[['Close']].reset_index()
    df.columns = ['Date', 'Exchange_Rate_NGN_USD']
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.strftime('%W').astype(int)
    
    # Weekly average
    weekly = df.groupby(['Year', 'Week'])['Exchange_Rate_NGN_USD'].mean().reset_index()
    return weekly

def fetch_data360_indicator(indicator_id: str, database_id: str, country_code: str = 'NGA') -> pd.DataFrame:
    """Fetches a specific indicator from the World Bank Data360 API."""
    print(f"Fetching {indicator_id} from Data360 ({database_id})...")
    url = f"https://data360api.worldbank.org/data360/data?DATABASE_ID={database_id}&INDICATOR={indicator_id}&REF_AREA={country_code}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'value' not in data:
            return pd.DataFrame()
            
        records = []
        for item in data['value']:
            if item['OBS_VALUE'] is not None:
                records.append({
                    'Date': item['TIME_PERIOD'],
                    'Value': float(item['OBS_VALUE'])
                })
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        return df
    except Exception as e:
        print(f"Failed to fetch {indicator_id}: {e}")
        return pd.DataFrame()

def fetch_historical_inflation() -> pd.DataFrame:
    """Fetches Nigeria's inflation data from World Bank Data360 (Monthly)."""
    print("Fetching Nigeria food and general inflation data from World Bank Data360...")
    
    # 1. Food Price Inflation (Monthly)
    df_food_inf = fetch_data360_indicator("FAO_CP_23014", "FAO_CP")
    # 2. General Consumer Price Inflation
    df_gen_inf = fetch_data360_indicator("WB_WDI_FP_CPI_TOTL_ZG", "WB_WDI")
    
    if df_food_inf.empty and df_gen_inf.empty:
        return pd.DataFrame()
        
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    def extend_monthly_and_fill(df, value_col):
        if df.empty: return df
        # Create a range of months from min to now
        min_date = df['Date'].min()
        target_date = datetime(current_year, current_month, 1)
        all_months = pd.date_range(start=min_date, end=target_date, freq='MS')
        df_all = pd.DataFrame({'Date': all_months})
        df_all['Year'] = df_all['Date'].dt.year
        df_all['Month'] = df_all['Date'].dt.month
        
        df = df_all.merge(df[['Year', 'Month', value_col]], on=['Year', 'Month'], how='left')
        df[value_col] = df[value_col].ffill().bfill()
        return df

    # Convert Monthly Food Inflation to Weekly
    df_weekly_food = pd.DataFrame()
    if not df_food_inf.empty:
        df_food_inf = df_food_inf.rename(columns={'Value': 'Food_Inflation_Rate_Percent'})
        df_food_inf = extend_monthly_and_fill(df_food_inf, 'Food_Inflation_Rate_Percent')
        df_weekly_food = utils.monthly_to_weekly(df_food_inf, value_columns=['Food_Inflation_Rate_Percent'], mode='mean')
        
    # Convert General Inflation to Weekly
    df_weekly_gen = pd.DataFrame()
    if not df_gen_inf.empty:
        df_gen_inf = df_gen_inf.rename(columns={'Value': 'General_Inflation_Rate_Percent'})
        # Check if it's monthly or annual by looking at Month distribution
        if df_gen_inf['Month'].nunique() > 1:
             df_gen_inf = extend_monthly_and_fill(df_gen_inf, 'General_Inflation_Rate_Percent')
             df_weekly_gen = utils.monthly_to_weekly(df_gen_inf, value_columns=['General_Inflation_Rate_Percent'], mode='mean')
        else:
             # Broadcast annual to weekly
             weekly_rows = []
             # Ffill annual data if missing years
             years = sorted(df_gen_inf['Year'].unique())
             all_years = range(min(years), current_year + 1)
             df_years = pd.DataFrame({'Year': all_years}).merge(df_gen_inf[['Year', 'General_Inflation_Rate_Percent']], on='Year', how='left')
             df_years['General_Inflation_Rate_Percent'] = df_years['General_Inflation_Rate_Percent'].ffill().bfill()
             
             for _, row in df_years.iterrows():
                 for week in range(53):
                     weekly_rows.append({
                         'Year': int(row['Year']), 
                         'Week': week, 
                         'General_Inflation_Rate_Percent': row['General_Inflation_Rate_Percent']
                     })
             df_weekly_gen = pd.DataFrame(weekly_rows)

    # Merge them
    if not df_weekly_food.empty and not df_weekly_gen.empty:
        return df_weekly_food.merge(df_weekly_gen, on=['Year', 'Week'], how='outer').ffill().bfill()
    elif not df_weekly_food.empty:
        return df_weekly_food
    else:
        return df_weekly_gen

def fetch_historical_weather(states: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical weather data from Open Meteo.
    Requires state_coords for mapping.
    """
    from agri_price.state_coords import state_coords
    
    all_weather = []
    for state in states:
        if state not in state_coords:
            continue
            
        lat, lon = state_coords[state]
        print(f"Fetching weather for {state}...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_mean,precipitation_sum,shortwave_radiation_sum&timezone=Africa%2FLagos"
        )
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'daily' not in data:
                continue
                
            df = pd.DataFrame(data['daily'])
            df['Date'] = pd.to_datetime(df['time'])
            df['Year'] = df['Date'].dt.year
            df['Week'] = df['Date'].dt.strftime('%W').astype(int)
            df['State'] = state
            
            weekly = df.groupby(['State', 'Year', 'Week']).agg({
                'temperature_2m_mean': 'mean',
                'precipitation_sum': 'sum',
                'shortwave_radiation_sum': 'mean'
            }).reset_index()
            
            weekly.rename(columns={
                'temperature_2m_mean': 'Avg_Temperature_C',
                'precipitation_sum': 'Precipitation_mm',
                'shortwave_radiation_sum': 'Solar_Radiation_MJ'
            }, inplace=True)
            
            all_weather.append(weekly)
        except Exception as e:
            print(f"Failed to fetch weather for {state}: {e}")
            
    if not all_weather:
        return pd.DataFrame()
        
    return pd.concat(all_weather, ignore_index=True)

def create_schema(db_path: str):
    """Ensures all required tables exist in the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tables for the data groups
    tables = {
        "weekly_weather": "State TEXT, Year INTEGER, Week INTEGER, Avg_Temperature_C REAL, Precipitation_mm REAL, Solar_Radiation_MJ REAL",
        "weekly_insecurity": "State TEXT, Year INTEGER, Week INTEGER, Regional_Events_Count INTEGER, Regional_Fatalities_Count INTEGER",
        "weekly_diesel": "State TEXT, Year INTEGER, Week INTEGER, Diesel_Price_NGN REAL",
        "weekly_crude_oil": "Year INTEGER, Week INTEGER, Crude_Oil_Price_USD REAL",
        "weekly_exchange_rate": "Year INTEGER, Week INTEGER, Exchange_Rate_NGN_USD REAL",
        "weekly_inflation": "Year INTEGER, Week INTEGER, General_Inflation_Rate_Percent REAL, Food_Inflation_Rate_Percent REAL",
        "weekly_news": "Year INTEGER, Week INTEGER, Weekly_Econ_Sentiment_Score REAL",
        "raw_news_sentiment": "id TEXT PRIMARY KEY, Sentiment_Score REAL",
        "historical_data": "State TEXT, Year INTEGER, Week INTEGER, Food_Item TEXT, Price_NGN REAL, Item_Type TEXT, Category TEXT, Vendor_Type TEXT"
    }
    
    for table_name, schema in tables.items():
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
        
    # Ensure current_market_state also exists (fallback if setup_db wasn't run)
    cursor.execute('''
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
    ''')
    
    conn.commit()
    conn.close()
    print("Database schema verified/created.")

def ingest_all_to_db(db_path: str, start_year: int = 2022, states: Optional[list[str]] = None):
    """Orchestrates fetching and saving all API data to DB."""
    create_schema(db_path)
    
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    
    # 1. Crude Oil
    df_crude = fetch_historical_crude_oil(start_date, end_date)
    if not df_crude.empty:
        df_crude.to_sql("weekly_crude_oil", conn, if_exists='replace', index=False)
        
    # 2. Exchange Rate
    df_ex = fetch_historical_exchange_rate(start_date, end_date)
    if not df_ex.empty:
        df_ex.to_sql("weekly_exchange_rate", conn, if_exists='replace', index=False)
        
    # 3. Inflation
    df_inf = fetch_historical_inflation()
    if not df_inf.empty:
        df_inf.to_sql("weekly_inflation", conn, if_exists='replace', index=False)
        
    # 4. Insecurity (ACLED)
    print("Fetching historical insecurity data from ACLED...")
    df_ins_monthly = insecurity.fetch_nigeria_insecurity(year=start_year)
    if not df_ins_monthly.empty:
        if states:
            # Filter for relevant states
            df_ins_monthly = df_ins_monthly[df_ins_monthly['State'].isin(states)]
            
        # Convert monthly to weekly using utils
        ins_dfs = []
        for state in df_ins_monthly['State'].unique():
            state_df = df_ins_monthly[df_ins_monthly['State'] == state].drop(columns=['State'])
            weekly = utils.monthly_to_weekly(state_df, value_columns=['Regional_Events_Count', 'Regional_Fatalities_Count'], mode='sum')
            weekly['State'] = state
            ins_dfs.append(weekly)
        
        if ins_dfs:
            df_insecurity = pd.concat(ins_dfs, ignore_index=True)
            df_insecurity.to_sql("weekly_insecurity", conn, if_exists='replace', index=False)

    # 5. Weather
    if not states:
        from agri_price.state_coords import state_coords
        states = list(state_coords.keys())
    
    df_weather = fetch_historical_weather(states, start_date, end_date)
    if not df_weather.empty:
        df_weather.to_sql("weekly_weather", conn, if_exists='replace', index=False)

    # 6. Diesel
    print("Fetching latest diesel prices...")
    df_diesel = fetch_latest_diesel_prices()
    if not df_diesel.empty:
        if states:
            df_diesel = df_diesel[df_diesel['State'].isin(states)]
            
        if not df_diesel.empty:
            now = datetime.now()
            df_diesel['Year'] = now.year
            df_diesel['Week'] = int(now.strftime('%W'))
            df_diesel.to_sql("weekly_diesel", conn, if_exists='append', index=False)
    
    conn.close()
    print("All API data ingested into DB.")
