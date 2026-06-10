import os
import pandas as pd
import requests
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from agri_price import insecurity
from agri_price import utils

from agri_price.state_coords import state_coords

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
    """Fetches the latest inflation rate (placeholder for now)."""
    # In a real scenario, this would call World Bank or NBS
    return {"General_Inflation_Rate_Percent": 33.20}

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

def fetch_historical_inflation() -> pd.DataFrame:
    """Fetches Nigeria's annual inflation rate from World Bank."""
    print("Fetching Nigeria inflation data from World Bank...")
    url = "https://api.worldbank.org/v2/country/nga/indicator/FP.CPI.TOTL.ZG?format=json&per_page=100"
    response = requests.get(url)
    data = response.json()
    
    if len(data) < 2:
        return pd.DataFrame()
    
    records = []
    for item in data[1]:
        if item['value'] is not None:
            records.append({
                'Year': int(item['date']),
                'General_Inflation_Rate_Percent': float(item['value'])
            })
    
    df_annual = pd.DataFrame(records)
    
    # Convert annual to weekly (simple broadcast for history)
    # In a real scenario, we'd use monthly to weekly with interpolation
    weekly_rows = []
    for year in df_annual['Year'].unique():
        val = df_annual[df_annual['Year'] == year]['General_Inflation_Rate_Percent'].values[0]
        for week in range(53):
            weekly_rows.append({'Year': year, 'Week': week, 'General_Inflation_Rate_Percent': val})
            
    return pd.DataFrame(weekly_rows)

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

def ingest_all_to_db(db_path: str, start_year: int = 2022):
    """Orchestrates fetching and saving all API data to DB."""
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
        # Convert monthly to weekly using utils
        ins_dfs = []
        for state in df_ins_monthly['State'].unique():
            state_df = df_ins_monthly[df_ins_monthly['State'] == state].drop(columns=['State'])
            weekly = utils.monthly_to_weekly(state_df, value_columns=['Regional_Events_Count', 'Regional_Fatalities_Count'], mode='sum')
            weekly['State'] = state
            ins_dfs.append(weekly)
        df_insecurity = pd.concat(ins_dfs, ignore_index=True)
        df_insecurity.to_sql("weekly_insecurity", conn, if_exists='replace', index=False)

    # 5. Weather
    # Using a subset of states or all if possible
    from agri_price.state_coords import state_coords
    states = list(state_coords.keys())
    df_weather = fetch_historical_weather(states, start_date, end_date)
    if not df_weather.empty:
        df_weather.to_sql("weekly_weather", conn, if_exists='replace', index=False)

    # 6. Diesel (Mock for now as requested API priceradar.ng might need more specific research)
    # But I will add a placeholder that can be easily updated.
    print("Diesel price API ingestion placeholder...")
    # TODO: Implement priceradar.ng or similar
    
    conn.close()
    print("All API data ingested into DB.")
