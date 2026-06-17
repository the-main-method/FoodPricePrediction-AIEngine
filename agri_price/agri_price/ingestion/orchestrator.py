import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional

from agri_price.data.db_manager import create_schema
from agri_price.core import utils
from agri_price.ingestion.fetchers import diesel, inflation, insecurity, market, weather

def ingest_all_to_db(db_path: str, start_year: int = 2022, states: Optional[list[str]] = None):
    """Orchestrates fetching and saving all API data to DB."""
    create_schema(db_path)
    
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    
    # 1. Crude Oil
    df_crude = market.fetch_historical_crude_oil(start_date, end_date)
    if not df_crude.empty:
        df_crude.to_sql("weekly_crude_oil", conn, if_exists='replace', index=False)
        
    # 2. Exchange Rate
    df_ex = market.fetch_historical_exchange_rate(start_date, end_date)
    if not df_ex.empty:
        df_ex.to_sql("weekly_exchange_rate", conn, if_exists='replace', index=False)
        
    # 3. Inflation
    df_inf = inflation.fetch_historical_inflation()
    if not df_inf.empty:
        df_inf.to_sql("weekly_inflation", conn, if_exists='replace', index=False)
        
    # 4. Insecurity (ACLED)
    print("Fetching historical insecurity data from ACLED...")
    df_ins_monthly = insecurity.fetch_nigeria_insecurity(year=start_year)
    if not df_ins_monthly.empty:
        if states:
            df_ins_monthly = df_ins_monthly[df_ins_monthly['State'].isin(states)]
            
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
        from agri_price.core.state_coords import state_coords
        states = list(state_coords.keys())
    
    df_weather = weather.fetch_historical_weather(states, start_date, end_date)
    if not df_weather.empty:
        df_weather.to_sql("weekly_weather", conn, if_exists='replace', index=False)

    # 6. Diesel
    print("Fetching latest diesel prices...")
    df_diesel = diesel.fetch_latest_diesel_prices()
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
