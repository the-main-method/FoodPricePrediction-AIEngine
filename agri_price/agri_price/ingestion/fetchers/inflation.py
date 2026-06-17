import pandas as pd
from datetime import datetime
from agri_price.core import utils
from agri_price.ingestion.base import fetch_data360_indicator

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
    
    return {
        "General_Inflation_Rate_Percent": gen_rate,
        "Food_Inflation_Rate_Percent": food_rate
    }

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
