import pandas as pd
import sqlite3
from typing import Tuple, List

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

def load_data(path: str, table_name: str = "historical_data") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # Load the latest dataset (from CSV or SQL)
    if path.endswith('.db'):
        conn = sqlite3.connect(path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
    else:
        df = pd.read_csv(path)

    cat_features = df.select_dtypes(include=['object', 'string']).columns.tolist()

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    if 'Target_Price_Change_1M_Percent' in df.columns:
        target_col = 'Target_Price_Change_1M_Percent'
    elif 'TARGET_Price_Change_1M' in df.columns:
        target_col = 'TARGET_Price_Change_1M'
    else:
        target_col = [col for col in df.columns if 'TARGET' in col.upper()][0]

    X = df.drop(columns=['Year', 'Month', 'Week', target_col], errors='ignore')
    y = df[target_col]

    return X, y, cat_features

def save_to_db(df: pd.DataFrame, db_path: str, table_name: str, if_exists: str = 'replace'):
    """Saves a DataFrame to a SQLite database."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"Successfully saved {len(df)} rows to {table_name} in {db_path}.")

def load_from_db(db_path: str, table_name: str) -> pd.DataFrame:
    """Loads a DataFrame from a SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def build_combined_dataset(
    db_path: str,
    news_path: str,
    food_path: str
) -> pd.DataFrame:
    """
    Combines pre-processed database tables and specific file-based raw data
    into a single weekly dataset.
    """
    from agri_price.core import utils
    from agri_price.ingestion.fetchers import news
    
    conn = sqlite3.connect(db_path)
    
    # 1. News Sentiment (Remains File-based, but now cached)
    print("Processing News...")
    df_news_raw = pd.read_excel(news_path)
    df_news_raw = df_news_raw[pd.to_numeric(df_news_raw['id'], errors='coerce').notna()].reset_index(drop=True)
    df_news = news.process_news_dataframe(df_news_raw, db_path=db_path)

    # 2. Insecurity (Load from DB)
    print("Loading Insecurity...")
    df_insecurity = pd.read_sql_query("SELECT * FROM weekly_insecurity", conn)

    # 3. Weather (Load from DB)
    print("Loading Weather...")
    df_weather = pd.read_sql_query("SELECT * FROM weekly_weather", conn)

    # 4. Food Prices (Remains File-based as requested)
    print("Processing Food Prices...")
    df_food_raw = pd.read_excel(food_path, sheet_name='Sheet1')
    df_food_raw.sort_values(by='date', inplace=True)
    df_food_raw['year'] = df_food_raw['date'].dt.year
    df_food_raw['week'] = df_food_raw['date'].dt.strftime('%W').astype(int)
    df_food_raw['location'] = utils.coords_to_region(df_food_raw['location'])
    df_food = (
        df_food_raw.drop(columns=['date'])
        .groupby(['year', 'week', 'food_item', 'location'])
        .agg(Price_NGN=('price', 'mean'), Item_Type=('item_type', 'first'), Category=('category', 'first'))
        .reset_index()
        .rename(columns={'year': 'Year', 'week': 'Week', 'location': 'State', 'food_item': 'Food_Item'})
    )

    # 5. Diesel (Load from DB)
    print("Loading Diesel...")
    df_diesel = pd.read_sql_query("SELECT * FROM weekly_diesel", conn)

    # 6. Crude Oil (Load from DB)
    print("Loading Crude Oil...")
    df_crude = pd.read_sql_query("SELECT * FROM weekly_crude_oil", conn)

    # 7. Exchange Rate (Load from DB)
    print("Loading Exchange Rates...")
    df_exchange = pd.read_sql_query("SELECT * FROM weekly_exchange_rate", conn)

    # 8. Inflation (Load from DB)
    print("Loading Inflation...")
    df_inflation = pd.read_sql_query("SELECT * FROM weekly_inflation", conn)

    conn.close()

    # Final Merge
    print("Merging all sources...")
    combined = (
        df_food
        .merge(df_weather, on=['State', 'Year', 'Week'], how='left')
        .merge(df_news, on=['Year', 'Week'], how='left')
        .merge(df_insecurity, on=['State', 'Year', 'Week'], how='left')
        .merge(df_diesel, on=['State', 'Year', 'Week'], how='left')
        .merge(df_crude, on=['Year', 'Week'], how='left')
        .merge(df_exchange, on=['Year', 'Week'], how='left')
        .merge(df_inflation, on=['Year', 'Week'], how='left')
    )

    return combined
