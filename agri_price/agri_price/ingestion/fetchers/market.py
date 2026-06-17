import pandas as pd
import yfinance as yf

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
