import requests
import pandas as pd

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
