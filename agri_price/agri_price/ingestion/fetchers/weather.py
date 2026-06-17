import pandas as pd
import requests
from agri_price.core.state_coords import state_coords

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

def fetch_historical_weather(states: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical weather data from Open Meteo.
    Requires state_coords for mapping.
    """
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
