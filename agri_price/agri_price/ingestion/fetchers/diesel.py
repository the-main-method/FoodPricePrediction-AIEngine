import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_depot_prices() -> pd.DataFrame:
    """
    Scrapes real-time ex-depot prices from DepotData.ng with retry logic and longer timeouts.
    """
    url = "https://www.depotdata.ng/"
    
    # Configure retries
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        print(f"Connecting to DepotData.ng (Site is known to be slow)...")
        response = session.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table = soup.find('table', class_='tabledesign')
        if not table:
            table = soup.find('table')
            
        if not table:
            print("Error: Could not find the price table on DepotData.ng")
            return pd.DataFrame()

        rows = []
        # Skip the header row
        for tr in table.find_all('tr')[1:]:
            # The first cell is often a <th> (Depot Name), others are <td>
            cols = tr.find_all(['td', 'th'])
            if len(cols) < 5:
                continue
                
            # 1. Depot Name & Location & Update Time
            # Use get_text(separator="|") to keep parts distinct
            depot_info_raw = cols[0].get_text(separator="|", strip=True)
            depot_info = depot_info_raw.split("|")
            
            # The structure seems to be: [Name, Address, Timestamp, UpdateStatus]
            name = depot_info[0] if len(depot_info) > 0 else "Unknown"
            location = depot_info[1] if len(depot_info) > 1 else "Unknown"
            updated = depot_info[2] if len(depot_info) > 2 else "Unknown"
            
            def clean_price(text):
                # Site shows change with +/- (e.g., "1,693+0.8")
                # Remove everything after the first '+' or '-'
                first_part = re.split(r'[+-]', text)[0].strip()
                clean = re.sub(r'[^\d.]', '', first_part)
                return float(clean) if clean else 0.0

            ago = clean_price(cols[1].get_text(strip=True))
            pms = clean_price(cols[2].get_text(strip=True))
            dpk = clean_price(cols[3].get_text(strip=True))
            atk = clean_price(cols[4].get_text(strip=True))
            
            rows.append({
                "Depot": name,
                "Location": location,
                "AGO": ago,
                "PMS": pms,
                "DPK": dpk,
                "ATK": atk,
                "Last_Updated": updated
            })

        if not rows:
            print("No price rows found in the table.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        
        def extract_state(loc):
            states = ["Lagos", "Rivers", "Delta", "Cross River", "Kaduna", "Kano", "Ogun", "Oyo", "Edo", "Akwa Ibom"]
            for s in states:
                if s.lower() in loc.lower():
                    return s
            return "Unknown"
            
        if 'Location' in df.columns:
            df['State'] = df['Location'].apply(extract_state)
        else:
            df['State'] = "Unknown"
            
        return df

    except requests.exceptions.Timeout:
        print("Error: DepotData.ng timed out after 60 seconds.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Scraping DepotData.ng failed: {e}")
        return pd.DataFrame()

def fetch_latest_diesel_prices() -> pd.DataFrame:
    """
    Fetches the latest diesel prices from DepotData.ng.
    """
    print("Fetching real-time diesel prices from DepotData.ng...")
    df_depot = fetch_depot_prices()
    if not df_depot.empty:
        # Standardize for the feature store
        df_depot = df_depot[df_depot['State'] != 'Unknown']
        if not df_depot.empty:
            # Average by State if multiple depots exist
            df_state = df_depot.groupby('State')['AGO'].mean().reset_index()
            df_state.rename(columns={'AGO': 'Diesel_Price_NGN'}, inplace=True)
            return df_state
            
    return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_depot_prices()
    if not df.empty:
        print(f"Successfully fetched {len(df)} depot records.")
        print(df.head())
    else:
        print("No data fetched.")
