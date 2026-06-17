import os
import pandas as pd
from acled import AcledClient
from typing import Optional

import dotenv
dotenv.load_dotenv()


def fetch_nigeria_insecurity(year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetches and processes ACLED insecurity data for Nigeria using the 'acled' library.
    Aggregates events to monthly level by Admin1 (State).
    
    Environment variables ACLED_EMAIL and ACLED_PASSWORD (or ACLED_USERNAME) 
    must be set for authentication.
    """
    if not (os.environ.get("ACLED_EMAIL") or os.environ.get("ACLED_USERNAME")) or not os.environ.get("ACLED_PASSWORD"):
        raise ValueError("ACLED_EMAIL (or ACLED_USERNAME) and ACLED_PASSWORD environment variables must be set.")
    
    client = AcledClient()
    
    all_events = []
    page = 1
    limit = 5000
    
    while True:
        events = client.get_data(
            country="Nigeria",
            year=year,
            limit=limit,
            page=page
        )
        if not events:
            break
        all_events.extend(events)
        if len(events) < limit:
            break
        page += 1
    
    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    
    # Convert event_date to datetime
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Month'] = df['event_date'].dt.month
    df['Year'] = df['event_date'].dt.year
    
    # Ensure fatalities is numeric
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
    
    # Aggregate by State (admin1), Year, Month
    df['Regional_Events_Count'] = 1
    
    agg_df = df.groupby(['admin1', 'Year', 'Month']).agg({
        'Regional_Events_Count': 'sum',
        'fatalities': 'sum'
    }).reset_index()
    
    agg_df.rename(columns={
        'admin1': 'State',
        'fatalities': 'Regional_Fatalities_Count'
    }, inplace=True)
    
    return agg_df


if __name__ == "__main__":
    # Example usage
    insecurity_df = fetch_nigeria_insecurity(year=2023)
    print(insecurity_df.head())