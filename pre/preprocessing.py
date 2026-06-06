from pathlib import Path
import calendar
import json
import time
import warnings
from typing import Sequence

import pandas as pd
import requests
import reverse_geocoder as rg
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# State coordinates
# ---------------------------------------------------------------------------

state_coords: dict[str, tuple[float, float]] = {
    "Abia": (5.416667, 7.5),
    "Adamawa": (9.333333, 12.5),
    "Akwa Ibom": (5, 7.833333),
    "Anambra": (6.333333, 7),
    "Bauchi": (10.306667, 9.8125),
    "Bayelsa": (4.75, 6.083333),
    "Benue": (7.333333, 8.75),
    "Borno": (11.5, 13),
    "Cross River": (5.75, 8.5),
    "Delta": (5.5, 6),
    "Ebonyi": (6.25, 8.083333),
    "Edo": (6.3381, 5.6074),
    "Ekiti": (7.6154, 5.2260),
    "Enugu": (6.4483, 7.5123),
    "Federal Capital Territory": (9.0720, 7.4394),
    "Gombe": (10.2784, 11.1684),
    "Imo": (5.5120, 7.0287),
    "Jigawa": (12.1957, 9.4039),
    "Kaduna": (10.5036, 7.4334),
    "Kano": (11.7862, 8.4780),
    "Katsina": (12.9775, 7.7067),
    "Kebbi": (12.4528, 4.2580),
    "Kogi": (7.8033, 6.7307),
    "Kwara": (8.5376, 4.5433),
    "Lagos": (6.5378, 3.2963),
    "Nassarawa": (8.5020, 8.5165),
    "Niger": (9.6218, 6.4258),
    "Ogun": (7.0880, 3.3480),
    "Ondo": (7.0585, 5.1019),
    "Osun": (7.5676, 4.4643),
    "Oyo": (8.1865, 3.5491),
    "Plateau": (9.1978, 9.3978),
    "Rivers": (4.8352, 6.9099),
    "Sokoto": (13.0355, 5.2316),
    "Taraba": (7.9739, 10.6150),
    "Yobe": (12.0355, 11.6430),
    "Zamfara": (12.0269, 6.2392),
}

states_coords_df = pd.DataFrame.from_dict(state_coords, orient='index', columns=['Latitude', 'Longitude'])
states_coords_df.index.name = 'State'
states_coords_df.reset_index(inplace=True)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def path(path: str) -> Path:
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    return _path

def parse_coords(coord_string: str) -> tuple[float, float]:
    parts = coord_string.split(",")
    return float(parts[0]), float(parts[1])

def latlon_to_region(coords: tuple[float, float]) -> str:
    return rg.search(coords)[0]["admin1"]

def coords_to_region(coord_series: pd.Series) -> pd.Series:
    results = rg.search(coord_series.apply(parse_coords).tolist())
    return pd.Series([r["admin1"] for r in results])

def month_to_num(month_name: str) -> int:
    month_dict = {m: i for i, m in enumerate(calendar.month_name) if m}
    return month_dict.get(month_name, 0)

def monthly_to_weekly(df, value_columns: list[str] | None = None, mode: str | list[str] = 'sum'):
    """Converts a monthly DataFrame into a weekly one (%W weeks).

    mode='sum'                - proportionally distribute totals (counts, events).
    mode='mean'               - day-weighted average (prices, rates).
    mode=['sum', 'mean', ...] - per-column modes; value_columns must be specified,
                                with one entry per column in the same order.
    """
    valid_modes = ('sum', 'mean')

    if isinstance(mode, list):
        if value_columns is None:
            raise ValueError("value_columns must be specified when mode is a list")
        if len(mode) != len(value_columns):
            raise ValueError(f"mode has {len(mode)} entries but value_columns has {len(value_columns)}")
        bad = [m for m in mode if m not in valid_modes]
        if bad:
            raise ValueError(f"Invalid mode values: {bad}. Each must be 'sum' or 'mean'")
        col_modes = dict(zip(value_columns, mode))
    else:
        if mode not in valid_modes:
            raise ValueError(f"mode must be 'sum' or 'mean', got {mode!r}")
        if value_columns is None:
            value_columns = [col for col in df.columns if col not in ['Year', 'Month']]
        col_modes = {col: mode for col in value_columns}

    monthly_lookup = {
        (row.Year, row.Month): {col: getattr(row, col) for col in value_columns}
        for row in df.itertuples(index=False)
    }

    weekly_rows = []

    for cal_year in df['Year'].unique():
        year_start = pd.Timestamp(year=cal_year, month=1, day=1)
        year_end   = pd.Timestamp(year=cal_year, month=12, day=31)

        weeks_in_year = sorted({
            int(day.strftime('%W'))
            for day in pd.date_range(year_start, year_end)
        })

        jan1 = year_start
        first_monday = jan1 + pd.Timedelta(days=(7 - jan1.weekday()) % 7)

        for week_num in weeks_in_year:
            if week_num == 0:
                week_start = jan1
                week_end   = first_monday - pd.Timedelta(days=1)
            else:
                week_start = first_monday + pd.Timedelta(weeks=week_num - 1)
                week_end   = week_start + pd.Timedelta(days=6)

            week_length = (week_end - week_start).days + 1
            weekly_row  = {'Year': cal_year, 'Week': week_num, **{col: 0.0 for col in value_columns}}

            cursor = week_start
            while cursor <= week_end:
                month_start   = cursor.replace(day=1)
                month_end     = month_start + pd.offsets.MonthEnd(0)
                days_in_month = month_end.day

                overlap_days = (min(week_end, month_end) - max(week_start, month_start)).days + 1

                key = (cursor.year, cursor.month)
                if key in monthly_lookup:
                    for col in value_columns:
                        weight = overlap_days / (days_in_month if col_modes[col] == 'sum' else week_length)
                        weekly_row[col] += monthly_lookup[key][col] * weight

                cursor = month_end + pd.Timedelta(days=1)

            weekly_rows.append(weekly_row)

    return (
        pd.DataFrame(weekly_rows)
          .astype({'Year': 'int64', 'Week': 'int64'} | {col: 'float' for col in value_columns})
          .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


class ListDataset(Dataset):
    def __init__(self, original_list: Sequence[str]):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def news():
    print("Loading data...")
    df = pd.read_excel('news 2022-2024.xlsx')
    df = df[pd.to_numeric(df['id'], errors='coerce').notna()].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    print("Parsing text data...")
    def parse_body_json(json_string):
        try:
            return json.loads(json_string).get('body', '')
        except:
            return ""

    df['Body_Text'] = df['article'].apply(parse_body_json)
    df['Combined_Text'] = df['title'] + ". " + df['Body_Text'].str[:300]

    print("Initializing FinBERT...")
    nlp = pipeline("text-classification", model="ProsusAI/finbert")

    print(f"Scoring {len(df)} news articles...")
    sentiments = list(tqdm(
        nlp(ListDataset(df['Combined_Text']), truncation=True, max_length=512)
    ))

    df['Sentiment_Score'] = [
        s['score'] if s['label'] == 'positive' else
        (-s['score'] if s['label'] == 'negative' else 0.0)
        for s in sentiments
    ]

    print("Aggregating to Weekly Data...")
    df['Year'] = df['date'].dt.year
    df['Week'] = df['date'].dt.strftime('%W').astype(int)

    weekly = df.groupby(['Year', 'Week'])['Sentiment_Score'].mean().reset_index()
    weekly.rename(columns={'Sentiment_Score': 'Weekly_Econ_Sentiment_Score'}, inplace=True)
    return weekly


def insecurity():
    print("Loading ACLED Data...")
    file_path = 'nigeria_hrp_political_violence_events_and_fatalities_by_month-year_as-of-29apr2026.xlsx'
    df = pd.read_excel(file_path, sheet_name='Data')

    print("Cleaning and standardizing...")
    df = df[['Admin1', 'Month', 'Year', 'Events', 'Fatalities']]
    df['Month'] = df['Month'].apply(month_to_num)
    df = df.sort_values(by=['Admin1', 'Year', 'Month'])

    print("Aggregating by State and Month...")
    state_monthly = df.groupby(['Admin1', 'Year', 'Month'])[['Events', 'Fatalities']].sum().reset_index()

    state_monthly.rename(columns={
        'Admin1': 'State'
    }, inplace=True)

    return state_monthly


NASA_CACHE_DIR = Path('.tmp/nasa')
START_DATE = "20160101"
END_DATE = "20260430"
PARAMETERS = "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"

def fetch_nasa_data(state, lat, lon):
    cache_path = NASA_CACHE_DIR / f"{state.replace(' ', '_')}.csv"

    if cache_path.exists():
        print(f"Loading cached data for {state}...")
        return pd.read_csv(cache_path), True

    print(f"Fetching data for {state} from NASA POWER...")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={PARAMETERS}&community=AG&longitude={lon}&latitude={lat}"
        f"&start={START_DATE}&end={END_DATE}&format=JSON"
    )

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {state}: HTTP {response.status_code}")
        return None, False

    data = response.json()
    temp_data   = data['properties']['parameter']['T2M']
    precip_data = data['properties']['parameter']['PRECTOTCORR']
    solar_data  = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']

    df = pd.DataFrame({
        'Date_Raw':          list(temp_data.keys()),
        'Avg_Temperature_C': list(temp_data.values()),
        'Precipitation_mm':  list(precip_data.values()),
        'Solar_Radiation_MJ':list(solar_data.values()),
    })
    df.replace(-999.0, pd.NA, inplace=True)
    df['State'] = state

    NASA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    return df, False


def weather():
    all_states_data = []

    for state, (lat, lon) in state_coords.items():
        df, from_cache = fetch_nasa_data(state, lat, lon)
        if df is not None:
            all_states_data.append(df)
        if not from_cache:
            time.sleep(2)

    master_df = pd.concat(all_states_data, ignore_index=True)

    print("\nProcessing and Aggregating to Weekly Timelines...")
    master_df['Date'] = pd.to_datetime(master_df['Date_Raw'], format='%Y%m%d')
    master_df['Year'] = master_df['Date'].dt.year
    master_df['Week'] = master_df['Date'].dt.strftime('%W').astype(int)

    return master_df.groupby(['State', 'Year', 'Week']).agg(
        Precipitation_mm   =('Precipitation_mm',   'sum'),
        Avg_Temperature_C  =('Avg_Temperature_C',  'mean'),
        Solar_Radiation_MJ =('Solar_Radiation_MJ', 'mean'),
    ).reset_index()


def food():
    df = pd.read_excel('Sample 2022-2024.xlsx', sheet_name='Sheet1')
    df.sort_values(by='date', inplace=True)

    df['year']     = df['date'].dt.year
    df['week']     = df['date'].dt.strftime('%W').astype(int)
    df['location'] = coords_to_region(df['location'])

    weekly_df = (
        df.drop(columns=['date'])
          .groupby(['year', 'week', 'food_item', 'location'])
          .agg(price=('price', 'mean'), item_type=('item_type', 'first'), category=('category', 'first'))
          .reset_index()
          .rename(columns={'year': 'Year', 'week': 'Week', 'location': 'State'})
    )
    return weekly_df


def diesel():
    df = pd.read_excel(
        'fpp-data/Diesel.xlsx',
        sheet_name='Automotive Gas Oil (Diesel) Pri',
        header=6,
    )
    df = df[df['State'] != 'Nigeria'].drop(columns=['Units'])
    df = df.melt(id_vars=['State'], var_name='Date', value_name='Diesel_Price_NGN')

    parsed = pd.to_datetime(df['Date'], format='%b %Y')
    df['Year']  = parsed.dt.year
    df['Month'] = parsed.dt.month
    df = df.drop(columns=['Date'])

    df = df.dropna(subset=['Diesel_Price_NGN'])
    return df.sort_values(['State', 'Year', 'Month']).reset_index(drop=True)


def crude_oil():
    df = pd.read_excel('fpp-data/Crude_Oil.xlsx')
    df.columns = ['Date', 'Crude_Oil_Price']
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.strftime('%W').astype(int)
    return df.groupby(['Year', 'Week'])['Crude_Oil_Price'].mean().reset_index()


def exchange_rate():
    df = pd.read_excel('fpp-data/Exchange_rates.xlsx')
    df['Date'] = pd.to_datetime(df['ratedate'], format='%B-%d-%Y')
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.strftime('%W').astype(int)
    return (
        df.groupby(['Year', 'Week'])
          .agg(Exchange_Rate=('weightedAvgRate', 'mean'))
          .reset_index()
    )


def inflation():
    df = pd.read_excel('fpp-data/Inflation.xlsx')
    return df.rename(columns={
        'tyear':                     'Year',
        'tmonth':                    'Month',
        'foodYearOn':                'Inflation_Food_YoY',
        'allItemsLessFrmProdYearOn': 'Inflation_Core_YoY',
    })[['Year', 'Month', 'Inflation_Food_YoY', 'Inflation_Core_YoY']]

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_state_monthly_to_weekly(monthly_df, name, **kwargs):
    """Saves the three standard artefacts for a state-level monthly dataset:
      processed/monthly/{name}.csv
      processed/weekly/{name}/{state}.csv
      processed/weekly/{name}.csv
    Returns the combined weekly DataFrame.
    """
    Path(f'processed/monthly').mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(f'processed/monthly/{name}.csv', index=False)

    state_folder = Path(f'processed/weekly/{name}')
    state_folder.mkdir(parents=True, exist_ok=True)

    dfs = []
    for state in monthly_df['State'].unique():
        state_monthly = monthly_df[monthly_df['State'] == state].drop(columns=['State'])
        weekly = monthly_to_weekly(state_monthly, **kwargs)
        weekly['State'] = state
        dfs.append(weekly)
        weekly.to_csv(state_folder / f'{state}.csv', index=False)

    weekly_df = pd.concat(dfs, ignore_index=True)
    weekly_df.to_csv(f'processed/weekly/{name}.csv', index=False)
    return weekly_df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    states_coords_df.to_csv(path("processed/states_coordinates.csv"), index=False)

    news_df = news()
    news_df.to_csv(path('processed/weekly/news.csv'), index=False)

    weekly_insecurity_df = save_state_monthly_to_weekly(insecurity(), 'insecurity')
    weekly_diesel_df     = save_state_monthly_to_weekly(diesel(), 'diesel', mode='mean')

    inflation_monthly = inflation()
    Path('processed/monthly').mkdir(parents=True, exist_ok=True)
    inflation_monthly.to_csv('processed/monthly/inflation.csv', index=False)
    Path('processed/weekly/inflation').mkdir(parents=True, exist_ok=True)
    inflation_df = monthly_to_weekly(inflation_monthly, mode='mean')
    inflation_df.to_csv('processed/weekly/inflation.csv', index=False)

    weather_df = weather()
    weather_df.to_csv(path('processed/weekly/weather.csv'), index=False)

    food_df = food()
    food_df.to_csv(path('processed/weekly/food.csv'), index=False)

    crude_oil_df = crude_oil()
    crude_oil_df.to_csv(path('processed/weekly/crude_oil.csv'), index=False)

    exchange_rate_df = exchange_rate()
    exchange_rate_df.to_csv(path('processed/weekly/exchange_rate.csv'), index=False)

    combined_df = (
        food_df
        .merge(weather_df,          on=['State', 'Year', 'Week'], how='left')
        .merge(news_df,             on=['Year', 'Week'],           how='left')
        .merge(weekly_insecurity_df,on=['State', 'Year', 'Week'], how='left')
        .merge(weekly_diesel_df,    on=['State', 'Year', 'Week'], how='left')
        .merge(crude_oil_df,        on=['Year', 'Week'],           how='left')
        .merge(exchange_rate_df,    on=['Year', 'Week'],           how='left')
        .merge(inflation_df,        on=['Year', 'Week'],           how='left')
    )
    combined_df.to_csv('processed/weekly/combined.csv', index=False)
    print(f"Done. Combined dataset: {combined_df.shape}")
