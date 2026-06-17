import calendar
import pandas as pd
import reverse_geocoder as rg
from pathlib import Path
from datetime import datetime

def path(path_str: str) -> Path:
    """Helper to ensure parent directory exists for a path."""
    _path = Path(path_str)
    _path.parent.mkdir(parents=True, exist_ok=True)
    return _path

def get_latest_file(pattern: str, base_path: Path) -> Path:
    """
    Finds the latest version of a file matching a pattern in a directory.
    Assumes versioning follows a timestamp pattern like _yymmdd or just alphanumeric sort.
    """
    files = list(base_path.glob(pattern))
    if not files:
        # Fallback to the exact pattern if no versioned files exist
        exact_file = base_path / pattern.replace("*", "")
        if exact_file.exists():
            return exact_file
        raise FileNotFoundError(f"No files matching {pattern} found in {base_path}")
    
    # Sort by name (which includes the timestamp)
    files.sort()
    return files[-1]

def get_versioned_path(base_name: str, extension: str, base_path: Path) -> Path:
    """Generates a versioned path with the current date."""
    now = datetime.now().strftime("%y%m%d")
    return base_path / f"{base_name}_{now}.{extension}"

def parse_coords(coord_string: str) -> tuple[float, float]:
    """Parses a 'lat,lon' string into a tuple of floats."""
    parts = coord_string.split(",")
    return float(parts[0]), float(parts[1])

def latlon_to_region(coords: tuple[float, float]) -> str:
    """Converts (lat, lon) to a region name (State) using reverse geocoding."""
    return rg.search(coords)[0]["admin1"]

def coords_to_region(coord_series: pd.Series) -> pd.Series:
    """Converts a series of 'lat,lon' strings to region names."""
    results = rg.search(coord_series.apply(parse_coords).tolist())
    return pd.Series([r["admin1"] for r in results])

def month_to_num(month_name: str) -> int:
    """Converts month name to its numeric representation (1-12)."""
    month_dict = {m: i for i, m in enumerate(calendar.month_name) if m}
    return month_dict.get(month_name, 0)

def monthly_to_weekly(df: pd.DataFrame, value_columns: list[str] | None = None, mode: str | list[str] = 'sum') -> pd.DataFrame:
    """
    Converts a monthly DataFrame into a weekly one (ISO %W weeks).
    
    Args:
        df: DataFrame with 'Year' and 'Month' columns.
        value_columns: Columns to resample. If None, uses all except Year/Month.
        mode: 'sum' for totals (counts), 'mean' for averages (prices/rates).
              Can be a list of modes corresponding to value_columns.
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

        # Get all ISO week numbers present in this calendar year
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
                        # Weighting logic:
                        # - sum: proportional to days in month
                        # - mean: proportional to days in the specific week
                        weight = overlap_days / (days_in_month if col_modes[col] == 'sum' else week_length)
                        weekly_row[col] += monthly_lookup[key][col] * weight

                cursor = month_end + pd.Timedelta(days=1)

            weekly_rows.append(weekly_row)

    return (
        pd.DataFrame(weekly_rows)
          .astype({'Year': 'int64', 'Week': 'int64'} | {col: 'float' for col in value_columns})
          .reset_index(drop=True)
    )

def get_season(month: int) -> str:
    """Determines the Nigerian season from the month."""
    if 4 <= month <= 10:
        return "Wet"
    return "Dry"
