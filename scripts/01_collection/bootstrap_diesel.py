import argparse
import pandas as pd
import sqlite3
from pathlib import Path
import sys

def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Bootstrap historical diesel data into the database from a CSV.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the historical Diesel CSV file")
    parser.add_argument("--db-path", type=str, default=str(repo_root / "data" / "feature_store.db"), help="Path to SQLite DB")
    
    args = parser.parse_args()
    input_path = Path(args.input_file)
    db_path = Path(args.db_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
        
    print(f"Reading historical diesel data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        
        # Required columns
        required_cols = {'State', 'Year', 'Week', 'Diesel_Price_NGN'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Error: CSV is missing required columns: {missing}")
            sys.exit(1)
            
        # Clean up and ensure data types
        df = df.copy()
        df['Year'] = df['Year'].astype(int)
        df['Week'] = df['Week'].astype(int)
        df['Diesel_Price_NGN'] = pd.to_numeric(df['Diesel_Price_NGN'], errors='coerce')
        df = df.dropna(subset=['Diesel_Price_NGN'])
        
        # Keep only required columns for the database
        df = df[['State', 'Year', 'Week', 'Diesel_Price_NGN']]
        
        print(f"Saving {len(df)} weekly records to database...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        # We use 'replace' to ensure we have a clean slate for the bootstrap
        df.to_sql("weekly_diesel", conn, if_exists='replace', index=False)
        conn.close()
        print(f"Historical diesel bootstrap complete. Data saved to {db_path}.")
        
    except Exception as e:
        print(f"Failed to bootstrap historical diesel data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
