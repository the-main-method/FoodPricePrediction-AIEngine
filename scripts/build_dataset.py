import pandas as pd
from pathlib import Path
from agri_price.data import build_combined_dataset, save_to_db
import os
import sys
import argparse

from agri_price.ingestion import ingest_all_to_db

def main():
    repo_root = Path(__file__).resolve().parents[1]
    
    parser = argparse.ArgumentParser(description="Orchestrate the full historical dataset build using the database-first pipeline.")
    parser.add_argument("--base-dir", type=str, default=str(repo_root / "data" / "raw"),
                        help="Base directory for Food Prices and News files (default: data/raw)")
    parser.add_argument("--db-path", type=str, default=str(repo_root / "data" / "feature_store.db"),
                        help="Path to the SQLite database (default: data/feature_store.db)")
    parser.add_argument("--output-csv", type=str, default=str(repo_root / "data" / "ml_ready_global_data.csv"),
                        help="Path to save the final CSV (default: data/ml_ready_global_data.csv)")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip API data ingestion")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    db_path = Path(args.db_path)
    
    if not args.skip_ingestion:
        print("Starting API-driven data ingestion...")
        ingest_all_to_db(str(db_path))

    # Define paths for remaining file-based data
    news_path = base_dir / "news 2022-2024.xlsx"
    food_path = base_dir / "Sample 2022-2024.xlsx"

    # Verify files exist
    for p in [news_path, food_path]:
        if not p.exists():
            print(f"Error: Required file not found at {p}")
            sys.exit(1)

    # Build the combined dataset using the new DB-centric logic
    print(f"Building combined dataset using database at {db_path}...")
    df = build_combined_dataset(
        db_path=str(db_path),
        news_path=str(news_path),
        food_path=str(food_path)
    )
    
    # Post-processing: Add target variable and cleaning
    print("Adding target variable (1-Month forward price change)...")
    group_cols = ['State', 'Food_Item', 'Item_Type', 'Category']
    df = df.sort_values(group_cols + ['Year', 'Week'])
    
    df['Target_Price_Change_1M_Percent'] = df.groupby(group_cols)['Price_NGN'].shift(-4)
    df['Target_Price_Change_1M_Percent'] = (df['Target_Price_Change_1M_Percent'] - df['Price_NGN']) / df['Price_NGN'] * 100
    
    initial_len = len(df)
    df = df.dropna(subset=['Target_Price_Change_1M_Percent'])
    print(f"Final dataset has {len(df)} rows (dropped {initial_len - len(df)} rows with missing targets).")
    
    # Save results
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved combined dataset to {output_csv}")
    
    save_to_db(df, str(db_path), "historical_data")
    print(f"Dataset successfully saved to database table 'historical_data' at {db_path}")

if __name__ == "__main__":
    main()
