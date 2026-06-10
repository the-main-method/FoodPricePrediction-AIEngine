import pandas as pd


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # Load the latest dataset
    df = pd.read_csv(path)

    # 1. DYNAMICALLY auto-detect all text/categorical columns! 
    # This grabs 'state', 'food_item', and any future text columns you add.
    cat_features = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Ensure categorical columns are perfectly clean strings
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Use our precise 1-Month target
    target_col = 'TARGET_Price_Change_1M'

    X = df.drop(columns=['Year', 'Month', 'Week', target_col], errors='ignore')
    y = df[target_col]

    return X, y, cat_features