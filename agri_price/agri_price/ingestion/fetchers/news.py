import json
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from torch.utils.data import Dataset
from typing import Sequence, Optional
import sqlite3

class ListDataset(Dataset):
    """Simple dataset wrapper for efficient inference."""
    def __init__(self, original_list: Sequence[str]):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def analyze_sentiment(texts: list[str], model_name: str = "ProsusAI/finbert") -> list[float]:
    """
    Analyzes sentiment of a list of texts using the specified model.
    Returns a list of scores (positive=score, negative=-score, neutral=0.0).
    """
    if not texts:
        return []
        
    print(f"Initializing sentiment pipeline: {model_name}")
    nlp = pipeline("text-classification", model=model_name)
    
    print(f"Scoring {len(texts)} articles...")
    dataset = ListDataset(texts)
    results = list(tqdm(nlp(dataset, truncation=True, max_length=512)))
    
    scores = []
    for s in results:
        score = s['score']
        label = s['label']
        if label == 'positive':
            scores.append(float(score))
        elif label == 'negative':
            scores.append(float(-score))
        else:
            scores.append(0.0)
            
    return scores

def process_news_dataframe(df: pd.DataFrame, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Standardizes a news DataFrame, runs sentiment analysis, 
    and aggregates to weekly granularity.
    
    Supports caching sentiment scores in a SQLite database to avoid 
    re-processing the same articles.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Check for Cached Sentiment
    cached_scores = {}
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            # Ensure table exists (though ingestion.py should have created it)
            conn.execute("CREATE TABLE IF NOT EXISTS raw_news_sentiment (id TEXT PRIMARY KEY, Sentiment_Score REAL)")
            
            # Load all cached scores into a dict
            cache_df = pd.read_sql_query("SELECT * FROM raw_news_sentiment", conn)
            cached_scores = dict(zip(cache_df['id'].astype(str), cache_df['Sentiment_Score']))
            conn.close()
            print(f"Loaded {len(cached_scores)} cached sentiment scores.")
        except Exception as e:
            print(f"Warning: Could not load sentiment cache: {e}")

    # 2. Identify Uncached Articles
    df['id'] = df['id'].astype(str)
    df['Sentiment_Score'] = df['id'].map(cached_scores)
    
    uncached_mask = df['Sentiment_Score'].isna()
    uncached_df = df[uncached_mask].copy()
    
    if not uncached_df.empty:
        print(f"Processing {len(uncached_df)} uncached news articles...")
        
        def parse_body_json(json_string):
            try:
                return json.loads(json_string).get('body', '')
            except (json.JSONDecodeError, AttributeError):
                return ""

        uncached_df['Body_Text'] = uncached_df['article'].apply(parse_body_json)
        uncached_df['Combined_Text'] = uncached_df['title'] + ". " + uncached_df['Body_Text'].str[:300]
        
        # Run sentiment analysis on uncached items only
        uncached_scores = analyze_sentiment(uncached_df['Combined_Text'].tolist())
        df.loc[uncached_mask, 'Sentiment_Score'] = uncached_scores
        
        # 3. Update Cache
        if db_path:
            try:
                print("Updating sentiment cache...")
                to_cache = pd.DataFrame({
                    'id': uncached_df['id'],
                    'Sentiment_Score': uncached_scores
                })
                conn = sqlite3.connect(db_path)
                to_cache.to_sql("raw_news_sentiment", conn, if_exists='append', index=False)
                conn.close()
            except Exception as e:
                print(f"Warning: Could not update sentiment cache: {e}")
    else:
        print("All news articles found in cache. Skipping FinBERT inference.")

    # 4. Aggregate to Weekly
    print("Aggregating to Weekly Data...")
    df['Year'] = df['date'].dt.year
    df['Week'] = df['date'].dt.strftime('%W').astype(int)

    weekly = df.groupby(['Year', 'Week'])['Sentiment_Score'].mean().reset_index()
    weekly.rename(columns={'Sentiment_Score': 'Weekly_Econ_Sentiment_Score'}, inplace=True)
    
    return weekly
