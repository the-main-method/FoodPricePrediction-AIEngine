import os
import sqlite3
import urllib.parse as urlparse
import pandas as pd
from sqlalchemy import create_engine

def get_db_url():
    return os.environ.get('DATABASE_URL') or os.environ.get('POSTGRES_URL')

def get_connection(db_path='data/feature_store.db'):
    db_url = get_db_url()
    if db_url and (db_url.startswith('postgres://') or db_url.startswith('postgresql://')):
        # Connect to PostgreSQL using pg8000
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
            
        import pg8000
        result = urlparse.urlparse(db_url)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port or 5432
        
        conn = pg8000.connect(
            user=username,
            password=password,
            host=hostname,
            port=port,
            database=database
        )
        return conn, True
    else:
        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        return conn, False

def execute_query(conn, is_postgres, sql, params=None):
    cursor = conn.cursor()
    if is_postgres:
        # Convert sqlite placeholders to postgres/pg8000 placeholders
        sql = sql.replace('?', '%s')
        
        # Dialect conversion for INSERT OR REPLACE (Postgres doesn't support this syntax)
        if "INSERT OR REPLACE" in sql.upper():
            if "CURRENT_MARKET_STATE" in sql.upper():
                del_cursor = conn.cursor()
                try:
                    del_cursor.execute("DELETE FROM current_market_state WHERE id = 1")
                except Exception:
                    pass  # Table might not exist yet
                del_cursor.close()
            sql = sql.replace("INSERT OR REPLACE", "INSERT").replace("insert or replace", "insert")
            
        # SQLite INTEGER PRIMARY KEY AUTOINCREMENT equivalent is serial primary key
        # In setup_db or create_schema we handle PRIMARY KEY. In Postgres INTEGER PRIMARY KEY is fine.
    
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)
    return cursor

def to_sql(df, table_name, db_path='data/feature_store.db', if_exists='replace'):
    db_url = get_db_url()
    if db_url and (db_url.startswith('postgres://') or db_url.startswith('postgresql://')):
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql+pg8000://', 1)
        else:
            db_url = db_url.replace('postgresql://', 'postgresql+pg8000://', 1)
        engine = create_engine(db_url)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    else:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()

def read_sql_query(sql, conn_or_path, is_postgres=False, params=None):
    if is_postgres:
        sql = sql.replace('?', '%s')
        # pg8000 connection works directly with pd.read_sql_query
        if isinstance(conn_or_path, str):
            conn, _ = get_connection(conn_or_path)
            res = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            return res
        else:
            return pd.read_sql_query(sql, conn_or_path, params=params)
    else:
        # SQLite
        if isinstance(conn_or_path, str):
            conn = sqlite3.connect(conn_or_path)
            res = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            return res
        else:
            return pd.read_sql_query(sql, conn_or_path, params=params)
