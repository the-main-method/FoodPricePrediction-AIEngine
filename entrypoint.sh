#!/bin/bash
set -e

# If the mounted /app/data directory is empty, seed it
if [ ! -f "/app/data/feature_store.db" ]; then
    echo "Setting up SQLite Feature Store database in persistent volume..."
    mkdir -p /app/data
    
    if [ -f "/app/data_seed/feature_store.db" ]; then
        cp /app/data_seed/feature_store.db /app/data/feature_store.db
        echo "Database seeded from pre-existing local copy."
    else
        echo "No seed database found. Recreating database from raw CSV..."
        python scripts/02_database/setup_db.py
    fi
fi

# Ensure final permissions are correct
chmod -R 777 /app/data

echo "Database checked/initialized. Starting API server..."
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
