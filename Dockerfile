# Use a clean Python base image
FROM python:3.13-slim-bookworm

# Install system dependencies needed for runtime and shell execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory inside the container
WORKDIR /app

# Copy package requirements
COPY pyproject.toml uv.lock ./
COPY agri_price/ ./agri_price/

# Sync dependencies (frozen)
RUN uv sync --frozen

# Copy the application source code
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Copy data folder to seed directory (to initialize persistent volume on first boot)
COPY data/ ./data_seed/

# Copy entrypoint script and make it executable
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables to run Python in unbuffered mode and add bin to path
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Run the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
