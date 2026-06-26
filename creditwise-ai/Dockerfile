FROM python:3.12-slim

# System deps for XGBoost/SHAP native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY app/ ./app/
COPY saved_models/ ./saved_models/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY .env.example ./.env.example

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# PORT is injected by Railway at runtime; default to 8000 for local Docker runs
ENV PORT=8000
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Use shell form so $PORT is expanded at runtime
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
