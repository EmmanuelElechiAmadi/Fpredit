# Football Predictor — production container
# Works on Railway, Render, Fly.io, Hugging Face Spaces (Docker), GCP Cloud Run.
FROM python:3.11-slim

WORKDIR /app

# Build tools (safety net if any wheel needs compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ app/
COPY src/ src/
COPY scripts/ scripts/
COPY webapp/ webapp/
COPY backtest.py predict.py config.yaml ./

# Football data (results + xG) baked into the image — required by the model
COPY data/ data/

ENV PORT=8000
EXPOSE 8000

# Bind 0.0.0.0 so the platform's router can reach us; honour $PORT when set.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
