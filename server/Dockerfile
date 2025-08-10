# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Needed to pip-install from a Git repo (QuantumBlur)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server/ /app/

# Fly will route to this port; keep it consistent with fly.toml
EXPOSE 8080

# Honor Fly's $PORT if present; default to 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
