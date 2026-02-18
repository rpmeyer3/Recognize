# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# ── Environment defaults (override in Render dashboard) ─────────────────────
ENV CONFIG_PATH=/app/configs/default.yaml
ENV CHECKPOINT_PATH=/app/checkpoints/best.pth
ENV ALLOWED_ORIGINS=*
ENV PORT=10000

EXPOSE ${PORT}

# Start uvicorn
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
