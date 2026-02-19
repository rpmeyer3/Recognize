# ── Build ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for opencv + curl for downloading checkpoint
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~3 GB vs CUDA version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (skip torch/torchvision — already installed as CPU-only)
COPY requirements.txt .
RUN grep -v -E '^torch' requirements.txt > requirements-deploy.txt && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy project
COPY . .

# Download checkpoint from Hugging Face (Xet storage requires ?download=true)
RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/best.pth \
    "https://huggingface.co/ryandoesai/pattern-dillineation/resolve/main/best.pth?download=true" && \
    echo "Downloaded checkpoint:" && ls -lh /app/checkpoints/best.pth

# ── Environment defaults ─────────────────────────────────────────────────────
ENV CONFIG_PATH=/app/configs/default.yaml
ENV CHECKPOINT_PATH=/app/checkpoints/best.pth
ENV ALLOWED_ORIGINS=*
ENV INFER_SIZE=256

# Railway sets PORT dynamically — default to 8000 as fallback
ENV PORT=8000

EXPOSE ${PORT}

# Start uvicorn — uses $PORT from Railway
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT