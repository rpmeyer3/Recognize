# Pattern Delineation — Noise-Robust Dot-Pattern Segmentation

> An Attention U-Net trained with curriculum learning to segment dot-filled organic shapes from images under extreme, variable noise — from clean inputs to near-zero SNR.

**[Live Demo](https://pattern-delineation.vercel.app)** · **[API](https://pattern-delineation-production.up.railway.app/health)** · **[Model Weights](https://huggingface.co/ryandoesai/pattern-dillineation)**

---

## Table of Contents

1. [Overview](#overview)
2. [Final Results](#final-results)
3. [Architecture](#architecture)
4. [Data Pipeline](#data-pipeline)
5. [Training Details](#training-details)
6. [Deployment Architecture](#deployment-architecture)
7. [Problems & Solutions](#problems--solutions)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [License](#license)

---

## Overview

Given a 512×512 grayscale image containing an organic blob shape filled with dot patterns and buried under heavy noise, the model produces a clean binary mask that delineates the shape boundary.

The model generalizes across:
- **Variable SNR**: from pristine images (SNR > 30 dB) to near-invisible signals (SNR < 0 dB)
- **5 noise types**: Gaussian, Poisson, Speckle, Salt-and-Pepper, and compound mixtures
- **Arbitrary organic shapes**: random Bézier blob contours, not restricted to any specific class

The full system includes a **FastAPI inference server** (Railway), a **glassmorphism web dashboard** (Vercel), and model weights hosted on **Hugging Face**.

---

## Final Results

Trained for **80 epochs** on an RTX 4070 Ti SUPER (16 GB VRAM) with curriculum learning:

| Metric | Score |
|---|---|
| **Val Dice** | 0.886 |
| **Clean Dice** | 0.926 |
| **IoU** | 0.796 |
| **Precision** | 0.881 |
| **Recall** | 0.892 |

> *Clean Dice* is evaluated on noise-free inputs to measure pure segmentation quality independent of noise robustness.

---

## Architecture

### Model: Attention U-Net (31.5M parameters)

| Component | Details |
|---|---|
| Encoder | 5 levels, base 64 filters (64 → 128 → 256 → 512 → 1024) |
| Decoder | Transposed convolutions + skip connections |
| Attention Gates | Learned gating on skip connections to suppress noise-activated features |
| CBAM | Channel & spatial attention on encoder stages |
| Anti-Alias | Blur-pool downsampling to prevent aliasing artifacts |
| Regularization | Dropout (0.1), Batch Normalization, gradient clipping (1.0) |

**Why Attention U-Net over vanilla U-Net?**

Standard U-Net skip connections faithfully propagate noisy encoder features to the decoder, which collapses at low SNR. Attention gates learn to weight only signal-relevant spatial regions, effectively acting as a learned noise gate. CBAM further improves selectivity — noise activates many channels uniformly while actual signal concentrates on fewer channels.

### Loss Function

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{Dice}} + \beta \cdot \mathcal{L}_{\text{BCE}} + \gamma \cdot \mathcal{L}_{\text{Boundary}}$$

| Component | Purpose | Weight Schedule |
|---|---|---|
| **Dice Loss** | Region overlap, handles class imbalance | α = 1.0 (constant) |
| **BCE with Logits** | Per-pixel calibration, gradient stability | β = 1.0 → 0.5 over training |
| **Boundary Tversky** | Asymmetric FP/FN penalty for sharp edges (α=0.7, β=0.3) | γ = 0.0 → 0.5 (ramped epochs 20–60) |

---

## Data Pipeline

### Synthetic Generation

All training data is synthesized on-the-fly — no external datasets required:

1. **Shape generation**: Random organic silhouettes via Bézier blobs with 5–15 control points
2. **Dot filling**: 15–80 dots of radius 2–6 px scattered inside the shape with configurable jitter
3. **Ground truth**: The original binary blob mask
4. **Noise injection**: Apply one or more noise types at curriculum-scaled intensity

### Noise Types

| Type | Distribution | Parameter Range |
|---|---|---|
| Gaussian | $\mathcal{N}(0, \sigma^2)$ | $\sigma \in [0.01, 1.5]$ |
| Poisson | $\text{Pois}(\lambda \cdot x)$ | $\lambda \in [1, 300]$ |
| Salt-and-Pepper | Bernoulli drops | $p \in [0.01, 0.5]$ |
| Speckle | Multiplicative Gaussian | $\sigma \in [0.05, 2.0]$ |
| Mixed | Compound of 2–3 above | Sampled per type |

---

## Training Details

### Curriculum Learning

The model trains in 4 progressive difficulty phases. A `noise_scale` factor controls the intensity range sampled during each phase, and `mixed_prob` controls how often compound noise is applied:

| Phase | Epochs | Noise Scale | Mixed Prob | Purpose |
|---|---|---|---|---|
| Easy | 1–5 | 0.10 | 0.0 | Learn basic shape priors |
| Medium | 6–20 | 0.35 | 0.2 | Develop noise tolerance |
| Hard | 21–45 | 0.70 | 0.4 | Robust delineation under moderate noise |
| Extreme | 46–80 | 1.00 | 0.6 | Full noise range including compound types |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | Cosine annealing with 3-epoch warmup |
| Batch size | 8 |
| Image size | 512×512 |
| Mixed precision | Enabled (fp16) |
| Early stopping | 15 epochs patience, min delta 0.001 |
| Training data | 5,000 samples (regenerated each epoch via synthesis) |
| Validation data | 2,000 samples |

---

## Deployment Architecture

```
┌──────────────┐       ┌───────────────────┐       ┌─────────────────┐
│   Vercel      │◄─────►│   Railway          │◄─────►│  Hugging Face   │
│   (Frontend)  │ CORS  │   (FastAPI + Model)│  curl │  (Checkpoint)   │
│   Static HTML │       │   CPU inference     │       │  379 MB .pth    │
│   + JS + CSS  │       │   256×256 infer     │       │  Xet storage    │
└──────────────┘       └───────────────────┘       └─────────────────┘
```

- **Frontend**: Vanilla HTML/CSS/JS on Vercel — dark glassmorphism dashboard with metric rings, dual-pane viewer, confidence heatmap overlay, run history, and download support
- **Backend**: FastAPI on Railway — loads model on startup, serves `/demo` (synthetic pattern generation + inference) and `/predict` (custom image inference)
- **Weights**: Hosted on Hugging Face, downloaded at Docker build time via `curl -L` with `?download=true` for Xet storage compatibility

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check, returns device info |
| `GET` | `/demo?num_dots=40&jitter=0.03` | Generate synthetic pattern, predict, return base64 PNGs + dice score |
| `POST` | `/predict` | Upload image → binary mask PNG |
| `POST` | `/predict/json` | Upload image → base64 mask + probability map |

---

## Problems & Solutions

### 1. Deployment Platform OOM Crashes

**Problem**: The initial deployment target (Render free tier) ran out of memory loading the 379 MB PyTorch model. Azure was attempted but blocked due to subscription restrictions.

**Solution**: Migrated to **Railway** (512 MB+ RAM). Reduced inference resolution from 512×512 to **256×256** (`INFER_SIZE=256`) and added explicit `gc.collect()` after each inference to free tensors. Installed **CPU-only PyTorch** in Docker (saves ~3 GB vs CUDA wheels).

### 2. Model Producing Empty Masks

**Problem**: After initial training, the model output completely blank masks — all zeros. Investigating the logit/probability ranges showed the sigmoid outputs were near-zero everywhere, meaning the model had learned to predict "background" for every pixel.

**Root cause**: **Config drift**. The `default.yaml` noise parameters had been silently modified between data generation and training. The training was using different noise ranges than what the data was generated with, causing a distribution mismatch. The checkpoint was essentially garbage.

**Solution**: Restored `default.yaml` to the original parameters, regenerated all training data with the corrected config, and retrained from scratch. Added a `clean_dice` validation metric to the trainer that evaluates on noise-free inputs, making it easier to detect this class of failure early.

### 3. Hugging Face Xet Storage Download Failure

**Problem**: After uploading the retrained checkpoint to Hugging Face and redeploying, Railway crashed with `_pickle.UnpicklingError: invalid load key, 'E'`. The downloaded "checkpoint" was actually an HTML page.

**Root cause**: Hugging Face migrated to **Xet storage** for large files. The standard resolve URL (`/resolve/main/best.pth`) returns a redirect page instead of the raw file unless `?download=true` is appended.

**Solution**: Added `?download=true` to the curl command in the Dockerfile:
```dockerfile
curl -L -o /app/checkpoints/best.pth \
  "https://huggingface.co/ryandoesai/pattern-dillineation/resolve/main/best.pth?download=true"
```

### 4. CORS and Frontend Connection Issues

**Problem**: The Vercel frontend couldn't reach the Railway backend — requests were blocked by CORS policy. Additionally, setting `allow_credentials=True` with `allow_origins=["*"]` is invalid per the CORS spec.

**Solution**: Set `allow_credentials=False` in FastAPI's CORS middleware (credentials aren't needed for this API). Made `ALLOWED_ORIGINS` configurable via environment variable.

### 5. Inference Hanging / Spinner Never Stopping

**Problem**: The frontend showed an infinite spinner when the backend took too long or crashed silently during inference.

**Solution**: Implemented `fetchWithTimeout()` in the frontend JS with a 120-second timeout. Added a processing overlay with animated spinner that's properly hidden in the `finally` block regardless of success/failure.

### 6. Uploaded Images Not Working

**Problem**: Users could upload arbitrary photos (dogs, landscapes, etc.) but the model produced meaningless masks. The upload feature gave the impression the model was broken.

**Root cause**: The model is **only trained on synthetic dot patterns** — it has no concept of natural images. Uploading a photo of a dog will never produce a useful segmentation mask.

**Solution**: Removed the drag-and-drop upload feature entirely. The frontend now exclusively uses the "Generate & Predict" demo flow, which synthesizes patterns matching the training distribution. The `/predict` API endpoint is kept for programmatic use but isn't exposed in the UI.

### 7. Attention Gate Key Mismatch

**Problem**: Loading the trained checkpoint threw `RuntimeError: Missing key(s)` — the state dict keys didn't match the model definition.

**Root cause**: The training code saved the model with keys like `attention_gates.W_g.weight`, but the model class defined the modules as `attn_gates`.

**Solution**: Added a key rename step during checkpoint loading:
```python
state = {k.replace("attention_gates.", "attn_gates."): v for k, v in state.items()}
```

---

## Project Structure

```
pattern-delineation/
├── api/
│   └── main.py                 # FastAPI inference server
├── checkpoints/
│   └── best.pth                # Trained model weights (379 MB)
├── configs/
│   └── default.yaml            # Training & data configuration
├── data/
│   ├── train/                  # Generated training data (.npy)
│   ├── val/                    # Validation data
│   └── test/                   # Test data
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation script
│   ├── generate_data.py        # Synthetic data generation
│   ├── inference.py            # CLI inference
│   └── app.py                  # Local Gradio/Streamlit app
├── src/
│   ├── models/
│   │   ├── attention_unet.py   # Attention U-Net with CBAM
│   │   ├── unet.py             # Vanilla U-Net baseline
│   │   └── layers.py           # Custom layers (blur-pool, CBAM, etc.)
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── synthesis.py        # Shape & dot pattern synthesizer
│   │   └── noise.py            # Noise injection functions
│   ├── losses/
│   │   └── losses.py           # Dice, BCE, Tversky, compound loss
│   ├── training/
│   │   ├── trainer.py          # Training loop with clean_dice metric
│   │   └── curriculum.py       # Curriculum phase scheduler
│   ├── preprocessing/
│   │   └── filters.py          # Bilateral, NLM filters
│   └── utils/
│       ├── metrics.py          # Dice, IoU, Hausdorff, Boundary F1
│       └── visualization.py    # Training visualization helpers
├── web/
│   ├── index.html              # Dashboard frontend
│   ├── style.css               # Dark glassmorphism theme
│   └── app.js                  # Frontend application logic
├── Dockerfile                  # Railway deployment image
├── vercel.json                 # Vercel static hosting config
├── render.yaml                 # Render config (deprecated, kept for reference)
└── requirements.txt            # Python dependencies
```

---

## Quick Start

### Local Development

```bash
# Clone and install
git clone https://github.com/yourusername/pattern-delineation.git
cd pattern-delineation
pip install -r requirements.txt

# Generate synthetic training data
python scripts/generate_data.py --num-train 5000 --num-val 2000 --output-dir data/

# Train with curriculum learning (GPU recommended)
python scripts/train.py --config configs/default.yaml --device cuda

# Run inference on a test image
python scripts/inference.py --checkpoint checkpoints/best.pth --input test.png --output mask.png
```

### Training Options

```bash
# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pth

# Override config values
python scripts/train.py --config configs/default.yaml --lr 1e-4 --batch-size 8
```

### Run the API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs for interactive API docs
```

### Docker

```bash
docker build -t pattern-delineation .
docker run -p 8000:8000 pattern-delineation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.x, Attention U-Net with CBAM |
| Training | Curriculum learning, AdamW, cosine scheduler, mixed precision |
| Backend | FastAPI, uvicorn, CPU-only PyTorch |
| Frontend | Vanilla HTML/CSS/JS, Inter + JetBrains Mono fonts |
| Hosting | Railway (API), Vercel (frontend), Hugging Face (weights) |
| Containerization | Docker (Python 3.11 slim) |

---

## License

MIT