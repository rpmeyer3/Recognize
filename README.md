# Pattern Delineation — Noise-Robust Shape Segmentation

> Teach a deep network to delineate arbitrary organic shapes (silhouettes) from images with extreme, varying noise levels — from clean to near-zero SNR. Using this as a steeping stone for noise-copyright development.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Technical Roadmap](#technical-roadmap)
   - [Architecture Selection](#1-architecture-selection)
   - [Data Strategy & Noise Simulation](#2-data-strategy--noise-simulation)
   - [Loss Functions](#3-loss-functions)
   - [Preprocessing Strategy](#4-preprocessing-strategy)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Training](#training)
6. [Inference](#inference)

---

## Problem Statement

Given a 512×512 image containing an **organic shape** (e.g. a shark silhouette, a leaf, a hand) 
buried under heavy, variable noise, produce a **clean, sharp binary mask** that precisely delineates 
the shape boundary. The model must generalize across:

- **Variable SNR**: from pristine images (SNR > 30 dB) to near-invisible signals (SNR < 0 dB)
- **Multiple noise types**: Gaussian, Poisson, Speckle, Salt-and-Pepper, and mixtures
- **Arbitrary organic shapes**: not restricted to a single class

---

## Technical Roadmap

### 1. Architecture Selection

| Architecture | Noise Robustness | Edge Sharpness | Training Cost | Verdict |
|---|---|---|---|---|
| Vanilla U-Net | Medium | Medium | Low | Baseline only |
| **Attention U-Net** | **High** | **High** | **Medium** | **Recommended** |
| U-Net + CBAM | High | High | Medium | Close second |
| Diffusion Segmentation | Very High | Very High | Very High | Overkill for this task |

**Why Attention U-Net?**

- **Attention gates** allow the decoder to suppress noise-activated features in skip connections. 
  In standard U-Net, skip connections faithfully propagate noisy encoder features to the decoder, 
  which hurts performance at low SNR. Attention gates learn to weight only the *signal-relevant* 
  spatial regions.
- **Channel & spatial attention** (via CBAM blocks bolted onto encoder stages) further improve 
  selectivity by learning per-channel importance — noise tends to activate many channels uniformly, 
  while signal concentrates on fewer channels.
- Diffusion-based segmentation (e.g., SegDiff) produces outstanding results but requires 
  iterative sampling at inference time (50–200 steps), making it impractical for most use-cases. 
  If you need the absolute best quality and can tolerate slow inference, the architecture is 
  provided but is not the default.

### 2. Data Strategy & Noise Simulation

#### Synthetic Data Pipeline

Since we control ground truth precisely, all training data is synthesized:

1. **Shape generation**: Render random organic silhouettes (Bézier blobs, SVG outlines, 
   real silhouette datasets like Silhouettes-1k) onto clean canvases.
2. **Clean image**: Binary mask → optional smooth gradient / texture fill inside the shape.
3. **Ground truth mask**: The original binary silhouette (0/1).
4. **Noise injection**: Apply one or more noise types at a random intensity.

#### Noise Types & Implementation

| Noise Type | Distribution | Parameter Range | Implementation |
|---|---|---|---|
| Gaussian | $\mathcal{N}(0, \sigma^2)$ | $\sigma \in [0.01, 1.5]$ | `torch.randn_like(x) * sigma` |
| Poisson | $\text{Pois}(\lambda \cdot x)$ | $\lambda \in [1, 300]$ | `torch.poisson(x * lam) / lam` |
| Salt-and-Pepper | Bernoulli drops | $p \in [0.01, 0.5]$ | Random pixel replacement |
| Speckle | Multiplicative Gaussian | $\sigma \in [0.05, 2.0]$ | `x + x * randn * sigma` |
| Mixed | Compound of 2-3 above | Sampled per type | Sequential application |

#### Curriculum Learning Strategy

Training proceeds in **4 phases**, each doubling in difficulty:

| Phase | Epochs | Noise σ/p Range | SNR (approx) | Purpose |
|---|---|---|---|---|
| 1 — Easy | 1–15 | σ ∈ [0.01, 0.15] | > 16 dB | Learn shape priors |
| 2 — Medium | 16–35 | σ ∈ [0.05, 0.5] | 6–16 dB | Develop noise tolerance |
| 3 — Hard | 36–60 | σ ∈ [0.1, 1.0] | 0–10 dB | Robust delineation |
| 4 — Extreme | 61–100 | σ ∈ [0.2, 1.5] + mixed | < 0 dB | Zero-visibility regime |

The scheduler also introduces mixed noise types starting in Phase 2 and increases the 
probability of compound noise over time.

### 3. Loss Functions

Standard BCE collapses when noise overwhelms the signal because the noisy input pushes 
predictions toward the prior class frequency. We use a **compound loss**:

```math
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{Dice}} + \beta \cdot \mathcal{L}_{\text{BCE}} + \gamma \cdot \mathcal{L}_{\text{Boundary}}
```

| Component | Purpose | Weight Schedule |
|---|---|---|
| **Dice Loss** | Region overlap — handles class imbalance naturally | α = 1.0 (constant) |
| **BCE with Logits** | Per-pixel calibration, gradient stability | β = 1.0 → 0.5 over training |
| **Boundary (Tversky)** | Penalizes boundary FP/FN asymmetrically | γ = 0.0 → 0.5 over training |

**Why this combination?**
- Dice alone produces smooth predictions but weak boundaries.
- BCE alone fails under class imbalance and heavy noise.
- Boundary-focused Tversky loss (with α=0.7, β=0.3) penalizes missed boundary pixels 
  more than false alarms, yielding crisp edges where it matters most.

### 4. Preprocessing Strategy

**Recommendation: Lightweight preprocessing + End-to-End training.** 

Do NOT use a fully separate denoising step — this discards signal along with noise. Instead:

1. **Mild spatial filtering** (optional, applied stochastically during training as augmentation):
   - Bilateral filter (preserves edges)
   - Non-local means (small window)
2. **Learned feature-level denoising**: The encoder's first two layers act as an implicit 
   denoiser when trained end-to-end with noisy inputs. We aid this with:
   - **Anti-aliased downsampling** (blur-pool) in encoder stride layers
   - **Residual connections** that allow the network to refine noisy features incrementally
3. **No separate denoising autoencoder** — empirically, end-to-end approaches trained with 
   curriculum learning outperform two-stage pipelines on this task.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python scripts/generate_data.py --num-train 10000 --num-val 2000 --output-dir data/

# 3. Train with curriculum learning
python scripts/train.py --config configs/default.yaml

# 4. Run inference on a noisy image
python scripts/inference.py --checkpoint checkpoints/best.pth --input noisy.png --output mask.png
```

---

## Training

```bash
# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pth

# Override config values from CLI
python scripts/train.py --config configs/default.yaml --lr 1e-4 --batch-size 8
```

## Inference

```bash
# Single image
python scripts/inference.py --checkpoint checkpoints/best.pth --input test.png

# Batch inference on a directory
python scripts/inference.py --checkpoint checkpoints/best.pth --input-dir test_images/ --output-dir results/
```

---

## License

MIT
