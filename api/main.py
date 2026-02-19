import io, os, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# ── allow imports from project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.data.synthesis import ShapeSynthesizer

# ── globals ──────────────────────────────────────────────────────────────────
MODEL = None
DEVICE = None
IMG_SIZE = 512
SYNTH = None
CFG = None

# ── config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(ROOT / "configs" / "default.yaml"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", str(ROOT / "checkpoints" / "best.pth"))
# Comma-separated allowed origins (set in Railway env vars)
_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_raw == "*" else [o.strip() for o in _origins_raw.split(",")]

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Pattern Delineation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── startup: load model once ────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global MODEL, DEVICE, IMG_SIZE, SYNTH, CFG

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = CFG.get("data", {}).get("image_size", 512)

    name = CFG.get("model", {}).get("name", "attention_unet")
    MODEL = build_attention_unet(CFG) if name == "attention_unet" else build_unet(CFG)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"]
    # Fix key mismatch from training checkpoint
    state = {k.replace("attention_gates.", "attn_gates."): v for k, v in state.items()}
    MODEL.load_state_dict(state)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    s = CFG.get("data", {}).get("shape", {})
    SYNTH = ShapeSynthesizer(
        image_size=IMG_SIZE,
        min_control_points=s.get("min_control_points", 5),
        max_control_points=s.get("max_control_points", 15),
        min_radius_frac=s.get("min_radius_frac", 0.15),
        max_radius_frac=s.get("max_radius_frac", 0.40),
        min_dots=s.get("min_dots", 15),
        max_dots=s.get("max_dots", 80),
        min_dot_radius=s.get("min_dot_radius", 2),
        max_dot_radius=s.get("max_dot_radius", 6),
        dot_intensity_range=tuple(s.get("dot_intensity_range", [0.6, 1.0])),
        bg_intensity_range=tuple(s.get("bg_intensity_range", [0.0, 0.4])),
        jitter_frac=s.get("jitter_frac", 0.03),
        fill_mode="binary",
    )
    print(f"✓ Model loaded on {DEVICE} ({MODEL.count_parameters():,} params)")


# ── helpers ──────────────────────────────────────────────────────────────────
def _predict(gray: np.ndarray, threshold: float = 0.5):
    """Run inference on a grayscale numpy image, return (mask, prob) as uint8."""
    h, w = gray.shape[:2]
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(t)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
    mask_np = cv2.resize(mask[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
    prob_np = cv2.resize(probs[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask_np * 255).astype(np.uint8), (prob_np * 255).astype(np.uint8)


def _encode_png(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
):
    """Upload an image, receive the predicted binary mask as PNG."""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    mask, _ = _predict(img, threshold)
    return StreamingResponse(io.BytesIO(_encode_png(mask)), media_type="image/png")


@app.post("/predict/json")
async def predict_json(
    file: UploadFile = File(...),
    threshold: float = 0.5,
):
    """Upload an image, receive mask + probability as base64 PNGs."""
    import base64

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    mask, prob = _predict(img, threshold)
    return JSONResponse({
        "mask": base64.b64encode(_encode_png(mask)).decode(),
        "probability": base64.b64encode(_encode_png(prob)).decode(),
    })


@app.get("/demo")
def demo(num_dots: int = 40, jitter: float = 0.03):
    """Generate a random dot pattern, predict, and return results as JSON."""
    import base64

    SYNTH.min_dots = max(5, int(num_dots * 0.8))
    SYNTH.max_dots = int(num_dots * 1.2)
    SYNTH.jitter_frac = jitter
    dot_img, gt = SYNTH.generate(num_blobs=1)
    dot_disp = (dot_img * 255).astype(np.uint8)
    mask, prob = _predict(dot_disp, 0.5)
    gt_u8 = (gt * 255).astype(np.uint8)

    # Dice score
    m = mask.astype(np.float32) / 255.0
    g = gt.astype(np.float32)
    inter = (m * g).sum()
    dice = float((2 * inter) / (m.sum() + g.sum() + 1e-7))

    return JSONResponse({
        "input": base64.b64encode(_encode_png(dot_disp)).decode(),
        "ground_truth": base64.b64encode(_encode_png(gt_u8)).decode(),
        "mask": base64.b64encode(_encode_png(mask)).decode(),
        "probability": base64.b64encode(_encode_png(prob)).decode(),
        "dice": round(dice, 4),
    })
