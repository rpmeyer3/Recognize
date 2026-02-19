import argparse, logging, os, sys
from pathlib import Path
import cv2, numpy as np, torch, yaml
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.preprocessing.filters import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_and_preprocess(path, size=512, preprocessor=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    orig = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    if preprocessor is not None:
        t[0] = preprocessor(t[0])
    return t, orig


def save_mask(mask, path, orig_size=None):
    if orig_size is not None:
        mask = cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, (mask * 255).astype(np.uint8))


def run_inference(model, tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor)
        probs = torch.sigmoid(logits)
        log.info(f"Raw logit range: {logits.min().item():.4f} to {logits.max().item():.4f}")
        log.info(f"Prob range: {probs.min().item():.4f} to {probs.max().item():.4f}")
        mask = (probs > threshold).float()
    return mask[0, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--input-dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--preprocess", action="store_true")
    args = p.parse_args()

    if args.input is None and args.input_dir is None:
        p.error("Provide either --input or --input-dir")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = cfg.get("model", {}).get("name", "attention_unet")
    model = build_attention_unet(cfg) if name == "attention_unet" else build_unet(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    state = {k.replace("attention_gates.", "attn_gates."): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    log.info(f"Model loaded from {args.checkpoint}")

    preprocessor = PreprocessingPipeline(cfg.get("preprocessing", {}), stochastic=False) if args.preprocess else None
    size = cfg.get("data", {}).get("image_size", 512)

    if args.input:
        t, orig = load_and_preprocess(args.input, size, preprocessor)
        mask = run_inference(model, t, device, args.threshold)
        out_path = args.output or args.input.replace(".", "_mask.")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_mask(mask, out_path, orig)
        log.info(f"Mask saved: {out_path}")
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [f for f in sorted(in_dir.iterdir()) if f.suffix.lower() in exts]
        log.info(f"Processing {len(files)} images from {in_dir}")
        for fp in files:
            t, orig = load_and_preprocess(str(fp), size, preprocessor)
            mask = run_inference(model, t, device, args.threshold)
            save_mask(mask, str(out_dir / f"{fp.stem}_mask.png"), orig)
        log.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
