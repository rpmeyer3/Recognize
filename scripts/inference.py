"""
Single-image and batch inference script.

Loads a trained model and runs inference on one or more noisy images,
producing clean binary mask outputs.

Usage:
    # Single image
    python scripts/inference.py --checkpoint checkpoints/best.pth --input noisy.png

    # Batch directory
    python scripts/inference.py --checkpoint checkpoints/best.pth --input-dir test/ --output-dir results/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.preprocessing.filters import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_preprocess(
    image_path: str,
    image_size: int = 512,
    preprocessor: PreprocessingPipeline = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load an image, resize, convert to grayscale tensor.

    Returns:
        tensor: (1, 1, H, W) float tensor in [0, 1].
        original_size: (H, W) of the original image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_size = img.shape[:2]
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    if preprocessor is not None:
        tensor[0] = preprocessor(tensor[0])

    return tensor, original_size


def save_mask(
    mask: np.ndarray,
    output_path: str,
    original_size: tuple[int, int] = None,
) -> None:
    """Save binary mask as PNG, optionally resized to original dimensions."""
    if original_size is not None:
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_uint8)


def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference on a single preprocessed tensor.

    Returns:
        Binary mask as (H, W) numpy array with values in {0, 1}.
    """
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        with autocast(enabled=device.type == "cuda"):
            logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
    return mask[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run inference on noisy images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path")
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory for batch")
    parser.add_argument("--output", type=str, default=None, help="Output mask path")
    parser.add_argument("--output-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing filters")
    args = parser.parse_args()

    # Validate inputs
    if args.input is None and args.input_dir is None:
        parser.error("Provide either --input or --input-dir")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model_name = cfg.get("model", {}).get("name", "attention_unet")
    if model_name == "attention_unet":
        model = build_attention_unet(cfg)
    else:
        model = build_unet(cfg)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint}")

    # Preprocessor
    preprocessor = None
    if args.preprocess:
        preprocessor = PreprocessingPipeline(
            cfg.get("preprocessing", {}), stochastic=False
        )

    image_size = cfg.get("data", {}).get("image_size", 512)

    # Single image
    if args.input:
        tensor, orig_size = load_and_preprocess(args.input, image_size, preprocessor)
        mask = run_inference(model, tensor, device, args.threshold)

        output_path = args.output or args.input.replace(".", "_mask.")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_mask(mask, output_path, orig_size)
        logger.info(f"Mask saved: {output_path}")

    # Batch inference
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]

        logger.info(f"Processing {len(image_files)} images from {input_dir}")

        for img_path in image_files:
            tensor, orig_size = load_and_preprocess(str(img_path), image_size, preprocessor)
            mask = run_inference(model, tensor, device, args.threshold)

            out_path = output_dir / f"{img_path.stem}_mask.png"
            save_mask(mask, str(out_path), orig_size)

        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
