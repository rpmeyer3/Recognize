"""
Evaluation script.

Loads a trained model and evaluates it on the test set at multiple
noise levels to produce a comprehensive noise-robustness report.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/default.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.data.dataset import PatternDataset
from src.data.noise import NoiseInjector, compute_snr
from src.utils.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_at_noise_level(
    model: torch.nn.Module,
    dataset: PatternDataset,
    noise_scale: float,
    device: torch.device,
    num_samples: int = 200,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate model at a specific noise difficulty level."""
    dataset.set_noise_scale(noise_scale)
    metrics = SegmentationMetrics(threshold=threshold)
    snr_values = []

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            noisy = sample["noisy"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            clean = sample["clean"]

            with autocast(enabled=device.type == "cuda"):
                logits = model(noisy)

            probs = torch.sigmoid(logits)
            metrics.update(probs, mask)

            # Compute SNR
            snr = compute_snr(clean, sample["noisy"])
            snr_values.append(snr)

    results = metrics.compute()
    results["avg_snr_db"] = sum(snr_values) / max(len(snr_values), 1)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model at multiple noise levels")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    args = parser.parse_args()

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
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build test dataset
    data_cfg = cfg.get("data", {})
    dataset = PatternDataset(
        root_dir=None,
        split="test",
        image_size=data_cfg.get("image_size", 512),
        noise_cfg=data_cfg.get("noise", {}),
        noise_scale=1.0,
        num_samples=args.num_samples,
        shape_cfg=data_cfg.get("shape", {}),
        seed=99,
    )

    # Evaluate at multiple noise levels
    noise_scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    report = {}

    for scale in noise_scales:
        logger.info(f"Evaluating at noise_scale={scale:.2f}...")
        results = evaluate_at_noise_level(
            model, dataset, scale, device,
            num_samples=args.num_samples,
            threshold=cfg.get("evaluation", {}).get("threshold", 0.5),
        )
        report[f"scale_{scale:.2f}"] = results
        logger.info(
            f"  SNR: {results['avg_snr_db']:.1f} dB | "
            f"Dice: {results['dice']:.4f} | "
            f"IoU: {results['iou']:.4f} | "
            f"HD95: {results['hausdorff_95']:.2f} | "
            f"BF1: {results['boundary_f1']:.4f}"
        )

    # Also evaluate with mixed noise
    logger.info("Evaluating with mixed noise...")
    dataset.set_noise_scale(1.0)
    dataset.set_mixed_prob(0.8)
    mixed_results = evaluate_at_noise_level(
        model, dataset, 1.0, device,
        num_samples=args.num_samples,
    )
    report["mixed_extreme"] = mixed_results
    logger.info(
        f"  Mixed extreme — Dice: {mixed_results['dice']:.4f} | "
        f"IoU: {mixed_results['iou']:.4f}"
    )

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
