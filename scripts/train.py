"""
Training entrypoint.

Loads config, builds model + dataloaders, and runs training with
curriculum learning.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pth
    python scripts/train.py --config configs/default.yaml --lr 1e-4 --batch-size 8
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_unet import AttentionUNet, build_attention_unet
from src.models.unet import UNet, build_unet
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer


def setup_logging(log_file: str = "training.log") -> None:
    """Configure logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    """Build model from config."""
    model_name = cfg.get("model", {}).get("name", "attention_unet")
    if model_name == "attention_unet":
        return build_attention_unet(cfg)
    elif model_name == "unet":
        return build_unet(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train pattern delineation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, choices=["attention_unet", "unet"])

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.model is not None:
        cfg["model"]["name"] = args.model

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Seed
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_model(cfg)
    logger.info(f"Model: {cfg['model']['name']} ({model.count_parameters():,} params)")

    # Build data
    train_loader, val_loader = create_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, device)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train!
    final_metrics = trainer.fit()

    # Summary
    logger.info("=" * 50)
    logger.info("Final Validation Metrics:")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
