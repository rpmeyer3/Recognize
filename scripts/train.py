import argparse, logging, sys
from pathlib import Path
import torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer


def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
    )


def build_model(cfg):
    name = cfg.get("model", {}).get("name", "attention_unet")
    if name == "attention_unet":
        return build_attention_unet(cfg)
    elif name == "unet":
        return build_unet(cfg)
    raise ValueError(f"Unknown model: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--model", type=str, default=None, choices=["attention_unet", "unet"])
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.lr: cfg["training"]["lr"] = args.lr
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.model: cfg["model"]["name"] = args.model

    setup_logging()
    log = logging.getLogger(__name__)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(cfg)
    log.info(f"Model: {cfg['model']['name']} ({model.count_parameters():,} params)")

    train_loader, val_loader = create_dataloaders(cfg)
    log.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    final = trainer.fit()
    log.info("=" * 50)
    log.info("Final Validation Metrics:")
    for k, v in final.items():
        log.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
