import argparse, json, logging, sys
from pathlib import Path
import torch, yaml
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.attention_unet import build_attention_unet
from src.models.unet import build_unet
from src.data.dataset import PatternDataset
from src.data.noise import compute_snr
from src.utils.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_at_scale(model, dataset, scale, device, n_samples=200, threshold=0.5):
    dataset.set_noise_scale(scale)
    metrics = SegmentationMetrics(threshold=threshold)
    snrs = []
    model.eval()
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            s = dataset[i]
            x = s["noisy"].unsqueeze(0).to(device)
            y = s["mask"].unsqueeze(0).to(device)
            with autocast(enabled=device.type == "cuda"):
                logits = model(x)
            metrics.update(torch.sigmoid(logits), y)
            snrs.append(compute_snr(s["clean"], s["noisy"]))
    results = metrics.compute()
    results["avg_snr_db"] = sum(snrs) / max(len(snrs), 1)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--output", type=str, default="evaluation_report.json")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = cfg.get("model", {}).get("name", "attention_unet")
    model = build_attention_unet(cfg) if name == "attention_unet" else build_unet(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    dcfg = cfg.get("data", {})
    dataset = PatternDataset(
        root_dir=None, split="test", image_size=dcfg.get("image_size", 512),
        noise_cfg=dcfg.get("noise", {}), noise_scale=1.0,
        num_samples=args.num_samples, shape_cfg=dcfg.get("shape", {}), seed=99,
    )

    scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    report = {}
    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    for s in scales:
        log.info(f"Evaluating at noise_scale={s:.2f}...")
        r = evaluate_at_scale(model, dataset, s, device, args.num_samples, threshold)
        report[f"scale_{s:.2f}"] = r
        log.info(f"  SNR: {r['avg_snr_db']:.1f} dB | Dice: {r['dice']:.4f} | IoU: {r['iou']:.4f} | HD95: {r['hausdorff_95']:.2f} | BF1: {r['boundary_f1']:.4f}")

    log.info("Evaluating with mixed noise...")
    dataset.set_noise_scale(1.0)
    dataset.set_mixed_prob(0.8)
    mixed = evaluate_at_scale(model, dataset, 1.0, device, args.num_samples)
    report["mixed_extreme"] = mixed
    log.info(f"  Mixed extreme — Dice: {mixed['dice']:.4f} | IoU: {mixed['iou']:.4f}")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
