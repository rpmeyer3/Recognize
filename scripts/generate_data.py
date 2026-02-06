import argparse, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.synthesis import ShapeSynthesizer


def generate_split(synth, out_dir, split, n):
    img_dir = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} {split} samples...")
    for i in tqdm(range(n), desc=split):
        img, mask = synth.generate()
        fname = f"{i:06d}.npy"
        np.save(img_dir / fname, img)
        np.save(mask_dir / fname, mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-train", type=int, default=10000)
    p.add_argument("--num-val", type=int, default=2000)
    p.add_argument("--num-test", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="data/")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--min-cp", type=int, default=5)
    p.add_argument("--max-cp", type=int, default=15)
    p.add_argument("--min-radius", type=float, default=0.15)
    p.add_argument("--max-radius", type=float, default=0.40)
    p.add_argument("--fill-mode", type=str, default="binary", choices=["binary", "gradient", "texture"])
    p.add_argument("--min-dots", type=int, default=15)
    p.add_argument("--max-dots", type=int, default=80)
    p.add_argument("--min-dot-radius", type=int, default=2)
    p.add_argument("--max-dot-radius", type=int, default=6)
    p.add_argument("--jitter-frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    synth = ShapeSynthesizer(
        image_size=args.image_size, min_control_points=args.min_cp,
        max_control_points=args.max_cp, min_radius_frac=args.min_radius,
        max_radius_frac=args.max_radius, min_dots=args.min_dots,
        max_dots=args.max_dots, min_dot_radius=args.min_dot_radius,
        max_dot_radius=args.max_dot_radius, jitter_frac=args.jitter_frac,
        fill_mode=args.fill_mode, seed=args.seed,
    )
    for split, count in [("train", args.num_train), ("val", args.num_val), ("test", args.num_test)]:
        if count > 0:
            generate_split(synth, out, split, count)
    print(f"\nData generated at: {out.resolve()}")
    print(f"  Train: {args.num_train}\n  Val:   {args.num_val}\n  Test:  {args.num_test}")


if __name__ == "__main__":
    main()
