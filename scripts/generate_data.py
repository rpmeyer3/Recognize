"""
Offline data generation script.

Generates clean images, ground truth masks, and saves them as .npy files
organized into train/val/test splits.

Usage:
    python scripts/generate_data.py --num-train 10000 --num-val 2000 --output-dir data/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthesis import ShapeSynthesizer


def generate_split(
    synthesizer: ShapeSynthesizer,
    output_dir: Path,
    split: str,
    num_samples: int,
) -> None:
    """Generate and save a single data split."""
    img_dir = output_dir / split / "images"
    mask_dir = output_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} {split} samples...")
    for i in tqdm(range(num_samples), desc=split):
        image, mask = synthesizer.generate()
        filename = f"{i:06d}.npy"
        np.save(img_dir / filename, image)
        np.save(mask_dir / filename, mask)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num-train", type=int, default=10000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=2000, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=500, help="Test samples")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    parser.add_argument("--min-cp", type=int, default=5, help="Min control points")
    parser.add_argument("--max-cp", type=int, default=15, help="Max control points")
    parser.add_argument("--min-radius", type=float, default=0.15, help="Min radius fraction")
    parser.add_argument("--max-radius", type=float, default=0.40, help="Max radius fraction")
    parser.add_argument("--fill-mode", type=str, default="binary", choices=["binary", "gradient", "texture"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    synthesizer = ShapeSynthesizer(
        image_size=args.image_size,
        min_control_points=args.min_cp,
        max_control_points=args.max_cp,
        min_radius_frac=args.min_radius,
        max_radius_frac=args.max_radius,
        fill_mode=args.fill_mode,
        seed=args.seed,
    )

    splits = [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test),
    ]

    for split, count in splits:
        if count > 0:
            generate_split(synthesizer, output_dir, split, count)

    print(f"\nData generated at: {output_dir.resolve()}")
    print(f"  Train: {args.num_train}")
    print(f"  Val:   {args.num_val}")
    print(f"  Test:  {args.num_test}")


if __name__ == "__main__":
    main()
