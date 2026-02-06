"""
PyTorch Dataset and DataLoader utilities.

Handles loading pre-generated data or on-the-fly synthesis, noise injection,
and optional preprocessing augmentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .noise import NoiseInjector
from .synthesis import ShapeSynthesizer


class PatternDataset(Dataset):
    """
    Dataset for noisy-image → clean-mask pairs.

    Operates in two modes:
    1. **Pregenerated**: Loads .npy files from disk (fast, reproducible).
    2. **On-the-fly**: Generates shapes + applies noise per sample (unlimited data).

    Args:
        root_dir: Path to pregenerated data (contains images/ and masks/ dirs).
                  If None, generates on-the-fly.
        split: "train" | "val" | "test".
        image_size: Spatial resolution.
        noise_cfg: Noise configuration dictionary.
        noise_scale: Curriculum difficulty scale passed to NoiseInjector.
        mixed_prob: Probability of using mixed noise.
        num_samples: Number of samples when using on-the-fly generation.
        shape_cfg: Shape synthesis config (for on-the-fly mode).
        transform: Optional additional transform (e.g. augmentations).
        seed: Random seed.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        image_size: int = 512,
        noise_cfg: Optional[dict] = None,
        noise_scale: float = 1.0,
        mixed_prob: float = 0.0,
        num_samples: int = 10000,
        shape_cfg: Optional[dict] = None,
        transform=None,
        seed: int = 42,
    ):
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.mixed_prob = mixed_prob

        # Noise injector
        noise_cfg = noise_cfg or {}
        self.noise_injector = NoiseInjector(noise_cfg, scale=noise_scale, rng=None)

        if root_dir is not None and os.path.isdir(root_dir):
            # Pregenerated mode
            self.mode = "pregenerated"
            split_dir = Path(root_dir) / split
            self.image_dir = split_dir / "images"
            self.mask_dir = split_dir / "masks"
            self.filenames = sorted(os.listdir(self.image_dir))
        else:
            # On-the-fly mode
            self.mode = "on_the_fly"
            self.num_samples = num_samples
            s_cfg = shape_cfg or {}
            self.synthesizer = ShapeSynthesizer(
                image_size=image_size,
                min_control_points=s_cfg.get("min_control_points", 5),
                max_control_points=s_cfg.get("max_control_points", 15),
                min_radius_frac=s_cfg.get("min_radius_frac", 0.15),
                max_radius_frac=s_cfg.get("max_radius_frac", 0.40),
                fill_mode=s_cfg.get("fill_mode", "binary"),
                seed=seed if split != "train" else None,  # Deterministic val/test
            )

    def set_noise_scale(self, scale: float) -> None:
        """Update noise difficulty for curriculum learning."""
        self.noise_injector.set_scale(scale)

    def set_mixed_prob(self, prob: float) -> None:
        """Update probability of mixed noise."""
        self.mixed_prob = prob

    def __len__(self) -> int:
        if self.mode == "pregenerated":
            return len(self.filenames)
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.mode == "pregenerated":
            image = np.load(self.image_dir / self.filenames[idx]).astype(np.float32)
            mask = np.load(self.mask_dir / self.filenames[idx]).astype(np.float32)
        else:
            image, mask = self.synthesizer.generate()

        # Convert to tensors: (1, H, W)
        clean = torch.from_numpy(image).unsqueeze(0)     # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)      # (1, H, W)

        # Apply noise
        noisy = self.noise_injector(clean, mixed_prob=self.mixed_prob)

        # Optional transform (augmentations)
        if self.transform is not None:
            noisy, mask_t = self.transform(noisy, mask_t)

        return {
            "noisy": noisy,       # Input to the model
            "clean": clean,       # Clean image (for visualization)
            "mask": mask_t,       # Ground truth mask
        }


def create_dataloaders(
    cfg: dict,
    noise_scale: float = 1.0,
    mixed_prob: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from config.

    Args:
        cfg: Full configuration dictionary.
        noise_scale: Current curriculum noise scale.
        mixed_prob: Current curriculum mixed noise probability.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = cfg.get("data", cfg)
    noise_cfg = data_cfg.get("noise", {})
    shape_cfg = data_cfg.get("shape", {})
    train_cfg = cfg.get("training", {})

    root_dir = data_cfg.get("output_dir", None)

    # Check if pregenerated data exists
    if root_dir and os.path.isdir(os.path.join(root_dir, "train")):
        train_root = root_dir
        val_root = root_dir
    else:
        train_root = None
        val_root = None

    train_ds = PatternDataset(
        root_dir=train_root,
        split="train",
        image_size=data_cfg.get("image_size", 512),
        noise_cfg=noise_cfg,
        noise_scale=noise_scale,
        mixed_prob=mixed_prob,
        num_samples=data_cfg.get("num_train", 10000),
        shape_cfg=shape_cfg,
        seed=train_cfg.get("seed", 42),
    )

    val_ds = PatternDataset(
        root_dir=val_root,
        split="val",
        image_size=data_cfg.get("image_size", 512),
        noise_cfg=noise_cfg,
        noise_scale=1.0,  # Always validate at full noise difficulty
        mixed_prob=0.3,
        num_samples=data_cfg.get("num_val", 2000),
        shape_cfg=shape_cfg,
        seed=train_cfg.get("seed", 42) + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader
