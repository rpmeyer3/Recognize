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
    def __init__(
        self, root_dir=None, split="train", image_size=512,
        noise_cfg=None, noise_scale=1.0, mixed_prob=0.0,
        num_samples=10000, shape_cfg=None, transform=None, seed=42,
    ):
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.mixed_prob = mixed_prob
        self.noise_injector = NoiseInjector(noise_cfg or {}, scale=noise_scale, rng=None)

        if root_dir is not None and os.path.isdir(root_dir):
            self.mode = "pregenerated"
            split_dir = Path(root_dir) / split
            self.image_dir = split_dir / "images"
            self.mask_dir = split_dir / "masks"
            self.filenames = sorted(os.listdir(self.image_dir))
            if num_samples < len(self.filenames):
                self.filenames = self.filenames[:num_samples]
        else:
            self.mode = "on_the_fly"
            self.num_samples = num_samples
            s = shape_cfg or {}
            self.synthesizer = ShapeSynthesizer(
                image_size=image_size,
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
                fill_mode=s.get("fill_mode", "binary"),
                seed=seed if split != "train" else None,
            )

    def set_noise_scale(self, scale):
        self.noise_injector.set_scale(scale)

    def set_mixed_prob(self, prob):
        self.mixed_prob = prob

    def __len__(self):
        return len(self.filenames) if self.mode == "pregenerated" else self.num_samples

    def __getitem__(self, idx):
        if self.mode == "pregenerated":
            image = np.load(self.image_dir / self.filenames[idx]).astype(np.float32)
            mask = np.load(self.mask_dir / self.filenames[idx]).astype(np.float32)
        else:
            image, mask = self.synthesizer.generate()

        input_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.noise_injector is not None and self.noise_injector.scale > 0:
            noisy = self.noise_injector(input_t, mixed_prob=self.mixed_prob)
        else:
            noisy = input_t

        if self.transform is not None:
            noisy, mask_t = self.transform(noisy, mask_t)

        return {"noisy": noisy, "clean": input_t, "mask": mask_t}


def create_dataloaders(cfg, noise_scale=1.0, mixed_prob=0.0):
    data_cfg = cfg.get("data", cfg)
    noise_cfg = data_cfg.get("noise", {})
    shape_cfg = data_cfg.get("shape", {})
    train_cfg = cfg.get("training", {})
    root_dir = data_cfg.get("output_dir", None)

    if root_dir and os.path.isdir(os.path.join(root_dir, "train")):
        train_root = val_root = root_dir
    else:
        train_root = val_root = None

    train_ds = PatternDataset(
        root_dir=train_root, split="train",
        image_size=data_cfg.get("image_size", 512),
        noise_cfg=noise_cfg, noise_scale=noise_scale, mixed_prob=mixed_prob,
        num_samples=data_cfg.get("num_train", 10000),
        shape_cfg=shape_cfg, seed=train_cfg.get("seed", 42),
    )
    val_ds = PatternDataset(
        root_dir=val_root, split="val",
        image_size=data_cfg.get("image_size", 512),
        noise_cfg=noise_cfg, noise_scale=1.0, mixed_prob=0.3,
        num_samples=data_cfg.get("num_val", 2000),
        shape_cfg=shape_cfg, seed=train_cfg.get("seed", 42) + 1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.get("batch_size", 4),
        shuffle=True, num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.get("batch_size", 4),
        shuffle=False, num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader
