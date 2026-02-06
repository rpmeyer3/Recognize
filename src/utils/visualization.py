"""
Visualization utilities for training monitoring and result analysis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class Visualizer:
    """
    Generates and saves visualization grids during training.

    Produces side-by-side plots of:
    [Noisy Input | Clean Image | Ground Truth | Prediction | Error Map]

    Args:
        log_every: Generate visualizations every N batches.
        num_samples: Number of samples to include in each grid.
    """

    def __init__(self, log_every: int = 50, num_samples: int = 4):
        self.log_every = log_every
        self.num_samples = num_samples

    def log_predictions(
        self,
        batch: dict[str, torch.Tensor],
        epoch: int,
        save_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Save a prediction visualization grid.

        Args:
            batch: Dict with 'noisy', 'clean', 'mask', 'pred' tensors.
            epoch: Current epoch number.
            save_dir: Directory to save images.
        """
        save_dir = Path(save_dir or "vis")
        save_dir.mkdir(parents=True, exist_ok=True)

        n = min(self.num_samples, batch["noisy"].size(0))

        fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        titles = ["Noisy Input", "Clean", "Ground Truth", "Prediction", "Error Map"]

        for i in range(n):
            noisy = batch["noisy"][i, 0].cpu().numpy()
            clean = batch["clean"][i, 0].cpu().numpy()
            mask = batch["mask"][i, 0].cpu().numpy()
            pred = (batch["pred"][i, 0].cpu().numpy() > 0.5).astype(float)

            # Error map: red = FP, blue = FN, green = TP
            error = np.zeros((*mask.shape, 3))
            tp = (pred > 0.5) & (mask > 0.5)
            fp = (pred > 0.5) & (mask < 0.5)
            fn = (pred < 0.5) & (mask > 0.5)
            error[tp] = [0, 0.8, 0]   # Green: correct
            error[fp] = [1, 0, 0]     # Red: false positive
            error[fn] = [0, 0, 1]     # Blue: false negative

            images = [noisy, clean, mask, pred, error]
            for j, (ax, img, title) in enumerate(zip(axes[i], images, titles)):
                if j < 4:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")

        plt.suptitle(f"Epoch {epoch}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f"epoch_{epoch:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        save_path: str | Path = "training_curves.png",
    ) -> None:
        """
        Plot training/validation loss and metric curves.

        Args:
            history: Dict mapping metric names to lists of per-epoch values.
            save_path: Output file path.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        if "train_loss" in history and "val_loss" in history:
            axes[0].plot(history["train_loss"], label="Train", color="blue")
            axes[0].plot(history["val_loss"], label="Val", color="red")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Dice curve
        if "dice" in history:
            axes[1].plot(history["dice"], label="Dice", color="green")
            if "iou" in history:
                axes[1].plot(history["iou"], label="IoU", color="orange")
            axes[1].set_title("Segmentation Metrics")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Hausdorff
        if "hausdorff_95" in history:
            axes[2].plot(history["hausdorff_95"], label="HD95", color="purple")
            axes[2].set_title("Hausdorff Distance (95%)")
            axes[2].set_xlabel("Epoch")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def visualize_noise_levels(
        self,
        clean_image: torch.Tensor,
        noise_injector,
        scales: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        save_path: str | Path = "noise_levels.png",
    ) -> None:
        """
        Visualize the same image at different noise levels.

        Useful for understanding the curriculum learning difficulty progression.
        """
        n = len(scales)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

        # Clean
        axes[0].imshow(clean_image.squeeze().numpy(), cmap="gray")
        axes[0].set_title("Clean")
        axes[0].axis("off")

        for i, scale in enumerate(scales):
            noise_injector.set_scale(scale)
            noisy = noise_injector(clean_image.clone())
            axes[i + 1].imshow(noisy.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i + 1].set_title(f"Scale={scale:.2f}")
            axes[i + 1].axis("off")

        plt.suptitle("Noise Curriculum Progression", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
