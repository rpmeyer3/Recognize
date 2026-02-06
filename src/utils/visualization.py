from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, log_every=50, num_samples=4):
        self.log_every = log_every
        self.num_samples = num_samples

    def log_predictions(self, batch, epoch, save_dir=None):
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
            err = np.zeros((*mask.shape, 3))
            err[(pred > 0.5) & (mask > 0.5)] = [0, 0.8, 0]
            err[(pred > 0.5) & (mask < 0.5)] = [1, 0, 0]
            err[(pred < 0.5) & (mask > 0.5)] = [0, 0, 1]
            imgs = [noisy, clean, mask, pred, err]
            for j, (ax, img, title) in enumerate(zip(axes[i], imgs, titles)):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1) if j < 4 else ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
        plt.suptitle(f"Epoch {epoch}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f"epoch_{epoch:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def plot_training_curves(self, history, save_path="training_curves.png"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        if "train_loss" in history and "val_loss" in history:
            axes[0].plot(history["train_loss"], label="Train", color="blue")
            axes[0].plot(history["val_loss"], label="Val", color="red")
            axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        if "dice" in history:
            axes[1].plot(history["dice"], label="Dice", color="green")
            if "iou" in history:
                axes[1].plot(history["iou"], label="IoU", color="orange")
            axes[1].set_title("Metrics"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        if "hausdorff_95" in history:
            axes[2].plot(history["hausdorff_95"], label="HD95", color="purple")
            axes[2].set_title("Hausdorff 95%"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def visualize_noise_levels(self, clean, injector, scales=[0.0, 0.25, 0.5, 0.75, 1.0], save_path="noise_levels.png"):
        n = len(scales)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        axes[0].imshow(clean.squeeze().numpy(), cmap="gray"); axes[0].set_title("Clean"); axes[0].axis("off")
        for i, s in enumerate(scales):
            injector.set_scale(s)
            noisy = injector(clean.clone())
            axes[i + 1].imshow(noisy.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i + 1].set_title(f"Scale={s:.2f}"); axes[i + 1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
