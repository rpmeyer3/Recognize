"""
Training loop with curriculum learning, mixed precision, and checkpointing.
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..losses import CompoundLoss
from ..utils.metrics import SegmentationMetrics
from ..utils.visualization import Visualizer
from .curriculum import CurriculumScheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training engine for noise-robust segmentation.

    Handles:
    - Curriculum learning (progressive noise difficulty)
    - Dynamic loss weight scheduling
    - Mixed-precision training (AMP)
    - Gradient clipping
    - Learning rate warmup + cosine schedule
    - Checkpointing and early stopping
    - Validation with comprehensive metrics

    Args:
        model: Segmentation model (AttentionUNet or UNet).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Full configuration dictionary.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        train_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})

        # Loss
        self.criterion = CompoundLoss(loss_cfg).to(device)

        # Optimizer
        opt_name = train_cfg.get("optimizer", "adamw").lower()
        lr = train_cfg.get("lr", 1e-4)
        wd = train_cfg.get("weight_decay", 1e-5)
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        self.epochs = train_cfg.get("epochs", 100)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)
        sched_name = train_cfg.get("scheduler", "cosine")
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - self.warmup_epochs
            )
        elif sched_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip = train_cfg.get("grad_clip_norm", 1.0)

        # Curriculum
        curriculum_cfg = cfg.get("curriculum", {})
        self.curriculum = CurriculumScheduler(curriculum_cfg)

        # Metrics
        eval_cfg = cfg.get("evaluation", {})
        self.metrics = SegmentationMetrics(threshold=eval_cfg.get("threshold", 0.5))

        # Visualization
        vis_cfg = cfg.get("visualization", {})
        self.visualizer = Visualizer(
            log_every=vis_cfg.get("log_every", 50),
            num_samples=vis_cfg.get("num_samples", 4),
        )

        # Checkpointing
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_cfg.get("save_every", 5)
        self.best_metric_name = train_cfg.get("save_best_metric", "dice")
        self.best_metric = 0.0

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stop_enabled = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 15)
        self.min_delta = es_cfg.get("min_delta", 0.001)
        self.epochs_without_improvement = 0

        # State
        self.start_epoch = 1
        self.global_step = 0

    def _warmup_lr(self, epoch: int) -> None:
        """Linear warmup from 0 to base LR."""
        if epoch > self.warmup_epochs:
            return
        warmup_factor = epoch / max(self.warmup_epochs, 1)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", self.cfg["training"]["lr"]) * warmup_factor

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        # Curriculum: update noise parameters
        noise_scale, mixed_prob = self.curriculum.get_noise_params(epoch)
        if hasattr(self.train_loader.dataset, "set_noise_scale"):
            self.train_loader.dataset.set_noise_scale(noise_scale)
            self.train_loader.dataset.set_mixed_prob(mixed_prob)

        # Update loss weights
        self.criterion.update_weights(epoch)

        total_loss = 0.0
        loss_components = {"dice": 0.0, "bce": 0.0, "boundary": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            noisy = batch["noisy"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(noisy)
                losses = self.criterion(logits, mask)

            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            for k in loss_components:
                loss_components[k] += losses[k].item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % 50 == 0:
                phase = self.curriculum.get_phase_name(epoch)
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Phase: {phase} | Scale: {noise_scale:.2f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

        return {"loss": avg_loss, **avg_components}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0
        sample_batch = None

        for batch_idx, batch in enumerate(self.val_loader):
            noisy = batch["noisy"].to(self.device)
            mask = batch["mask"].to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(noisy)
                losses = self.criterion(logits, mask)

            total_loss += losses["total"].item()
            num_batches += 1

            # Accumulate metrics
            probs = torch.sigmoid(logits)
            self.metrics.update(probs, mask)

            # Save first batch for visualization
            if sample_batch is None:
                sample_batch = {
                    "noisy": batch["noisy"],
                    "clean": batch["clean"],
                    "mask": batch["mask"],
                    "pred": probs.cpu(),
                }

        avg_loss = total_loss / max(num_batches, 1)
        metric_results = self.metrics.compute()

        # Visualization
        if sample_batch is not None:
            self.visualizer.log_predictions(
                sample_batch, epoch, save_dir=self.ckpt_dir / "vis"
            )

        return {"val_loss": avg_loss, **metric_results}

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": self.cfg,
        }

        torch.save(state, self.ckpt_dir / "last.pth")

        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
            logger.info(f"New best model saved (dice={metrics.get('dice', 0):.4f})")

        if epoch % self.save_every == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_metric = ckpt.get("best_metric", 0.0)
        self.start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best={self.best_metric:.4f})")

    def fit(self) -> dict:
        """
        Full training loop with curriculum learning.

        Returns:
            Dictionary of final metrics.
        """
        logger.info(f"Training for {self.epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(self.curriculum.summary())

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()

            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.best_metric_name, 0))
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]
            phase = self.curriculum.get_phase_name(epoch)

            logger.info(
                f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) | "
                f"Phase: {phase} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Dice: {val_metrics.get('dice', 0):.4f} | "
                f"Val IoU: {val_metrics.get('iou', 0):.4f}"
            )

            # Checkpoint
            current_metric = val_metrics.get(self.best_metric_name, 0)
            is_best = current_metric > self.best_metric + self.min_delta
            if is_best:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if (
                self.early_stop_enabled
                and self.epochs_without_improvement >= self.patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        logger.info(f"Training complete. Best {self.best_metric_name}: {self.best_metric:.4f}")
        return val_metrics
