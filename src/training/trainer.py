from __future__ import annotations
import os, time, logging
from pathlib import Path
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
    def __init__(self, model, train_loader, val_loader, cfg, device=torch.device("cuda")):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tcfg = cfg.get("training", {})
        lcfg = cfg.get("loss", {})

        self.criterion = CompoundLoss(lcfg).to(device)

        lr = tcfg.get("lr", 1e-4)
        wd = tcfg.get("weight_decay", 1e-5)
        opt = tcfg.get("optimizer", "adamw").lower()
        if opt == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        self.epochs = tcfg.get("epochs", 100)
        self.warmup_epochs = tcfg.get("warmup_epochs", 3)

        sched = tcfg.get("scheduler", "cosine")
        if sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.warmup_epochs)
        elif sched == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.use_amp = tcfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip = tcfg.get("grad_clip_norm", 1.0)

        self.curriculum = CurriculumScheduler(cfg.get("curriculum", {}))
        self.metrics = SegmentationMetrics(threshold=cfg.get("evaluation", {}).get("threshold", 0.5))

        vis_cfg = cfg.get("visualization", {})
        self.visualizer = Visualizer(log_every=vis_cfg.get("log_every", 50), num_samples=vis_cfg.get("num_samples", 4))

        self.ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints/"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = tcfg.get("save_every", 5)
        self.best_metric_name = tcfg.get("save_best_metric", "dice")
        self.best_metric = 0.0

        es = tcfg.get("early_stopping", {})
        self.early_stop = es.get("enabled", True)
        self.patience = es.get("patience", 15)
        self.min_delta = es.get("min_delta", 0.001)
        self.no_improve = 0
        self.start_epoch = 1
        self.global_step = 0

    def _warmup_lr(self, epoch):
        if epoch > self.warmup_epochs:
            return
        factor = epoch / max(self.warmup_epochs, 1)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", self.cfg["training"]["lr"]) * factor

    def train_epoch(self, epoch):
        self.model.train()
        ns, mp = self.curriculum.get_noise_params(epoch)
        if hasattr(self.train_loader.dataset, "set_noise_scale"):
            self.train_loader.dataset.set_noise_scale(ns)
            self.train_loader.dataset.set_mixed_prob(mp)
        self.criterion.update_weights(epoch)

        total_loss = 0.0
        comps = {"dice": 0.0, "bce": 0.0, "boundary": 0.0}
        n_batches = 0

        for bi, batch in enumerate(self.train_loader):
            x = batch["noisy"].to(self.device)
            y = batch["mask"].to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                losses = self.criterion(logits, y)

            self.scaler.scale(losses["total"]).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            for k in comps:
                comps[k] += losses[k].item()
            n_batches += 1
            self.global_step += 1

            if bi % 50 == 0:
                phase = self.curriculum.get_phase_name(epoch)
                logger.info(f"Epoch {epoch} [{bi}/{len(self.train_loader)}] Loss: {losses['total'].item():.4f} | Phase: {phase} | Scale: {ns:.2f}")

        avg = total_loss / max(n_batches, 1)
        avg_c = {k: v / max(n_batches, 1) for k, v in comps.items()}
        return {"loss": avg, **avg_c}

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        n_batches = 0
        sample = None

        for bi, batch in enumerate(self.val_loader):
            x = batch["noisy"].to(self.device)
            y = batch["mask"].to(self.device)
            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                losses = self.criterion(logits, y)
            total_loss += losses["total"].item()
            n_batches += 1
            probs = torch.sigmoid(logits)
            self.metrics.update(probs, y)
            if sample is None:
                sample = {"noisy": batch["noisy"], "clean": batch["clean"], "mask": batch["mask"], "pred": probs.cpu()}

        avg = total_loss / max(n_batches, 1)
        results = self.metrics.compute()
        if sample is not None:
            self.visualizer.log_predictions(sample, epoch, save_dir=self.ckpt_dir / "vis")
        return {"val_loss": avg, **results}

    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric, "metrics": metrics, "config": self.cfg,
        }
        torch.save(state, self.ckpt_dir / "last.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
            logger.info(f"New best model saved (dice={metrics.get('dice', 0):.4f})")
        if epoch % self.save_every == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_metric = ckpt.get("best_metric", 0.0)
        self.start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best={self.best_metric:.4f})")

    def fit(self):
        logger.info(f"Training for {self.epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(self.curriculum.summary())

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            self._warmup_lr(epoch)
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_m.get(self.best_metric_name, 0))
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            phase = self.curriculum.get_phase_name(epoch)
            logger.info(
                f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) | Phase: {phase} | LR: {lr:.2e} | "
                f"Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['val_loss']:.4f} | "
                f"Val Dice: {val_m.get('dice', 0):.4f} | Val IoU: {val_m.get('iou', 0):.4f}"
            )

            cur = val_m.get(self.best_metric_name, 0)
            is_best = cur > self.best_metric + self.min_delta
            if is_best:
                self.best_metric = cur
                self.no_improve = 0
            else:
                self.no_improve += 1
            self.save_checkpoint(epoch, val_m, is_best=is_best)

            if self.early_stop and self.no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        logger.info(f"Training complete. Best {self.best_metric_name}: {self.best_metric:.4f}")
        return val_m
