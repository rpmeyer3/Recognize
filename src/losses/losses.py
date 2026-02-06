from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        tgt = targets.view(targets.size(0), -1)
        inter = (probs * tgt).sum(dim=1)
        union = probs.sum(dim=1) + tgt.sum(dim=1)
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        tgt = targets.view(targets.size(0), -1)
        tp = (probs * tgt).sum(dim=1)
        fn = (tgt * (1 - probs)).sum(dim=1)
        fp = ((1 - tgt) * probs).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, kernel_size=3):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.kernel_size = kernel_size
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer("morph_kernel", kernel)

    def _extract_boundary(self, mask):
        pad = self.kernel_size // 2
        dilated = (F.conv2d(mask, self.morph_kernel, padding=pad) > 0).float()
        eroded = (F.conv2d(mask, self.morph_kernel, padding=pad) >= self.morph_kernel.sum()).float()
        return dilated - eroded

    def forward(self, logits, targets):
        bw = self._extract_boundary(targets) + 0.1
        return self.tversky(logits * bw, targets * bw)


class CompoundLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss(cfg.get("tversky_alpha", 0.7), cfg.get("tversky_beta", 0.3))
        self.dice_weight = cfg.get("dice_weight", 1.0)
        self.bce_weight = cfg.get("bce_weight", 1.0)
        self.bce_weight_init = cfg.get("bce_weight", 1.0)
        self.bce_weight_min = cfg.get("bce_weight_min", 0.5)
        self.boundary_weight = cfg.get("boundary_weight", 0.0)
        self.boundary_weight_max = cfg.get("boundary_weight_max", 0.5)
        self.ramp_start = cfg.get("boundary_ramp_start_epoch", 20)
        self.ramp_end = cfg.get("boundary_ramp_end_epoch", 60)

    def update_weights(self, epoch):
        if epoch < self.ramp_start:
            progress = 0.0
        elif epoch >= self.ramp_end:
            progress = 1.0
        else:
            progress = (epoch - self.ramp_start) / (self.ramp_end - self.ramp_start)
        self.bce_weight = self.bce_weight_init - progress * (self.bce_weight_init - self.bce_weight_min)
        self.boundary_weight = progress * self.boundary_weight_max

    def forward(self, logits, targets):
        ld = self.dice_loss(logits, targets)
        lb = self.bce_loss(logits, targets)
        lbound = self.boundary_loss(logits, targets)
        total = self.dice_weight * ld + self.bce_weight * lb + self.boundary_weight * lbound
        return {"total": total, "dice": ld.detach(), "bce": lb.detach(), "boundary": lbound.detach()}
