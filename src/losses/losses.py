"""
Loss functions for noise-robust segmentation.

Implements:
- DiceLoss: Region-based overlap loss
- TverskyLoss: Asymmetric overlap loss for boundary emphasis
- BoundaryLoss: Edge-focused variant using distance maps
- CompoundLoss: Weighted combination with dynamic scheduling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2·|A∩B| / (|A| + |B|)

    Naturally handles class imbalance and provides strong gradients even
    when the target region is small. Operates on sigmoid probabilities.

    Args:
        smooth: Smoothing term to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            targets: (B, 1, H, W) binary ground truth.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — asymmetric generalization of Dice.

    Tversky(α, β) = TP / (TP + α·FN + β·FP)

    With α > β, the loss penalizes false negatives (missed boundary pixels)
    more than false positives, which is critical for sharp delineation.

    Recommended: α=0.7, β=0.3 for boundary-sensitive segmentation.

    Args:
        alpha: Weight for false negatives.
        beta: Weight for false positives.
        smooth: Smoothing factor.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fn = (targets_flat * (1 - probs_flat)).sum(dim=1)
        fp = ((1 - targets_flat) * probs_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-focused loss using Tversky on edge pixels.

    Extracts boundary regions from the ground truth mask via morphological
    gradient (dilation - erosion), then computes Tversky loss on the
    boundary strip. This directly optimizes edge sharpness.

    Args:
        alpha: FN weight in Tversky.
        beta: FP weight in Tversky.
        kernel_size: Size of the morphological kernel for boundary extraction.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, kernel_size: int = 3):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.kernel_size = kernel_size

        # Fixed morphological kernel for boundary extraction
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer("morph_kernel", kernel)

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary strip via morphological gradient."""
        pad = self.kernel_size // 2
        dilated = F.conv2d(mask, self.morph_kernel, padding=pad)
        dilated = (dilated > 0).float()
        eroded = F.conv2d(mask, self.morph_kernel, padding=pad)
        eroded = (eroded >= self.morph_kernel.sum()).float()
        return dilated - eroded

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        boundary_mask = self._extract_boundary(targets)
        # Only compute loss on boundary pixels (expand mask slightly for context)
        boundary_weight = boundary_mask + 0.1  # Avoid zeroing out non-boundary gradients
        weighted_logits = logits * boundary_weight
        weighted_targets = targets * boundary_weight
        return self.tversky(weighted_logits, weighted_targets)


class CompoundLoss(nn.Module):
    """
    Dynamic compound loss: Dice + BCE + Boundary.

    L = α·Dice + β·BCE + γ·Boundary(Tversky)

    The weights β and γ are scheduled over training:
    - BCE weight decays from bce_weight to bce_weight_min
    - Boundary weight ramps from 0 to boundary_weight_max

    Args:
        cfg: Loss configuration dictionary.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss(
            alpha=cfg.get("tversky_alpha", 0.7),
            beta=cfg.get("tversky_beta", 0.3),
        )

        # Static weights
        self.dice_weight = cfg.get("dice_weight", 1.0)

        # Dynamic weights (will be updated by trainer)
        self.bce_weight = cfg.get("bce_weight", 1.0)
        self.bce_weight_init = cfg.get("bce_weight", 1.0)
        self.bce_weight_min = cfg.get("bce_weight_min", 0.5)

        self.boundary_weight = cfg.get("boundary_weight", 0.0)
        self.boundary_weight_max = cfg.get("boundary_weight_max", 0.5)

        self.ramp_start = cfg.get("boundary_ramp_start_epoch", 20)
        self.ramp_end = cfg.get("boundary_ramp_end_epoch", 60)

    def update_weights(self, epoch: int) -> None:
        """Update dynamic loss weights based on current epoch."""
        if epoch < self.ramp_start:
            progress = 0.0
        elif epoch >= self.ramp_end:
            progress = 1.0
        else:
            progress = (epoch - self.ramp_start) / (self.ramp_end - self.ramp_start)

        # BCE decays linearly
        self.bce_weight = self.bce_weight_init - progress * (
            self.bce_weight_init - self.bce_weight_min
        )

        # Boundary ramps linearly
        self.boundary_weight = progress * self.boundary_weight_max

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute compound loss.

        Returns:
            Dictionary with 'total', 'dice', 'bce', 'boundary' loss values.
        """
        l_dice = self.dice_loss(logits, targets)
        l_bce = self.bce_loss(logits, targets)
        l_boundary = self.boundary_loss(logits, targets)

        total = (
            self.dice_weight * l_dice
            + self.bce_weight * l_bce
            + self.boundary_weight * l_boundary
        )

        return {
            "total": total,
            "dice": l_dice.detach(),
            "bce": l_bce.detach(),
            "boundary": l_boundary.detach(),
        }
