"""
Segmentation evaluation metrics.

Computes Dice, IoU, Hausdorff Distance, and Boundary F1-score
for binary segmentation evaluation.
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from scipy import ndimage


class SegmentationMetrics:
    """
    Accumulates and computes segmentation metrics over a dataset.

    Supports:
    - Dice Coefficient (F1)
    - Intersection over Union (IoU / Jaccard)
    - Hausdorff Distance (95th percentile)
    - Boundary F1-score (precision/recall on boundary pixels)

    Args:
        threshold: Binarization threshold for probability maps.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._hausdorff_distances: list[float] = []
        self._boundary_f1_scores: list[float] = []
        self._count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: (B, 1, H, W) probability maps in [0, 1].
            targets: (B, 1, H, W) binary ground truth.
        """
        pred_binary = (preds > self.threshold).float()
        targets = targets.float()

        # Flatten for TP/FP/FN counting
        pred_flat = pred_binary.view(-1)
        target_flat = targets.view(-1)

        self._tp += (pred_flat * target_flat).sum().item()
        self._fp += (pred_flat * (1 - target_flat)).sum().item()
        self._fn += ((1 - pred_flat) * target_flat).sum().item()
        self._tn += ((1 - pred_flat) * (1 - target_flat)).sum().item()
        self._count += preds.size(0)

        # Per-sample Hausdorff and Boundary F1
        for i in range(preds.size(0)):
            pred_np = pred_binary[i, 0].cpu().numpy()
            target_np = targets[i, 0].cpu().numpy()

            hd = self._hausdorff_95(pred_np, target_np)
            if hd is not None:
                self._hausdorff_distances.append(hd)

            bf1 = self._boundary_f1(pred_np, target_np)
            self._boundary_f1_scores.append(bf1)

    def compute(self) -> dict[str, float]:
        """Compute all accumulated metrics."""
        eps = 1e-7

        dice = (2 * self._tp + eps) / (2 * self._tp + self._fp + self._fn + eps)
        iou = (self._tp + eps) / (self._tp + self._fp + self._fn + eps)
        precision = (self._tp + eps) / (self._tp + self._fp + eps)
        recall = (self._tp + eps) / (self._tp + self._fn + eps)

        hausdorff = (
            np.mean(self._hausdorff_distances)
            if self._hausdorff_distances
            else 0.0
        )
        boundary_f1 = (
            np.mean(self._boundary_f1_scores)
            if self._boundary_f1_scores
            else 0.0
        )

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "hausdorff_95": hausdorff,
            "boundary_f1": boundary_f1,
        }

    @staticmethod
    def _hausdorff_95(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
        """
        95th percentile Hausdorff distance between two binary masks.

        Measures the worst-case boundary error (ignoring the top 5% outliers).
        """
        pred_boundary = pred - ndimage.binary_erosion(pred).astype(pred.dtype)
        target_boundary = target - ndimage.binary_erosion(target).astype(target.dtype)

        pred_pts = np.argwhere(pred_boundary > 0)
        target_pts = np.argwhere(target_boundary > 0)

        if len(pred_pts) == 0 or len(target_pts) == 0:
            return None

        # Forward distances (pred → target)
        from scipy.spatial.distance import cdist

        d_pred_to_target = cdist(pred_pts, target_pts, metric="euclidean").min(axis=1)
        d_target_to_pred = cdist(target_pts, pred_pts, metric="euclidean").min(axis=1)

        all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
        return float(np.percentile(all_distances, 95))

    @staticmethod
    def _boundary_f1(
        pred: np.ndarray, target: np.ndarray, tolerance: int = 2
    ) -> float:
        """
        Boundary F1-score with a pixel tolerance.

        Measures how well predicted boundaries match ground truth boundaries,
        allowing a small tolerance for slight misalignment.

        Args:
            tolerance: Number of pixels of allowed misalignment.
        """
        pred_boundary = pred - ndimage.binary_erosion(pred).astype(pred.dtype)
        target_boundary = target - ndimage.binary_erosion(target).astype(target.dtype)

        if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
            return 1.0
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return 0.0

        # Dilate boundaries by tolerance
        struct = ndimage.generate_binary_structure(2, 1)
        pred_dilated = ndimage.binary_dilation(
            pred_boundary, structure=struct, iterations=tolerance
        )
        target_dilated = ndimage.binary_dilation(
            target_boundary, structure=struct, iterations=tolerance
        )

        # Precision: fraction of predicted boundary within tolerance of target
        precision = (pred_boundary * target_dilated).sum() / max(pred_boundary.sum(), 1)
        # Recall: fraction of target boundary within tolerance of prediction
        recall = (target_boundary * pred_dilated).sum() / max(target_boundary.sum(), 1)

        eps = 1e-7
        return float(2 * precision * recall / (precision + recall + eps))
