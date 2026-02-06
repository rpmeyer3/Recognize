from __future__ import annotations
from typing import Optional
import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._tp = self._fp = self._fn = self._tn = 0
        self._hd_scores = []
        self._bf1_scores = []
        self._count = 0

    def update(self, preds, targets):
        pred_bin = (preds > self.threshold).float()
        tgt = targets.float()
        pf = pred_bin.view(-1)
        tf = tgt.view(-1)
        self._tp += (pf * tf).sum().item()
        self._fp += (pf * (1 - tf)).sum().item()
        self._fn += ((1 - pf) * tf).sum().item()
        self._tn += ((1 - pf) * (1 - tf)).sum().item()
        self._count += preds.size(0)

        for i in range(preds.size(0)):
            p = pred_bin[i, 0].cpu().numpy()
            t = tgt[i, 0].cpu().numpy()
            hd = self._hausdorff_95(p, t)
            if hd is not None:
                self._hd_scores.append(hd)
            self._bf1_scores.append(self._boundary_f1(p, t))

    def compute(self):
        eps = 1e-7
        dice = (2 * self._tp + eps) / (2 * self._tp + self._fp + self._fn + eps)
        iou = (self._tp + eps) / (self._tp + self._fp + self._fn + eps)
        prec = (self._tp + eps) / (self._tp + self._fp + eps)
        rec = (self._tp + eps) / (self._tp + self._fn + eps)
        hd = np.mean(self._hd_scores) if self._hd_scores else 0.0
        bf1 = np.mean(self._bf1_scores) if self._bf1_scores else 0.0
        return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "hausdorff_95": hd, "boundary_f1": bf1}

    @staticmethod
    def _hausdorff_95(pred, target):
        pb = pred - ndimage.binary_erosion(pred).astype(pred.dtype)
        tb = target - ndimage.binary_erosion(target).astype(target.dtype)
        pp = np.argwhere(pb > 0)
        tp = np.argwhere(tb > 0)
        if len(pp) == 0 or len(tp) == 0:
            return None
        d1 = cdist(pp, tp, metric="euclidean").min(axis=1)
        d2 = cdist(tp, pp, metric="euclidean").min(axis=1)
        return float(np.percentile(np.concatenate([d1, d2]), 95))

    @staticmethod
    def _boundary_f1(pred, target, tolerance=2):
        pb = pred - ndimage.binary_erosion(pred).astype(pred.dtype)
        tb = target - ndimage.binary_erosion(target).astype(target.dtype)
        if pb.sum() == 0 and tb.sum() == 0:
            return 1.0
        if pb.sum() == 0 or tb.sum() == 0:
            return 0.0
        struct = ndimage.generate_binary_structure(2, 1)
        pd = ndimage.binary_dilation(pb, structure=struct, iterations=tolerance)
        td = ndimage.binary_dilation(tb, structure=struct, iterations=tolerance)
        prec = (pb * td).sum() / max(pb.sum(), 1)
        rec = (tb * pd).sum() / max(tb.sum(), 1)
        return float(2 * prec * rec / (prec + rec + 1e-7))
