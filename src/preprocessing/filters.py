"""
Optional preprocessing filters.

Lightweight spatial filters that can be applied stochastically during
training as augmentation, or deterministically at inference time.
These are NOT a separate denoising stage — they provide mild spatial
smoothing that aids the early encoder layers.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class PreprocessingPipeline:
    """
    Lightweight preprocessing with edge-preserving spatial filters.

    Applied stochastically during training (as augmentation) or
    deterministically during inference.

    Args:
        cfg: Preprocessing configuration dictionary.
        stochastic: If True, randomly apply/skip filters during training.
        apply_prob: Probability of applying each filter when stochastic=True.
    """

    def __init__(self, cfg: dict, stochastic: bool = True, apply_prob: float = 0.3):
        self.cfg = cfg
        self.stochastic = stochastic
        self.apply_prob = apply_prob

        # Bilateral filter params
        bil = cfg.get("bilateral", {})
        self.bil_d = bil.get("d", 9)
        self.bil_sigma_color = bil.get("sigma_color", 75)
        self.bil_sigma_space = bil.get("sigma_space", 75)

        # Non-local means params
        nlm = cfg.get("nlm", {})
        self.nlm_h = nlm.get("h", 10)
        self.nlm_template_window = nlm.get("template_window", 7)
        self.nlm_search_window = nlm.get("search_window", 21)

    def bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Edge-preserving bilateral filter.

        Smooths noise while preserving sharp edges — ideal as a mild
        preprocessing step before encoder input.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(
            img_uint8, self.bil_d, self.bil_sigma_color, self.bil_sigma_space
        )
        return filtered.astype(np.float32) / 255.0

    def nlm_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising.

        Exploits self-similarity in the image for denoising. Effective
        but computationally expensive — use sparingly.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        filtered = cv2.fastNlMeansDenoising(
            img_uint8,
            None,
            h=self.nlm_h,
            templateWindowSize=self.nlm_template_window,
            searchWindowSize=self.nlm_search_window,
        )
        return filtered.astype(np.float32) / 255.0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing pipeline.

        Args:
            image: (1, H, W) or (H, W) float tensor in [0, 1].
        Returns:
            Processed image tensor with same shape.
        """
        squeeze = False
        if image.dim() == 3:
            image_np = image.squeeze(0).cpu().numpy()
            squeeze = True
        else:
            image_np = image.cpu().numpy()

        import random

        if self.stochastic:
            if random.random() < self.apply_prob:
                image_np = self.bilateral_filter(image_np)
            if random.random() < self.apply_prob * 0.5:  # NLM less frequently
                image_np = self.nlm_filter(image_np)
        else:
            image_np = self.bilateral_filter(image_np)

        result = torch.from_numpy(image_np)
        if squeeze:
            result = result.unsqueeze(0)
        return result.to(image.device)
