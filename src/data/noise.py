"""
Noise injection module.

Implements Gaussian, Poisson, Salt-and-Pepper, Speckle, and mixed noise
with controllable intensity. Designed to be used both at dataset creation
time and as on-the-fly augmentation during training.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import numpy as np


class NoiseInjector:
    """
    Applies configurable noise to a clean image tensor.

    The ``scale`` parameter (0.0–1.0) controls what fraction of the
    configured parameter range is active, enabling curriculum learning.

    Args:
        cfg: Noise configuration dictionary with parameter ranges.
        scale: Current curriculum difficulty scale (0.0 = no noise, 1.0 = max).
        rng: Optional random generator for reproducibility.
    """

    def __init__(
        self,
        cfg: dict,
        scale: float = 1.0,
        rng: Optional[random.Random] = None,
    ):
        self.cfg = cfg
        self.scale = max(0.0, min(1.0, scale))
        self.rng = rng or random.Random()

        # Parse parameter ranges
        self.gaussian_sigma_range = cfg.get("gaussian_sigma", [0.01, 1.5])
        self.poisson_lambda_range = cfg.get("poisson_lambda", [1.0, 300.0])
        self.sp_prob_range = cfg.get("salt_pepper_prob", [0.01, 0.5])
        self.speckle_sigma_range = cfg.get("speckle_sigma", [0.05, 2.0])

    def set_scale(self, scale: float) -> None:
        """Update the curriculum difficulty scale."""
        self.scale = max(0.0, min(1.0, scale))

    def _scaled_range(self, lo: float, hi: float) -> tuple[float, float]:
        """Return the active sub-range given the current scale."""
        span = hi - lo
        effective_hi = lo + span * self.scale
        return lo, effective_hi

    def _sample_uniform(self, lo: float, hi: float) -> float:
        """Sample uniformly from [lo, hi]."""
        return self.rng.uniform(lo, hi)

    # ------------------------------------------------------------------
    # Individual noise types
    # ------------------------------------------------------------------

    def add_gaussian(self, image: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Additive white Gaussian noise: x' = x + N(0, σ²).

        Args:
            image: (C, H, W) float tensor in [0, 1].
            sigma: Noise standard deviation. If None, sampled from scaled range.
        Returns:
            Noisy image tensor (clamped to [0, 1]).
        """
        if sigma is None:
            lo, hi = self._scaled_range(*self.gaussian_sigma_range)
            sigma = self._sample_uniform(lo, hi)
        noise = torch.randn_like(image) * sigma
        return (image + noise).clamp(0.0, 1.0)

    def add_poisson(self, image: torch.Tensor, lam: Optional[float] = None) -> torch.Tensor:
        """
        Poisson (shot) noise: x' = Poisson(λ·x) / λ.

        Lower λ = more noise. We invert the range so that higher scale = lower λ.

        Args:
            image: (C, H, W) float tensor in [0, 1].
            lam: Poisson scaling factor. If None, sampled from scaled range.
        Returns:
            Noisy image tensor (clamped to [0, 1]).
        """
        lam_lo, lam_hi = self.poisson_lambda_range
        if lam is None:
            # Higher scale → lower lambda → more noise
            effective_lam_lo = lam_hi - (lam_hi - lam_lo) * self.scale
            lam = self._sample_uniform(effective_lam_lo, lam_hi)
        noisy = torch.poisson(image * lam) / lam
        return noisy.clamp(0.0, 1.0)

    def add_salt_pepper(self, image: torch.Tensor, prob: Optional[float] = None) -> torch.Tensor:
        """
        Salt-and-pepper noise: random pixels set to 0 or 1.

        Args:
            image: (C, H, W) float tensor in [0, 1].
            prob: Probability of each pixel being corrupted. If None, sampled.
        Returns:
            Noisy image tensor.
        """
        if prob is None:
            lo, hi = self._scaled_range(*self.sp_prob_range)
            prob = self._sample_uniform(lo, hi)

        noisy = image.clone()
        mask = torch.rand_like(image)
        salt = mask < (prob / 2)
        pepper = mask > (1 - prob / 2)
        noisy[salt] = 1.0
        noisy[pepper] = 0.0
        return noisy

    def add_speckle(self, image: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Multiplicative speckle noise: x' = x + x · N(0, σ²).

        Args:
            image: (C, H, W) float tensor in [0, 1].
            sigma: Speckle standard deviation. If None, sampled from scaled range.
        Returns:
            Noisy image tensor (clamped to [0, 1]).
        """
        if sigma is None:
            lo, hi = self._scaled_range(*self.speckle_sigma_range)
            sigma = self._sample_uniform(lo, hi)
        noise = torch.randn_like(image) * sigma
        return (image + image * noise).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Mixed / compound noise
    # ------------------------------------------------------------------

    def add_mixed(self, image: torch.Tensor, num_types: int = 2) -> torch.Tensor:
        """
        Apply a random combination of 2-3 noise types sequentially.

        The order is randomized, and each type uses independently sampled params.

        Args:
            image: (C, H, W) float tensor in [0, 1].
            num_types: Number of noise types to combine (2 or 3).
        Returns:
            Noisy image tensor.
        """
        noise_fns = [self.add_gaussian, self.add_poisson, self.add_salt_pepper, self.add_speckle]
        chosen = self.rng.sample(noise_fns, k=min(num_types, len(noise_fns)))
        for fn in chosen:
            image = fn(image)
        return image

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        noise_type: Optional[str] = None,
        mixed_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply noise to an image.

        Args:
            image: (C, H, W) float tensor in [0, 1].
            noise_type: Specific type, or None to sample randomly.
            mixed_prob: Probability of applying mixed noise instead of single type.
        Returns:
            Noisy image.
        """
        # Decide if we use mixed noise
        if self.rng.random() < mixed_prob:
            return self.add_mixed(image, num_types=self.rng.choice([2, 3]))

        # Pick noise type
        if noise_type is None:
            noise_type = self.rng.choice(["gaussian", "poisson", "salt_pepper", "speckle"])

        dispatch = {
            "gaussian": self.add_gaussian,
            "poisson": self.add_poisson,
            "salt_pepper": self.add_salt_pepper,
            "speckle": self.add_speckle,
            "mixed": self.add_mixed,
        }
        fn = dispatch.get(noise_type, self.add_gaussian)
        return fn(image)


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    SNR = 10 · log₁₀(‖signal‖² / ‖noise‖²)
    """
    signal_power = (clean ** 2).mean()
    noise_power = ((noisy - clean) ** 2).mean()
    if noise_power < 1e-10:
        return float("inf")
    return 10.0 * torch.log10(signal_power / noise_power).item()
