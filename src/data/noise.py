from __future__ import annotations
import random
from typing import Optional
import torch
import numpy as np


class NoiseInjector:
    def __init__(self, cfg: dict, scale: float = 1.0, rng=None):
        self.cfg = cfg
        self.scale = max(0.0, min(1.0, scale))
        self.rng = rng or random.Random()
        self.gaussian_sigma_range = cfg.get("gaussian_sigma", [0.01, 1.5])
        self.poisson_lambda_range = cfg.get("poisson_lambda", [1.0, 300.0])
        self.sp_prob_range = cfg.get("salt_pepper_prob", [0.01, 0.5])
        self.speckle_sigma_range = cfg.get("speckle_sigma", [0.05, 2.0])

    def set_scale(self, scale: float):
        self.scale = max(0.0, min(1.0, scale))

    def _scaled_range(self, lo, hi):
        return lo, lo + (hi - lo) * self.scale

    def add_gaussian(self, x, sigma=None):
        if sigma is None:
            lo, hi = self._scaled_range(*self.gaussian_sigma_range)
            sigma = self.rng.uniform(lo, hi)
        return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)

    def add_poisson(self, x, lam=None):
        lo, hi = self.poisson_lambda_range
        if lam is None:
            eff_lo = hi - (hi - lo) * self.scale
            lam = self.rng.uniform(eff_lo, hi)
        return (torch.poisson(x * lam) / lam).clamp(0.0, 1.0)

    def add_salt_pepper(self, x, prob=None):
        if prob is None:
            lo, hi = self._scaled_range(*self.sp_prob_range)
            prob = self.rng.uniform(lo, hi)
        out = x.clone()
        mask = torch.rand_like(x)
        out[mask < (prob / 2)] = 1.0
        out[mask > (1 - prob / 2)] = 0.0
        return out

    def add_speckle(self, x, sigma=None):
        if sigma is None:
            lo, hi = self._scaled_range(*self.speckle_sigma_range)
            sigma = self.rng.uniform(lo, hi)
        return (x + x * torch.randn_like(x) * sigma).clamp(0.0, 1.0)

    def add_mixed(self, x, num_types=2):
        fns = [self.add_gaussian, self.add_poisson, self.add_salt_pepper, self.add_speckle]
        for fn in self.rng.sample(fns, k=min(num_types, len(fns))):
            x = fn(x)
        return x

    def __call__(self, x, noise_type=None, mixed_prob=0.0):
        if self.rng.random() < mixed_prob:
            return self.add_mixed(x, num_types=self.rng.choice([2, 3]))
        if noise_type is None:
            noise_type = self.rng.choice(["gaussian", "poisson", "salt_pepper", "speckle"])
        dispatch = {
            "gaussian": self.add_gaussian, "poisson": self.add_poisson,
            "salt_pepper": self.add_salt_pepper, "speckle": self.add_speckle,
            "mixed": self.add_mixed,
        }
        return dispatch.get(noise_type, self.add_gaussian)(x)


def compute_snr(clean, noisy):
    sig = (clean ** 2).mean()
    noise = ((noisy - clean) ** 2).mean()
    if noise < 1e-10:
        return float("inf")
    return 10.0 * torch.log10(sig / noise).item()
