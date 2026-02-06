from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseConfig:
    name: str
    start_epoch: int
    end_epoch: int
    noise_scale: float
    mixed_prob: float


class CurriculumScheduler:
    def __init__(self, cfg):
        self.enabled = cfg.get("enabled", True)
        self.phases = sorted(
            [PhaseConfig(**p) for p in cfg.get("phases", [])],
            key=lambda p: p.start_epoch,
        )

    def get_phase(self, epoch):
        for phase in reversed(self.phases):
            if epoch >= phase.start_epoch:
                return phase
        return self.phases[0] if self.phases else None

    def get_noise_params(self, epoch):
        if not self.enabled or not self.phases:
            return 1.0, 0.3
        phase = self.get_phase(epoch)
        if phase is None:
            return 1.0, 0.3
        prev = None
        for p in self.phases:
            if p.start_epoch < phase.start_epoch:
                prev = p
        if prev is None or epoch <= phase.start_epoch:
            return phase.noise_scale, phase.mixed_prob
        progress = min(1.0, max(0.0, (epoch - phase.start_epoch) / max(phase.end_epoch - phase.start_epoch, 1)))
        ns = prev.noise_scale + progress * (phase.noise_scale - prev.noise_scale)
        mp = prev.mixed_prob + progress * (phase.mixed_prob - prev.mixed_prob)
        return ns, mp

    def get_phase_name(self, epoch):
        phase = self.get_phase(epoch)
        return phase.name if phase else "unknown"

    def summary(self):
        lines = ["Curriculum Learning Schedule:", "-" * 50]
        for p in self.phases:
            lines.append(f"  Phase '{p.name}': epochs {p.start_epoch}-{p.end_epoch}, "
                         f"noise_scale={p.noise_scale:.2f}, mixed_prob={p.mixed_prob:.2f}")
        return "\n".join(lines)
