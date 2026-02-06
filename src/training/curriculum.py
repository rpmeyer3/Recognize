"""
Curriculum Learning scheduler.

Controls training difficulty by gradually increasing noise intensity
and complexity across defined phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    start_epoch: int
    end_epoch: int
    noise_scale: float
    mixed_prob: float


class CurriculumScheduler:
    """
    Manages curriculum learning phases for noise-robust training.

    The scheduler controls:
    1. **noise_scale** (0.0–1.0): Fraction of the max noise parameter range
       that is active. At scale=0.1, only the easiest 10% of the noise range
       is sampled.
    2. **mixed_prob** (0.0–1.0): Probability of applying compound noise
       (2-3 noise types simultaneously).

    The key insight is that starting with easy examples lets the network
    first learn robust shape priors, then progressively learn to extract
    those shapes from increasingly degraded inputs.

    Args:
        cfg: Curriculum configuration dictionary containing phase list.
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", True)
        self.phases: list[PhaseConfig] = []

        for p in cfg.get("phases", []):
            self.phases.append(PhaseConfig(
                name=p["name"],
                start_epoch=p["start_epoch"],
                end_epoch=p["end_epoch"],
                noise_scale=p["noise_scale"],
                mixed_prob=p["mixed_prob"],
            ))

        # Sort phases by start epoch
        self.phases.sort(key=lambda p: p.start_epoch)

    def get_phase(self, epoch: int) -> Optional[PhaseConfig]:
        """Get the phase config for a given epoch."""
        for phase in reversed(self.phases):
            if epoch >= phase.start_epoch:
                return phase
        return self.phases[0] if self.phases else None

    def get_noise_params(self, epoch: int) -> tuple[float, float]:
        """
        Get noise parameters for the current epoch.

        Within a phase, parameters are linearly interpolated from the
        previous phase's values to the current phase's target values.
        This avoids jarring transitions between phases.

        Args:
            epoch: Current training epoch.

        Returns:
            (noise_scale, mixed_prob) tuple.
        """
        if not self.enabled or not self.phases:
            return 1.0, 0.3

        phase = self.get_phase(epoch)
        if phase is None:
            return 1.0, 0.3

        # Find previous phase for interpolation
        prev_phase = None
        for p in self.phases:
            if p.start_epoch < phase.start_epoch:
                prev_phase = p

        if prev_phase is None or epoch <= phase.start_epoch:
            return phase.noise_scale, phase.mixed_prob

        # Smooth interpolation within phase
        progress = (epoch - phase.start_epoch) / max(
            phase.end_epoch - phase.start_epoch, 1
        )
        progress = min(1.0, max(0.0, progress))

        noise_scale = prev_phase.noise_scale + progress * (
            phase.noise_scale - prev_phase.noise_scale
        )
        mixed_prob = prev_phase.mixed_prob + progress * (
            phase.mixed_prob - prev_phase.mixed_prob
        )

        return noise_scale, mixed_prob

    def get_phase_name(self, epoch: int) -> str:
        """Get the human-readable name of the current phase."""
        phase = self.get_phase(epoch)
        return phase.name if phase else "unknown"

    def summary(self) -> str:
        """Print a summary of all curriculum phases."""
        lines = ["Curriculum Learning Schedule:", "-" * 50]
        for p in self.phases:
            lines.append(
                f"  Phase '{p.name}': epochs {p.start_epoch}-{p.end_epoch}, "
                f"noise_scale={p.noise_scale:.2f}, mixed_prob={p.mixed_prob:.2f}"
            )
        return "\n".join(lines)
