"""
Shape synthesis module.

Generates random organic silhouettes and renders them as dot-pattern inputs
(dots placed along the shape outline) paired with ground truth filled masks.

This trains the model to "connect the dots" — seeing only sparse boundary
points and inferring the complete shape.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import cv2


class ShapeSynthesizer:
    """
    Generates random organic 2D shapes and renders them as dot patterns.

    Shapes are constructed from smooth closed Bézier curves with randomized
    control points, producing natural silhouettes (similar to sharks, leaves,
    hands, etc.) rather than rigid geometric primitives.

    The input image is a dot pattern (dots placed along the outline),
    and the ground truth is the filled binary mask.

    Args:
        image_size: Output image dimensions (square).
        min_control_points: Minimum number of Bézier control points.
        max_control_points: Maximum number of control points.
        min_radius_frac: Minimum shape radius as fraction of image size.
        max_radius_frac: Maximum shape radius as fraction of image size.
        min_dots: Minimum number of dots along the outline.
        max_dots: Maximum number of dots along the outline.
        min_dot_radius: Minimum dot circle radius in pixels.
        max_dot_radius: Maximum dot circle radius in pixels.
        dot_intensity_range: (min, max) intensity of dots in [0, 1].
        bg_intensity_range: (min, max) background brightness in [0, 1].
        jitter_frac: Positional jitter as fraction of shape radius.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        image_size: int = 512,
        min_control_points: int = 5,
        max_control_points: int = 15,
        min_radius_frac: float = 0.15,
        max_radius_frac: float = 0.40,
        min_dots: int = 15,
        max_dots: int = 80,
        min_dot_radius: int = 2,
        max_dot_radius: int = 6,
        dot_intensity_range: tuple[float, float] = (0.6, 1.0),
        bg_intensity_range: tuple[float, float] = (0.0, 0.4),
        jitter_frac: float = 0.03,
        fill_mode: str = "binary",
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.min_cp = min_control_points
        self.max_cp = max_control_points
        self.min_radius = int(image_size * min_radius_frac)
        self.max_radius = int(image_size * max_radius_frac)
        self.min_dots = min_dots
        self.max_dots = max_dots
        self.min_dot_radius = min_dot_radius
        self.max_dot_radius = max_dot_radius
        self.dot_intensity_range = dot_intensity_range
        self.bg_intensity_range = bg_intensity_range
        self.jitter_frac = jitter_frac
        self.fill_mode = fill_mode
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed if seed is not None else None)

    def _random_bezier_blob(self) -> np.ndarray:
        """
        Generate a closed Bézier blob as a polygon of (x, y) points.

        Strategy:
        1. Pick a random center within the safe zone.
        2. Generate N control points at random angles with random radii.
        3. Sort by angle to ensure a non-self-intersecting contour.
        4. Smooth with cubic Bézier interpolation.
        """
        cx = self.rng.randint(self.max_radius + 10, self.image_size - self.max_radius - 10)
        cy = self.rng.randint(self.max_radius + 10, self.image_size - self.max_radius - 10)

        n_points = self.rng.randint(self.min_cp, self.max_cp)
        angles = sorted([self.rng.uniform(0, 2 * math.pi) for _ in range(n_points)])

        # Generate radii with smooth variation
        base_radius = self.rng.uniform(self.min_radius, self.max_radius)
        radii = []
        for _ in range(n_points):
            r = base_radius * self.rng.uniform(0.5, 1.5)
            radii.append(r)

        # Convert to Cartesian control points
        control_pts = []
        for angle, radius in zip(angles, radii):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            control_pts.append((x, y))

        # Interpolate smooth curve through control points
        return self._catmull_rom_spline(control_pts, num_interp=200)

    def _catmull_rom_spline(
        self, points: list[tuple[float, float]], num_interp: int = 200
    ) -> np.ndarray:
        """
        Catmull-Rom spline interpolation through a closed set of points.
        Produces a smooth, C1-continuous closed curve.
        """
        pts = np.array(points, dtype=np.float64)
        n = len(pts)
        curve_points = []

        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]

            steps = max(num_interp // n, 10)
            for t_idx in range(steps):
                t = t_idx / steps
                t2 = t * t
                t3 = t2 * t

                # Catmull-Rom basis
                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                curve_points.append((x, y))

        return np.array(curve_points, dtype=np.int32)

    def _generate_multi_blob(self, num_blobs: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
        """Generate a mask with one or more organic blobs. Returns mask and contours."""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        contours = []
        for _ in range(num_blobs):
            contour = self._random_bezier_blob()
            # Clip to valid image bounds
            contour[:, 0] = np.clip(contour[:, 0], 0, self.image_size - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, self.image_size - 1)
            cv2.fillPoly(mask, [contour], color=255)
            contours.append(contour)
        return mask, contours

    def _sample_contour_points(self, contour: np.ndarray, num_dots: int) -> np.ndarray:
        """
        Sample evenly-spaced points along a contour with optional jitter.

        Returns (num_dots, 2) array of (x, y) positions.
        """
        # Compute cumulative arc length along contour
        diffs = np.diff(contour, axis=0).astype(np.float64)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = cum_lengths[-1]

        if total_length < 1:
            return contour[:num_dots]

        # Sample at equal arc-length intervals
        target_lengths = np.linspace(0, total_length, num_dots, endpoint=False)
        sampled_points = []

        for target in target_lengths:
            idx = np.searchsorted(cum_lengths, target, side="right") - 1
            idx = max(0, min(idx, len(contour) - 2))

            seg_start = cum_lengths[idx]
            seg_len = seg_lengths[idx] if idx < len(seg_lengths) else 1.0
            t = (target - seg_start) / max(seg_len, 1e-6)
            t = np.clip(t, 0, 1)

            pt = contour[idx].astype(np.float64) * (1 - t) + contour[idx + 1].astype(np.float64) * t

            # Add jitter
            if self.jitter_frac > 0:
                jitter_px = self.jitter_frac * self.image_size
                pt[0] += self.rng.gauss(0, jitter_px)
                pt[1] += self.rng.gauss(0, jitter_px)

            pt[0] = np.clip(pt[0], 0, self.image_size - 1)
            pt[1] = np.clip(pt[1], 0, self.image_size - 1)
            sampled_points.append(pt)

        return np.array(sampled_points, dtype=np.float64)

    def _render_dot_pattern(
        self, contours: list[np.ndarray], bg_value: float, dot_value: float
    ) -> np.ndarray:
        """
        Render dots along shape contours on a background.

        Produces an image that looks like a connect-the-dots puzzle.
        """
        image = np.full((self.image_size, self.image_size), bg_value, dtype=np.float32)

        for contour in contours:
            num_dots = self.rng.randint(self.min_dots, self.max_dots)
            points = self._sample_contour_points(contour, num_dots)

            for pt in points:
                x, y = int(round(pt[0])), int(round(pt[1]))
                r = self.rng.randint(self.min_dot_radius, self.max_dot_radius)

                # Slight per-dot intensity variation
                intensity = dot_value * self.rng.uniform(0.8, 1.0)

                cv2.circle(image, (x, y), r, float(intensity), -1, lineType=cv2.LINE_AA)

        return image

    def _apply_fill(self, mask: np.ndarray) -> np.ndarray:
        """Fill the mask interior according to the fill mode."""
        if self.fill_mode == "binary":
            return (mask > 0).astype(np.float32)

        elif self.fill_mode == "gradient":
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return mask.astype(np.float32) / 255.0
            cx, cy = xs.mean(), ys.mean()
            max_dist = max(
                np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max(), 1.0
            )
            image = np.zeros_like(mask, dtype=np.float32)
            yy, xx = np.mgrid[0 : self.image_size, 0 : self.image_size]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            gradient = 1.0 - (dist / (max_dist * 1.2)).clip(0, 1)
            image[mask > 0] = gradient[mask > 0]
            return image

        elif self.fill_mode == "texture":
            noise = self.np_rng.randn(self.image_size, self.image_size).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (31, 31), 7.0)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            image = np.zeros_like(mask, dtype=np.float32)
            image[mask > 0] = noise[mask > 0] * 0.7 + 0.3
            return image

        else:
            return (mask > 0).astype(np.float32)

    def generate(self, num_blobs: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a single (dot_pattern_image, filled_mask) pair.

        Args:
            num_blobs: Number of shapes per image. If None, randomly 1-2.

        Returns:
            image: (H, W) float32 array in [0, 1] — dot pattern input.
            mask: (H, W) float32 array in {0, 1} — filled ground truth mask.
        """
        if num_blobs is None:
            num_blobs = self.rng.randint(1, 2)

        raw_mask, contours = self._generate_multi_blob(num_blobs)
        mask = (raw_mask > 0).astype(np.float32)

        # Random background and dot intensities (supports any "color" combo)
        bg_value = self.rng.uniform(*self.bg_intensity_range)
        dot_value = self.rng.uniform(*self.dot_intensity_range)

        # Make sure dots contrast with background
        if abs(dot_value - bg_value) < 0.3:
            dot_value = 1.0 - bg_value  # flip to ensure contrast

        image = self._render_dot_pattern(contours, bg_value, dot_value)

        return image, mask

    def generate_batch(
        self, batch_size: int, num_blobs: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of (images, masks).

        Returns:
            images: (B, H, W) float32 array.
            masks: (B, H, W) float32 array.
        """
        images, masks = [], []
        for _ in range(batch_size):
            img, msk = self.generate(num_blobs)
            images.append(img)
            masks.append(msk)
        return np.stack(images), np.stack(masks)
