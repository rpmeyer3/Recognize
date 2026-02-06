from __future__ import annotations
import math, random
from typing import Optional
import numpy as np
import cv2


class ShapeSynthesizer:
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
        pad = self.max_radius + 10
        cx = self.rng.randint(pad, self.image_size - pad)
        cy = self.rng.randint(pad, self.image_size - pad)
        n = self.rng.randint(self.min_cp, self.max_cp)
        angles = sorted([self.rng.uniform(0, 2 * math.pi) for _ in range(n)])
        base_r = self.rng.uniform(self.min_radius, self.max_radius)
        radii = [base_r * self.rng.uniform(0.5, 1.5) for _ in range(n)]
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a, r in zip(angles, radii)]
        return self._catmull_rom_spline(pts, 200)

    def _catmull_rom_spline(self, points, num_interp=200):
        pts = np.array(points, dtype=np.float64)
        n = len(pts)
        out = []
        for i in range(n):
            p0, p1 = pts[(i - 1) % n], pts[i]
            p2, p3 = pts[(i + 1) % n], pts[(i + 2) % n]
            steps = max(num_interp // n, 10)
            for ti in range(steps):
                t = ti / steps
                t2, t3 = t * t, t * t * t
                x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
                y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
                out.append((x, y))
        return np.array(out, dtype=np.int32)

    def _generate_multi_blob(self, num_blobs=1):
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        contours = []
        for _ in range(num_blobs):
            c = self._random_bezier_blob()
            c[:, 0] = np.clip(c[:, 0], 0, self.image_size - 1)
            c[:, 1] = np.clip(c[:, 1], 0, self.image_size - 1)
            cv2.fillPoly(mask, [c], color=255)
            contours.append(c)
        return mask, contours

    def _sample_contour_points(self, contour, num_dots):
        diffs = np.diff(contour, axis=0).astype(np.float64)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total < 1:
            return contour[:num_dots]
        targets = np.linspace(0, total, num_dots, endpoint=False)
        pts = []
        for tgt in targets:
            idx = max(0, min(np.searchsorted(cum, tgt, side="right") - 1, len(contour) - 2))
            sl = seg_lens[idx] if idx < len(seg_lens) else 1.0
            t = np.clip((tgt - cum[idx]) / max(sl, 1e-6), 0, 1)
            pt = contour[idx].astype(np.float64) * (1 - t) + contour[idx + 1].astype(np.float64) * t
            if self.jitter_frac > 0:
                j = self.jitter_frac * self.image_size
                pt[0] += self.rng.gauss(0, j)
                pt[1] += self.rng.gauss(0, j)
            pt[0] = np.clip(pt[0], 0, self.image_size - 1)
            pt[1] = np.clip(pt[1], 0, self.image_size - 1)
            pts.append(pt)
        return np.array(pts, dtype=np.float64)

    def _render_dot_pattern(self, contours, bg_val, dot_val):
        img = np.full((self.image_size, self.image_size), bg_val, dtype=np.float32)
        for contour in contours:
            ndots = self.rng.randint(self.min_dots, self.max_dots)
            points = self._sample_contour_points(contour, ndots)
            for pt in points:
                x, y = int(round(pt[0])), int(round(pt[1]))
                r = self.rng.randint(self.min_dot_radius, self.max_dot_radius)
                intensity = dot_val * self.rng.uniform(0.8, 1.0)
                cv2.circle(img, (x, y), r, float(intensity), -1, lineType=cv2.LINE_AA)
        return img

    def _apply_fill(self, mask):
        if self.fill_mode == "binary":
            return (mask > 0).astype(np.float32)
        elif self.fill_mode == "gradient":
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return mask.astype(np.float32) / 255.0
            cx, cy = xs.mean(), ys.mean()
            max_d = max(np.sqrt((xs - cx)**2 + (ys - cy)**2).max(), 1.0)
            out = np.zeros_like(mask, dtype=np.float32)
            yy, xx = np.mgrid[0:self.image_size, 0:self.image_size]
            grad = 1.0 - (np.sqrt((xx - cx)**2 + (yy - cy)**2) / (max_d * 1.2)).clip(0, 1)
            out[mask > 0] = grad[mask > 0]
            return out
        elif self.fill_mode == "texture":
            noise = self.np_rng.randn(self.image_size, self.image_size).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (31, 31), 7.0)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            out = np.zeros_like(mask, dtype=np.float32)
            out[mask > 0] = noise[mask > 0] * 0.7 + 0.3
            return out
        return (mask > 0).astype(np.float32)

    def generate(self, num_blobs=None):
        if num_blobs is None:
            num_blobs = self.rng.randint(1, 2)
        raw_mask, contours = self._generate_multi_blob(num_blobs)
        mask = (raw_mask > 0).astype(np.float32)
        bg = self.rng.uniform(*self.bg_intensity_range)
        dot = self.rng.uniform(*self.dot_intensity_range)
        if abs(dot - bg) < 0.3:
            dot = 1.0 - bg
        img = self._render_dot_pattern(contours, bg, dot)
        return img, mask

    def generate_batch(self, batch_size, num_blobs=None):
        imgs, masks = [], []
        for _ in range(batch_size):
            img, msk = self.generate(num_blobs)
            imgs.append(img)
            masks.append(msk)
        return np.stack(imgs), np.stack(masks)
