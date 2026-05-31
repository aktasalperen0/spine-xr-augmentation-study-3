"""ROI (region-of-interest) patch utilities for the bbox-crop pipeline.

The dataset's lesions occupy 0.15%-3.1% of a full ~3000x2500 spine X-ray. Classifying or
generating from whole images throws the signal away. These helpers crop tight lesion patches
(with a small context margin) and produce matched-size "No Finding" patches so the classifier
cannot exploit crop geometry as a shortcut (see plan decision R3).

All functions operate on single-channel (grayscale) uint8 numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _clamp_box(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> tuple[int, int, int, int]:
    xmin = int(max(0, min(round(xmin), w - 1)))
    ymin = int(max(0, min(round(ymin), h - 1)))
    xmax = int(max(xmin + 1, min(round(xmax), w)))
    ymax = int(max(ymin + 1, min(round(ymax), h)))
    return xmin, ymin, xmax, ymax


def expand_box(box: tuple[float, float, float, float], margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Expand (xmin,ymin,xmax,ymax) by `margin` fraction of its size on each side, clamped."""
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    dx, dy = bw * margin, bh * margin
    return _clamp_box(xmin - dx, ymin - dy, xmax + dx, ymax + dy, w, h)


def crop_with_margin(img: np.ndarray, box: tuple[float, float, float, float], margin: float) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = expand_box(box, margin, w, h)
    return img[y0:y1, x0:x1]


def square_resize(patch: np.ndarray, size: int) -> np.ndarray:
    """Stretch-resize a patch to (size, size) — fills the frame with lesion content.

    Aspect ratio is intentionally not preserved: for tiny lesions, filling the frame with the
    pathology is more valuable than preserving anatomical proportions, and the identical
    transform is applied to positive and NF patches (so no geometry shortcut).
    """
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class BBoxSizeSampler:
    """Empirical sampler over (width, height) of a case's positive boxes in one split.

    Used to generate No-Finding patches whose size distribution matches the lesion patches,
    so the classifier cannot separate classes by crop tightness (plan R3).
    """
    widths: np.ndarray
    heights: np.ndarray

    @classmethod
    def from_boxes(cls, boxes: list[tuple[float, float, float, float]]) -> "BBoxSizeSampler":
        if not boxes:
            raise ValueError("cannot build size sampler from zero boxes")
        arr = np.array(boxes, dtype=np.float64)
        return cls(widths=arr[:, 2] - arr[:, 0], heights=arr[:, 3] - arr[:, 1])

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        i = int(rng.integers(0, len(self.widths)))
        return float(self.widths[i]), float(self.heights[i])

    def median(self) -> tuple[float, float]:
        return float(np.median(self.widths)), float(np.median(self.heights))


def sample_nf_patch(
    img: np.ndarray,
    sampler: BBoxSizeSampler,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Crop a random region from an NF image whose (w,h) is drawn from the lesion size dist."""
    h, w = img.shape[:2]
    bw, bh = sampler.sample(rng)
    # Apply the same margin expansion used for positive patches.
    bw = min(bw * (1 + 2 * margin), w)
    bh = min(bh * (1 + 2 * margin), h)
    bw = max(int(round(bw)), 1)
    bh = max(int(round(bh)), 1)
    x0 = int(rng.integers(0, max(1, w - bw + 1)))
    y0 = int(rng.integers(0, max(1, h - bh + 1)))
    return img[y0:y0 + bh, x0:x0 + bw]
