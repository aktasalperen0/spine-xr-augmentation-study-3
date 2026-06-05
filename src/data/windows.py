"""Fixed-size window utilities for the patient-level pipeline.

Training and test now both use FIXED 128x128 native-resolution windows (no box-stretching), so
the classifier sees the same distribution at train and test time:
- training positives  : 128 windows centred (with jitter) on a lesion box
- training hard-negs   : 128 windows from abnormal images that overlap NO box
- training NF negs      : 128 windows from normal images
- test                 : 128 windows tiled with stride 96 across the whole image (no box used)

All functions take/return single-channel uint8 arrays.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]   # xmin, ymin, xmax, ymax


def _clip_window(cx: int, cy: int, win: int, w: int, h: int) -> tuple[int, int]:
    """Return top-left (x0,y0) of a win×win window centred at (cx,cy), clipped inside the image."""
    x0 = int(np.clip(cx - win // 2, 0, max(0, w - win)))
    y0 = int(np.clip(cy - win // 2, 0, max(0, h - win)))
    return x0, y0


def _crop(img: np.ndarray, x0: int, y0: int, win: int) -> np.ndarray:
    """Crop win×win; pad with 0 if the image is smaller than win."""
    h, w = img.shape[:2]
    patch = np.zeros((win, win), dtype=img.dtype)
    x1, y1 = min(x0 + win, w), min(y0 + win, h)
    patch[: y1 - y0, : x1 - x0] = img[y0:y1, x0:x1]
    return patch


def window_overlaps_box(x0: int, y0: int, win: int, box: Box, min_frac: float = 0.1) -> bool:
    """True if box∩window area >= min_frac * min(box_area, window_area)."""
    bx0, by0, bx1, by1 = box
    ix0, iy0 = max(x0, bx0), max(y0, by0)
    ix1, iy1 = min(x0 + win, bx1), min(y0 + win, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return False
    box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    win_area = float(win * win)
    return inter >= min_frac * min(box_area, win_area)


def positive_windows(img: np.ndarray, box: Box, n: int, win: int, jitter: float, rng) -> list[tuple[int, int, np.ndarray]]:
    """n windows centred on the box centre with up to `jitter`*win random offset."""
    h, w = img.shape[:2]
    bx0, by0, bx1, by1 = box
    cx, cy = int((bx0 + bx1) / 2), int((by0 + by1) / 2)
    out = []
    for _ in range(n):
        jx = int(rng.uniform(-jitter, jitter) * win)
        jy = int(rng.uniform(-jitter, jitter) * win)
        x0, y0 = _clip_window(cx + jx, cy + jy, win, w, h)
        out.append((x0, y0, _crop(img, x0, y0, win)))
    return out


def hard_negative_windows(img: np.ndarray, boxes: list[Box], n: int, win: int, rng, max_tries: int = 50) -> list[tuple[int, int, np.ndarray]]:
    """n random windows that overlap none of `boxes` (background of an abnormal image)."""
    h, w = img.shape[:2]
    out = []
    tries = 0
    while len(out) < n and tries < n * max_tries:
        tries += 1
        x0 = int(rng.integers(0, max(1, w - win + 1)))
        y0 = int(rng.integers(0, max(1, h - win + 1)))
        if any(window_overlaps_box(x0, y0, win, b, min_frac=0.02) for b in boxes):
            continue
        out.append((x0, y0, _crop(img, x0, y0, win)))
    return out


def sliding_windows(img: np.ndarray, win: int = 128, stride: int = 96):
    """Yield (x0, y0, patch) tiling the whole image; the last row/col is clamped to the edge."""
    h, w = img.shape[:2]
    xs = list(range(0, max(1, w - win + 1), stride))
    ys = list(range(0, max(1, h - win + 1), stride))
    if not xs or xs[-1] != w - win:
        xs.append(max(0, w - win))
    if not ys or ys[-1] != h - win:
        ys.append(max(0, h - win))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, _crop(img, x0, y0, win)
