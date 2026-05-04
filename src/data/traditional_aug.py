"""Offline geometric augmentation operators (paper-faithful).

Per the paper's individual-augmentation analysis (§5.3) and decision D6, only the three
best-performing transforms are applied:

- **Rotation 270°** (all cases)
- **Shearing 30°**  (all cases)
- **Rotation 90°** (Case 1) or **Rotation 45°** (Cases 2, 3, 4) — the case's "second rotation"

Vertical flip and zoom hurt performance in the paper and are excluded. Horizontal flip and
cropping are neutral and excluded for parsimony (and to keep traditional and hybrid sharing
the same transform set, so hybrid is a clean WGAN-only delta on top of traditional).

All transforms are deterministic, take and return a uint8 BGR/RGB image (cv2-loaded) of the
same H×W as the input, and fill exposed border pixels with black (border value 0).
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np


def rotate_270(img: np.ndarray) -> np.ndarray:
    """270° == 90° counter-clockwise. Output is rotated; H/W swapped."""
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_90(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate_arbitrary(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate around the image centre, keep canvas size; black-fill exposed corners."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def shear_x(img: np.ndarray, degrees: float) -> np.ndarray:
    """Affine x-shear by `degrees` (paper uses 30°). Keeps canvas size; centred."""
    h, w = img.shape[:2]
    sh = math.tan(math.radians(degrees))
    # Translate so the centre stays put after shear.
    tx = -sh * h / 2.0
    M = np.array([[1.0, sh, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def case_aug_recipe(second_rotation_deg: int, shear_deg: float = 30.0) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Return [(tag, fn), ...] for the case's three best transforms.

    `tag` is used in the output filename so each augmented PNG is identifiable.
    """
    if second_rotation_deg == 90:
        second = ("rot90", rotate_90)
    else:
        second = (f"rot{second_rotation_deg}", lambda im, d=second_rotation_deg: rotate_arbitrary(im, d))

    return [
        ("rot270", rotate_270),
        ("shear30", lambda im, d=shear_deg: shear_x(im, d)),
        second,
    ]
