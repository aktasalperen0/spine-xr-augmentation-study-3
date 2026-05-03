"""Image transforms. Grayscale X-ray PNG -> 3-channel tensor with CLAHE + normalize.

The "baseline_no_aug" preset is paper-faithful: no augmentation at all (resize/pad/CLAHE/normalize).
Stronger presets are added in later phases.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pretrained backbones expect ImageNet-style normalization. X-rays are grayscale-replicated
# to 3 channels; we use a symmetric (0.5, 0.5, 0.5) normalization that has worked well in
# medical-imaging transfer learning (matches study-2's choice).
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def _resize_pad(image_size: int) -> list[A.BasicTransform]:
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0),
    ]


def _clahe() -> A.BasicTransform:
    return A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)


def _normalize_to_tensor() -> list[A.BasicTransform]:
    return [A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]


def val_transform(image_size: int) -> A.Compose:
    return A.Compose([*_resize_pad(image_size), _clahe(), *_normalize_to_tensor()])


def baseline_no_aug_transform(image_size: int) -> A.Compose:
    """Paper-faithful baseline: no augmentation. Identical pipeline to val."""
    return val_transform(image_size)


def build_transform(name: str, image_size: int) -> A.Compose:
    if name in ("val", "eval"):
        return val_transform(image_size)
    if name == "baseline_no_aug":
        return baseline_no_aug_transform(image_size)
    raise ValueError(f"unknown transform preset: {name}")
