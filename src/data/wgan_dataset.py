"""Image-only Dataset for WGAN training — ROI lesion patches (the bbox-crop pivot).

The pool is a per-case ROI patch manifest (built by scripts/02b_build_roi_patches.py). We filter
to rows where the target class column == 1, i.e. patches of that lesion. Each patch PNG is already
a tight, store_size-square lesion crop, so the WGAN models lesion *texture* instead of a whole
spine — this is the keystone fix for the blurry-blob problem.

Pre-processing: read grayscale patch, square-resize to image_size, scale to [-1, 1] (Tanh range).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WGANClassDataset(Dataset):
    """Returns (image_tensor,) — labels intentionally not exposed."""

    def __init__(self, pool_csv: Path | str, target_class: str, image_size: int = 128):
        df = pd.read_csv(pool_csv)
        if target_class not in df.columns:
            raise ValueError(f"{target_class!r} not in {pool_csv}")
        self.df = df[df[target_class] == 1].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"empty patch pool for {target_class!r} in {pool_csv}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["path"])
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).unsqueeze(0)
