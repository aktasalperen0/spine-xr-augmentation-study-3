"""Image-only Dataset for WGAN training (one class pool).

The pool is built by selecting rows from a case's `train.csv` where the target class column
is 1. Multi-label co-occurrence is allowed (decision D2 + plan §"Open risks" mitigation):
e.g., the WGAN for "Vertebral collapse" sees images that also have DSN positive — we accept
that synthetic outputs may carry secondary-class features.

Pre-processing:
- Read PNG as grayscale.
- Resize so the longer side fits in `image_size`, then centre-pad with 0 to (image_size, image_size).
- Convert to float32 in [-1, 1] (matches Generator's Tanh output range).
- Return a (1, H, W) tensor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resize_and_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=img.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


class WGANClassDataset(Dataset):
    """Returns (image_tensor,) — labels intentionally not exposed."""

    def __init__(self, pool_csv: Path | str, target_class: str, image_size: int = 256):
        df = pd.read_csv(pool_csv)
        if target_class not in df.columns:
            raise ValueError(f"{target_class!r} not in {pool_csv}")
        self.df = df[df[target_class] == 1].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"empty pool for {target_class!r} in {pool_csv}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["path"])
        img = _resize_and_pad(img, self.image_size)
        # uint8 [0, 255] -> float32 [-1, 1]
        x = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).unsqueeze(0)
