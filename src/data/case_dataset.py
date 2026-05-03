"""PyTorch Dataset for a single Case split CSV.

Each row in the case split CSV has: image_id, study_id, path, source, <label_columns...>.
The label columns are passed in via `label_columns` (e.g. ["Disc space narrowing",
"Vertebral collapse", "No finding"]). The dataset is multi-label: an image with
DSN=1, VC=1, NF=0 returns label tensor [1.0, 1.0, 0.0].
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CaseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_columns: list[str],
        transform: Callable | None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["path"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        label = torch.tensor([float(row[c]) for c in self.label_columns], dtype=torch.float32)
        return img, label


def compute_pos_weight(
    df: pd.DataFrame, label_columns: list[str], clip: float = 10.0
) -> torch.Tensor:
    """Per-class neg/pos ratio for BCEWithLogitsLoss `pos_weight`. Clipped to avoid blow-ups."""
    weights: list[float] = []
    n = len(df)
    for c in label_columns:
        pos = int(df[c].sum())
        neg = n - pos
        w = neg / max(pos, 1)
        weights.append(min(w, clip))
    return torch.tensor(weights, dtype=torch.float32)
