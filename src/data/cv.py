"""Study-disjoint 3-fold assignment over ROI patches (advisor F2).

Each ROI patch is single-label (one lesion class or No finding). We stratify folds by that
single class while keeping every study_id wholly within one fold (no patient leakage across
folds) via sklearn's StratifiedGroupKFold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def patch_class(df: pd.DataFrame, label_columns: list[str]) -> np.ndarray:
    """Single-label class index per patch = argmax over the one-hot label columns."""
    return df[label_columns].to_numpy().astype(int).argmax(axis=1)


def assign_folds(df: pd.DataFrame, label_columns: list[str], n_splits: int = 3, seed: int = 42) -> pd.DataFrame:
    y = patch_class(df, label_columns)
    groups = df["study_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, val_idx) in enumerate(sgkf.split(df, y, groups), start=1):
        fold[val_idx] = k
    out = df.copy()
    out["fold"] = fold

    # Invariants.
    assert (out["fold"] > 0).all(), "some patches got no fold"
    spans = out.groupby("study_id")["fold"].nunique()
    assert spans.max() == 1, "a study_id spans multiple folds (leakage)"
    for k in range(1, n_splits + 1):
        present = out.loc[out["fold"] == k, label_columns].sum(axis=0)
        missing = [c for c in label_columns if int(present[c]) == 0]
        if missing:
            raise RuntimeError(f"fold {k} missing classes {missing} — reduce n_splits or check data")
    return out
