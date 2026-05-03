"""Multi-label classification metrics.

For each output class we compute precision/recall/F1 at threshold 0.5, plus AP and AUROC
on the raw probabilities. Macro F1 — the primary model-selection metric (D8) — is the
unweighted mean of per-class F1 across the case's output classes.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def per_class_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Iterable[str], threshold: float = 0.5
) -> dict:
    """y_true and y_prob: shape (N, C), float in [0,1] for y_prob, 0/1 for y_true."""
    class_names = list(class_names)
    y_pred = (y_prob >= threshold).astype(np.int32)
    out: dict = {"per_class": {}, "threshold": threshold}

    for i, c in enumerate(class_names):
        yt, yp, ypr = y_true[:, i], y_pred[:, i], y_prob[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        try:
            ap = float(average_precision_score(yt, ypr))
        except ValueError:
            ap = float("nan")
        try:
            auroc = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else float("nan")
        except ValueError:
            auroc = float("nan")
        out["per_class"][c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "ap": ap,
            "auroc": auroc,
            "n_pos": int(yt.sum()),
        }

    f1s = [out["per_class"][c]["f1"] for c in class_names]
    out["macro_f1"] = float(np.mean(f1s))
    return out


def format_metrics_md(metrics: dict, class_names: Iterable[str]) -> str:
    lines = ["| class | F1 | precision | recall | AP | AUROC | n_pos |", "|---|---|---|---|---|---|---|"]
    for c in class_names:
        m = metrics["per_class"][c]
        lines.append(
            f"| {c} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['ap']:.4f} | {m['auroc']:.4f} | {m['n_pos']} |"
        )
    lines.append(f"| **macro** | **{metrics['macro_f1']:.4f}** |  |  |  |  |  |")
    return "\n".join(lines)
