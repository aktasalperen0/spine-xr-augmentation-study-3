"""Phase 1: build per-image multi-label tables for the official train and test splits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.audit import build_label_table, class_counts, co_occurrence, lesion_classes
from src.utils.config import load_config
from src.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    classes = cfg["classes"]
    out_dir = Path(cfg["paths"]["outputs_root"]) / "01_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", "abnormal_train_annotations", "normal_train_metadata",
         "abnormal_train_pngs", "normal_train_pngs"),
        ("test", "abnormal_test_annotations", "normal_test_metadata",
         "abnormal_test_pngs", "normal_test_pngs"),
    ]

    for split, ann_key, meta_key, ab_key, nm_key in splits:
        log.info(f"Building {split} label table...")
        df = build_label_table(
            ROOT / cfg["paths"][ann_key],
            ROOT / cfg["paths"][meta_key],
            ROOT / cfg["paths"][ab_key],
            ROOT / cfg["paths"][nm_key],
            classes,
        )
        missing = df[~df["path"].apply(lambda p: Path(p).exists())]
        if len(missing):
            log.warning(f"{split}: {len(missing)} missing PNGs; dropping. First 3: {missing['path'].head(3).tolist()}")
            df = df[df["path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        df.to_csv(out_dir / f"{split}_labels.csv", index=False)
        log.info(f"{split}_labels.csv: {len(df)} rows")

        class_counts(df, classes).to_csv(out_dir / f"{split}_class_counts.csv", index=False)
        co_occurrence(df, classes).to_csv(out_dir / f"{split}_co_occurrence.csv")

    train = pd.read_csv(out_dir / "train_labels.csv")
    test = pd.read_csv(out_dir / "test_labels.csv")
    lesions = lesion_classes(classes)
    lines = ["# Audit Report", ""]
    for split_name, df in [("Train", train), ("Test", test)]:
        n = len(df)
        n_abn = int((df["source"] == "abnormal").sum())
        n_nrm = int((df["source"] == "normal").sum())
        multi = int((df[lesions].sum(axis=1) > 1).sum())
        lines += [
            f"## {split_name}",
            f"- Total images: {n}",
            f"- Abnormal: {n_abn}  Normal: {n_nrm}",
            f"- Multi-label images (>=2 lesions): {multi} ({multi / max(n_abn,1):.2%} of abnormal)",
            "- Per-class positives:",
        ]
        for c in classes:
            lines.append(f"  - {c}: {int(df[c].sum())}")
        lines.append("")
    (out_dir / "audit_report.md").write_text("\n".join(lines))
    log.info(f"Wrote {out_dir/'audit_report.md'}")


if __name__ == "__main__":
    main()
