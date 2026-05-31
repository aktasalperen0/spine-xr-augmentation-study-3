"""Phase 04: build the offline traditional-augmentation training set per Case.

For each case:
1. Load outputs/02_splits/<case>/train.csv  (real train images, NF capped, internal_val removed)
2. Identify abnormal rows (any case-positive class == 1). NF rows are NOT augmented (D6).
3. Apply the case's three best transforms (rot270, shear30, second-rotation) to each abnormal row.
4. Save augmented PNGs to outputs/04_traditional/<case>/aug_pngs/<image_id>__<tag>.png.
5. Emit train_traditional.csv = original train rows + augmented rows (each aug row has a new
   path and a `transform` column; labels are inherited from the real source row).
6. Write manifest.json with counts.

internal_val.csv and test.csv are NOT touched — they still live under outputs/02_splits/<case>/
and are loaded from there by the trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd

from src.data.traditional_aug import case_aug_recipe
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def is_abnormal_row(row: pd.Series, positives: list[str]) -> bool:
    return any(int(row[c]) == 1 for c in positives)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None)
    ap.add_argument("--splits-tag", default="02_splits",
                    help="Sub-dir under outputs/ holding per-case train.csv to augment. "
                         "ROI pivot: pass 02b_roi to augment lesion patches.")
    ap.add_argument("--out-tag", default="04_traditional",
                    help="Output sub-dir under outputs/. ROI pivot: pass 04_roi_traditional.")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()

    splits_root = Path(base["paths"]["outputs_root"]) / args.splits_tag
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    overall_counts: dict = {}
    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue

        positives = list(case_body["positives"])
        label_cols = case_label_columns(case_body)
        recipe = case_aug_recipe(int(case_body["second_rotation_deg"]))

        case_out = out_root / case_name
        png_out = case_out / "aug_pngs"
        png_out.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(splits_root / case_name / "train.csv")
        log.info(f"[{case_name}] real train rows = {len(train_df)}; recipe = {[t[0] for t in recipe]}")

        # Mark transform tag on real rows for traceability.
        real_rows = train_df.copy()
        real_rows["transform"] = "real"

        aug_rows: list[dict] = []
        n_abn = 0
        for _, row in train_df.iterrows():
            if not is_abnormal_row(row, positives):
                continue
            n_abn += 1
            img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warning(f"[{case_name}] skipping unreadable {row['path']}")
                continue

            for tag, fn in recipe:
                out_img = fn(img)
                out_path = png_out / f"{row['image_id']}__{tag}.png"
                cv2.imwrite(str(out_path), out_img)
                aug_rows.append({
                    "image_id": f"{row['image_id']}__{tag}",
                    "study_id": row["study_id"],
                    "path": str(out_path),
                    "source": "abnormal_aug",
                    **{c: int(row[c]) for c in label_cols},
                    "transform": tag,
                })

        aug_df = pd.DataFrame(aug_rows, columns=real_rows.columns)
        combined = pd.concat([real_rows, aug_df], ignore_index=True)
        combined.to_csv(case_out / "train_traditional.csv", index=False)

        counts = {
            "real_rows": int(len(real_rows)),
            "abnormal_real_rows": int(n_abn),
            "aug_rows": int(len(aug_df)),
            "total_rows": int(len(combined)),
            "transforms": [t[0] for t in recipe],
            **{f"real__{c}": int(real_rows[c].sum()) for c in label_cols},
            **{f"total__{c}": int(combined[c].sum()) for c in label_cols},
        }
        (case_out / "manifest.json").write_text(json.dumps({
            "case": case_name,
            "positives": positives,
            "second_rotation_deg": int(case_body["second_rotation_deg"]),
            "counts": counts,
        }, indent=2))
        overall_counts[case_name] = counts

        log.info(
            f"[{case_name}] abnormal_real={n_abn} -> aug_rows={len(aug_df)} "
            f"(x{len(recipe)})   total_rows={len(combined)}"
        )

    # Aggregate summary
    if overall_counts:
        lines = ["# Traditional augmentation — build summary", "",
                 "| case | real_rows | abnormal_real | aug_rows | total_rows | transforms |",
                 "|---|---|---|---|---|---|"]
        for case, c in overall_counts.items():
            lines.append(
                f"| {case} | {c['real_rows']} | {c['abnormal_real_rows']} | "
                f"{c['aug_rows']} | {c['total_rows']} | {', '.join(c['transforms'])} |"
            )
        (out_root / "summary.md").write_text("\n".join(lines))
        log.info(f"Wrote {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
