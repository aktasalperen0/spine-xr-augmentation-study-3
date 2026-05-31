"""Phase 02b: build ROI lesion-patch manifests per case (the bbox-crop pivot).

For each case and each split (train / internal_val / test):
  - Read the split's image membership from outputs/02_splits/<case>/<split>.csv (this preserves
    the NF cap D1, internal-val carve-out D8, and official test D3 — no new leakage, R4).
  - Positive patches (R1): for each abnormal image in the split, for every bbox whose
    lesion_type is one of the case's positives, crop with margin (R2), stretch-resize to
    store_size, save PNG. Label = that lesion_type (single-label over the case classes).
  - No-Finding patches (R3): for each NF image, emit `nf_patches_per_image` random crops whose
    (w,h) is drawn from THIS split's positive-box size distribution, same margin, same resize.

Outputs per case under outputs/02b_roi/<case>/:
  - patch_pngs/<patch_id>.png
  - train.csv, internal_val.csv, test.csv   (schema compatible with CaseDataset)
  - manifest.json, summary.md

The CSV schema matches the whole-image splits (image_id, study_id, path, source, <case cols>)
plus lesion_type + src_image_id, so the existing trainer/dataset consume it unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from src.data.roi import BBoxSizeSampler, crop_with_margin, sample_nf_patch, square_resize
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"
SPLITS = ["train", "internal_val", "test"]


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def load_annotations(base: dict) -> dict[str, list[tuple]]:
    """image_id -> list of (lesion_type, (xmin,ymin,xmax,ymax)) from BOTH train and test."""
    cols = ["image_id", "lesion_type", "xmin", "ymin", "xmax", "ymax"]
    frames = []
    for key in ("abnormal_train_annotations", "abnormal_test_annotations"):
        frames.append(pd.read_csv(ROOT / base["paths"][key])[cols])
    ann = pd.concat(frames, ignore_index=True)
    out: dict[str, list[tuple]] = {}
    for r in ann.itertuples(index=False):
        out.setdefault(r.image_id, []).append(
            (r.lesion_type, (float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax)))
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()

    roi = cases_cfg["roi"]
    margin = float(roi["margin"])
    nf_per_img = int(roi["nf_patches_per_image"])
    store_size = int(roi["store_size"])
    rng = np.random.default_rng(int(roi["seed"]))

    ann_map = load_annotations(base)
    splits_root = Path(base["paths"]["outputs_root"]) / "02_splits"
    out_root = Path(base["paths"]["outputs_root"]) / "02b_roi"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        positives = list(case_body["positives"])
        label_cols = case_label_columns(case_body)
        case_out = out_root / case_name
        png_dir = case_out / "patch_pngs"
        png_dir.mkdir(parents=True, exist_ok=True)

        manifest_counts: dict = {}
        for split in SPLITS:
            split_csv = splits_root / case_name / f"{split}.csv"
            sdf = pd.read_csv(split_csv)

            # Collect this split's positive boxes (for patches AND the NF size sampler).
            pos_boxes_for_sampler: list[tuple] = []
            patch_rows: list[dict] = []

            abn = sdf[sdf["source"] == "abnormal"]
            for r in abn.itertuples(index=False):
                img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning(f"[{case_name}/{split}] unreadable {r.path}")
                    continue
                for lesion_type, box in ann_map.get(r.image_id, []):
                    if lesion_type not in positives:
                        continue
                    pos_boxes_for_sampler.append(box)
                    crop = crop_with_margin(img, box, margin)
                    if crop.size == 0:
                        continue
                    patch = square_resize(crop, store_size)
                    bi = len(patch_rows)
                    pid = f"{r.image_id}__roi{bi:04d}"
                    ppath = png_dir / f"{pid}.png"
                    cv2.imwrite(str(ppath), patch)
                    row = {
                        "image_id": pid, "study_id": r.study_id, "path": str(ppath),
                        "source": "abnormal_roi", "lesion_type": lesion_type, "src_image_id": r.image_id,
                    }
                    for c in label_cols:
                        row[c] = 1 if c == lesion_type else 0
                    patch_rows.append(row)

            n_pos = len(patch_rows)
            if not pos_boxes_for_sampler:
                raise RuntimeError(f"[{case_name}/{split}] no positive boxes found — cannot size NF patches")
            sampler = BBoxSizeSampler.from_boxes(pos_boxes_for_sampler)

            # NF patches, matched-size (R3).
            nf = sdf[sdf["source"] == "normal"]
            nf_sizes: list[tuple[float, float]] = []
            for r in nf.itertuples(index=False):
                img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning(f"[{case_name}/{split}] unreadable {r.path}")
                    continue
                for k in range(nf_per_img):
                    crop = sample_nf_patch(img, sampler, margin, rng)
                    if crop.size == 0:
                        continue
                    nf_sizes.append((crop.shape[1], crop.shape[0]))
                    patch = square_resize(crop, store_size)
                    pid = f"{r.image_id}__nf{k:02d}"
                    ppath = png_dir / f"{pid}.png"
                    cv2.imwrite(str(ppath), patch)
                    row = {
                        "image_id": pid, "study_id": r.study_id, "path": str(ppath),
                        "source": "nf_roi", "lesion_type": NO_FINDING, "src_image_id": r.image_id,
                    }
                    for c in label_cols:
                        row[c] = 1 if c == NO_FINDING else 0
                    patch_rows.append(row)

            df = pd.DataFrame(patch_rows, columns=["image_id", "study_id", "path", "source"] + label_cols + ["lesion_type", "src_image_id"])
            df.to_csv(case_out / f"{split}.csv", index=False)

            # Shortcut guard: log positive vs NF size medians (should be close by construction).
            pm_w, pm_h = sampler.median()
            nf_arr = np.array(nf_sizes) if nf_sizes else np.zeros((1, 2))
            counts = {
                "n_pos": int(n_pos), "n_nf": int(len(patch_rows) - n_pos), "n_total": int(len(df)),
                "pos_median_wh": [round(pm_w, 1), round(pm_h, 1)],
                "nf_median_wh": [round(float(np.median(nf_arr[:, 0])), 1), round(float(np.median(nf_arr[:, 1])), 1)],
            }
            for c in label_cols:
                counts[c] = int(df[c].sum())
            manifest_counts[split] = counts
            log.info(f"[{case_name}/{split}] pos={counts['n_pos']} nf={counts['n_nf']} "
                     f"pos_med={counts['pos_median_wh']} nf_med={counts['nf_median_wh']}")
            summary_rows.append({"case": case_name, "split": split, **{k: counts[k] for k in ['n_pos','n_nf','n_total']}})

        (case_out / "manifest.json").write_text(json.dumps({
            "case": case_name, "positives": positives, "margin": margin,
            "nf_patches_per_image": nf_per_img, "store_size": store_size, "counts": manifest_counts,
        }, indent=2))

    sm = pd.DataFrame(summary_rows)
    lines = ["# ROI patch build summary", "", sm.to_markdown(index=False)]
    (out_root / "summary.md").write_text("\n".join(lines))
    log.info(f"Wrote {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
