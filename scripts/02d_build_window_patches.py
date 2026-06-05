"""Phase 02d: build fixed 128x128 native-resolution training windows (patient-level pipeline).

Per case, for the train and internal_val splits (membership from outputs/02_splits/<case>/*.csv):
  - positives     : windows centred (jittered) on each case-lesion box; multi-hot within the case
  - hard negatives : windows from abnormal images overlapping NO box  -> No finding
  - NF negatives    : windows from normal images                       -> No finding

This makes the training distribution match the test sliding-window distribution (stride-96, no
box at test) and — crucially — teaches the model to reject abnormal-image background (the #1 fix
for false-positive flooding at patient-level test).
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

from src.data.windows import hard_negative_windows, positive_windows, window_overlaps_box
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"
SPLITS = ["train", "internal_val"]


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def load_annotations(base: dict) -> dict[str, list[tuple]]:
    cols = ["image_id", "lesion_type", "xmin", "ymin", "xmax", "ymax"]
    frames = [pd.read_csv(ROOT / base["paths"][k])[cols]
              for k in ("abnormal_train_annotations", "abnormal_test_annotations")]
    ann = pd.concat(frames, ignore_index=True)
    out: dict[str, list[tuple]] = {}
    for r in ann.itertuples(index=False):
        out.setdefault(r.image_id, []).append(
            (r.lesion_type, (float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--n-pos", type=int, default=2, help="windows per lesion box")
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--n-hardneg", type=int, default=3, help="background windows per abnormal image")
    ap.add_argument("--n-nf", type=int, default=2, help="windows per normal image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()
    rng = np.random.default_rng(args.seed)
    win = args.win

    ann_map = load_annotations(base)
    splits_root = Path(base["paths"]["outputs_root"]) / "02_splits"
    out_root = Path(base["paths"]["outputs_root"]) / "02d_windows"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        positives = list(case_body["positives"])
        label_cols = case_label_columns(case_body)
        case_out = out_root / case_name
        png_dir = case_out / "win_pngs"
        png_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            sdf = pd.read_csv(splits_root / case_name / f"{split}.csv")
            rows: list[dict] = []

            def emit(patch, x0, y0, src_id, study_id, source, label_vec):
                pid = f"{src_id}__{source}_{x0}_{y0}"
                p = png_dir / f"{pid}.png"
                cv2.imwrite(str(p), patch)
                row = {"image_id": pid, "study_id": study_id, "path": str(p),
                       "source": source, "src_image_id": src_id, "win_x": x0, "win_y": y0}
                for c, v in zip(label_cols, label_vec):
                    row[c] = int(v)
                rows.append(row)

            abn = sdf[sdf["source"] == "abnormal"]
            for r in abn.itertuples(index=False):
                img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                boxes_all = [b for _, b in ann_map.get(r.image_id, [])]
                case_boxes = [(lt, b) for lt, b in ann_map.get(r.image_id, []) if lt in positives]
                # positives
                for lt, box in case_boxes:
                    for x0, y0, patch in positive_windows(img, box, args.n_pos, win, args.jitter, rng):
                        vec = [1 if (c != NO_FINDING and any(
                            window_overlaps_box(x0, y0, win, bb, 0.1)
                            for clt, bb in case_boxes if clt == c)) else 0 for c in label_cols]
                        # NF column stays 0 for positives
                        emit(patch, x0, y0, r.image_id, r.study_id, "pos_win", vec)
                # hard negatives (background of abnormal image)
                for x0, y0, patch in hard_negative_windows(img, boxes_all, args.n_hardneg, win, rng):
                    vec = [1 if c == NO_FINDING else 0 for c in label_cols]
                    emit(patch, x0, y0, r.image_id, r.study_id, "hardneg_win", vec)

            nf = sdf[sdf["source"] == "normal"]
            for r in nf.itertuples(index=False):
                img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                for x0, y0, patch in hard_negative_windows(img, [], args.n_nf, win, rng):
                    vec = [1 if c == NO_FINDING else 0 for c in label_cols]
                    emit(patch, x0, y0, r.image_id, r.study_id, "nf_win", vec)

            df = pd.DataFrame(rows, columns=["image_id", "study_id", "path", "source", "src_image_id", "win_x", "win_y"] + label_cols)
            df.to_csv(case_out / f"{split}.csv", index=False)
            counts = {"n": len(df), **{c: int(df[c].sum()) for c in label_cols},
                      "pos_win": int((df.source == "pos_win").sum()),
                      "hardneg_win": int((df.source == "hardneg_win").sum()),
                      "nf_win": int((df.source == "nf_win").sum())}
            summary.append({"case": case_name, "split": split, **counts})
            log.info(f"[{case_name}/{split}] {counts}")

    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    (out_root / "summary.md").write_text("# 02d window patches\n\n" + pd.DataFrame(summary).to_markdown(index=False))
    log.info(f"Wrote {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
