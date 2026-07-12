"""Phase 08: patient-level inference on FULL images (no GT box at test) — honest test.

COUNT-BASED aggregation (fix for the max-aggregation false-positive flood):
  An image is predicted positive for disease d iff the NUMBER of windows scoring >= tau_win
  is >= K_d. A real lesion fires a CLUSTER of overlapping windows; isolated false positives do
  not reach K. (tau_win, K_d) are calibrated per disease on a train-side validation set (full
  images), never on test.

  --mode calibrate : internal_val full images (GT from train_labels.csv) -> per-disease (tau,K)
  --mode test       : official test full images (GT from test_labels.csv) -> patient-level F1

The test path never opens an annotations (box) file.
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
import torch

from src.data.transforms import build_transform
from src.data.windows import sliding_windows
from src.models.classifier import build_classifier
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"
DISEASES = ["Osteophytes", "Disc space narrowing", "Other lesions", "Foraminal stenosis",
            "Surgical implant", "Spondylolysthesis", "Vertebral collapse"]
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # window-level probability thresholds
K_GRID = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]     # min #firing-windows for an image-level positive


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def case_label_columns(b): return list(b["positives"]) + [NO_FINDING]


def load_case_models(cond_root: Path, cases_cfg, backbone, device, log):
    models, owner = {}, {}
    for case, body in cases_cfg["cases"].items():
        label_cols = case_label_columns(body)
        bp = cond_root / case / backbone / "best.pth"
        if not bp.exists():
            log.warning(f"missing {bp}"); continue
        ck = torch.load(bp, map_location=device, weights_only=False)
        m = build_classifier(backbone, num_classes=len(label_cols)).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
        models[case] = (m, label_cols)
        for d in body["positives"]:
            owner[d] = case
    return models, owner


@torch.no_grad()
def score_image(path, models, owner, tf, device, win, stride, batch=256):
    """Return {disease: [count at each TAU_GRID level]} over all windows of the image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    zero = {d: [0] * len(TAU_GRID) for d in DISEASES}
    if img is None:
        return zero
    tiles = [cv2.cvtColor(p, cv2.COLOR_GRAY2RGB) for _, _, p in sliding_windows(img, win, stride)]
    if not tiles:
        return zero
    # per-case, per-class counts at each tau level
    case_counts = {case: np.zeros((len(lc), len(TAU_GRID)), dtype=np.int64) for case, (_, lc) in models.items()}
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = torch.stack([tf(image=t)["image"] for t in chunk]).to(device)
        for case, (m, lc) in models.items():
            p = torch.sigmoid(m(x)).cpu().numpy()           # (B, n_classes)
            for ti, tau in enumerate(TAU_GRID):
                case_counts[case][:, ti] += (p >= tau).sum(axis=0)
    out = {}
    for d in DISEASES:
        case = owner.get(d)
        if case is None or case not in models:
            out[d] = [0] * len(TAU_GRID); continue
        _, lc = models[case]
        out[d] = case_counts[case][lc.index(d)].tolist()
    return out


def build_image_set(labels_csv):
    df = pd.read_csv(labels_csv)
    rows = []
    for _, r in df.iterrows():
        gt = {d: (int(r[d]) if d in df.columns else 0) for d in DISEASES}
        rows.append({"image_id": r["image_id"], "path": r["path"], "gt": gt})
    return rows


def _f1(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--classifier", default="configs/classifier.yaml")
    ap.add_argument("--cond-tag", required=True)
    ap.add_argument("--mode", choices=["calibrate", "test"], required=True)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    cls_cfg = load_config(args.classifier)
    log = get_logger()
    device = _device()
    backbone = next(iter(cls_cfg["backbones"].keys()))
    image_size = int(cls_cfg["backbones"][backbone]["image_size"])
    tf = build_transform("val", image_size)

    outputs_root = Path(base["paths"]["outputs_root"])
    cond_root = outputs_root / args.cond_tag
    out_tag = args.out_tag or args.cond_tag.replace("_win", "_patient")
    out_dir = outputs_root / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    models, owner = load_case_models(cond_root, cases_cfg, backbone, device, log)
    if not models:
        log.error("no case models loaded"); return

    audit = outputs_root / "01_audit"
    labels_csv = audit / ("train_labels.csv" if args.mode == "calibrate" else "test_labels.csv")
    images = build_image_set(labels_csv)
    if args.mode == "calibrate":
        ival_ids = set()
        for case in cases_cfg["cases"]:
            f = outputs_root / "02_splits" / case / "internal_val.csv"
            if f.exists():
                ival_ids |= set(pd.read_csv(f)["image_id"].tolist())
        images = [im for im in images if im["image_id"] in ival_ids]
    if args.limit:
        images = images[:args.limit]
    log.info(f"[{args.mode}] {args.cond_tag}: scoring {len(images)} full images (stride={args.stride}, count-based)")

    # score all images -> counts
    rows = []
    for k, im in enumerate(images):
        c = score_image(im["path"], models, owner, tf, device, args.win, args.stride)
        row = {"image_id": im["image_id"]}
        for d in DISEASES:
            for ti in range(len(TAU_GRID)):
                row[f"cnt__{d}__{ti}"] = c[d][ti]
            row[f"gt__{d}"] = im["gt"][d]
        rows.append(row)
        if (k + 1) % 50 == 0:
            log.info(f"  {k+1}/{len(images)}")
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out_dir / f"scores_{args.mode}.csv", index=False)

    if args.mode == "calibrate":
        thresholds = {}
        for d in DISEASES:
            gt = sdf[f"gt__{d}"].to_numpy()
            best = {"f1": -1.0, "ti": 0, "tau": TAU_GRID[0], "K": 1}
            for ti, tau in enumerate(TAU_GRID):
                cnt = sdf[f"cnt__{d}__{ti}"].to_numpy()
                for K in K_GRID:
                    f1, *_ = _f1((cnt >= K).astype(int), gt)
                    if f1 > best["f1"]:
                        best = {"f1": f1, "ti": ti, "tau": tau, "K": K}
            thresholds[d] = {"tau_win": best["tau"], "ti": best["ti"], "K": best["K"]}
            log.info(f"  [{d}] tau_win={best['tau']} K={best['K']}  (val F1={best['f1']:.3f})")
        (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        log.info(f"wrote {out_dir/'thresholds.json'}")
        return

    # test
    th_path = out_dir / "thresholds.json"
    if not th_path.exists():
        log.warning("no thresholds.json (run --mode calibrate first); defaulting tau=0.9,K=5")
        thresholds = {d: {"tau_win": 0.9, "ti": TAU_GRID.index(0.9), "K": 5} for d in DISEASES}
    else:
        thresholds = json.loads(th_path.read_text())
    per = {}; f1s = []
    for d in DISEASES:
        th = thresholds[d]; ti = int(th["ti"]); K = int(th["K"])
        cnt = sdf[f"cnt__{d}__{ti}"].to_numpy(); gt = sdf[f"gt__{d}"].to_numpy()
        f1, prec, rec, tp, fp, fn = _f1((cnt >= K).astype(int), gt)
        per[d] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                  "tp": tp, "fp": fp, "fn": fn, "n_pos": int(gt.sum()),
                  "tau_win": th["tau_win"], "K": K}
        f1s.append(f1)
    macro = float(np.mean(f1s))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"condition": args.cond_tag, "stride": args.stride, "aggregation": "count",
         "patient_macro_f1": macro, "per_disease": per}, indent=2))
    md = ["| disease | F1 | precision | recall | n_pos | tau_win | K |", "|---|---|---|---|---|---|---|"]
    for d in DISEASES:
        p = per[d]; md.append(f"| {d} | {p['f1']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {p['n_pos']} | {p['tau_win']} | {p['K']} |")
    md.append(f"| **macro** | **{macro:.4f}** |  |  |  |  |  |")
    (out_dir / "per_disease.md").write_text(f"# Patient-level (count-agg) — {args.cond_tag}\n\n" + "\n".join(md))
    log.info(f"[{args.cond_tag}] PATIENT-LEVEL macro-F1 = {macro:.4f}")


if __name__ == "__main__":
    main()
