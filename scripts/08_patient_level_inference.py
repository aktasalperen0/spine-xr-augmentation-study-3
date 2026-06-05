"""Phase 08: patient-level inference on FULL images (no GT box at test) — the honest test.

For a condition, load the 4 case window-classifiers (outputs/<cond>_win/<case>/densenet121/best.pth).
For each full image, tile 128x128 windows at stride 96, score every window with each case model,
and set image_score[disease] = max window probability (from the case that owns that disease).

  --mode calibrate : on train-side internal_val FULL images (GT from train_labels.csv) -> per-disease
                     threshold τ_d maximising F1.  Writes thresholds.json.
  --mode test       : on the official test FULL images (GT from test_labels.csv) -> apply τ_d ->
                     patient-level per-disease P/R/F1 + macro.  Writes metrics.json/per_disease.md.

The test path NEVER opens an annotations (box) file — verified by construction.
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


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def case_label_columns(b): return list(b["positives"]) + [NO_FINDING]


def load_case_models(cond_root: Path, cases_cfg, backbone, device, log):
    """Return {case: (model, label_cols)} and disease->case owner map."""
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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {d: 0.0 for d in DISEASES}
    # gather windows
    tiles = [cv2.cvtColor(p, cv2.COLOR_GRAY2RGB) for _, _, p in sliding_windows(img, win, stride)]
    if not tiles:
        return {d: 0.0 for d in DISEASES}
    # per-case max prob over all windows
    case_max = {case: np.zeros(len(lc)) for case, (_, lc) in models.items()}
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = torch.stack([tf(image=t)["image"] for t in chunk]).to(device)
        for case, (m, lc) in models.items():
            p = torch.sigmoid(m(x)).cpu().numpy()           # (B, n_classes)
            case_max[case] = np.maximum(case_max[case], p.max(axis=0))
    scores = {}
    for d in DISEASES:
        case = owner.get(d)
        if case is None or case not in models:
            scores[d] = 0.0; continue
        _, lc = models[case]
        scores[d] = float(case_max[case][lc.index(d)])
    return scores


def build_image_set(labels_csv):
    df = pd.read_csv(labels_csv)
    rows = []
    for _, r in df.iterrows():
        gt = {d: (int(r[d]) if d in df.columns else 0) for d in DISEASES}
        rows.append({"image_id": r["image_id"], "path": r["path"], "gt": gt})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--classifier", default="configs/classifier.yaml")
    ap.add_argument("--cond-tag", required=True, help="e.g. baseline_win / traditional_win")
    ap.add_argument("--mode", choices=["calibrate", "test"], required=True)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="cap #images (smoke)")
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
    log.info(f"[{args.cond_tag}] loaded cases={list(models)} owner={owner}")

    # GT label tables already have per-image one-hot + path
    audit = outputs_root / "01_audit"
    labels_csv = audit / ("train_labels.csv" if args.mode == "calibrate" else "test_labels.csv")
    images = build_image_set(labels_csv)

    if args.mode == "calibrate":
        # restrict to internal_val image_ids (union across cases) so we don't tune on trained windows
        ival_ids = set()
        for case in cases_cfg["cases"]:
            f = outputs_root / "02_splits" / case / "internal_val.csv"
            if f.exists():
                ival_ids |= set(pd.read_csv(f)["image_id"].tolist())
        images = [im for im in images if im["image_id"] in ival_ids]
    if args.limit:
        images = images[:args.limit]
    log.info(f"[{args.mode}] scoring {len(images)} full images (stride={args.stride})")

    score_rows = []
    for k, im in enumerate(images):
        s = score_image(im["path"], models, owner, tf, device, args.win, args.stride)
        score_rows.append({"image_id": im["image_id"], **{f"score__{d}": s[d] for d in DISEASES},
                           **{f"gt__{d}": im["gt"][d] for d in DISEASES}})
        if (k + 1) % 50 == 0:
            log.info(f"  {k+1}/{len(images)}")
    sdf = pd.DataFrame(score_rows)
    sdf.to_csv(out_dir / f"scores_{args.mode}.csv", index=False)

    if args.mode == "calibrate":
        thresholds = {}
        for d in DISEASES:
            sc = sdf[f"score__{d}"].to_numpy(); gt = sdf[f"gt__{d}"].to_numpy()
            best_t, best_f1 = 0.5, -1.0
            for t in np.linspace(0.05, 0.95, 19):
                pred = (sc >= t).astype(int)
                tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
                fn = int(((pred == 0) & (gt == 1)).sum())
                f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thresholds[d] = best_t
            log.info(f"  τ[{d}]={best_t:.2f} (val F1={best_f1:.3f})")
        (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        log.info(f"wrote {out_dir/'thresholds.json'}")
        return

    # test mode: apply thresholds, patient-level metrics
    th_path = out_dir / "thresholds.json"
    if not th_path.exists():
        log.warning("no thresholds.json found in test out-dir; run --mode calibrate first. Using 0.5.")
        thresholds = {d: 0.5 for d in DISEASES}
    else:
        thresholds = json.loads(th_path.read_text())
    per = {}
    f1s = []
    for d in DISEASES:
        sc = sdf[f"score__{d}"].to_numpy(); gt = sdf[f"gt__{d}"].to_numpy()
        pred = (sc >= thresholds.get(d, 0.5)).astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[d] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                  "tp": tp, "fp": fp, "fn": fn, "n_pos": int(gt.sum()), "tau": thresholds.get(d, 0.5)}
        f1s.append(f1)
    macro = float(np.mean(f1s))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"condition": args.cond_tag, "stride": args.stride, "patient_macro_f1": macro, "per_disease": per}, indent=2))
    md = ["| disease | F1 | precision | recall | n_pos | τ |", "|---|---|---|---|---|---|"]
    for d in DISEASES:
        p = per[d]; md.append(f"| {d} | {p['f1']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {p['n_pos']} | {p['tau']:.2f} |")
    md.append(f"| **macro** | **{macro:.4f}** |  |  |  |  |")
    (out_dir / "per_disease.md").write_text(f"# Patient-level — {args.cond_tag}\n\n" + "\n".join(md))
    log.info(f"[{args.cond_tag}] PATIENT-LEVEL macro-F1 = {macro:.4f}  -> {out_dir/'per_disease.md'}")


if __name__ == "__main__":
    main()
