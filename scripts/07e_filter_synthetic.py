"""Phase 07e: confidence-based rejection sampling of synthetic patches (jury filter).

Jury = ensemble of the 3 baseline CV fold models for the case (mean sigmoid probability). A
synthetic patch is ACCEPTED if jury prob[target] >= tau AND argmax == target (Q2). From the
accepted set we RANDOM-sample N (Q2 diversity), where N is dynamically capped (Q3):
    N = clamp(target_total - n_real, 0, n_accepted),  target_total = min(case_max_class, mult * n_real)

Writes a filtered manifest.csv (07 schema → consumable by 03_train_cv --synth-manifest) plus an
acceptance_report.json. Works for any pool (diffusion or progan).

Feedback-loop note: filtering toward jury-confident images risks confirmation bias; we use a
floor (not an ultra-high cutoff) + random sampling, report acceptance rate, and — decisively —
evaluate the final classifier on the untouched real held-out test, which this filtering cannot game.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.case_dataset import CaseDataset
from src.data.transforms import build_transform
from src.models.classifier import build_classifier
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def load_jury(baseline_dir: Path, case: str, backbone: str, label_cols: list[str], device):
    models = []
    for fold_dir in sorted((baseline_dir / case / backbone).glob("fold_*")):
        bp = fold_dir / "best.pth"
        if not bp.exists():
            continue
        ck = torch.load(bp, map_location=device, weights_only=False)
        m = build_classifier(backbone, num_classes=len(label_cols)).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError(f"no jury models under {baseline_dir/case/backbone}/fold_*/best.pth")
    return models


@torch.no_grad()
def jury_probs(models, df, label_cols, image_size, device, batch_size=128):
    tf = build_transform("val", image_size)
    ds = CaseDataset(df, label_cols, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    probs = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.zeros(x.size(0), len(label_cols), device=device)
        for m in models:
            p += torch.sigmoid(m(x))
        probs.append((p / len(models)).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--case", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--pool-manifest", required=True, help="big generated pool manifest.csv")
    ap.add_argument("--baseline-dir", default="outputs/03cv_baseline")
    ap.add_argument("--backbone", default="densenet121")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--target-multiplier", type=float, default=3.0)
    ap.add_argument("--out-tag", required=True, help="e.g. 07f_diffusion_filtered")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()
    rng = np.random.default_rng(args.seed)
    device = _device()

    case_body = cases_cfg["cases"][args.case]
    label_cols = case_label_columns(case_body)
    if args.class_name not in label_cols:
        raise ValueError(f"{args.class_name!r} not in case classes {label_cols}")
    target_idx = label_cols.index(args.class_name)

    roi_root = Path(base["paths"]["outputs_root"]) / "02b_roi"
    train = pd.read_csv(roi_root / args.case / "train.csv")
    n_real = int(train[args.class_name].sum())
    case_max = int(train[label_cols].sum(axis=0).max())
    target_total = min(case_max, int(round(args.target_multiplier * n_real)))

    pool = pd.read_csv(args.pool_manifest)
    log.info(f"[{args.class_name}] pool={len(pool)} n_real={n_real} case_max={case_max} "
             f"target_total={target_total} (mult={args.target_multiplier})")

    jury = load_jury(Path(args.baseline_dir), args.case, args.backbone, label_cols, device)
    probs = jury_probs(jury, pool, label_cols, args.image_size, device)
    pred = probs.argmax(axis=1)
    accept_mask = (probs[:, target_idx] >= args.tau) & (pred == target_idx)
    accepted = pool[accept_mask].reset_index(drop=True)
    n_accepted = len(accepted)
    acc_rate = n_accepted / max(1, len(pool))

    n_keep = int(min(max(target_total - n_real, 0), n_accepted))
    if n_keep < n_accepted:
        idx = rng.choice(n_accepted, size=n_keep, replace=False)
        kept = accepted.iloc[sorted(idx)].reset_index(drop=True)
    else:
        kept = accepted
        if n_keep > n_accepted:
            log.warning(f"[{args.class_name}] wanted {n_keep} but only {n_accepted} accepted")

    slug = args.class_name.lower().replace(" ", "_")
    out_dir = Path(base["paths"]["outputs_root"]) / args.out_tag / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_dir / "manifest.csv", index=False)
    report = {
        "class_name": args.class_name, "case": args.case, "tau": args.tau,
        "pool_size": int(len(pool)), "n_accepted": int(n_accepted),
        "acceptance_rate": round(acc_rate, 4),
        "n_real": n_real, "case_max_class": case_max, "target_multiplier": args.target_multiplier,
        "target_total": target_total, "n_kept": int(len(kept)),
        "mean_accepted_conf": float(probs[accept_mask, target_idx].mean()) if n_accepted else float("nan"),
    }
    (out_dir / "acceptance_report.json").write_text(json.dumps(report, indent=2))
    log.info(f"[{args.class_name}] accepted {n_accepted}/{len(pool)} ({acc_rate:.1%}), "
             f"kept {len(kept)} -> {out_dir/'manifest.csv'}")


if __name__ == "__main__":
    main()
