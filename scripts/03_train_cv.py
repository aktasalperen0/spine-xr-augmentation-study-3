"""Phase 03 (CV): 3-fold cross-validation classifier driver (advisor F2/F3/F5).

For each case (and the single backbone in configs/classifier.yaml):
  - load outputs/02b_roi/<case>/cv_folds.csv (real ROI patches with a `fold` column)
  - optionally append synthetic patches (--synth-manifest) to the TRAIN side only (fold=0)
  - for each fold k: train on fold!=k, validate (model-selection) on fold==k, and report on the
    untouched official test (outputs/02b_roi/<case>/test.csv)
  - aggregate macro-F1 across folds (mean±std) on the held-out test

Conditions are distinguished by --out-tag + flags:
  baseline    : --train-transform baseline_no_aug                              (no sampler)
  traditional : --train-transform roi_train_online --balanced-sampler
  diffusion   : --train-transform roi_train_online --balanced-sampler --synth-manifest <ddpm manifests...>
  progan      : --train-transform roi_train_online --balanced-sampler --synth-manifest <progan manifests...>
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

from src.train.classifier_trainer import TrainerConfig, train_one_cell
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

NO_FINDING = "No finding"


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def load_synth(manifests: list[str], label_cols: list[str], pool_cols: list[str]) -> pd.DataFrame:
    """Load synthetic-patch manifests, keep only rows whose positive class is in this case,
    project onto the case's pool schema, and tag fold=0 (always train, never val)."""
    rows = []
    for mp in manifests:
        m = pd.read_csv(mp)
        # Keep rows that are positive for one of this case's lesion classes (not No finding).
        case_pos = [c for c in label_cols if c != NO_FINDING]
        keep = m[m[case_pos].sum(axis=1) > 0] if all(c in m.columns for c in case_pos) else m.iloc[0:0]
        rows.append(keep)
    if not rows:
        return pd.DataFrame(columns=pool_cols)
    syn = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame()
    out["image_id"] = syn["image_id"]
    out["study_id"] = syn["study_id"]
    out["path"] = syn["path"]
    out["source"] = syn["source"]
    for c in label_cols:
        out[c] = syn[c].astype(int) if c in syn.columns else 0
    out["fold"] = 0  # synthetic rows live only in train
    return out.reindex(columns=pool_cols, fill_value=pd.NA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--classifier", default="configs/classifier.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-transform", default=None, help="override classifier.yaml train_transform")
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--synth-manifest", nargs="*", default=None,
                    help="synthetic patch manifest CSV(s) appended to TRAIN folds only")
    ap.add_argument("--out-tag", required=True, help="e.g. 03cv_baseline / 04cv_traditional / 05cv_diffusion")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    cls_cfg = load_config(args.classifier)
    log = get_logger()
    seed = int(cases_cfg.get("seed", base["project"]["seed"]))
    set_seed(seed)

    roi_root = Path(base["paths"]["outputs_root"]) / "02b_roi"
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs if args.epochs is not None else int(cls_cfg["train"]["epochs"])
    train_tf = args.train_transform or cls_cfg.get("train_transform", "baseline_no_aug")

    all_summaries = []
    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        label_cols = case_label_columns(case_body)
        cdir = roi_root / case_name
        folded = pd.read_csv(cdir / "cv_folds.csv")
        test_df = pd.read_csv(cdir / "test.csv")
        pool_cols = list(folded.columns)

        synth = load_synth(args.synth_manifest, label_cols, pool_cols) if args.synth_manifest else None
        if synth is not None and len(synth):
            folded = pd.concat([folded, synth], ignore_index=True)
            log.info(f"[{case_name}] appended {len(synth)} synthetic patches to train pool")

        for backbone, bcfg in cls_cfg["backbones"].items():
            fold_results = []
            for k in range(1, args.n_splits + 1):
                tr = folded[folded["fold"] != k].reset_index(drop=True)
                va = folded[folded["fold"] == k].reset_index(drop=True)
                cell_out = out_root / case_name / backbone / f"fold_{k}"
                cfg = TrainerConfig(
                    case_name=f"{case_name}|f{k}", backbone=backbone, label_columns=label_cols,
                    image_size=int(bcfg["image_size"]), batch_size=int(bcfg["batch_size"]),
                    epochs=epochs, lr=float(cls_cfg["train"]["lr"]),
                    weight_decay=float(cls_cfg["train"]["weight_decay"]),
                    warmup_epochs=int(cls_cfg["train"]["warmup_epochs"]),
                    grad_clip_norm=float(cls_cfg["train"]["grad_clip_norm"]),
                    num_workers=int(cls_cfg["train"]["num_workers"]),
                    train_transform=train_tf, eval_transform=cls_cfg["eval_transform"],
                    pos_weight_clip=float(cls_cfg["pos_weight_clip"]),
                    out_dir=cell_out, seed=seed,
                    balanced_sampler=bool(args.balanced_sampler),
                    select_metric_split="val",
                )
                log.info(f"=== {case_name} / {backbone} / fold {k}/{args.n_splits} "
                         f"(train={len(tr)} val={len(va)} test={len(test_df)} tf={train_tf}) ===")
                s = train_one_cell(tr, va, test_df, cfg, log)
                fold_results.append(s)

            macro = [s["best_test_macro_f1"] for s in fold_results]
            per_class = {c: [s["best_test_metrics"]["per_class"][c]["f1"] for s in fold_results]
                         for c in label_cols}
            agg = {
                "case": case_name, "backbone": backbone, "condition": args.out_tag,
                "n_splits": args.n_splits,
                "test_macro_f1_mean": float(np.mean(macro)), "test_macro_f1_std": float(np.std(macro)),
                "per_fold_test_macro_f1": macro,
                "per_class_test_f1_mean": {c: float(np.mean(v)) for c, v in per_class.items()},
            }
            (out_root / case_name / backbone / "cv_summary.json").write_text(json.dumps(agg, indent=2))
            all_summaries.append(agg)
            log.info(f"[{case_name}/{backbone}] test macro-F1 = "
                     f"{agg['test_macro_f1_mean']:.4f} ± {agg['test_macro_f1_std']:.4f}")

    if all_summaries:
        df = pd.DataFrame([{
            "case": s["case"], "backbone": s["backbone"],
            "test_macro_f1": f"{s['test_macro_f1_mean']:.4f} ± {s['test_macro_f1_std']:.4f}",
        } for s in all_summaries])
        (out_root / "cv_summary.md").write_text(
            f"# CV summary — {args.out_tag} (train_transform={train_tf}, "
            f"balanced_sampler={args.balanced_sampler}, synth={bool(args.synth_manifest)})\n\n"
            + df.to_markdown(index=False))
        (out_root / "cv_summary.json").write_text(json.dumps(all_summaries, indent=2))
        log.info(f"Wrote {out_root/'cv_summary.md'}")


if __name__ == "__main__":
    main()
