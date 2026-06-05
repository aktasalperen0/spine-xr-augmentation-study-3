"""Phase 03w: train one window-classifier per (condition x case) for the patient-level pipeline.

Train on outputs/02d_windows/<case>/train.csv (128 native windows: pos + hard-neg + nf), select on
internal_val. Optionally append filtered synthetic 128 patches (--synth-manifest) to TRAIN.

Conditions (out-tag + flags):
  baseline    : --train-transform baseline_no_aug
  traditional : --train-transform roi_train_online --balanced-sampler
  diffusion   : --train-transform roi_train_online --balanced-sampler --synth-manifest <ddpm filtered...>
  progan      : --train-transform roi_train_online --balanced-sampler --synth-manifest <progan filtered...>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.train.classifier_trainer import TrainerConfig, train_one_cell
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

NO_FINDING = "No finding"


def case_label_columns(b): return list(b["positives"]) + [NO_FINDING]


def load_synth(manifests, label_cols, schema_cols):
    if not manifests:
        return pd.DataFrame(columns=schema_cols)
    case_pos = [c for c in label_cols if c != NO_FINDING]
    parts = []
    for mp in manifests:
        m = pd.read_csv(mp)
        if all(c in m.columns for c in case_pos):
            parts.append(m[m[case_pos].sum(axis=1) > 0])
    if not parts:
        return pd.DataFrame(columns=schema_cols)
    syn = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"image_id": syn["image_id"], "study_id": syn["study_id"],
                        "path": syn["path"], "source": "synth_win"})
    for c in label_cols:
        out[c] = syn[c].astype(int) if c in syn.columns else 0
    return out.reindex(columns=schema_cols, fill_value=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--classifier", default="configs/classifier.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-transform", default=None)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--synth-manifest", nargs="*", default=None)
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    cls_cfg = load_config(args.classifier)
    log = get_logger()
    seed = int(cases_cfg.get("seed", 42)); set_seed(seed)

    win_root = Path(base["paths"]["outputs_root"]) / "02d_windows"
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs if args.epochs is not None else int(cls_cfg["train"]["epochs"])
    train_tf = args.train_transform or cls_cfg.get("train_transform", "baseline_no_aug")
    backbone, bcfg = next(iter(cls_cfg["backbones"].items()))

    for case_name, body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        label_cols = case_label_columns(body)
        tr = pd.read_csv(win_root / case_name / "train.csv")
        va = pd.read_csv(win_root / case_name / "internal_val.csv")
        synth = load_synth(args.synth_manifest, label_cols, list(tr.columns))
        if len(synth):
            tr = pd.concat([tr, synth], ignore_index=True)
            log.info(f"[{case_name}] +{len(synth)} synthetic windows")
        cell_out = out_root / case_name / backbone
        cfg = TrainerConfig(
            case_name=case_name, backbone=backbone, label_columns=label_cols,
            image_size=int(bcfg["image_size"]), batch_size=int(bcfg["batch_size"]),
            epochs=epochs, lr=float(cls_cfg["train"]["lr"]),
            weight_decay=float(cls_cfg["train"]["weight_decay"]),
            warmup_epochs=int(cls_cfg["train"]["warmup_epochs"]),
            grad_clip_norm=float(cls_cfg["train"]["grad_clip_norm"]),
            num_workers=int(cls_cfg["train"]["num_workers"]),
            train_transform=train_tf, eval_transform=cls_cfg["eval_transform"],
            pos_weight_clip=float(cls_cfg["pos_weight_clip"]), out_dir=cell_out, seed=seed,
            balanced_sampler=bool(args.balanced_sampler), select_metric_split="val",
        )
        log.info(f"=== {args.out_tag} / {case_name} (train={len(tr)} val={len(va)} tf={train_tf}) ===")
        # test_df = internal_val (placeholder; patient-level eval is separate)
        train_one_cell(tr, va, va, cfg, log)


if __name__ == "__main__":
    main()
