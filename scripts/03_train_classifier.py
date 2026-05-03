"""Phase 03: train baseline (no-aug) classifier per Case x Backbone.

For each Case in configs/cases.yaml and each backbone in configs/classifier.yaml:
- load the case's train / internal_val / test CSVs from outputs/02_splits/
- train a multi-label sigmoid classifier (BCE) on train
- pick best.pth by Macro F1 on the official test set (D8)
- write outputs/03_baseline/<case>/<backbone>/{best.pth, metrics.json, log.csv, test_metrics.md}
- write outputs/03_baseline/summary.md aggregating all cells
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


def case_label_columns(case_cfg: dict) -> list[str]:
    return list(case_cfg["positives"]) + [NO_FINDING]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--classifier", default="configs/classifier.yaml")
    ap.add_argument("--cases-filter", nargs="*", default=None,
                    help="Limit to specific cases, e.g. --cases-filter case_1 case_4")
    ap.add_argument("--backbones-filter", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke tests)")
    ap.add_argument("--out-tag", default="03_baseline",
                    help="Sub-dir under outputs/ to write into")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    cls_cfg = load_config(args.classifier)
    log = get_logger()

    seed = int(cases_cfg.get("seed", base["project"]["seed"]))
    set_seed(seed)

    splits_root = Path(base["paths"]["outputs_root"]) / "02_splits"
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else int(cls_cfg["train"]["epochs"])

    summaries: list[dict] = []
    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        case_dir = splits_root / case_name
        train_df = pd.read_csv(case_dir / "train.csv")
        ival_df = pd.read_csv(case_dir / "internal_val.csv")
        test_df = pd.read_csv(case_dir / "test.csv")
        label_columns = case_label_columns(case_body)
        log.info(
            f"[{case_name}] train={len(train_df)} internal_val={len(ival_df)} "
            f"test={len(test_df)} labels={label_columns}"
        )

        for backbone, bcfg in cls_cfg["backbones"].items():
            if args.backbones_filter and backbone not in args.backbones_filter:
                continue
            cell_out = out_root / case_name / backbone
            cfg = TrainerConfig(
                case_name=case_name,
                backbone=backbone,
                label_columns=label_columns,
                image_size=int(bcfg["image_size"]),
                batch_size=int(bcfg["batch_size"]),
                epochs=epochs,
                lr=float(cls_cfg["train"]["lr"]),
                weight_decay=float(cls_cfg["train"]["weight_decay"]),
                warmup_epochs=int(cls_cfg["train"]["warmup_epochs"]),
                grad_clip_norm=float(cls_cfg["train"]["grad_clip_norm"]),
                num_workers=int(cls_cfg["train"]["num_workers"]),
                train_transform=cls_cfg["train_transform"],
                eval_transform=cls_cfg["eval_transform"],
                pos_weight_clip=float(cls_cfg["pos_weight_clip"]),
                out_dir=cell_out,
                seed=seed,
            )
            log.info(f"=== Training {case_name}/{backbone} (epochs={epochs}) ===")
            summary = train_one_cell(train_df, ival_df, test_df, cfg, log)
            summaries.append(summary)

    # Aggregate
    if summaries:
        rows = []
        for s in summaries:
            row = {"case": s["case"], "backbone": s["backbone"],
                   "best_epoch": s["best_epoch"],
                   "best_test_macro_f1": s["best_test_macro_f1"],
                   "best_val_macro_f1": s["best_val_macro_f1"]}
            for c in s["label_columns"]:
                row[f"test_f1__{c}"] = s["best_test_metrics"]["per_class"][c]["f1"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "summary.csv", index=False)
        lines = ["# Summary — " + args.out_tag, "", df.to_markdown(index=False, floatfmt=".4f")]
        (out_root / "summary.md").write_text("\n".join(lines))
        (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
        log.info(f"Wrote {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
