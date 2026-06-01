"""Phase 02c: assign study-disjoint 3-fold CV labels over the ROI train pool (advisor F2).

For each case: merge outputs/02b_roi/<case>/{train,internal_val}.csv (the train pool; official
test stays untouched), assign a `fold` column (StratifiedGroupKFold on study_id, stratified by
patch class), and write outputs/02b_roi/<case>/cv_folds.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.cv import assign_folds
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--cases-filter", nargs="*", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()
    seed = int(cases_cfg.get("seed", 42))
    roi_root = Path(base["paths"]["outputs_root"]) / "02b_roi"

    for case_name, case_body in cases_cfg["cases"].items():
        if args.cases_filter and case_name not in args.cases_filter:
            continue
        label_cols = case_label_columns(case_body)
        cdir = roi_root / case_name
        train = pd.read_csv(cdir / "train.csv")
        ival = pd.read_csv(cdir / "internal_val.csv")
        pool = pd.concat([train, ival], ignore_index=True)
        folded = assign_folds(pool, label_cols, n_splits=args.n_splits, seed=seed)
        folded.to_csv(cdir / "cv_folds.csv", index=False)

        counts = []
        for k in range(1, args.n_splits + 1):
            fk = folded[folded["fold"] == k]
            counts.append(f"fold{k}: n={len(fk)} " + " ".join(f"{c}={int(fk[c].sum())}" for c in label_cols))
        log.info(f"[{case_name}] pool={len(pool)} -> cv_folds.csv\n    " + "\n    ".join(counts))


if __name__ == "__main__":
    main()
