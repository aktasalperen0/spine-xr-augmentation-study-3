"""Phase 2: produce per-Case train / internal_val / test CSVs.

Reads:  outputs/01_audit/{train,test}_labels.csv  (built by scripts/01_audit.py)
        configs/base.yaml, configs/cases.yaml
Writes: outputs/02_splits/case_{1..4}/{train,internal_val,test}.csv + manifest.json
        outputs/02_splits/splits_summary.md

See the plan and src/data/splitter.py for invariants and the decision log.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.splitter import (
    assert_split_invariants,
    build_case_split,
    load_cases,
    render_summary_md,
    write_case_split,
)
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()

    seed = int(cases_cfg.get("seed", base["project"]["seed"]))
    set_seed(seed)

    audit_dir = Path(base["paths"]["outputs_root"]) / "01_audit"
    out_root = Path(base["paths"]["outputs_root"]) / "02_splits"
    out_root.mkdir(parents=True, exist_ok=True)

    train_labels = pd.read_csv(audit_dir / "train_labels.csv")
    test_labels = pd.read_csv(audit_dir / "test_labels.csv")

    cap = int(cases_cfg["no_finding_cap_train"])
    ival_frac = float(cases_cfg["internal_val_fraction"])

    manifests = []
    for case in load_cases(cases_cfg):
        log.info(f"Building splits for {case.name} ...")
        splits = build_case_split(
            train_labels=train_labels,
            test_labels=test_labels,
            case=case,
            no_finding_cap_train=cap,
            internal_val_fraction=ival_frac,
            seed=seed,
        )
        assert_split_invariants(case, splits)
        manifest = write_case_split(out_root / case.name, case, splits, seed=seed, no_finding_cap_train=cap)
        manifests.append(manifest)
        log.info(
            f"  {case.name}: train={manifest['counts']['train']['n_rows']}  "
            f"internal_val={manifest['counts']['internal_val']['n_rows']}  "
            f"test={manifest['counts']['test']['n_rows']}"
        )

    (out_root / "splits_summary.md").write_text(render_summary_md(manifests))
    log.info(f"Wrote {out_root/'splits_summary.md'}")


if __name__ == "__main__":
    main()
