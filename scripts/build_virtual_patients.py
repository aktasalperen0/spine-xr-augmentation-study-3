"""Build "virtual patients" from filtered synthetic 128 patches (advisor's multi-label request).

Groups a case's accepted synthetic patches (possibly of different case-diseases) under shared
virtual_patient_id's, so the synthetic training data mirrors the real dataset's multi-lesion-per-
image structure. Emits one combined manifest per case (07-style schema + virtual_patient_id),
consumable by scripts/03w_train_window.py --synth-manifest.

Each patch is still an independent 128 training window; the grouping is bookkeeping that makes the
multi-label co-occurrence explicit for the report/narrative.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"
SYNTH_SLUGS = {  # slug -> (class, case)
    "disc_space_narrowing": ("Disc space narrowing", "case_1"),
    "vertebral_collapse": ("Vertebral collapse", "case_1"),
    "foraminal_stenosis": ("Foraminal stenosis", "case_2"),
    "spondylolysthesis": ("Spondylolysthesis", "case_2"),
    "surgical_implant": ("Surgical implant", "case_3"),
    "other_lesions": ("Other lesions", "case_4"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--filtered-tag", required=True, help="e.g. 07f_diffusion_filtered or 07f_progan_filtered")
    ap.add_argument("--out-tag", required=True, help="e.g. virtual_patients_diffusion")
    ap.add_argument("--group-size", type=int, default=2, help="avg patches per virtual patient")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = load_config(args.config)
    log = get_logger()
    rng = np.random.default_rng(args.seed)
    froot = Path(base["paths"]["outputs_root"]) / args.filtered_tag
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag

    by_case: dict[str, list[pd.DataFrame]] = {}
    for slug, (cls, case) in SYNTH_SLUGS.items():
        mp = froot / slug / "manifest.csv"
        if not mp.exists():
            continue
        by_case.setdefault(case, []).append(pd.read_csv(mp))

    for case, frames in by_case.items():
        df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        # assign virtual patient ids: chunks of ~group_size
        vids = []
        i = 0; vp = 0
        while i < len(df):
            g = max(1, int(rng.poisson(args.group_size)) or 1)
            for _ in range(min(g, len(df) - i)):
                vids.append(f"vp_{case}_{vp:05d}")
            vp += 1; i += g
        df["virtual_patient_id"] = vids[:len(df)]
        df["study_id"] = df["virtual_patient_id"]
        out_dir = out_root / case
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "manifest.csv", index=False)
        log.info(f"[{case}] {len(df)} synthetic patches -> {vp} virtual patients ({out_dir/'manifest.csv'})")


if __name__ == "__main__":
    main()
