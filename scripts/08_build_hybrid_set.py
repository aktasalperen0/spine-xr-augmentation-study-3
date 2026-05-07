"""Phase 08: build the hybrid training set for one case (paper §5.5).

Hybrid recipe per case (paper-faithful):
    train_hybrid = real train rows
                 + WGAN-generated rows for one or more minority classes in this case
                 + 3 geometric transforms applied to ALL abnormal images (real abn + WGAN gens)

NF rows are NEVER augmented (D6).

Real-image aug PNGs are reused from `outputs/04_traditional/<case>/aug_pngs/` — we read the
already-built `train_traditional.csv` directly (which includes both real and real-aug rows).
WGAN-aug PNGs are written fresh under `outputs/08_hybrid_<variant>/<case>/aug_pngs/`.

Variant tag lets us run partial hybrids (e.g. `dsn_only` while we test the WGAN→F1 hypothesis
before committing to the remaining 5 classes). Output structure:

    outputs/08_hybrid_<variant_tag>/<case>/
        aug_pngs/<image_id>__<tag>.png        # transforms of WGAN gens only
        train_hybrid.csv                       # real + real-aug + wgan + wgan-aug
        manifest.json

Trainer call (reuses scripts/03_train_classifier.py with the existing CLI overrides):

    python scripts/03_train_classifier.py \\
        --config configs/base.yaml --cases configs/cases.yaml --classifier configs/classifier.yaml \\
        --train-csv-root outputs/08_hybrid_dsn_only \\
        --train-csv-name train_hybrid.csv \\
        --cases-filter case_1 \\
        --out-tag 08_hybrid_dsn_only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd

from src.data.traditional_aug import case_aug_recipe
from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"


def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--case", required=True, help="Single case to build for, e.g. case_1")
    ap.add_argument("--wgan-classes", nargs="+", required=True,
                    help="WGAN class slugs to include, e.g. disc_space_narrowing. "
                         "Resolves to outputs/07_wgan_generated/<slug>/manifest.csv")
    ap.add_argument("--wgan-root", default="outputs/07_wgan_generated",
                    help="Root holding per-class manifest.csv (default outputs/07_wgan_generated)")
    ap.add_argument("--traditional-root", default="outputs/04_traditional",
                    help="Root holding train_traditional.csv per case (default outputs/04_traditional)")
    ap.add_argument("--variant-tag", required=True,
                    help="Short tag distinguishing hybrid variants, e.g. dsn_only or full")
    ap.add_argument("--max-wgan-per-class", type=int, default=None,
                    help="Cap how many WGAN PNGs to draw from each manifest. "
                         "Default: take all rows in the manifest. "
                         "Use this for dose-reduction experiments (e.g. 400 instead of 800).")
    ap.add_argument("--no-transforms-on-wgan", action="store_true",
                    help="Skip writing rot/shear copies of WGAN PNGs. "
                         "WGAN-real rows still go in; WGAN-aug rows do not.")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()

    if args.case not in cases_cfg["cases"]:
        raise ValueError(f"unknown case {args.case!r}; choices: {list(cases_cfg['cases'])}")
    case_body = cases_cfg["cases"][args.case]
    positives = list(case_body["positives"])
    label_cols = case_label_columns(case_body)
    recipe = case_aug_recipe(int(case_body["second_rotation_deg"]))

    out_root = Path(base["paths"]["outputs_root"]) / f"08_hybrid_{args.variant_tag}"
    case_out = out_root / args.case
    png_out = case_out / "aug_pngs"
    png_out.mkdir(parents=True, exist_ok=True)

    # ---- 1) Real + real-aug rows (already built in Phase 04) ----
    trad_csv = Path(args.traditional_root) / args.case / "train_traditional.csv"
    if not trad_csv.exists():
        raise FileNotFoundError(f"missing {trad_csv}; run scripts/04_build_traditional_set.py first")
    trad_df = pd.read_csv(trad_csv)
    log.info(f"[{args.case}] real + real-aug rows from {trad_csv}: {len(trad_df)}")

    # ---- 2) WGAN gen rows (project onto case columns) ----
    wgan_real_rows: list[pd.DataFrame] = []
    wgan_real_paths: list[tuple[str, str, str, str]] = []  # (image_id, study_id, path, target_pos_class)
    for slug in args.wgan_classes:
        man = Path(args.wgan_root) / slug / "manifest.csv"
        if not man.exists():
            raise FileNotFoundError(f"missing WGAN manifest {man}")
        m = pd.read_csv(man)
        # Determine which case-positive class this WGAN slug supplies.
        # The manifest has one-hot columns over ALL 8 classes; we pick the column that is in
        # this case's positives and is set to 1.
        active_pos = None
        for pos in positives:
            if pos in m.columns and (m[pos] == 1).all():
                active_pos = pos
                break
        if active_pos is None:
            raise ValueError(
                f"WGAN slug {slug!r} positives don't match any of case {args.case}'s positives "
                f"({positives}); check class wiring."
            )
        # Optional dose cap (deterministic head-of-file slice).
        if args.max_wgan_per_class is not None and len(m) > int(args.max_wgan_per_class):
            m = m.head(int(args.max_wgan_per_class)).copy()
        log.info(f"[{args.case}] WGAN class {slug!r} -> case-positive {active_pos!r} "
                 f"({len(m)} synthetic rows)")
        # Project to case columns.
        proj = pd.DataFrame({
            "image_id": m["image_id"],
            "study_id": m["study_id"],
            "path": m["path"],
            "source": "wgan_synth",
            **{c: (1 if c == active_pos else 0) for c in label_cols},
            "transform": "wgan_real",
        })
        wgan_real_rows.append(proj)
        for _, row in m.iterrows():
            wgan_real_paths.append((row["image_id"], row["study_id"], row["path"], active_pos))

    wgan_real_df = pd.concat(wgan_real_rows, ignore_index=True) if wgan_real_rows else pd.DataFrame()

    # ---- 3) WGAN-aug rows: apply 3 transforms to each WGAN PNG (skipped if --no-transforms-on-wgan) ----
    wgan_aug_rows: list[dict] = []
    if args.no_transforms_on_wgan:
        log.info(f"[{args.case}] --no-transforms-on-wgan: skipping rot/shear copies of WGAN PNGs")
        wgan_real_paths = []  # short-circuit the loop below
    for image_id, study_id, src_path, active_pos in wgan_real_paths:
        # Read the source WGAN PNG (relative path from project root expected).
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning(f"[{args.case}] could not read {src_path}; skipping")
            continue
        for tag, fn in recipe:
            out_img = fn(img)
            new_id = f"{image_id}__{tag}"
            out_path = png_out / f"{new_id}.png"
            cv2.imwrite(str(out_path), out_img)
            wgan_aug_rows.append({
                "image_id": new_id,
                "study_id": study_id,
                "path": str(out_path),
                "source": "wgan_synth_aug",
                **{c: (1 if c == active_pos else 0) for c in label_cols},
                "transform": tag,
            })

    wgan_aug_df = pd.DataFrame(wgan_aug_rows, columns=trad_df.columns)
    log.info(f"[{args.case}] WGAN-aug rows written: {len(wgan_aug_df)} (= {len(wgan_real_paths)} × {len(recipe)})")

    # ---- 4) Combine and write ----
    parts = [trad_df]
    if len(wgan_real_df):
        parts.append(wgan_real_df[trad_df.columns])
    if len(wgan_aug_df):
        parts.append(wgan_aug_df[trad_df.columns])
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(case_out / "train_hybrid.csv", index=False)

    counts = {
        "real_plus_real_aug_rows": int(len(trad_df)),
        "wgan_real_rows": int(len(wgan_real_df)),
        "wgan_aug_rows": int(len(wgan_aug_df)),
        "total_rows": int(len(combined)),
        "transforms": [t[0] for t in recipe],
        **{f"total__{c}": int(combined[c].sum()) for c in label_cols},
    }
    (case_out / "manifest.json").write_text(json.dumps({
        "case": args.case,
        "variant_tag": args.variant_tag,
        "wgan_classes": list(args.wgan_classes),
        "max_wgan_per_class": args.max_wgan_per_class,
        "transforms_on_wgan": (not args.no_transforms_on_wgan),
        "positives": positives,
        "second_rotation_deg": int(case_body["second_rotation_deg"]),
        "counts": counts,
    }, indent=2))

    summary = (
        f"# Hybrid build summary — {args.case} / variant '{args.variant_tag}'\n\n"
        f"- WGAN classes used: {', '.join(args.wgan_classes)}\n"
        f"- Transforms: {', '.join(t[0] for t in recipe)}\n"
        f"- real + real-aug rows: {counts['real_plus_real_aug_rows']}\n"
        f"- WGAN gen rows:        {counts['wgan_real_rows']}\n"
        f"- WGAN aug rows:        {counts['wgan_aug_rows']}\n"
        f"- **total**:            **{counts['total_rows']}**\n\n"
        + "\n".join(f"- {c}: {counts[f'total__{c}']}" for c in label_cols)
    )
    (case_out / "summary.md").write_text(summary)
    log.info(
        f"[{args.case}] wrote train_hybrid.csv ({len(combined)} rows) -> {case_out/'train_hybrid.csv'}"
    )


if __name__ == "__main__":
    main()
