"""Phase 05: train one WGAN per minority class (paper §4.6.2; plan D5).

Per class entry in configs/wgan.yaml:
- read pool CSV (a case's train.csv) and filter to rows where the target class column == 1
- run WGAN training (weight-clip default, GP optional via --loss gp)
- write outputs/05_wgan/<case>/<class_slug>/{checkpoints/, samples/, log.csv, manifest.json}

Osteophytes is intentionally excluded (already abnormal-majority in case_3).

Usage:
    python scripts/05_train_wgan.py                                   # all 6 minority classes
    python scripts/05_train_wgan.py --classes-filter "Vertebral collapse" "Other lesions"
    python scripts/05_train_wgan.py --iterations 1000 --classes-filter "Vertebral collapse"   # smoke
    python scripts/05_train_wgan.py --loss gp                          # WGAN-GP escape hatch (full sweep)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wgan_dataset import WGANClassDataset
from src.train.wgan_trainer import WGANConfig, train_wgan
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--wgan", default="configs/wgan.yaml")
    ap.add_argument("--classes-filter", nargs="*", default=None,
                    help="Limit to specific class names, e.g. --classes-filter \"Vertebral collapse\"")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Override total_iterations (smoke runs)")
    ap.add_argument("--loss", default=None, choices=["weight_clip", "gp"],
                    help="Override loss kind. Default = configs/wgan.yaml's train.loss (weight_clip).")
    ap.add_argument("--out-tag", default="05_wgan",
                    help="Sub-dir under outputs/. Default 05_wgan.")
    args = ap.parse_args()

    base = load_config(args.config)
    wcfg = load_config(args.wgan)
    log = get_logger()
    set_seed(int(wcfg["train"].get("seed", 42)))

    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    iterations = args.iterations if args.iterations is not None else int(wcfg["train"]["total_iterations"])
    loss_kind = args.loss if args.loss is not None else wcfg["train"]["loss"]

    summaries: list[dict] = []
    for entry in wcfg["classes"]:
        if args.classes_filter and entry["name"] not in args.classes_filter:
            continue
        out_dir = out_root / entry["case"] / entry["slug"]

        ds = WGANClassDataset(
            pool_csv=entry["pool_csv"],
            target_class=entry["name"],
            image_size=int(wcfg["generator"]["image_size"]),
        )

        cfg = WGANConfig(
            class_name=entry["name"],
            class_slug=entry["slug"],
            case=entry["case"],
            pool_csv=entry["pool_csv"],
            out_dir=out_dir,
            latent_dim=int(wcfg["generator"]["latent_dim"]),
            image_size=int(wcfg["generator"]["image_size"]),
            g_base_channels=int(wcfg["generator"]["base_channels"]),
            out_channels=int(wcfg["generator"]["out_channels"]),
            d_base_channels=int(wcfg["critic"]["base_channels"]),
            batch_size=int(wcfg["train"]["batch_size"]),
            n_critic=int(wcfg["train"]["n_critic"]),
            loss=loss_kind,
            weight_clip_value=float(wcfg["train"]["weight_clip_value"]),
            gp_lambda=float(wcfg["train"]["gp_lambda"]),
            optim_clip=dict(wcfg["train"]["optimizer_clip"]),
            optim_gp=dict(wcfg["train"]["optimizer_gp"]),
            total_iterations=iterations,
            log_every=int(wcfg["train"]["log_every"]),
            snapshot_every=int(wcfg["train"]["snapshot_every"]),
            first_snapshot_at=int(wcfg["train"]["first_snapshot_at"]),
            sample_every=int(wcfg["train"]["sample_every"]),
            num_sample_images=int(wcfg["train"]["num_sample_images"]),
            num_workers=int(wcfg["train"]["num_workers"]),
            prefetch_factor=int(wcfg["train"]["prefetch_factor"]),
            persistent_workers=bool(wcfg["train"]["persistent_workers"]),
            pin_memory=bool(wcfg["train"]["pin_memory"]),
            amp=bool(wcfg["train"]["amp"]),
            seed=int(wcfg["train"].get("seed", 42)),
        )
        log.info(f"=== Training WGAN for {entry['name']} (pool={len(ds)}, loss={loss_kind}) ===")
        summary = train_wgan(cfg, ds, log)
        (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)

    if summaries:
        lines = ["# WGAN training summary", "",
                 "| class | case | n_pool | loss | iterations | n_snapshots |",
                 "|---|---|---|---|---|---|"]
        for s in summaries:
            lines.append(
                f"| {s['class_name']} | {s['case']} | {s['n_pool']} | {s['loss']} | "
                f"{s['total_iterations']} | {len(s['snapshots'])} |"
            )
        (out_root / "summary.md").write_text("\n".join(lines))
        (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
        log.info(f"Wrote {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
