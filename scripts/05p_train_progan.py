"""Phase 05p: train a Progressive-GAN per minority class on ROI patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wgan_dataset import WGANClassDataset
from src.train.progan_trainer import ProGANConfig, train_progan
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--progan", default="configs/progan.yaml")
    ap.add_argument("--classes-filter", nargs="*", default=None)
    ap.add_argument("--fade-iters", type=int, default=None)
    ap.add_argument("--stab-iters", type=int, default=None)
    ap.add_argument("--out-tag", default="05p_progan")
    args = ap.parse_args()

    base = load_config(args.config)
    pcfg = load_config(args.progan)
    log = get_logger()
    set_seed(int(pcfg["train"].get("seed", 42)))
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for entry in pcfg["classes"]:
        if args.classes_filter and entry["name"] not in args.classes_filter:
            continue
        ds = WGANClassDataset(entry["pool_csv"], entry["name"], image_size=128)
        cfg = ProGANConfig(
            class_name=entry["name"], class_slug=entry["slug"], case=entry["case"],
            pool_csv=entry["pool_csv"], out_dir=out_root / entry["case"] / entry["slug"],
            latent_dim=int(pcfg["model"]["latent_dim"]), out_channels=int(pcfg["model"]["out_channels"]),
            max_depth=int(pcfg["model"]["max_depth"]),
            fade_iters=args.fade_iters if args.fade_iters is not None else int(pcfg["train"]["fade_iters"]),
            stab_iters=args.stab_iters if args.stab_iters is not None else int(pcfg["train"]["stab_iters"]),
            gp_lambda=float(pcfg["train"]["gp_lambda"]), drift_eps=float(pcfg["train"]["drift_eps"]),
            lr=float(pcfg["train"]["lr"]), betas=tuple(pcfg["train"]["betas"]),
            batch_per_res={int(k): int(v) for k, v in pcfg["train"]["batch_per_res"].items()},
            log_every=int(pcfg["train"]["log_every"]), sample_every=int(pcfg["train"]["sample_every"]),
            num_sample_images=int(pcfg["train"]["num_sample_images"]),
            num_workers=int(pcfg["train"]["num_workers"]), seed=int(pcfg["train"].get("seed", 42)),
        )
        log.info(f"=== ProGAN for {entry['name']} (pool={len(ds)}) ===")
        s = train_progan(cfg, ds, log)
        (cfg.out_dir / "manifest.json").write_text(json.dumps(s, indent=2))
        summaries.append(s)

    if summaries:
        (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
        log.info(f"Wrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
