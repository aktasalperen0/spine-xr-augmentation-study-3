"""Phase 05d: train a MONAI pixel-space DDPM per minority class on ROI patches.

Usage:
    python scripts/05d_train_diffusion.py                                  # all classes
    python scripts/05d_train_diffusion.py --classes-filter "Other lesions"
    python scripts/05d_train_diffusion.py --iterations 300 --classes-filter "Vertebral collapse"  # smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wgan_dataset import WGANClassDataset   # reused: per-class patch loader → [-1,1]
from src.train.diffusion_trainer import DiffusionConfig, train_diffusion
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--diffusion", default="configs/diffusion.yaml")
    ap.add_argument("--classes-filter", nargs="*", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--out-tag", default="05d_diffusion")
    args = ap.parse_args()

    base = load_config(args.config)
    dcfg = load_config(args.diffusion)
    log = get_logger()
    set_seed(int(dcfg["train"].get("seed", 42)))
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)
    iters = args.iterations if args.iterations is not None else int(dcfg["train"]["total_iterations"])

    summaries = []
    for entry in dcfg["classes"]:
        if args.classes_filter and entry["name"] not in args.classes_filter:
            continue
        ds = WGANClassDataset(entry["pool_csv"], entry["name"], image_size=int(dcfg["unet"]["image_size"]))
        cfg = DiffusionConfig(
            class_name=entry["name"], class_slug=entry["slug"], case=entry["case"],
            pool_csv=entry["pool_csv"], out_dir=out_root / entry["case"] / entry["slug"],
            image_size=int(dcfg["unet"]["image_size"]), in_channels=int(dcfg["unet"]["in_channels"]),
            channels=tuple(dcfg["unet"]["channels"]), attention_levels=tuple(dcfg["unet"]["attention_levels"]),
            num_res_blocks=int(dcfg["unet"]["num_res_blocks"]), num_head_channels=int(dcfg["unet"]["num_head_channels"]),
            num_train_timesteps=int(dcfg["scheduler"]["num_train_timesteps"]),
            beta_start=float(dcfg["scheduler"]["beta_start"]), beta_end=float(dcfg["scheduler"]["beta_end"]),
            batch_size=int(dcfg["train"]["batch_size"]), lr=float(dcfg["train"]["lr"]),
            total_iterations=iters, log_every=int(dcfg["train"]["log_every"]),
            sample_every=int(dcfg["train"]["sample_every"]), ckpt_every=int(dcfg["train"]["ckpt_every"]),
            first_ckpt_at=int(dcfg["train"]["first_ckpt_at"]), num_sample_images=int(dcfg["train"]["num_sample_images"]),
            num_workers=int(dcfg["train"]["num_workers"]), amp=bool(dcfg["train"]["amp"]),
            seed=int(dcfg["train"].get("seed", 42)),
        )
        log.info(f"=== DDPM for {entry['name']} (pool={len(ds)}) ===")
        s = train_diffusion(cfg, ds, log)
        (cfg.out_dir / "manifest.json").write_text(json.dumps(s, indent=2))
        summaries.append(s)

    if summaries:
        (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
        log.info(f"Wrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
