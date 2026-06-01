"""Phase 07d: generate synthetic ROI patches from a trained MONAI DDPM checkpoint.

Writes PNGs + a manifest.csv with the same schema as the WGAN generator (07), so the CV driver
(scripts/03_train_cv.py --synth-manifest) can append them to train folds.

Usage:
    python scripts/07d_generate_diffusion.py \
        --checkpoint outputs/05d_diffusion/case_4/other_lesions/checkpoints/iter_008000.pt \
        --n-samples 800 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.train.diffusion_trainer import import_monai_generative, sample_images
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

ALL_CLASSES = [
    "Osteophytes", "Disc space narrowing", "Other lesions", "Foraminal stenosis",
    "Surgical implant", "Spondylolysthesis", "Vertebral collapse", "No finding",
]


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--out-tag", default="07d_diffusion_generated")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    base = load_config(args.config)
    log = get_logger()
    set_seed(args.seed)
    device = _device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    class_name = args.class_name or cfg["class_name"]
    if class_name not in ALL_CLASSES:
        raise ValueError(f"{class_name!r} not in {ALL_CLASSES}")
    slug = cfg.get("class_slug") or class_name.lower().replace(" ", "_")

    DiffusionModelUNet, DDPMScheduler = import_monai_generative()
    unet = DiffusionModelUNet(
        spatial_dims=2, in_channels=int(cfg["in_channels"]), out_channels=int(cfg["in_channels"]),
        channels=tuple(cfg["channels"]), attention_levels=tuple(cfg["attention_levels"]),
        num_res_blocks=int(cfg["num_res_blocks"]), num_head_channels=int(cfg["num_head_channels"]),
    ).to(device)
    unet.load_state_dict(ckpt["unet_state"]); unet.eval()
    scheduler = DDPMScheduler(num_train_timesteps=int(cfg["num_train_timesteps"]),
                              schedule="linear_beta",
                              beta_start=float(cfg["beta_start"]), beta_end=float(cfg["beta_end"]))

    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag / slug
    pngs = out_root / "pngs"; pngs.mkdir(parents=True, exist_ok=True)
    iter_tag = f"iter_{int(ckpt.get('iter', 0)):06d}"
    size = int(cfg["image_size"]); inch = int(cfg["in_channels"])

    rows = []
    written = 0
    log.info(f"generating {args.n_samples} DDPM samples for {class_name!r} -> {pngs}")
    while written < args.n_samples:
        bs = min(args.batch_size, args.n_samples - written)
        imgs = sample_images(unet, scheduler, bs, size, inch, device)  # [-1,1]
        x = imgs[:, 0] if imgs.size(1) == 1 else imgs.mean(1)
        x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for i in range(bs):
            iid = f"ddpm_{slug}_{iter_tag}_{written+i:06d}"
            p = pngs / f"{iid}.png"; cv2.imwrite(str(p), x[i])
            row = {"image_id": iid, "study_id": f"ddpm_synth_{slug}", "path": str(p), "source": "ddpm_synth"}
            for c in ALL_CLASSES:
                row[c] = 1 if c == class_name else 0
            row["transform"] = "synth_real"; row["synth_iter"] = int(ckpt.get("iter", 0))
            rows.append(row)
        written += bs
        log.info(f"  {written}/{args.n_samples}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_root / "manifest.csv", index=False)
    (out_root / "manifest.json").write_text(json.dumps(
        {"class_name": class_name, "class_slug": slug, "checkpoint": args.checkpoint,
         "n_samples": args.n_samples, "image_size": size}, indent=2))
    log.info(f"wrote manifest {out_root/'manifest.csv'}")


if __name__ == "__main__":
    main()
